# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
DSL prefill SDPA kernel — classic pipeline, per-tensor FP8 (E4M3 / E5M2), d=128, SM100.

Classic 2-sub-tile pipeline (TILES_Q=2, two softmax warpgroups, four correction
warps, persistent try_cancel scheduler).  Per-tensor descales are LOADED
IN-KERNEL from 1-element device tensors and folded into scale_softmax_log2 /
o_scale_fused (Rule 3 — no host readback); o_scale_fused feeds the correction
epilogue's threshold_beta.  descale_s/scale_s are accepted and ignored (P is
cast unscaled; unsupported knobs on this cell).
Optional output dtype (DTYPE_O): FP8 in → E4M3 / E5M2 / BF16 / FP16 O.  Shares the
f16 / mxfp8 SM100 d=128 layout plus the FP8-on-Blackwell K-path:
  1. **cga2-only, STAGES_KV=4** (config; FP8 BPE=1 → 128..160 KiB SMEM).
  2. **512-col TMEM** with per-sub-tile stats on the S_acc heads (col 0/128;
     FP8 P is 4:1-packed at the S_acc tails 96/224, so the heads are free).
  3. **Manual row-max** (no LDTM.STAT).
  4. **FP8 MMA uses the Blackwell K=32 QMMA path** — idesc ``k_dim=0`` +
     ``TILE_K_HW=32`` (config).  ``NUM_KPHASES_PV`` derives from
     ``CFG.TILE_K_HW_BMM2`` (→ 4 k-steps at TILE_K_HW=32).  Confirmed by the
     cuDNN f8 reference (UTCMMA_TILE_K=32, BMM_XMMAS_K=4, kind::f8f6f4).

THD / varlen (``CFG.THD_VARLEN=1``) follows the device-built-metadata +
plan-time-envelope design (``write_thd_meta``, issue #552 / PRs #606, #608)
used by the f16 kernels — FP8 is element-addressed (no block-scale SF), so it
rides the same path: packed ``[1,T,H,D]`` with DYNAMIC token extents (the
packed totals never reach the host), the setup kernel builds the
[kv|cu_q|cu_k] metadata + per-batch O TMA descriptors device-side, and the
launch grid is the plan-time envelope (units past a sequence's live tiles
decode the batch == n_batch sentinel and drain without loads or stores).
Ragged Stats ride the caller's declared layout (token-major (T, H) or
head-major); the amax_o atomicMax is gated on live rows. Dense path
byte-identical.
"""

import os
import sys
from functools import lru_cache
from typing import Callable, Optional, Tuple


from cutlass.base_dsl.typing import Pointer  # was the legacy DSL Pointer pre-DKG-bump
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.primitives import vote_sync, VoteSync
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass._mlir.dialects import arith

import cutlass
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver  # noqa: F401  (cute.compile pulls cuda)

from dataclasses import dataclass

from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d128

# The template loader (api_dsl._load_kernel_module) injects FROST_TEMPLATE_PARAMS
# as a module global before this body runs; the default keeps direct import usable.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
CFG, _TMA = make_cfg_d128(PARAMS)
if PARAMS.softmax_f16:
    raise ValueError("prefill_d128_fp8_sm100: softmax_f16 is served by the SM107 sibling only (softmax_precision knob domain)")
Cfg = type(CFG)
TMA_QK_ITERS = _TMA.QK_ITERS
TMA_VO_ITERS = _TMA.VO_ITERS
TMA_QK_GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _TMA.VO_GRANU_ELEMS

# O's TMA box follows O's swizzle, not V's (under cga2 V may drop to a narrower swizzle).
# Sized in BPE_O (output dtype) — O may be written at BF16/FP16 when DTYPE_O != DTYPE_QKV.
TMA_O_GRANU_ELEMS_HOST = CFG.O_SWZ_BYTES // CFG.BPE_O
TMA_O_ITERS_HOST = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES

from typing import NamedTuple

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
    read_tile_id_arrive,
    SCHED_NATURAL,
    SCHED_LPT,
    SCHED_LPT_L2,
)
from cudnn.frost.tile_dsl.pointwise import (
    # SM100: no LDTM.STAT — the MASK_NONE fast path uses manual tcgen05_ld +
    # row_max_reduction (see _softmax_kv_body); tmem_load_max_reduction_tile
    # is not imported.
    row_reduction_pair,
    row_max_reduction,
    vec_scale_pair,
    fp32_to_fp8_pack,
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

# Storage dtype + MMA kind dispatch keyed off CFG.DTYPE_QKV.
if CFG.DTYPE_QKV == 0:
    STORAGE_DTYPE = cutlass.Float8E4M3FN
    P_STORAGE_DTYPE = cutlass.Float8E4M3FN
    MMA_KIND = nvvm.Tcgen05MMAKind.F8F6F4
elif CFG.DTYPE_QKV == 1:
    STORAGE_DTYPE = cutlass.Float8E5M2
    P_STORAGE_DTYPE = cutlass.Float8E5M2
    MMA_KIND = nvvm.Tcgen05MMAKind.F8F6F4
else:
    raise ValueError(
        f"prefill_sdpa_fp8: DTYPE_QKV={CFG.DTYPE_QKV} not supported "
        f"(expected 0=E4M3 or 1=E5M2; use the DSL backend with --bf16/--fp16 "
        f"for the f16 kernel, or extend this dispatch for new fp8 variants)"
    )

# P -> fp8 cast bias (BAKED constant — NOT cuDNN's Scale_S; that pair is
# accepted and ignored). P is quantized as P * 2**P_CAST_LOG2_SCALE: the
# lazy-rescale skip bounds P by 2**RESCALE_THRESHOLD (4.0 for fp8, see
# config_sm100.rescale_threshold), so the cast peaks at 2^(4+4) = 256 < 448
# (e4m3 max) — no saturation — while flat-row entries (P ~ 1/S) sit four
# binades above e4m3's subnormal cliff (quantization stays normal out to
# S ~ 2^13). The bias rides the exp2 argument, so total_sum accumulates in
# the SAME 2^4-scaled units and the O normalization (O_acc / total_sum)
# cancels it exactly; only the LSE subtracts the constant (and the sink
# denominator term is scaled up to match). Invariant:
# RESCALE_THRESHOLD + P_CAST_LOG2_SCALE <= log2(448).
P_CAST_LOG2_SCALE = 4.0

# DTYPE_O is independent of DTYPE_QKV (mirrors C++ Cfg::DTYPE_O — defaults to
# DTYPE_QKV but may be promoted to BF16/FP16 so a downstream consumer skips a
# dequant).  BPE_O ∈ {1, 2}.  The epilogue keeps the bit-identical hand-rolled
# 16:4 FP8 pack for DTYPE_O ∈ {0, 1}; BF16/FP16 take a generic cast + swizzled
# store (see _correction_warp_group).
if CFG.DTYPE_O == 0:
    OUT_STORAGE_DTYPE = cutlass.Float8E4M3FN
elif CFG.DTYPE_O == 1:
    OUT_STORAGE_DTYPE = cutlass.Float8E5M2
elif CFG.DTYPE_O == 2:
    OUT_STORAGE_DTYPE = cutlass.BFloat16
elif CFG.DTYPE_O == 3:
    OUT_STORAGE_DTYPE = cutlass.Float16
else:
    raise ValueError(f"prefill_sdpa_fp8: DTYPE_O={CFG.DTYPE_O} not supported (expected 0=E4M3 / 1=E5M2 / 2=BF16 / 3=FP16)")


from cudnn.sdpa.fwd.kernels._common_sm100 import (
    make_split_helpers,
    Bars,
    KvLoopBounds,
    make_classic_bars,
    compute_kv_loop_bounds,
    lpt_tile_coords,
    make_sdpa_helpers,
)

CGA_SIZE = CFG.CGA_M * CFG.CGA_N

CTA_GROUP_KIND = nvvm.CTAGroup.CTA_2 if CFG.CTA_MMA == 2 else nvvm.CTAGroup.CTA_1

# K (split along seq rows) and V (split along d_v cols) shrink by CTA_MMA;
# Q / O are per-CTA full.  At cga2 leader's expect_tx = per-CTA bytes × CTA_MMA.
qBufferElems = CFG.TILE_M * CFG.TILE_K
kBufferElems = CFG.TILE_N * CFG.TILE_K // CFG.CTA_MMA
vBufferElems = CFG.TILE_O * CFG.TILE_N // CFG.CTA_MMA
oBufferElems = CFG.TILE_M * CFG.TILE_O

qTmaTransactionBytes = qBufferElems * CFG.BPE * CFG.CTA_MMA
kTmaTransactionBytes = kBufferElems * CFG.BPE * CFG.CTA_MMA
vTmaTransactionBytes = vBufferElems * CFG.BPE * CFG.CTA_MMA


CGA_TILE_M = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA


# SM100 llama is always cga2 → LPT reverse-row count in CGA-tile units.
_sdpa_h = make_sdpa_helpers(CFG, lpt_q_tiles_in_cga_units=True)
_decode_initial = _sdpa_h.decode_initial
_decode_payload = _sdpa_h.decode_payload
_bounds_for_tile = _sdpa_h.bounds_for_tile
_resolve_seqlen_kv = _sdpa_h.resolve_seqlen_kv
_resolve_seqlen_q = _sdpa_h.resolve_seqlen_q

# THD / varlen — shared helpers (FP8 element-addressed like f16, per-tensor
# dequant scales, no block-scale SF).  Gated by CFG.THD_VARLEN (folds out:
# _thd_tma_offsets is (0, 0, batch_idx) dense — TMA coords byte-identical).
# TILES_Q=2: q_seq_off applies to BOTH Q slabs + both O-store slabs.
# seq_kv_lens overloaded as the THD metadata buffer (int32 len 3B+2):
#   [0..B-1]=seq_kv_lens  [B..2B]=cu_q(B+1)  [2B+1..3B+1]=cu_k(B+1)
# The setup kernel builds it DEVICE-side from the caller's length tensors and
# the adapter launches the plan-time envelope grid (issue #552) — no length
# ever reaches the host.
from cudnn.sdpa.fwd.kernels.thd_sm100 import build_thd_meta_o_kv_descs_kernel as _build_thd_meta_o_kv_descs_kernel, TENSOR_MAP_QWORDS

_TENSOR_MAP_QWORDS = TENSOR_MAP_QWORDS
_dispatch_decode_initial = _sdpa_h.dispatch_decode_initial
_dispatch_decode_payload = _sdpa_h.dispatch_decode_payload
_thd_tma_offsets = _sdpa_h.thd_tma_offsets

# === PackGQA ===
HEADS_PER_TILE = CFG.QH_PER_KH if CFG.PACK_GQA else 1
TOKENS_PER_TILE = CFG.TILE_M // HEADS_PER_TILE

# === KV split ===
#
# Mechanics live in _common_sm100.make_split_helpers, shared with the other
# SM100 prefill flavors: each Q tile's KV loop range is cut into SPLIT_KV
# contiguous chunks, each run as its own persistent tile, and each writing a
# normalized partial O + its own LSE into a split-major workspace that
# split_combine_sm100 folds with the exact log-sum-exp identity.  At
# SPLIT_KV == 1 every closure folds away and this is the classic kernel.


@cute.jit
def _bounds_for_tile_uniform(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, seq_q_lens_tensor, batch_idx, qh_per_kh: int = 1):
    """Uniform bounds signature for make_split_helpers.

    This flavor has no dead-Q-tile trim, so the trailing two args are
    accepted and ignored — callers pass None for them.
    """
    return _bounds_for_tile(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, qh_per_kh)


_split_h = make_split_helpers(
    CFG,
    bounds_for_tile=_bounds_for_tile_uniform,
    dispatch_decode_initial=_dispatch_decode_initial,
    dispatch_decode_payload=_dispatch_decode_payload,
)
SPLIT_KV = _split_h.SPLIT_KV
MAY_BE_EMPTY = _split_h.MAY_BE_EMPTY
_decode_initial_split = _split_h.decode_initial_split
_decode_payload_split = _split_h.decode_payload_split
_bounds_for_tile_split = _split_h.bounds_for_tile_split
_nomask_range_split = _split_h.nomask_range_split
_partial_batch = _split_h.partial_batch


@dataclass(frozen=True)
class KernelTmemLayout:
    """Column offsets for the classic 2-sub-tile SDPA pipeline (FP8).

    Stats sit outside S_acc/O ranges so BMM1's next-iter write doesn't clobber them.
    P aliases the tail of S_acc[qs] (BMM1 finishes before P is written).
    """

    # Blackwell SM10.0 512-col TMEM cap.
    TOTAL_COLS: int = 512

    S0_OFF: int = 0
    S1_OFF: int = 128

    O0_OFF: int = 256
    O1_OFF: int = 384

    # P (FP8 4:1 packed, 32 cols each) aliases S_acc tail at 96 / 224.
    P0_OFF: int = 96
    P1_OFF: int = 224

    # SM100: stats ride the head of sub-tile qs's S_acc slot (col 0 / 128); FP8
    # P is 4:1-packed at the tails (96 / 224), so the heads are free after S is
    # read.  stats_off = STATS_OFF + qs*STATS_STRIDE.
    STATS_OFF: int = 0
    STATS_STRIDE: int = 128


LAYOUT = KernelTmemLayout()


# === Kernel ===


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
    # 1-element fp32 DEVICE scales (Rule 3): the descale/scale factors are
    # loaded and folded HERE — scale_softmax_log2 arrives as attn_scale*log2(e)
    # and o_scale_fused as 1.0; no host readback exists anywhere. descale_s /
    # scale_s are NOT taken: this kernel casts P unscaled, so their values are
    # accepted by the adapter and ignored (unsupported knobs on this cell).
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

    # Q/K/V/O order matters: Tcgen05SmemDesc.build truncates start_address past
    # ~256 KiB so high-offset tiles alias to low SMEM in BMM2.
    sQ_raw = cutlass.Array(STORAGE_DTYPE, CFG.TILES_Q * qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sK_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * kBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sV_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * vBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sO_raw = cutlass.Array(OUT_STORAGE_DTYPE, CFG.TILES_Q * oBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)

    sQ = SmemTile(
        base=sQ_raw,
        elems_per_stage=qBufferElems,
        stages=CFG.TILES_Q,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QKO,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_QK_GRANU_ELEMS,
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
    )
    sV = SmemTile(
        base=sV_raw,
        elems_per_stage=vBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_PV,
        stride_byte_offset=STRIDE_BYTE_OFFSET_PV,
        layout=SMEM_LAYOUT_V,
        # V split along d_v under cga2 → tma_loads_per_tile shrinks by CTA_MMA.
        tma_loads_per_tile=TMA_VO_ITERS // CFG.CTA_MMA,
        tma_granu_elems=TMA_VO_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_N * TMA_VO_GRANU_ELEMS,
    )
    sO = SmemTile(
        base=sO_raw,
        elems_per_stage=oBufferElems,
        stages=CFG.TILES_Q,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=SMEM_LAYOUT_QKO,
        tma_loads_per_tile=TMA_O_ITERS_HOST,
        tma_granu_elems=TMA_O_GRANU_ELEMS_HOST,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_O_GRANU_ELEMS_HOST,
    )

    bars = make_classic_bars(CFG)

    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)

    # tile_id_smem stride 8 Int32/stage: 16 B try_cancel.async payload + 16 B padding.
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

    # CGA-aware mbar init counts; at CTA_MMA=1 collapse to cga1 baselines.
    READ_TILE_ARRIVERS_TOTAL = ((CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS) + CFG.CORRECTION_WARPS + 1 + 1) * CGA_SIZE + (CFG.CGA_M // CFG.CTA_MMA)

    if warp_idx == 0:
        if nvvm.elect_sync():
            # range_constexpr → Python-int loop var (required for the
            # mb_bmm2_ready tuple-init lookup).  Bounds are small —
            # unroll is free.
            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                bars.mb_q_full[qs].init()
                bars.mb_q_empty[qs].init()
                bars.mb_bmm1_done[qs].init()
                bars.mb_bmm2_done[qs].init()
                bars.mb_stat_full[qs].init()
                bars.mb_stat_empty[qs].init()
                bars.mb_stats_read[qs].init()
                bars.mb_o_full[qs].init()
                bars.mb_o_empty[qs].init()
                for c in cutlass.range_constexpr(CFG.N_BMM2_CHUNKS):
                    bars.mb_bmm2_ready[qs * CFG.N_BMM2_CHUNKS + c].init()
            for ks in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_k_full[ks].init()
                bars.mb_k_empty[ks].init()
                bars.mb_v_full[ks].init()
                bars.mb_v_empty[ks].init()
            for s in range(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), CFG.ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), READ_TILE_ARRIVERS_TOTAL)
            bars.mb_tmem_dealloc.init()
            bars.mb_empty_mainloop.init()

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    # P4 cluster fence — gates cga2 cross-CTA arrives on peer init.
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        cga_arrive()
        cga_wait()

    # const_expr wrap — without it the DSL stages if/else and post-branch reads NameError.
    cta_id_x = cute.arch.block_idx_in_cluster() if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    cta_in_pair = (cta_id_x & cutlass.Int32(1)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    leader_cta_id = (cta_id_x & cutlass.Int32(~1 & 0xFFFFFFFF)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    mcast_mask = (cutlass.Int32(3) << leader_cta_id) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    # tma_mcast_mask = 1 << cta_rank — cta_group::2 routing strips bit-24 onto leader.
    tma_mcast_mask = (cutlass.Int16(1) << cta_in_pair) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int16(0)
    is_leader = cta_in_pair == cutlass.Int32(0)

    # Device-scale fold: every thread loads the four 1-element fp32 scales
    # (same addresses -> L2 broadcast, negligible) and folds them into the
    # softmax scale and the output scale.
    _dsc_q = cutlass.Float32(cutlass.make_array_view(descale_q_t)[0])
    _dsc_k = cutlass.Float32(cutlass.make_array_view(descale_k_t)[0])
    _dsc_v = cutlass.Float32(cutlass.make_array_view(descale_v_t)[0])
    _scl_o = cutlass.Float32(cutlass.make_array_view(scale_o_t)[0])
    scale_softmax_log2 = scale_softmax_log2 * _dsc_q * _dsc_k
    o_scale_fused = o_scale_fused * _dsc_v * _scl_o

    if warp_idx >= CFG.SOFTMAX_WG0_BASE and warp_idx < CFG.SOFTMAX_WG0_BASE + CFG.SOFTMAX_WG_WARPS:
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _softmax_warp_group(
            sub_tile_id=0,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            scale_log2=scale_softmax_log2,
            tmem_ptr_i32=tmem_ptr_i32,
            sQ=sQ,
            bars=bars,
            sched=sched,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            leader_cta_id=leader_cta_id,
            cta_in_pair=cta_in_pair,
            qh_per_kh=qh_per_kh,
        )

    elif warp_idx >= CFG.SOFTMAX_WG1_BASE and warp_idx < CFG.SOFTMAX_WG1_BASE + CFG.SOFTMAX_WG_WARPS:
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _softmax_warp_group(
            sub_tile_id=1,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            scale_log2=scale_softmax_log2,
            tmem_ptr_i32=tmem_ptr_i32,
            sQ=sQ,
            bars=bars,
            sched=sched,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            leader_cta_id=leader_cta_id,
            cta_in_pair=cta_in_pair,
            qh_per_kh=qh_per_kh,
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
            qh_per_kh=qh_per_kh,
        )

    # cga2 non-leader runs quiet body (alloc+dealloc only); cga1 folds to full path.
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
                    qh_per_kh=qh_per_kh,
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
                qh_per_kh=qh_per_kh,
            )

    elif warp_idx == CFG.TMALDG_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        # Warm descriptor cache.
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        _tmaldg_warp_group(
            tma_q_desc=tma_q_desc,
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            sQ=sQ,
            sK=sK,
            sV=sV,
            bars=bars,
            sched=sched,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            o_desc_words=o_desc_words,
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
            qh_per_kh=qh_per_kh,
            seqlen_kv=seqlen_kv,
        )

    else:  # warp_idx == CFG.SCHED_WARP_ID
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        # try_cancel.multicast::cluster::all — only CGA's (0,0,0) CTA issues.
        is_cga_first_cta = cta_id_x == cutlass.Int32(0)
        scheduler_warp_loop(sched, CFG.SCHEDULER_STAGES, is_cga_first_cta)


# === TMA-LDG warp ===


@cute.jit
def _tmaldg_warp_group(
    tma_q_desc,
    tma_k_desc,
    tma_v_desc,
    sQ,
    sK,
    sV,
    bars,
    sched,
    seqlen_q,
    seqlen_kv,
    seq_kv_lens_tensor,
    o_desc_words,
    n_q_supers,
    n_qh,
    n_batch,
    qh_per_kh,
    is_leader,
    cta_in_pair,
    tma_mcast_mask,
):
    """Unified TMA-LDG warp — cga1/cga2 × MASK_NONE/PADDED/CAUSAL/SWA.

    Spill-free in OTHER_REGS (40) via the q_row_base trick: pre-multiplying
    q_super_idx into q_row_base after each scheduler decode keeps the LDS.128
    result in uniform registers instead of getting STL'd to local stack.
    """
    q_empty_phase = cutlass.Int32(1)
    kv_state = PipelineState.start(phase=1)

    tma_q = GmemTileTma(tma_q_desc)
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD: K/V ride the setup kernel's RUNTIME descriptors (o_desc_words
        # slots n_batch+1 / n_batch+2), whose seq extent is clamped to the
        # packed KV total cu_k[B]. The last sequence's tile steps past that
        # total into the buffer's capacity tail; through the clamped
        # descriptors those rows are TMA-OOB and land as EXACT ZEROS — a NaN
        # tail (test_mhas_v2 poisons it) would otherwise wipe the tile via
        # BMM2's P·V (0 · NaN == NaN) and, on cc10.3, the pre-mask fused-LDTM
        # row-max. Same closure shape as the dense GmemTileTma, so every load
        # site below is branch-free.
        _k_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(1)) * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        _v_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(2)) * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        tma_k = lambda *coords: tma_slice_runtime_desc(_k_rt_ptr, *coords)  # noqa: E731
        tma_v = lambda *coords: tma_slice_runtime_desc(_v_rt_ptr, *coords)  # noqa: E731
    else:
        tma_k = GmemTileTma(tma_k_desc)
        tma_v = GmemTileTma(tma_v_desc)

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
        qh_per_kh,
        seqlen_kv,
    )
    # GQA: K/V are indexed by kv-head, not Q-head; with PackGQA the decoded
    # head_idx is the PACKED head (Q head base = head_idx * G) and q_row_base is
    # in TOKEN units (rows // G).
    q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
    kv_head_idx = cute.arch.make_warp_uniform(head_idx if cutlass.const_expr(CFG.PACK_GQA) else head_idx // qh_per_kh)
    q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE))
    q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)

    if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    elif cutlass.const_expr(CFG.MASK_FLAGS == 0):
        kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
    else:
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
        bounds_init = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, None, None, split_idx, CFG.QH_PER_KH)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    # The DSL TMA descriptor coord is in ELEMENTS, not bytes (C++ uses UINT8 desc).
    K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_N // CFG.CTA_MMA)
    V_COL_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_O // CFG.CTA_MMA)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            pass
        else:
            # Prologue interleave: Q[0] → K[first] → Q[1] → V[first] → mainloop.
            kv_row_base = kv_left * CFG.TILE_N

            bars.mb_q_empty[0].wait(q_empty_phase)
            if cutlass.const_expr(CFG.CTA_MMA == 2):
                bars.mb_q_full[0].arrive(n_bytes=qTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            else:
                bars.mb_q_full[0].arrive(n_bytes=qTmaTransactionBytes, pred=nvvm.elect_sync())
            tma_load_tile(
                sQ[0],
                tma_q(
                    cutlass.Int32(0),
                    q_head_idx,
                    q_row_base + cutlass.Int32(0 * TOKENS_PER_TILE) + q_seq_off,
                    tma_batch,
                ),
                bars.mb_q_full[0].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )

            bars.mb_k_empty[kv_state.idx].wait(kv_state.phase)
            if cutlass.const_expr(CFG.CTA_MMA == 2):
                bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            else:
                bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=nvvm.elect_sync())
            tma_load_tile(
                sK[kv_state.idx],
                # THD: prologue K load MUST apply the per-sequence kv offset
                # (kv_seq_off) + packed batch coord (tma_batch), like the mainloop
                # K load + the V loads (dense-identity fold at THD_VARLEN=0).
                tma_k(cutlass.Int32(0), kv_head_idx, kv_row_base + K_ROW_OFFSET_PEER + kv_seq_off, tma_batch),
                bars.mb_k_full[kv_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )

            bars.mb_q_empty[1].wait(q_empty_phase)
            if cutlass.const_expr(CFG.CTA_MMA == 2):
                bars.mb_q_full[1].arrive(n_bytes=qTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            else:
                bars.mb_q_full[1].arrive(n_bytes=qTmaTransactionBytes, pred=nvvm.elect_sync())
            tma_load_tile(
                sQ[1],
                tma_q(
                    cutlass.Int32(0),
                    q_head_idx,
                    q_row_base + cutlass.Int32(1 * TOKENS_PER_TILE) + q_seq_off,
                    tma_batch,
                ),
                bars.mb_q_full[1].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            q_empty_phase = q_empty_phase ^ 1

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

            for kv_loop in cutlass.range(kv_left + cutlass.Int32(1), kv_right, 1, unroll=1):
                kv_row_base = kv_loop * CFG.TILE_N

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

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0))).load())
        nxt_hb = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1))).load())
        nxt_v = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load())
        q_super_idx, head_idx, batch_idx, split_idx = _decode_payload_split(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
            qh_per_kh,
            seqlen_kv,
        )
        q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
        kv_head_idx = cute.arch.make_warp_uniform(head_idx if cutlass.const_expr(CFG.PACK_GQA) else head_idx // qh_per_kh)
        # q_row_base after decode drives ptxas R2UR (keeps nxt_q live before back-edge).
        q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE))
        q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV > 1):
            kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
        elif cutlass.const_expr(CFG.MASK_FLAGS != 0):
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
            bounds_next = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, None, None, split_idx, CFG.QH_PER_KH)
            kv_left = bounds_next.left
            kv_right = bounds_next.right

    # cga2: drain trailing empty mbar arrives before SMEM teardown.
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        for _qs in cutlass.range_constexpr(CFG.TILES_Q):
            bars.mb_q_empty[_qs].wait(q_empty_phase)
        q_empty_phase = q_empty_phase ^ cutlass.Int32(1)
        for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
            bars.mb_k_empty[kv_state.idx].wait(kv_state.phase)
            bars.mb_v_empty[kv_state.idx].wait(kv_state.phase)
            kv_state = advance(kv_state, CFG.STAGES_KV)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)


# === TMA-STG warp ===


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
    seqlen_kv,
    qh_per_kh,
):
    """Persistent O-store warp; tiles claimed via scheduler's try_cancel.async."""
    o_full_phase = cutlass.Int32(0)

    tma_o = GmemTileTma(tma_o_desc)

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
        qh_per_kh,
        seqlen_kv,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        q_row_base = q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE)
        q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
        # KV split: partials are stacked split-major on the workspace BATCH axis
        # (extent B*SPLIT_KV), so the store needs no new descriptor — only a
        # shifted batch coord.  Folds to batch_idx at SPLIT_KV == 1.
        o_batch = _partial_batch(batch_idx, split_idx, n_batch)

        for qs in cutlass.range_constexpr(CFG.TILES_Q):
            bars.mb_o_full[qs].wait(o_full_phase)

            # O TMA params follow O's swizzle, not V's (V and O swizzles may differ).
            if cutlass.const_expr(CFG.THD_VARLEN):
                # THD: store each Q slab through this batch's pre-built descriptor
                # (base at the sequence's packed row, seq extent = S_q_b → a box
                # past S_q_b is OOB-clipped).  q_row coord is sequence-local; the
                # batch coord collapses to 0.  Both slabs share one descriptor.
                # (split_kv is dense-only — the config backstop rejects THD —
                # so o_batch never applies here.)
                # DEAD unit (batch == n_batch, over-launched envelope grid —
                # issue #552): no O rows exist and descriptor slot n_batch is
                # never built, so skip the store; the barrier protocol below
                # still runs.
                if batch_idx < n_batch:
                    o_desc_ptr = (o_desc_words.iterator.raw_ptr() + batch_idx * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
                    o_slice = tma_slice_runtime_desc(o_desc_ptr, cutlass.Int32(0), head_idx, q_row_base + cutlass.Int32(qs * CFG.TILE_M), cutlass.Int32(0))
                    tma_store_tile(sO[qs], o_slice)
            else:
                tma_store_tile(
                    sO[qs],
                    tma_o(cutlass.Int32(0), q_head_idx, q_row_base + cutlass.Int32(qs * TOKENS_PER_TILE), o_batch),
                )

            tma_store_commit()
            tma_store_wait(0)

            bars.mb_o_empty[qs].arrive()

        o_full_phase = o_full_phase ^ 1

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0))).load()
        nxt_hb = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1))).load()
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        q_super_idx, head_idx, batch_idx, split_idx = _decode_payload_split(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
            qh_per_kh,
            seqlen_kv,
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


# Swizzle byte width → nvvm.Tcgen05SmemSwizzle enum.
_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
SMEM_LAYOUT_Q = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_K = _SWZ_ENUM[CFG.K_SWZ_BYTES]
SMEM_LAYOUT_V = _SWZ_ENUM[CFG.V_SWZ_BYTES]
SMEM_LAYOUT_O = _SWZ_ENUM[CFG.O_SWZ_BYTES]
SMEM_LAYOUT_QKO = SMEM_LAYOUT_Q

# O SMEM Swizzle preset (B, 4, 3): Swz128B=(3,4,3) etc.  Third param is XOR shift, NOT B.
_O_SWZ_B = {128: 3, 64: 2, 32: 1}[CFG.O_SWZ_BYTES]
_O_SMEM_SWIZZLE = cutlass.Swizzle(_O_SWZ_B, 4, 3)
LEADING_BYTE_OFFSET_QK = 0
# SM100/Blackwell FP8: K=32 QMMA path.  Derive from
# CFG.TILE_K_HW_BMM2 (=32) so NUM_KPHASES_PV = TILE_N/32 = 4 k-steps; a hardcoded
# 64 would issue only 2 steps and silently drop half of V's K (cf. rules §16).
_MMA_K_FP8 = CFG.TILE_K_HW_BMM2
STRIDE_BYTE_OFFSET_QK = 8 * CFG.Q_SWZ_BYTES

# leading_byte_offset = 0 when (TILE_O/CTA_MMA)/8 <= 8 else TILE_N*V_SWZ_BYTES.
_CORE_MATRIX_ROWS = 8
_V_PC_COLS = CFG.TILE_O // CFG.CTA_MMA
LEADING_BYTE_OFFSET_PV = 0 if (_V_PC_COLS // _CORE_MATRIX_ROWS) <= 8 else CFG.TILE_N * CFG.V_SWZ_BYTES
STRIDE_BYTE_OFFSET_PV = 8 * CFG.V_SWZ_BYTES

NUM_KPHASES_PV = CFG.TILE_N // _MMA_K_FP8
NUM_KPHASES_PV_PER_CHUNK = NUM_KPHASES_PV // CFG.N_BMM2_CHUNKS


@cute.jit
def _mma_warp_quiet(tmem_ptr_i32, bars):
    """Non-leader CTA's MMA-warp body under cga2: alloc + named-bar arrive +
    tmem_dealloc wait + dealloc.  Peer's TMEM stays allocated because leader
    reads through the cluster crossbar during collective MMA.
    """
    # All-lanes warp-collective ops — NO elect_sync gating.
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    # Must match lead MMA's +1 arrive on softmax (id=1) and correction (id=2) bars.
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
    qh_per_kh,
):
    """Unified MMA warp (cga1 / cga2-leader; MASK_NONE/PADDED/CAUSAL/SWA).

    Spill-free in OTHER_REGS (40).  Non-leader cga2 CTA uses _mma_warp_quiet.
    """
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    nvvm.barrier_cta_arrive(2, 32 * (CFG.CORRECTION_WARPS + 1))

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    # SM100/Blackwell FP8: k_dim=0 = K=32 QMMA path (NOT the k_dim=1 K=64
    # fast path, which is silently WRONG on Blackwell — cuda-kernels rules §16).
    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_N,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        k_dim=0,
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_O,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        b_major=1,
        k_dim=0,
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

    desc_Q0 = sQ[0].desc()
    desc_Q1 = sQ[1].desc()

    if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    else:
        q_super_idx, _hd, batch_idx, split_idx = _decode_initial_split(
            sched.bidx_init,
            sched.bidy_init,
            sched.bidz_init,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
            qh_per_kh,
            seqlen_kv,
        )
        if cutlass.const_expr(CFG.MASK_FLAGS == 0):
            kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
        else:
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
            bounds_init = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, None, None, split_idx, CFG.QH_PER_KH)
            kv_left = bounds_init.left
            kv_right = bounds_init.right

    q_full_phase = cutlass.Int32(0)
    kv_state = PipelineState.start(phase=0)
    bmm2_ready_phase = cutlass.Int32(0)
    # Init unconditionally so type stays stable across const_expr branches.
    empty_mainloop_phase = cutlass.Int32(0)
    # Stats-consumed gate (one flip per tile per sub-tile): the prologue BMM1
    # overwrites the S_acc HEAD where the PREVIOUS tile's final
    # (total_max, total_sum) stats live until the correction epilogue has read
    # them.  q_full/k_full alone do NOT order that read before the BMM1 (Q/K
    # can be resident well before correction finishes its epilogue — the fp8
    # epilogue's amax reductions + LSE/O gmem stores widen the window on
    # multi-wave grids, corrupting the next tile's stats/O nondeterministically).
    # Bootstrap phase 1: the first tile has no prior stats to protect, so the
    # wait passes immediately.
    stats_read_phase = cutlass.Int32(1)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            # Empty-kv tile: fire bmm2_done so softmax/corr phases stay in lockstep.
            bars.mb_empty_mainloop.wait(empty_mainloop_phase)
            empty_mainloop_phase = empty_mainloop_phase ^ cutlass.Int32(1)
            # Keep the stats-read gate in lockstep — correction's epilogue
            # (and its mb_stats_read arrive) runs for empty tiles too.
            bars.mb_stats_read[0].wait(stats_read_phase)
            bars.mb_stats_read[1].wait(stats_read_phase)
            elect_p = nvvm.elect_sync()
            bars.mb_bmm2_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_bmm2_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        else:
            # mb_stats_read[qs] gates each prologue BMM1 on the correction
            # epilogue having READ the previous tile's final stats from the
            # S_acc head this BMM1 is about to overwrite (q_full/k_full don't
            # order that).
            bars.mb_q_full[0].wait(q_full_phase)
            bars.mb_k_full[kv_state.idx].wait(kv_state.phase)
            bars.mb_stats_read[0].wait(stats_read_phase)
            desc_K = sK[kv_state.idx].desc()
            mma_ss(bmm1_desc, desc_Q0, desc_K, (tmem_raw.subview(LAYOUT.S0_OFF)))
            elect_p = nvvm.elect_sync()
            bars.mb_bmm1_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

            bars.mb_q_full[1].wait(q_full_phase)
            bars.mb_stats_read[1].wait(stats_read_phase)
            mma_ss(bmm1_desc, desc_Q1, desc_K, (tmem_raw.subview(LAYOUT.S1_OFF)))
            elect_p = nvvm.elect_sync()
            bars.mb_bmm1_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_k_empty[kv_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

            q_full_phase = q_full_phase ^ 1

            for kv_loop in cutlass.range(kv_left + cutlass.Int32(1), kv_right, 1, unroll=1):
                old_state = kv_state
                kv_state = advance(kv_state, CFG.STAGES_KV)

                bars.mb_v_full[old_state.idx].wait(old_state.phase)
                desc_V = sV[old_state.idx].desc()
                is_not_first_bmm2 = cutlass.Boolean(kv_loop != (kv_left + cutlass.Int32(1)))

                bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 0].wait(bmm2_ready_phase)
                accum_b2 = is_not_first_bmm2
                for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                    mma_ts_step(bmm2_desc, (tmem_raw.subview(LAYOUT.P0_OFF)), desc_V, (tmem_raw.subview(LAYOUT.O0_OFF)), local_k, accum_b2)
                    accum_b2 = cutlass.Boolean(True)
                # Chunk 1 folds out at N_BMM2_CHUNKS=1 (full NUM_KPHASES_PV in chunk 0).
                if cutlass.const_expr(CFG.N_BMM2_CHUNKS == 2):
                    bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 1].wait(bmm2_ready_phase)
                    for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                        mma_ts_step(
                            bmm2_desc,
                            (tmem_raw.subview(LAYOUT.P0_OFF)),
                            desc_V,
                            (tmem_raw.subview(LAYOUT.O0_OFF)),
                            NUM_KPHASES_PV_PER_CHUNK + local_k,
                            cutlass.Boolean(True),
                        )
                elect_p = nvvm.elect_sync()
                bars.mb_bmm2_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

                bars.mb_k_full[kv_state.idx].wait(kv_state.phase)
                desc_K = sK[kv_state.idx].desc()
                mma_ss(bmm1_desc, desc_Q0, desc_K, (tmem_raw.subview(LAYOUT.S0_OFF)))
                elect_p = nvvm.elect_sync()
                bars.mb_bmm1_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

                bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 0].wait(bmm2_ready_phase)
                accum_b2 = is_not_first_bmm2
                for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                    mma_ts_step(bmm2_desc, (tmem_raw.subview(LAYOUT.P1_OFF)), desc_V, (tmem_raw.subview(LAYOUT.O1_OFF)), local_k, accum_b2)
                    accum_b2 = cutlass.Boolean(True)
                if cutlass.const_expr(CFG.N_BMM2_CHUNKS == 2):
                    bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 1].wait(bmm2_ready_phase)
                    for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                        mma_ts_step(
                            bmm2_desc,
                            (tmem_raw.subview(LAYOUT.P1_OFF)),
                            desc_V,
                            (tmem_raw.subview(LAYOUT.O1_OFF)),
                            NUM_KPHASES_PV_PER_CHUNK + local_k,
                            cutlass.Boolean(True),
                        )
                elect_p = nvvm.elect_sync()
                bars.mb_bmm2_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_v_empty[old_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

                mma_ss(bmm1_desc, desc_Q1, desc_K, (tmem_raw.subview(LAYOUT.S1_OFF)))
                elect_p = nvvm.elect_sync()
                bars.mb_bmm1_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_k_empty[kv_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

                bmm2_ready_phase = bmm2_ready_phase ^ 1

            # Epilogue BMM2 always runs (n_kv >= 1).
            elect_p = nvvm.elect_sync()
            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                bars.mb_q_empty[qs].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

            bars.mb_v_full[kv_state.idx].wait(kv_state.phase)
            desc_V = sV[kv_state.idx].desc()
            is_not_first_bmm2_epi = cutlass.Boolean((kv_right - kv_left) != cutlass.Int32(1))

            bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 0].wait(bmm2_ready_phase)
            accum_b2 = is_not_first_bmm2_epi
            for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                mma_ts_step(bmm2_desc, (tmem_raw.subview(LAYOUT.P0_OFF)), desc_V, (tmem_raw.subview(LAYOUT.O0_OFF)), local_k, accum_b2)
                accum_b2 = cutlass.Boolean(True)
            if cutlass.const_expr(CFG.N_BMM2_CHUNKS == 2):
                bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 1].wait(bmm2_ready_phase)
                for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                    mma_ts_step(
                        bmm2_desc,
                        (tmem_raw.subview(LAYOUT.P0_OFF)),
                        desc_V,
                        (tmem_raw.subview(LAYOUT.O0_OFF)),
                        NUM_KPHASES_PV_PER_CHUNK + local_k,
                        cutlass.Boolean(True),
                    )
            elect_p = nvvm.elect_sync()
            bars.mb_bmm2_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

            bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 0].wait(bmm2_ready_phase)
            accum_b2 = is_not_first_bmm2_epi
            for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                mma_ts_step(bmm2_desc, (tmem_raw.subview(LAYOUT.P1_OFF)), desc_V, (tmem_raw.subview(LAYOUT.O1_OFF)), local_k, accum_b2)
                accum_b2 = cutlass.Boolean(True)
            if cutlass.const_expr(CFG.N_BMM2_CHUNKS == 2):
                bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 1].wait(bmm2_ready_phase)
                for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                    mma_ts_step(
                        bmm2_desc,
                        (tmem_raw.subview(LAYOUT.P1_OFF)),
                        desc_V,
                        (tmem_raw.subview(LAYOUT.O1_OFF)),
                        NUM_KPHASES_PV_PER_CHUNK + local_k,
                        cutlass.Boolean(True),
                    )
            elect_p = nvvm.elect_sync()
            bars.mb_bmm2_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_v_empty[kv_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

            bmm2_ready_phase = bmm2_ready_phase ^ 1
            kv_state = advance(kv_state, CFG.STAGES_KV)

        # One correction-epilogue arrive per tile per sub-tile — flip once per tile.
        stats_read_phase = stats_read_phase ^ 1

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
            nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
            is_valid_tile = nxt_v & cutlass.Int32(1)
        else:
            nxt_q = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0))).load())
            nxt_hb = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1))).load())
            nxt_v = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load())
            q_super_idx, _hd, batch_idx, split_idx = _decode_payload_split(
                nxt_q,
                nxt_hb,
                cta_in_pair,
                n_q_supers,
                n_qh,
                n_batch,
                seq_kv_lens_tensor,
                qh_per_kh,
                seqlen_kv,
            )
            is_valid_tile = nxt_v & cutlass.Int32(1)
            if cutlass.const_expr(CFG.MASK_FLAGS == 0):
                kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
            else:
                eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
                eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
                bounds_next = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, None, None, split_idx, CFG.QH_PER_KH)
                kv_left = bounds_next.left
                kv_right = bounds_next.right
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _softmax_kv_body(
    apply_mask: cutlass.Constexpr[bool],
    sub_tile_id: cutlass.Constexpr[int],
    kv_loop,
    tmem_ptr_i32,
    bars,
    q_abs,
    eff_seqlen_kv,
    eff_seqlen_q,
    scale_log2,
    total_max,
    total_sum,
    leader_cta_id,
):
    """Per-kv-iter softmax body; returns updated (total_max, total_sum).

    Compile-time apply_mask picks the load+max strategy:
    - False: tcgen05.ld.red.f32.max fast path (fused HW row-max).
    - True: chunked load + apply_mask_chunk + sw row_max_reduction.
    HW max can't observe NEG_INFINITY written after the load, so masked
    iters fall back; 3-segment kv-loop keeps the fast path on interior iters.

    Phase tracking for mb_bmm1_done / mb_stat_empty is hoisted to the caller
    so ptxas places phases in URs (lowers to USYNCS.PHASECHK.TRANS64 instead
    of per-thread SYNCS+NANOSLEEP).
    """
    tmem_S_off = LAYOUT.S0_OFF if sub_tile_id == 0 else LAYOUT.S1_OFF
    tmem_P_off = LAYOUT.P0_OFF if sub_tile_id == 0 else LAYOUT.P1_OFF
    stats_off = LAYOUT.STATS_OFF + sub_tile_id * LAYOUT.STATS_STRIDE
    CHUNK = 64
    P_COLS_PER_CHUNK = CHUNK // 4
    N_CHUNKS = CFG.N_BMM2_CHUNKS
    NEG_INF = cutlass.Float32(-3.4028235e38)
    RESCALE_THRESHOLD = cutlass.Float32(CFG.RESCALE_THRESHOLD)

    # tcgen05.ld/st auto-derives row from warp_id; address needs col only.
    tmem_base = tmem_ptr_i32.load()
    s_addr_base = tmem_base + cutlass.Int32(tmem_S_off)
    p_addr_base = tmem_base + cutlass.Int32(tmem_P_off)
    stats_addr = tmem_base + cutlass.Int32(stats_off)

    # const_expr wrap — without it MLIR cf.if NameErrors reg_S_vec post-branch.
    if cutlass.const_expr(apply_mask):
        # Comprehensions (not for+append) so the tracer sees fully-formed lists.
        kv_col_base = kv_loop * cutlass.Int32(CFG.TILE_N)
        raw_chunks = [
            nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * CHUNK), cutlass.Float32),
                num=CHUNK,
            )
            for c in range(N_CHUNKS)
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
    else:
        # SM100: manual row-max (no LDTM.STAT / tmem_load_max_reduction_tile) —
        # the masked path's pattern sans mask.  S_acc is FP32 regardless of dtype.
        raw_chunks = [
            nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * CHUNK), cutlass.Float32),
                num=CHUNK,
            )
            for c in range(N_CHUNKS)
        ]
        chunks_max = [row_max_reduction(raw_chunks[c]) for c in range(N_CHUNKS)]
        reg_S_vec = vec_concat(raw_chunks)
        current_max_unscaled = chunks_max[0]
        for m in chunks_max[1:]:
            current_max_unscaled = cute.math.max(current_max_unscaled, m)

    # size= explicit — Vector.shape[0] is MLIR-typed after vec_concat.
    reg_S = RegTile(reg_S_vec, size=CFG.TILE_N)
    current_max = current_max_unscaled * scale_log2

    # sync the warpgroups before the stat-store.
    if sub_tile_id == 1:
        nvvm.barrier_cta_sync(barrier_id=8, thread_count=256)

    # Online softmax with RESCALE_THRESHOLD skip.
    old_total_max = total_max
    is_first = total_max == NEG_INF
    update_cond = is_first | ((current_max - total_max) > RESCALE_THRESHOLD)
    total_max = cutlass.Float32(
        arith.select(
            update_cond.ir_value(),
            current_max.ir_value(),
            total_max.ir_value(),
        )
    )
    exp_input = cutlass.Float32(
        arith.select(
            is_first.ir_value(),
            NEG_INF.ir_value(),
            (old_total_max - total_max).ir_value(),
        )
    )
    alpha = cute.math.exp2(exp_input, fastmath=True)
    new_total_max = total_max

    alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
    bars.mb_stat_full[sub_tile_id].arrive()

    # Manual unroll — tracer intercepts range_constexpr in ways that break
    # slice.indices() inside RegTile[]; N_CHUNKS ∈ {1,2} so explicit is cleaner.
    # P-cast bias: exp2(x + P_CAST_LOG2_SCALE) = 2^4 * P — the sum picks up
    # the same factor, so normalization cancels it (see P_CAST_LOG2_SCALE).
    reg_S = reg_S * scale_log2 - (new_total_max - cutlass.Float32(P_CAST_LOG2_SCALE))

    chunk_S_0 = reg_S[0:CHUNK].vec
    chunk_P_0 = cute.math.exp2(chunk_S_0, fastmath=True)
    # Hoist chunk-0 sum before cast to overlap with cast's FFMA chain.
    hoisted_sum = row_reduction_pair(chunk_P_0)
    chunk_P_0_fp8 = chunk_P_0.to(STORAGE_DTYPE)
    nvvm.tcgen05_st(
        "32x32b",
        nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32),
        chunk_P_0_fp8,
    )
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
    bars.mb_bmm2_ready[sub_tile_id * N_CHUNKS + 0].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

    deferred_P_1 = None
    if cutlass.const_expr(N_CHUNKS == 2):
        chunk_S_1 = reg_S[CHUNK : 2 * CHUNK].vec
        deferred_P_1 = cute.math.exp2(chunk_S_1, fastmath=True)
        chunk_P_1_fp8 = deferred_P_1.to(STORAGE_DTYPE)
        nvvm.tcgen05_st(
            "32x32b",
            nvvm.make_tmem_ptr(
                p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK),
                cutlass.Float32,
            ),
            chunk_P_1_fp8,
        )
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        bars.mb_bmm2_ready[sub_tile_id * N_CHUNKS + 1].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

    if sub_tile_id == 0:
        nvvm.barrier_cta_sync(barrier_id=8, thread_count=256)

    new_p_sum_pair = hoisted_sum
    if cutlass.const_expr(N_CHUNKS == 2):
        new_p_sum_pair = new_p_sum_pair + row_reduction_pair(deferred_P_1)
    alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
    total_sum = total_sum * alpha_pair + new_p_sum_pair

    return total_max, total_sum


@cute.jit
def _softmax_warp_group(
    sub_tile_id: cutlass.Constexpr[int],
    seqlen_q,
    seqlen_kv,
    scale_log2: cutlass.Float32,
    tmem_ptr_i32,
    sQ,
    bars,
    sched,
    seq_kv_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    leader_cta_id,
    cta_in_pair,
    qh_per_kh,
):
    """Softmax warp group: online softmax per kv iter, one lane per S_acc row.

    Tracks (total_max, total_sum) with RESCALE_THRESHOLD skip, publishes alpha
    to corr, writes P at S_acc tail, fires bmm2_ready[sub][chunk].
    """
    # Wait on MMA's TMEM-publish bar BEFORE tmem_ptr_i32.load() — else stale base.
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))

    tmem_S_off = LAYOUT.S0_OFF if sub_tile_id == 0 else LAYOUT.S1_OFF
    tmem_P_off = LAYOUT.P0_OFF if sub_tile_id == 0 else LAYOUT.P1_OFF

    NEG_INF = cutlass.Float32(-3.4028235e38)

    CHUNK = 64
    P_COLS_PER_CHUNK = CHUNK // 4
    stats_off = LAYOUT.STATS_OFF + sub_tile_id * LAYOUT.STATS_STRIDE

    # Phase trackers persist (XOR) across tile boundaries.
    bmm1_phase = cutlass.Int32(0)
    stat_empty_phase = cutlass.Int32(1)  # bootstrap pre-armed at phase 1 so first wait passes
    # BOTH softmax wgs wait on mb_o_empty[0]; init phase=1, XOR after.
    epilogue_state = cutlass.Int32(1)

    # total_sum is Vector[Float32, 2] (even/odd partials) so per-iter update
    # lowers to packed FMUL2+FADD2; folded to scalar once outside the kv loop.
    total_max = NEG_INF
    total_sum = cutlass.Vector.from_elements(
        (cutlass.Float32(0.0), cutlass.Float32(0.0)),
        cutlass.Float32,
    )

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
        qh_per_kh,
        seqlen_kv,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)

    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
    bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, None, None, split_idx, CFG.QH_PER_KH)

    softmax_wg_base_const = CFG.SOFTMAX_WG0_BASE if sub_tile_id == 0 else CFG.SOFTMAX_WG1_BASE
    tid_in_wg = cute.arch.thread_idx()[0] - cutlass.Int32(softmax_wg_base_const * 32)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        bars.mb_o_empty[0].wait(epilogue_state)
        epilogue_state = epilogue_state ^ cutlass.Int32(1)

        total_max = NEG_INF
        total_sum = cutlass.Vector.from_elements(
            (cutlass.Float32(0.0), cutlass.Float32(0.0)),
            cutlass.Float32,
        )
        # PackGQA: q_abs is the row's TOKEN index (row // G): every mask
        # predicate downstream is a token-space compare, and all G rows of one
        # token share it.
        q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE)
        q_abs = q_row_coord + cutlass.Int32(sub_tile_id * TOKENS_PER_TILE) + (tid_in_wg // cutlass.Int32(HEADS_PER_TILE))
        # Bootstrap stat_empty wait lifts wait off per-iter critical path so
        # α publish + stat_full fire back-to-back.
        bars.mb_stat_empty[sub_tile_id].wait(stat_empty_phase)
        stat_empty_phase = stat_empty_phase ^ 1
        # 3-segment kv loop: LEFT-masked | unmasked (fast HW max) | RIGHT-masked.
        # MASK_NONE folds masked sub-loops out at trace time.
        if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
            for kv_loop in cutlass.range(bounds.left, bounds.right, 1, unroll=1):
                bars.mb_bmm1_done[sub_tile_id].wait(bmm1_phase)
                bmm1_phase = bmm1_phase ^ 1
                total_max, total_sum = _softmax_kv_body(
                    False,
                    sub_tile_id,
                    kv_loop,
                    tmem_ptr_i32,
                    bars,
                    q_abs,
                    eff_seqlen_kv,
                    eff_seqlen_q,
                    scale_log2,
                    total_max,
                    total_sum,
                    leader_cta_id,
                )
                bars.mb_stat_empty[sub_tile_id].wait(stat_empty_phase)
                stat_empty_phase = stat_empty_phase ^ 1
        else:
            for kv_loop in cutlass.range(bounds.left, bounds.unmasked_lo, 1, unroll=1):
                bars.mb_bmm1_done[sub_tile_id].wait(bmm1_phase)
                bmm1_phase = bmm1_phase ^ 1
                total_max, total_sum = _softmax_kv_body(
                    True,
                    sub_tile_id,
                    kv_loop,
                    tmem_ptr_i32,
                    bars,
                    q_abs,
                    eff_seqlen_kv,
                    eff_seqlen_q,
                    scale_log2,
                    total_max,
                    total_sum,
                    leader_cta_id,
                )
                bars.mb_stat_empty[sub_tile_id].wait(stat_empty_phase)
                stat_empty_phase = stat_empty_phase ^ 1
            for kv_loop in cutlass.range(bounds.unmasked_lo, bounds.unmasked_hi, 1, unroll=1):
                bars.mb_bmm1_done[sub_tile_id].wait(bmm1_phase)
                bmm1_phase = bmm1_phase ^ 1
                total_max, total_sum = _softmax_kv_body(
                    False,
                    sub_tile_id,
                    kv_loop,
                    tmem_ptr_i32,
                    bars,
                    q_abs,
                    eff_seqlen_kv,
                    eff_seqlen_q,
                    scale_log2,
                    total_max,
                    total_sum,
                    leader_cta_id,
                )
                bars.mb_stat_empty[sub_tile_id].wait(stat_empty_phase)
                stat_empty_phase = stat_empty_phase ^ 1
            for kv_loop in cutlass.range(bounds.unmasked_hi, bounds.right, 1, unroll=1):
                bars.mb_bmm1_done[sub_tile_id].wait(bmm1_phase)
                bmm1_phase = bmm1_phase ^ 1
                total_max, total_sum = _softmax_kv_body(
                    True,
                    sub_tile_id,
                    kv_loop,
                    tmem_ptr_i32,
                    bars,
                    q_abs,
                    eff_seqlen_kv,
                    eff_seqlen_q,
                    scale_log2,
                    total_max,
                    total_sum,
                    leader_cta_id,
                )
                bars.mb_stat_empty[sub_tile_id].wait(stat_empty_phase)
                stat_empty_phase = stat_empty_phase ^ 1

        # End-of-kv: publish (total_max, total_sum_final) — corr does LSE.
        total_sum_scalar = total_sum[0] + total_sum[1]

        stats_addr_epi = tmem_ptr_i32.load() + cutlass.Int32(stats_off)
        stats_vec_epi = cutlass.Vector.from_elements((total_max, total_sum_scalar), cutlass.Float32)
        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr_epi, cutlass.Float32), stats_vec_epi)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        bars.mb_stat_full[sub_tile_id].arrive()

        # make_warp_uniform keeps scheduler payload in URs across the back-edge.
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0))).load())
        nxt_hb = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1))).load())
        nxt_v = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load())
        q_super_idx, head_idx, batch_idx, split_idx = _decode_payload_split(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
            qh_per_kh,
            seqlen_kv,
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
        bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, None, None, split_idx, CFG.QH_PER_KH)


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
    qh_per_kh,
):
    """Correction warp group: 4 warps × 32 lanes = 128, one lane per O row.

    Per-kv rescales O by α (skipped via all_alpha_one ballot).  Per-tile
    epilogue normalizes O by 1/total_sum, casts to fp8, swizzled-stores to sO,
    fires o_full.  Holds the P14 end-of-tile catch-up flip on bmm2_done_phase.
    """
    # Wait on MMA TMEM-publish bar BEFORE tmem_ptr_i32.load() — else stale base.
    nvvm.barrier_cta_sync(barrier_id=2, thread_count=32 * (CFG.CORRECTION_WARPS + 1))

    tid_raw = cute.arch.thread_idx()[0]
    tid_in_wg = tid_raw - cutlass.Int32(CFG.CORR_WARP_BASE * 32)

    # O_CHUNK=16 keeps live range short — O_CHUNK=32 spilled correction-warp regs.
    O_CHUNK = 16
    N_CHUNKS_O = CFG.TILE_O // O_CHUNK
    O_CHUNK_EPI = 64
    N_CHUNKS_O_EPI = CFG.TILE_O // O_CHUNK_EPI
    # D_BLOCK_SIZE must use O_SWZ_B not V_SWZ_B (under cga2 V may drop swizzle).
    # Sized in BPE_O so it stays consistent with the BPE_O-derived O_SWZ_BYTES
    # (these feed the FP8 store branch only; the BF16/FP16 branch is self-contained).
    TMA_O_ITERS = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES
    D_BLOCK_SIZE = CFG.TILE_O // TMA_O_ITERS
    TMA_O_GRANU_ELEMS = CFG.TILE_M * D_BLOCK_SIZE

    stat_full_phase = cutlass.Int32(0)
    # bmm2_done starts at phase=0; iter 0 skipped — first wait at kv_loop=1.
    bmm2_done_phase = cutlass.Int32(0)
    o_empty_phase = cutlass.Int32(1)  # bootstrap pre-armed at phase 1 so first wait passes

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
        qh_per_kh,
        seqlen_kv,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)

    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
    bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, None, None, split_idx, CFG.QH_PER_KH)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        # Iter-0 skip: MMA's iter-0 BMM2 uses init_d=False so α-rescale is unneeded.
        if bounds.right > bounds.left:
            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                bars.mb_bmm2_ready[qs * CFG.N_BMM2_CHUNKS + 0].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                bars.mb_stat_full[qs].wait(stat_full_phase)
                bars.mb_stat_empty[qs].arrive()
            stat_full_phase = stat_full_phase ^ 1
        else:
            bars.mb_empty_mainloop.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

        for kv_loop in cutlass.range(bounds.left + cutlass.Int32(1), bounds.right, 1, unroll=1):
            tmem_base_iter = tmem_ptr_i32.load()
            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                stats_off = LAYOUT.STATS_OFF + qs * LAYOUT.STATS_STRIDE
                tmem_O_off = LAYOUT.O0_OFF if qs == 0 else LAYOUT.O1_OFF

                bars.mb_stat_full[qs].wait(stat_full_phase)

                stats_addr = tmem_base_iter + cutlass.Int32(stats_off)
                stats_vec = nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(stats_addr, cutlass.Float32),
                    num=2,
                )
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                alpha = stats_vec[0]

                # all_alpha_one ballot: rescale skipped once softmax stops bumping max.
                alpha_is_one = alpha == cutlass.Float32(1.0)
                all_alpha_one = vote_sync(0xFFFFFFFF, alpha_is_one, VoteSync.ALL)

                bars.mb_stat_empty[qs].arrive()

                bars.mb_bmm2_done[qs].wait(bmm2_done_phase)

                # Split O_CHUNK=16 into 2× O_HALF=8 LDTM/STTM issued back-to-back —
                # distinct scoreboard slots overlap second LDTM with first's wait.
                # vec_scale_pair emits mul_packed_f32x2 → SASS FMUL2.
                O_HALF = O_CHUNK // 2
                if ~all_alpha_one:
                    for chunk_idx in cutlass.range_constexpr(N_CHUNKS_O):
                        o_addr_a = tmem_base_iter + cutlass.Int32(tmem_O_off + chunk_idx * O_CHUNK)
                        o_addr_b = tmem_base_iter + cutlass.Int32(tmem_O_off + chunk_idx * O_CHUNK + O_HALF)
                        o_a = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(o_addr_a, cutlass.Float32),
                            num=O_HALF,
                        )
                        o_b = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(o_addr_b, cutlass.Float32),
                            num=O_HALF,
                        )
                        s_a = vec_scale_pair(o_a, alpha, O_HALF)
                        s_b = vec_scale_pair(o_b, alpha, O_HALF)
                        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(o_addr_a, cutlass.Float32), s_a)
                        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(o_addr_b, cutlass.Float32), s_b)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)

                bars.mb_bmm2_ready[qs * CFG.N_BMM2_CHUNKS + 0].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            stat_full_phase = stat_full_phase ^ 1
            bmm2_done_phase = bmm2_done_phase ^ 1

        tmem_base_epi = tmem_ptr_i32.load()
        for qs in cutlass.range_constexpr(CFG.TILES_Q):
            stats_off = LAYOUT.STATS_OFF + qs * LAYOUT.STATS_STRIDE
            tmem_O_off = LAYOUT.O0_OFF if qs == 0 else LAYOUT.O1_OFF

            bars.mb_bmm2_done[qs].wait(bmm2_done_phase)

            # softmax fires stat_full once more after kv loop for (total_max, total_sum_final).
            bars.mb_stat_full[qs].wait(stat_full_phase)

            stats_addr = tmem_base_epi + cutlass.Int32(stats_off)
            stats_vec = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(stats_addr, cutlass.Float32),
                num=2,
            )
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            total_max_scaled = stats_vec[0]  # log2-units
            total_sum = stats_vec[1]

            bars.mb_stat_empty[qs].arrive()
            # Release MMA's NEXT-tile prologue BMM1 into this S_acc slot: the
            # final stats ride the slot HEAD and are now safely in registers
            # (tcgen05_wait LOAD above).  Cross-CTA arrive on the leader under
            # cga2 — the collective BMM1 writes both peers' TMEM.
            bars.mb_stats_read[qs].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            inv_sum = cutlass.Float32(0.0)  # pre-declare for DSL if-staging
            lse_val = cutlass.Float32(0.0)  # pre-declare; computed in both branches
            q_row_global = (
                q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE) + cutlass.Int32(qs * TOKENS_PER_TILE) + (tid_in_wg // cutlass.Int32(HEADS_PER_TILE))
            )
            row_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE) + (tid_in_wg % cutlass.Int32(HEADS_PER_TILE))
            LN2 = cutlass.Float32(0.6931471805599453)
            total_max_nat = total_max_scaled * LN2
            # Dead row (no valid KV column at all): O := 0 in BOTH branches
            # (with a sink the denominator is finite but the O numerator is an
            # empty sum).  total_sum >= 2^P_CAST_LOG2_SCALE for any alive row
            # so this never fires spuriously.
            row_dead = total_sum <= cutlass.Float32(0.0)
            if cutlass.const_expr(CFG.HAS_SINK):
                sinks_arr = cutlass.make_array_view(sinks_tensor)
                sink_logit = sinks_arr[row_head_idx]
                new_max = cute.math.max(total_max_nat, sink_logit)
                scale = cute.math.exp(total_max_nat - new_max, fastmath=True)
                # total_sum is in 2^P_CAST_LOG2_SCALE units — lift the sink term
                # into the same units, then take the constant back out of the LSE.
                new_sum = total_sum * scale + cute.math.exp(sink_logit - new_max, fastmath=True) * cutlass.Float32(2.0**P_CAST_LOG2_SCALE)
                lse_val = new_max + cute.math.log(new_sum, fastmath=True) - cutlass.Float32(P_CAST_LOG2_SCALE) * LN2
                inv_sum = (scale * o_scale_fused) / new_sum
            else:
                # total_sum carries 2^P_CAST_LOG2_SCALE — subtract the constant.
                lse_val = total_max_nat + cute.math.log(total_sum, fastmath=True) - cutlass.Float32(P_CAST_LOG2_SCALE) * LN2
                # Safe inverse: avoid div by 0 on fully-masked rows.
                inv_sum = o_scale_fused / cute.math.max(total_sum, cutlass.Float32(1e-30))
                # Dead row without a sink: LSE := -inf on top of O := 0.
                neg_inf_lse = cutlass.Float32(float("-inf"))
                lse_val = cutlass.Float32(arith.select(row_dead.ir_value(), neg_inf_lse.ir_value(), lse_val.ir_value()))
                inv_sum = cutlass.Float32(arith.select(row_dead.ir_value(), cutlass.Float32(0.0).ir_value(), inv_sum.ir_value()))

            # cga2 OOB-row guard: cluster Q rows can exceed seqlen_q.
            if cutlass.const_expr(CFG.THD_VARLEN):
                # THD: q_row_global is sequence-local; the row is valid against
                # the per-sequence Q length from the device metadata (negative
                # for the dead-unit sentinel batch == n_batch — issue #552 —
                # so no dead-unit row ever writes LSE or feeds amax_o). The
                # packed ragged-Stats LSE is written in the caller's declared
                # layout — token-major rank-2 [T, QH] (index
                # [cu_q[b] + local, head]) or head-major rank-3
                # [1, QH, head_stride] (index [0, head, cu_q[b] + local]).
                _cu = cutlass.make_array_view(seq_kv_lens_tensor)
                _cu_q_b = cutlass.Int32(_cu[n_batch + batch_idx])
                _s_q_b = cutlass.Int32(_cu[n_batch + batch_idx + cutlass.Int32(1)]) - _cu_q_b
                _row_valid = q_row_global < _s_q_b
                if cutlass.const_expr(lse_tensor is not None):
                    if _row_valid:
                        lse_arr = cutlass.make_array_view(lse_tensor)
                        if cutlass.const_expr(len(lse_tensor.shape) == 2):
                            lse_row = lse_arr[_cu_q_b + q_row_global, :]
                            lse_row[head_idx] = lse_val
                        else:
                            lse_row = lse_arr[cutlass.Int32(0), head_idx, :]
                            lse_row[_cu_q_b + q_row_global] = lse_val
            else:
                _row_valid = q_row_global < seqlen_q
                if _row_valid:
                    if cutlass.const_expr(lse_tensor is not None):
                        lse_arr = cutlass.make_array_view(lse_tensor)
                        # This chunk's LSE goes to its own split-major slot, matching where
                        # TMA-STG put the chunk's O.  The pair (O_s, lse_s) is everything
                        # the combine needs.  Folds to batch_idx at SPLIT_KV == 1.
                        lse_batch = _partial_batch(batch_idx, split_idx, n_batch)
                        lse_arr[lse_batch, row_head_idx, q_row_global] = lse_val

            # amax_o = max over valid rows of |o_scaled| (the fp32 pre-cast output). Divided
            # by scale_o in api to give the pre-quant output amax (cuDNN FP8 ref, in-kernel).
            _amax_o_ptr = Pointer(amax_o_tensor.iterator.raw_ptr(), dtype=cutlass.Int32)
            _amax_o_local = cutlass.Float32(0.0)

            sO_sub_base = sO[qs].base

            if cutlass.const_expr(CFG.DTYPE_O <= 1):
                # FP8 output (DTYPE_O ∈ {0,1}): hand-rolled 16:4 fp8 pack +
                # STS.128 — forces F2FP outputs into a register quad so STS.128
                # needs no PRMT to gather them (vs the DSL store_swizzled which
                # folds the swizzle XOR via byte-level PRMT).  Bit-identical to
                # the DTYPE_O == DTYPE_QKV path.
                _SWZ_BYTES_C = CFG.O_SWZ_BYTES
                _SWZ_SHIFT_C = 3 - _O_SWZ_B
                _SWZ_MASK_C = (1 << _O_SWZ_B) - 1
                _SUBTILE_BYTES_C = CFG.TILE_M * _SWZ_BYTES_C

                row_base_bytes = tid_in_wg * cutlass.Int32(_SWZ_BYTES_C)
                row_xor_field = ((tid_in_wg >> cutlass.Int32(_SWZ_SHIFT_C)) & cutlass.Int32(_SWZ_MASK_C)) << cutlass.Int32(4)

                for chunk_idx in cutlass.range_constexpr(N_CHUNKS_O):
                    o_addr = tmem_base_epi + cutlass.Int32(tmem_O_off + chunk_idx * O_CHUNK)
                    o_chunk = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                        num=O_CHUNK,
                    )
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                    o_scaled = o_chunk * inv_sum
                    # Dead-row sanitize: an empty mainloop never writes O TMEM,
                    # so o_chunk is garbage (possibly NaN) and `* inv_sum(=0)`
                    # cannot zero it — select 0 explicitly (keeps amax_o clean).
                    _zero_f = cutlass.Float32(0.0)
                    o_elems = [cutlass.Float32(arith.select(row_dead.ir_value(), _zero_f.ir_value(), o_scaled[i].ir_value())) for i in range(O_CHUNK)]

                    for _i in cutlass.range_constexpr(O_CHUNK):
                        _e = o_elems[_i]
                        _amax_o_local = cute.math.max(_amax_o_local, cute.math.max(_e, -_e))

                    # Plain range (not range_constexpr) — extraction at Python trace time.
                    o_packed_v = fp32_to_fp8_pack(
                        [o_elems[i] for i in range(16)],
                        dtype=OUT_STORAGE_DTYPE,
                    )

                    col_elems = chunk_idx * O_CHUNK
                    block_idx = col_elems // D_BLOCK_SIZE
                    col_in_block_bytes = (col_elems % D_BLOCK_SIZE) * CFG.BPE_O
                    block_off_bytes = block_idx * _SUBTILE_BYTES_C

                    swizzled_in_row = row_xor_field ^ cutlass.Int32(col_in_block_bytes)
                    addr_off_bytes = cutlass.Int32(block_off_bytes) + row_base_bytes + swizzled_in_row

                    # Re-type pointer to Int32 so 4-i32 store maps to one st.shared.v4.b32.
                    fp8_ptr = sO_sub_base.subview(addr_off_bytes).data_ptr()
                    i32_ptr = Pointer(fp8_ptr, dtype=cutlass.Int32)
                    if chunk_idx == 0:
                        bars.mb_o_empty[qs].wait(o_empty_phase)
                    i32_ptr.store(o_packed_v, alignment=16)
            else:
                # BF16 / FP16 output (DTYPE_O ∈ {2,3}): the 16:4 fp8 pack does
                # not apply — cast fp32 → OUT_STORAGE_DTYPE and swizzled-store,
                # matching the BPE_O-sized TMA-O box.  Mirrors the dsv4 d512 fp8
                # generic epilogue (prefill_sdpa_d512_fp8.py).
                O_EPI_BLOCK_SIZE = 64 // CFG.BPE_O  # 32 elems (half-out)
                O_D_BLOCK = CFG.O_SWZ_BYTES // CFG.BPE_O  # TMA chunk elems
                O_TMA_GRANU_ELEMS = CFG.TILE_M * O_D_BLOCK
                for b in cutlass.range_constexpr(CFG.TILE_O // O_EPI_BLOCK_SIZE):
                    o_addr = tmem_base_epi + cutlass.Int32(tmem_O_off + b * O_EPI_BLOCK_SIZE)
                    o_chunk = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                        num=O_EPI_BLOCK_SIZE,
                    )
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                    o_scaled_h = o_chunk * inv_sum
                    # Dead-row sanitize — see the fp8-pack path above.
                    _zero_f = cutlass.Float32(0.0)
                    o_scaled_h = cutlass.Vector.from_elements(
                        tuple(
                            cutlass.Float32(arith.select(row_dead.ir_value(), _zero_f.ir_value(), o_scaled_h[i].ir_value())) for i in range(O_EPI_BLOCK_SIZE)
                        ),
                        cutlass.Float32,
                    )
                    for _i in cutlass.range_constexpr(O_EPI_BLOCK_SIZE):
                        _e = o_scaled_h[_i]
                        _amax_o_local = cute.math.max(_amax_o_local, cute.math.max(_e, -_e))
                    o_half = o_scaled_h.to(OUT_STORAGE_DTYPE)

                    col_offset_const = (b * O_EPI_BLOCK_SIZE) % O_D_BLOCK
                    block_idx_const = (b * O_EPI_BLOCK_SIZE) // O_D_BLOCK
                    block_offset_const = block_idx_const * O_TMA_GRANU_ELEMS
                    smem_offset = cutlass.Int32(block_offset_const + col_offset_const) + tid_in_wg * cutlass.Int32(O_D_BLOCK)
                    smem_ptr = sO_sub_base.subview(smem_offset).data_ptr()
                    if b == 0:
                        bars.mb_o_empty[qs].wait(o_empty_phase)
                    smem_ptr.store_swizzled(o_half, alignment=64, swizzle=_O_SMEM_SWIZZLE)

            # Under KV split this epilogue sees only its OWN partial, and the
            # recombined O is a convex combination of the partials -- so a max
            # over partials over-reports the output amax (~2.9x at 8 splits).
            # split_combine_sm100 computes it over the recombined O instead;
            # this write has to stay out of the way, since atomicMax only grows.
            if cutlass.const_expr(SPLIT_KV == 1):
                if _row_valid:
                    nvvm.atomicrmw(nvvm.AtomicOp.MAX, _amax_o_ptr, _amax_o_local.bitcast(cutlass.Int32))

            # fence_proxy needed before TMA reads SMEM written by stores above.
            nvvm.fence_proxy("async.shared", space="cta")

            bars.mb_o_full[qs].arrive()

        stat_full_phase = stat_full_phase ^ 1
        o_empty_phase = o_empty_phase ^ 1
        # P14 catch-up flip — bmm2_done_phase ^= 1 AFTER epilogue wait.
        # n_kv=1 multi-wave deadlocks on the 2nd tile without this.
        bmm2_done_phase = bmm2_done_phase ^ 1

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0))).load()
        nxt_hb = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1))).load()
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        q_super_idx, head_idx, batch_idx, split_idx = _decode_payload_split(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
            qh_per_kh,
            seqlen_kv,
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
        bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, None, None, split_idx, CFG.QH_PER_KH)

    # tmem_dealloc fan-out: fire one arrive per lane; cga2 also DSMEM-arrives
    # on the peer so peer's local mbar accumulates the full CGA count.
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
    # 1-element fp32 device scales (Rule 3 — see _kernel).
    descale_q_t: cute.Tensor,
    descale_k_t: cute.Tensor,
    descale_v_t: cute.Tensor,
    scale_o_t: cute.Tensor,
    amax_o_tensor: cute.Tensor,
    # THD device metadata build (issue #552): the CALLER's Q/KV length
    # tensors — (B,) per-batch lengths or (B+1,) cu prefix sums, per side via
    # thd_lens_form (bit 0: Q is cu, bit 1: KV is cu) — consumed only by the
    # setup kernel, which writes the [kv|cu_q|cu_k] metadata buffer
    # (seq_kv_lens_tensor) device-side. None (folded out of the ABI) for
    # dense graphs.
    thd_q_lens_tensor: Optional[cute.Tensor] = None,
    thd_kv_lens_tensor: Optional[cute.Tensor] = None,
    thd_lens_form: Optional[cutlass.Int32] = None,
    stream: _cuda_driver.CUstream = None,
) -> None:
    B, QH, KH, SQ, SKV, _ = problem_size
    if cutlass.const_expr(CFG.THD_VARLEN):
        # Packed token totals are runtime values (dynamic extents); the
        # problem_size slots are 0 by contract.
        SQ = q_tensor.shape[1]
        SKV = k_tensor.shape[1]

    # K box rows are per-CTA (TILE_N/CTA_MMA); O box inner must match O's swizzle, not V's.
    # O box sized in BPE_O — O may be written at BF16/FP16 (DTYPE_O != DTYPE_QKV).
    _O_GRANU_ELEMS = CFG.O_SWZ_BYTES // CFG.BPE_O
    if cutlass.const_expr(CFG.PACK_GQA and q_tensor.shape[2] != k_tensor.shape[2] * CFG.QH_PER_KH):
        raise ValueError(f"CFG.QH_PER_KH ({CFG.QH_PER_KH}) does not match tensor head extents H_q={q_tensor.shape[2]}, H_kv={k_tensor.shape[2]}")
    qk_box_q = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, TMA_QK_GRANU_ELEMS)
    qk_box_k = (1, CFG.TILE_N // CFG.CTA_MMA, 1, TMA_QK_GRANU_ELEMS)
    vo_box_v = (1, CFG.TILE_N, 1, TMA_VO_GRANU_ELEMS)
    vo_box_o = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, _O_GRANU_ELEMS)
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
    # V TMA swizzle tracks per-CTA inner bytes.
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

    # Cluster-wide divisor mandatory: without it cga2 over-launches and OOB
    # clusters collide with valid clusters' GMEM slots at all but smallest SQ.
    # PackGQA: SQ*G packed rows per packed head, and QH/G packed heads.
    rows_per_cluster = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA
    q_clusters = (SQ * HEADS_PER_TILE + rows_per_cluster - 1) // rows_per_cluster
    grid_q_supers = q_clusters * CFG.CTA_MMA
    q_supers = grid_q_supers
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD setup launch: build the [kv|cu_q|cu_k] metadata buffer
        # DEVICE-side from the caller's length tensors (no host cumsum, no
        # H2D — issue #552), then the per-batch O descriptor array (reuse
        # tma_o_desc over the packed [1,T,QH,D_v] O as base). Main grid: the
        # PLAN-TIME ENVELOPE (n_thd_units = B * ceil(S_q_decl/CGA_TILE_M) * QH,
        # from the DECLARED S_q — no runtime length reaches the host); units
        # past a sequence's live tiles decode the batch == n_batch sentinel
        # and drain without loads or stores. grid_x = n_thd_units * CGA_M.
        # Per-token element stride of packed O (o_tensor.stride[1] = QH * d_v)
        # — NOT CFG.TILE_O, which is only coincidentally right at QH == 1.
        # The FP8 setup variant also clamps runtime K/V descriptors to the
        # packed KV total (slots n_batch+1/+2 of o_desc_words) so tile-tail
        # loads past it zero-fill instead of reading the buffer's capacity
        # tail — a NaN tail would poison BMM2's P·V (0 · NaN == NaN).
        _build_thd_meta_o_kv_descs_kernel(
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
        ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
        grid_shape = (n_thd_units * cutlass.Int32(CFG.CGA_M), cutlass.Int32(1), cutlass.Int32(1))
    else:
        # KV split rides the BATCH axis: z = batch + split*B.  The decode
        # already recovers the batch coord on both the blockIdx and the
        # scheduler-handout paths, so the split travels with it for free.
        grid_shape = (
            (grid_q_supers, QH // HEADS_PER_TILE, B * SPLIT_KV)
            if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL)
            else (grid_q_supers * (QH // HEADS_PER_TILE) * B * SPLIT_KV, 1, 1)
        )
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
        cutlass.Int32(QH // HEADS_PER_TILE),
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
    has_lse: bool = True,
    lse_head_major: bool = False,
    lse_head_stride: int = 0,
    lse_stride: Optional[tuple[int, int, int]] = None,
    d_qk: int = CFG.TILE_K,
    d_v: int = CFG.TILE_O,
) -> Callable:
    """Compile with ALL dims concrete — pins TMA strides at compile time.

    ``d_qk``/``d_v`` <= the d128 tile serve the dense ENVELOPE: the TMA
    descriptors carry the ACTUAL extents while the tile box stays the
    compile-time D, so loads past them hardware zero-fill (exact in FP8 —
    S/softmax/P·V are bit-identical to the unpadded problem) and O stores
    past ``d_v`` are OOB-clipped. Head dims like the ViT d=72-in-80 contract
    run without caller-side re-padding. THD serves native tile dims only
    (the packed compile key carries no head-dim entries).

    THD/varlen: q/k/v/o/lse PACKED with batch dim 1 ([1,T,H,D]); ``b`` is the
    LOGICAL batch (sequence count) driving n_batch / metadata + O-desc sizes.
    ``sq``/``skv`` are IGNORED under THD — the packed token totals are runtime
    values, so the token extents compile DYNAMIC (``cute.sym_int``) and the
    cache key stays plan-time-only (issue #552). THD Stats layouts: token-major
    packed rank-2 (T, H) by default (cuDNN's TH1 ragged Stats recipe);
    ``lse_head_major=True`` = rank-3 [1, QH, head_stride] (0 → compact).
    ``has_lse=False`` compiles the LSE store out (the kernel specializes on a
    ``None`` LSE argument) — callers without a Stats output pass no LSE buffer
    at all; the amax_o atomicMax write is independent and unchanged."""
    if SPLIT_KV > 1 and not has_lse:
        # Each split's LSE is not optional under KV split — it IS the weight
        # the combine reduces with.  Without it the partials cannot be recombined.
        raise ValueError("split_kv > 1 requires has_lse=True (the per-split LSE drives the combine)")
    if not (0 < d_qk <= CFG.TILE_K and 0 < d_v <= CFG.TILE_O):
        raise ValueError(f"fp8 d128 envelope: need 0 < d_qk <= {CFG.TILE_K} and 0 < d_v <= {CFG.TILE_O}; got ({d_qk}, {d_v})")
    if (d_qk * CFG.BPE) % 16 != 0 or (d_v * CFG.BPE) % 16 != 0:
        # d_v strides BOTH V (BPE) and O (BPE_O >= BPE); the fp8 input side is
        # the binding TMA 16-byte global-stride constraint.
        raise ValueError(f"fp8 d128 envelope: d_qk/d_v global strides must be 16-byte multiples (TMA rule at BPE={CFG.BPE}); got ({d_qk}, {d_v})")
    if CFG.THD_VARLEN and (d_qk != CFG.TILE_K or d_v != CFG.TILE_O):
        raise ValueError("THD/varlen serves native tile dims only (the packed compile key carries no head-dim entries); leave d_qk/d_v at the defaults")
    _fake_batch = 1 if CFG.THD_VARLEN else b
    # KV split: O and LSE are the PARTIAL workspaces, stacked split-major on
    # the batch axis (B*SPLIT_KV).  Q/K/V keep the real batch.  THD packs the
    # batch away (dim 1) and split_kv is dense-only (config backstop), so the
    # THD fakes see SPLIT_KV == 1.
    _o_batch = _fake_batch * SPLIT_KV
    _lse_batch = b * SPLIT_KV
    if CFG.THD_VARLEN:
        # Dynamic packed token totals: one symbol per ragged group (Q/O and a
        # token-major LSE share t_q; K/V share t_kv), so a new total re-binds
        # the same compiled artifact instead of minting a new one (issue #552).
        sq = cute.sym_int(divisibility=1)
        skv = cute.sym_int(divisibility=1)
    fake_q = cute.runtime.make_fake_compact_tensor(
        STORAGE_DTYPE,
        (_fake_batch, sq, qh, d_qk),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_k = cute.runtime.make_fake_compact_tensor(
        STORAGE_DTYPE,
        (_fake_batch, skv, kh, d_qk),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_v = cute.runtime.make_fake_compact_tensor(
        STORAGE_DTYPE,
        (_fake_batch, skv, kh, d_v),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_o = cute.runtime.make_fake_compact_tensor(
        OUT_STORAGE_DTYPE,
        (_o_batch, sq, qh, d_v),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    if not has_lse:
        # No Stats output: the LSE argument is None-specialized and the store
        # is compiled out entirely — no dummy buffer exists at any level.
        if lse_head_major or lse_head_stride:
            raise ValueError("lse_head_major / lse_head_stride require has_lse=True")
        fake_lse = None
    elif CFG.THD_VARLEN:
        # Packed ragged-Stats LSE in the caller's declared layout (align 4:
        # the store is scalar f32 and the caller's Stats buffer only
        # guarantees element alignment). The epilogue store branches on the
        # STATIC rank, so the layout is fully encoded in this fake tensor.
        if lse_head_major:
            _lse_hs = lse_head_stride if lse_head_stride else sq
            fake_lse = cute.runtime.make_fake_compact_tensor(
                cutlass.Float32,
                (1, qh, _lse_hs),
                stride_order=(2, 1, 0),
                assumed_align=4,
            )
        else:
            if lse_head_stride:
                raise ValueError("lse_head_stride is head-major-only (token-major (T, H) is compact)")
            fake_lse = cute.runtime.make_fake_compact_tensor(
                cutlass.Float32,
                (sq, qh),
                stride_order=(1, 0),
                assumed_align=4,
            )
    else:
        if lse_head_major or lse_head_stride:
            raise ValueError("lse_head_major / lse_head_stride are THD-only (dense LSE is compact (B, H, Sq))")
        fake_lse = (
            cute.runtime.make_fake_tensor(cutlass.Float32, (_lse_batch, qh, sq), lse_stride, assumed_align=4)
            if lse_stride is not None and SPLIT_KV == 1
            else cute.runtime.make_fake_compact_tensor(
                cutlass.Float32,
                (_lse_batch, qh, sq),
                stride_order=(2, 1, 0),
                assumed_align=16,
            )
        )
    # Always part of the ABI; unread when CFG.HAS_SINK == 0 (compile-time fold).
    fake_sinks = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (qh,),
        stride_order=(0,),
        assumed_align=16,
    )
    # Always part of the ABI; unread when CFG.SEQ_KV_LENS_PRESENT == 0.  THD
    # overloads it as [seq_kv_lens(B)|cu_q(B+1)|cu_k(B+1)] (len 3B+2).
    _skv_len = (3 * b + 2) if CFG.THD_VARLEN else b
    fake_seq_kv_lens = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (_skv_len,),
        stride_order=(0,),
        assumed_align=16,
    )
    # Per-batch O TMA-descriptor array (16 int64 = 128 B each) + 1 pad slot +
    # the packed-total-clamped K and V runtime descriptors (slots n_batch+1 /
    # n_batch+2 — see the THD tma_k/tma_v closures); dummy 1-elem when THD
    # off (kernel never reads it).
    _odesc_len = ((b + 3) * _TENSOR_MAP_QWORDS) if CFG.THD_VARLEN else 1
    fake_o_desc = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64,
        (_odesc_len,),
        stride_order=(0,),
        assumed_align=16,
    )
    fake_amax_o = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        stride_order=(0,),
        assumed_align=16,
    )

    def _fake_scale():
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (1,),
            stride_order=(0,),
            assumed_align=4,
        )

    # THD: the caller's Q/KV length tensors, consumed by the setup kernel's
    # device-side metadata build. DYNAMIC extents — (B,) per-batch lengths and
    # (B+1,) cu prefix sums bind the same artifact; the form rides the runtime
    # thd_lens_form bitmask, so no compile key grows (Rule 4). align 4: bound
    # directly, only natural int32 alignment is guaranteed.
    if CFG.THD_VARLEN:
        fake_thd_q_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(divisibility=1),), stride_order=(0,), assumed_align=4)
        fake_thd_kv_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(divisibility=1),), stride_order=(0,), assumed_align=4)
        fake_thd_lens_form = cutlass.Int32(0)
    else:
        fake_thd_q_lens = None
        fake_thd_kv_lens = None
        fake_thd_lens_form = None
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
        # THD: the packed totals are runtime values carried by the (dynamic)
        # tensor extents — _host reads them from the views' shapes.
        (b, qh, kh, 0, 0, 0) if CFG.THD_VARLEN else (b, qh, kh, sq, skv, 0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Int32(0),
        _fake_scale(),
        _fake_scale(),
        _fake_scale(),
        _fake_scale(),
        fake_amax_o,
        fake_thd_q_lens,
        fake_thd_kv_lens,
        fake_thd_lens_form,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )
