# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
DSL SDPA forward kernel — d128 DECODE tile, d_qk = d_v = 128, FP16/BF16, SM100.

The d128 prefill pipeline (``sm100/prefill_d128_f16.py``) computes 512 Q rows
per cga2 cluster (TILES_Q=2 x TILE_M=128 x CTA_MMA=2).  A decode step has one
token per sequence -- with PackGQA, G rows out of those 512 are live -- so the
prefill tile spends its time issuing BMM1/BMM2 for dead rows and exponentiating
dead S rows; measured on B200 at B=32, H=64/4, S_kv=4096, page 16, bf16: 119 us
for 268 MB of KV (2.3 TB/s), against 44 us for the cuDNN backend's decode engine.

This tile, same shapes, graph path, CUDA-graph replay (B200, SM100, 148 SMs):

    64/4  S_q=1                 119.2 us -> 48.9 us  (2.44x; 5.5 TB/s)
    64/4  S_q=1, mixed lengths  117.9 us -> 46.6 us
    64/4  S_q=4 MTP bottom-right causal, NATURAL  127.5 us -> 50.6 us
    64/8  S_q=1                 234.6 us -> 96.4 us
    96/8  S_q=1 (G=12 packs 4: partial PackGQA)  615 us -> 225 us  (unpacked on this tile: 875 us)
    96/8  S_q=4 MTP bottom-right causal (G=12 packs 4)  652 us -> 228 us
    d64 64/8 S_q=1 (d128 envelope)  230.5 us -> 80.2 us
    b=4 64/4 S_q=1 (split-KV 8 by the existing wave model)  61.6 us -> 18.6 us

With the KV cache L2-resident the same launch takes 39 us: the tile is issue /
latency-bound in its single softmax warpgroup (~1.2 us per KV tile per CTA, NCU:
0.31 issued warps per scheduler, 168 registers x 384 threads = one CTA per SM),
not HBM-bound -- the next step toward the backend's 44 us is a second softmax
warpgroup on the odd KV tiles with an in-CTA (max, sum, O) merge.

This template is that kernel's body with everything the second Q sub-tile
needed removed, at the geometry ``config_sm100.CfgD128Decode`` fixes:

  1. **TILES_Q=1, cga1** (``CTA_MMA=1``, no cluster): one Q slab, one S/P slot,
     one O accumulator; BMM1 and BMM2 are M=128 per KV tile -- a quarter of the
     prefill cluster's MMA issue per tile -- and no cross-CTA arrives.
  2. **One softmax warpgroup** (12 warps / 384 threads: softmax 0-3,
     correction 4-7, MMA 8, TMA-LDG 9, TMA-STG 10, scheduler 11 -- the d256
     flavor's layout).  The prefill's second warpgroup existed for sub-tile 1.
  3. **STAGES_KV=3 with the Q u O alias**: 32 KiB (Q u O) + 3 x (32 K + 32 V)
     = 224 KiB, under the SM100 227 KiB cap.  At cga1 every CTA streams whole
     K/V tiles, and the deeper ring keeps more of the KV cache in flight per SM
     while the tile is memory-bound.
  4. **Two S/P TMEM slots** (cols 0..127 and 128..255; P aliases each slot's
     tail at +64), alternating per KV tile: BMM1(i+1) is issued into the other
     slot right after K(i+1) lands, so it runs on the tensor core while the
     softmax warpgroup is still on S(i) -- with one slot the pipeline was
     BMM1 -> softmax -> BMM2 -> BMM1 in series (53 us on the 64/4 shape, direct
     template, against 48 us with two slots).  O at
     256..383; the (alpha | max, sum) stats at a dedicated column 384 that no
     MMA ever writes.

Everything else is the prefill kernel's contract, shared through
``_common_blackwell`` / ``config_sm100``: paged KV (``CFG.PAGED_KV``: block-table
indirection on the K/V TMA loads, HND and NHD pools, per-batch lengths read on
device, boxes past the live pages TMA-OOB zero-filled), the padded / causal /
bottom-right / SWA band masks, the dense padded-Q trim (``SEQ_Q_LENS_PRESENT``),
the attention sink fold, Stats (natural or base-2), PackGQA (row r <-> token
r // PACK_G, head r % PACK_G -- the whole group when it divides 128, else its
largest divisor that does: partial PackGQA, 96/8 packs 4 of its 12 heads and
three packed heads read one KV head), KV split with fp32 partials for
``sm100/split_combine.py``,
and the CLC try_cancel persistent scheduler (NATURAL / LPT / LPT_L2).  THD is
NOT wired: the decode tile is dense-only (``make_cfg_d128_decode`` rejects it and
the engine row declines cga=1 for THD graphs).

Selection: the (128, 128) f16/bf16 flavor at ``TILE_CGA_M=1`` IS this tile
(``api_dsl._load_sm100_kernel_module``); the heuristics propose cga=1 exactly
when ``S_q * pack_g <= 128`` (``pack_g`` = the packed group ``CFG.PACK_G`` for
a PackGQA plan -- G, or ``gcd(G, 128)`` under partial PackGQA -- 1 for an
unpacked one), i.e. when one 128-row tile covers a packed head's live Q rows.
The kernel is correct for any S_q -- larger S_q simply launches
``ceil(S_q * PACK_G / 128)`` independent CTAs per packed head, each walking the
KV range.
The graph adapter and direct kernel callers use one explicit pointer ABI.
``compile()`` specializes only head dimensions, Stats presence/layout and the
paged-pool layout; shapes and strides are bound at launch.
"""

from cudnn.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
from functools import lru_cache
from typing import Callable, Optional, Tuple

from cutlass.experimental import primitives as nvvm
from cutlass.experimental.primitives import vote_sync, VoteSync
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass._mlir.dialects import arith

import cutlass
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver  # noqa: F401  (cute.compile pulls cuda)

from dataclasses import dataclass

from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d128_decode

# The template loader (api_dsl._load_kernel_module) injects FROST_TEMPLATE_PARAMS
# as a module global before this body executes; a plain import falls back to
# the decode tile's minimal legal params so the file stays importable on its own.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams(cta_mma=1))
CFG, _TMA = make_cfg_d128_decode(PARAMS)
Cfg = type(CFG)
TMA_QK_ITERS = _TMA.QK_ITERS
TMA_VO_ITERS = _TMA.VO_ITERS
TMA_QK_GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _TMA.VO_GRANU_ELEMS

# O TMA box / store params follow O's swizzle.
TMA_O_GRANU_ELEMS_HOST = CFG.O_SWZ_BYTES // CFG.BPE
TMA_O_ITERS_HOST = (CFG.TILE_O * CFG.BPE) // CFG.O_SWZ_BYTES

from cudnn.frost.tile_dsl.barrier import (
    PipelineState,
    advance,
    wait,
)
from cudnn.frost.tile_dsl.scheduler import (
    Sched,
    scheduler_warp_loop,
    read_tile_id_arrive,
    read_clc_payload,
    SCHED_NATURAL,
)
from cudnn.frost.tile_dsl.pointwise import (
    row_reduction_pair,
    row_max_reduction,
    vec_scale_pair,
)
from cudnn.frost.tile_dsl.regtile import RegTile, vec_concat
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts_step
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_tile, tma_store_commit, tma_store_wait
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, GmemTileTma
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.frost.tile_dsl.mask import (
    apply_mask_chunk,
    MASK_NONE,
)

# Storage dtype + MMA kind dispatch — folded at trace time on CFG.DTYPE_QKV.
if CFG.DTYPE_QKV == 2:
    STORAGE_DTYPE = cutlass.BFloat16
    MMA_KIND = nvvm.Tcgen05MMAKind.F16
elif CFG.DTYPE_QKV == 3:
    STORAGE_DTYPE = cutlass.Float16
    MMA_KIND = nvvm.Tcgen05MMAKind.F16
else:
    raise ValueError(f"decode_d128_f16_sm100: DTYPE_QKV={CFG.DTYPE_QKV} not supported (expected 2=BF16 or 3=FP16)")


from cudnn.sdpa.fwd.kernels._common_blackwell import (
    make_split_helpers,
    sdpa_operand_tensors,
    store_fp32_partial_tile as _store_fp32_partial_tile,
    make_classic_bars,
    row_max_for_exp2,
    make_sdpa_helpers,
)

if CFG.CTA_MMA != 1 or CFG.TILES_Q != 1 or CFG.THD_VARLEN:
    # make_cfg_d128_decode already enforces this; the module-level check keeps
    # the body's assumptions (no peer CTA, one sub-tile, dense) visible.
    raise ValueError("decode_d128_f16_sm100: the decode tile is cga1, TILES_Q=1 and dense-only")

CGA_SIZE = 1
CTA_GROUP_KIND = nvvm.CTAGroup.CTA_1

# Per-CTA buffer element counts + TMA transaction byte counts (cga1: whole tiles).
qBufferElems = CFG.TILE_M * CFG.TILE_K
kBufferElems = CFG.TILE_N * CFG.TILE_K
vBufferElems = CFG.TILE_O * CFG.TILE_N
oBufferElems = CFG.TILE_M * CFG.TILE_O

# Q u O alias: O reuses Q's SMEM slab once BMM1 has consumed Q (the slab is
# sized to the larger; d_qk == d_v here so both are qBufferElems).  TMA-STG
# arrives mb_q_o_alias after O drains; TMA-LDG waits it before reloading Q.
QO_SLAB_ELEMS = max(qBufferElems, oBufferElems)

qTmaTransactionBytes = qBufferElems * CFG.BPE
kTmaTransactionBytes = kBufferElems * CFG.BPE
vTmaTransactionBytes = vBufferElems * CFG.BPE

# Q rows one CTA (== one cluster at cga1) covers: 128.
CGA_TILE_M = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA

_sdpa_h = make_sdpa_helpers(CFG, lpt_q_tiles_in_cga_units=True)
# qtrim variant: collapses the KV loop for tiles entirely past the per-batch
# actual Q length (SEQ_Q_LENS_PRESENT; folds to plain bounds otherwise).
_bounds_for_tile = _sdpa_h.bounds_for_tile_qtrim
_resolve_seqlen_kv = _sdpa_h.resolve_seqlen_kv
_resolve_seqlen_q = _sdpa_h.resolve_seqlen_q
_dispatch_decode_initial = _sdpa_h.dispatch_decode_initial
_dispatch_decode_payload = _sdpa_h.dispatch_decode_payload

# === PackGQA ===
#
# HEADS_PER_TILE = CFG.PACK_G heads share one token row-group of the Q tile
# (row r <-> token r // PACK_G, head r % PACK_G): the whole GQA group when it
# divides TILE_M, else its largest divisor that does (partial PackGQA -- 96/8
# packs 4 of its 12 heads, so the grid's head axis holds QH / 4 packed heads
# and PACKED_HEADS_PER_KV = 3 of them read the same KV head).  CFG.QH_PER_KH
# stays the graph's GQA ratio (it drives the bottom-right diagonal).  The
# bounds helpers take HEADS_PER_TILE (the packed group), never QH_PER_KH --
# the same mapping as sm100/prefill_d128_f16.py.
HEADS_PER_TILE = CFG.PACK_G if CFG.PACK_GQA else 1
TOKENS_PER_TILE = CFG.TILE_M // HEADS_PER_TILE
PACKED_HEADS_PER_KV = CFG.QH_PER_KH // HEADS_PER_TILE if CFG.PACK_GQA else 1

# === KV split === (shared mechanics; see prefill_d128_f16.py and _common_blackwell)
_split_h = make_split_helpers(
    CFG,
    bounds_for_tile=_bounds_for_tile,
    dispatch_decode_initial=_dispatch_decode_initial,
    dispatch_decode_payload=_dispatch_decode_payload,
)
SPLIT_KV = _split_h.SPLIT_KV
_FP32_PARTIALS = SPLIT_KV > 1
MAY_BE_EMPTY = _split_h.MAY_BE_EMPTY
_decode_initial_split = _split_h.decode_initial_split
_decode_payload_split = _split_h.decode_payload_split
_bounds_for_tile_split = _split_h.bounds_for_tile_split
_nomask_range_split = _split_h.nomask_range_split
_partial_batch = _split_h.partial_batch

# === Paged KV ===
#
# At cga1 a K tile is the full TILE_N rows (no pair to split it with) and so is
# V; each is loaded as a stack of ``KV_BOXES`` row boxes of ``KV_BOX_ROWS`` rows so
# that no box straddles a page (page_size | 128 or 128 | page_size, validated).
PAGED_KV = bool(CFG.PAGED_KV)
PAGE_SIZE = CFG.PAGE_SIZE if PAGED_KV else 0
KV_BOX_ROWS = min(PAGE_SIZE, CFG.TILE_N) if PAGED_KV else CFG.TILE_N
KV_BOXES = CFG.TILE_N // KV_BOX_ROWS


@dataclass(frozen=True)
class KernelTmemLayout:
    """Column offsets for the single-sub-tile, two-S-slot decode pipeline.

    S slot ``s`` (s in 0..S_STAGES-1) is the 128 columns at ``s * S_SLOT_COLS``;
    fp16/bf16 P packs 2 probs per fp32 cell, so P (64 cols) aliases each slot's
    tail at ``+P_IN_SLOT_OFF`` exactly as the prefill layout does.  KV tile i
    uses slot i % 2 (all three roles walk the same 2-stage PipelineState), so
    BMM1(i+1) can be issued into the free slot while softmax still reads S(i).
    O accumulates at O0_OFF.  The per-iteration alpha and the per-tile
    (total_max, total_sum) stats live at STATS_OFF, a column no MMA writes --
    the prefill's head-of-S-slot placement would alias the alternating slots.
    """

    TOTAL_COLS: int = 512
    S_STAGES: int = 2
    S_SLOT_COLS: int = 128
    P_IN_SLOT_OFF: int = 64
    O0_OFF: int = 256
    STATS_OFF: int = 384


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
    # Dense padded-Q trim: separate (B,)-int32 per-batch Q lengths (cuDNN
    # SEQLEN_Q / FA seqused_q style); reads fold out unless CFG.SEQ_Q_LENS_PRESENT.
    seq_q_lens_addr: cutlass.Int64 = 0,
    o_partial_f32: Optional[cute.Tensor] = None,
    # Paged KV: [B, max_pages] int32 page ids per batch, one table for K and
    # one for V. None (folded out of the ABI) unless CFG.PAGED_KV.
    block_table_tensor: Optional[cute.Tensor] = None,
    block_table_v_tensor: Optional[cute.Tensor] = None,
    # Paged KV: HND pool (row stride below head stride) -> descriptor dims
    # (D, row, H_kv, page); derived by _host from the bound strides.
    paged_hnd: cutlass.Constexpr[bool] = False,
) -> None:
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    # SMEM in Q u O / K / V order — Tcgen05SmemDesc.build truncates start_address
    # past ~256 KiB so this order keeps the MMA operands in low SMEM.
    sQ_raw = cutlass.Array(STORAGE_DTYPE, CFG.TILES_Q * QO_SLAB_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    sO_raw = sQ_raw
    sK_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * kBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sV_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * vBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)

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
        tma_subtile_stride_elems=CFG.TILE_N * TMA_QK_GRANU_ELEMS,
    )
    sV = SmemTile(
        base=sV_raw,
        elems_per_stage=vBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_PV,
        stride_byte_offset=STRIDE_BYTE_OFFSET_PV,
        layout=SMEM_LAYOUT_V,
        tma_loads_per_tile=TMA_VO_ITERS,
        tma_granu_elems=TMA_VO_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_N * TMA_VO_GRANU_ELEMS,
    )
    sO = SmemTile(
        base=sO_raw,
        elems_per_stage=QO_SLAB_ELEMS,
        stages=CFG.TILES_Q,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=SMEM_LAYOUT_QKO,
        tma_loads_per_tile=TMA_O_ITERS_HOST,
        tma_granu_elems=TMA_O_GRANU_ELEMS_HOST,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_O_GRANU_ELEMS_HOST,
    )

    bars = make_classic_bars(CFG, s_stages=LAYOUT.S_STAGES)

    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)

    # tile_id_smem stride 8 Int32/stage (32 B) = 16 B try_cancel payload + 16 B padding.
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

    # 4 softmax + 4 corr + MMA + TMALDG + TMASTG warps arrive per tile (cga1).
    READ_TILE_ARRIVERS_TOTAL = CFG.READ_TILE_ARRIVERS

    if warp_idx == 0:
        if nvvm.elect_sync():
            bars.mb_q_full[0].init()
            bars.mb_q_empty[0].init()
            bars.mb_bmm2_done[0].init()
            bars.mb_stat_full[0].init()
            bars.mb_stat_empty[0].init()
            bars.mb_stats_read[0].init()
            bars.mb_o_full[0].init()
            bars.mb_o_empty[0].init()
            bars.mb_q_o_alias[0].init()
            bars.mb_qo_slab_free[0].init()
            for ss in cutlass.range_constexpr(LAYOUT.S_STAGES):
                bars.mb_bmm1_done[ss].init()
                for c in cutlass.range_constexpr(CFG.N_BMM2_CHUNKS):
                    bars.mb_bmm2_ready[ss * CFG.N_BMM2_CHUNKS + c].init()
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

    # cga1: no peer.  The shared helpers take these as values; all fold to "self".
    cta_in_pair = cutlass.Int32(0)
    leader_cta_id = cutlass.Int32(0)
    mcast_mask = cutlass.Int32(0)
    tma_mcast_mask = cutlass.Int16(0)

    # === Per-warp role dispatch ===
    if warp_idx >= CFG.SOFTMAX_WG0_BASE and warp_idx < CFG.SOFTMAX_WG0_BASE + CFG.SOFTMAX_WG_WARPS:
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _softmax_warp_group(
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            scale_log2=scale_softmax_log2,
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            sched=sched,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
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
            bars=bars,
            sched=sched,
            lse_tensor=lse_tensor,
            sinks_tensor=sinks_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            leader_cta_id=leader_cta_id,
            cta_in_pair=cta_in_pair,
            qh_per_kh=qh_per_kh,
            o_partial_f32=o_partial_f32,
        )

    elif warp_idx == CFG.MMA_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
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
            seq_q_lens_addr=seq_q_lens_addr,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            mcast_mask=mcast_mask,
            cta_in_pair=cta_in_pair,
            qh_per_kh=qh_per_kh,
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
            sQ=sQ,
            sK=sK,
            sV=sV,
            bars=bars,
            sched=sched,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            qh_per_kh=qh_per_kh,
            cta_in_pair=cta_in_pair,
            tma_mcast_mask=tma_mcast_mask,
            block_table_tensor=block_table_tensor,
            block_table_v_tensor=block_table_v_tensor,
            paged_hnd=paged_hnd,
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
            qh_per_kh=qh_per_kh,
            seqlen_kv=seqlen_kv,
        )

    else:  # warp_idx == CFG.SCHED_WARP_ID
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        # cga1: this CTA is always the cluster's first CTA.
        is_cga_first_cta = cutlass.Int32(1) == cutlass.Int32(1)
        scheduler_warp_loop(sched, CFG.SCHEDULER_STAGES, is_cga_first_cta, CGA_SIZE)


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


# === TMA-LDG warp ===


@cute.jit
def _paged_load_tile(
    smem_tile,
    tma,
    block_table_tensor,
    batch_idx,
    kv_head_idx,
    n_pages_b,
    kv_tile,
    mbar,
    granu_elems: cutlass.Constexpr[int],
):
    """Issue one paged K or V tile (TILE_N rows) as ``KV_BOXES`` row boxes through the block table.

    Box ``j`` covers sequence rows ``[kv_tile*TILE_N + j*KV_BOX_ROWS, +KV_BOX_ROWS)``
    and lands at that row offset in SMEM (128 B swizzle repeats every 8 rows and
    box rows are a multiple of 8, so stacked boxes equal one tall box).  A box
    whose page slot is at or past this batch's live page count gets page ``-1``:
    TMA-OOB, zero-filled, bytes still credited to ``mbar``.  The block-table read
    itself is clamped into the live range so it never leaves the row.
    """
    bt = cutlass.make_array_view(block_table_tensor)
    last_live = cute.math.max(n_pages_b - cutlass.Int32(1), cutlass.Int32(0))
    for j in cutlass.range_constexpr(KV_BOXES):
        g = kv_tile * cutlass.Int32(CFG.TILE_N) + cutlass.Int32(j * KV_BOX_ROWS)
        slot = g // cutlass.Int32(PAGE_SIZE)
        row_in_page = g % cutlass.Int32(PAGE_SIZE)
        page_live = cutlass.Int32(bt[batch_idx, cute.math.min(slot, last_live)])
        in_range = slot < n_pages_b
        page = cutlass.Int32(arith.select(in_range.ir_value(), page_live.ir_value(), cutlass.Int32(-1).ir_value()))
        tma_load_tile(
            smem_tile.shifted(j * KV_BOX_ROWS * granu_elems),
            tma(cutlass.Int32(0), kv_head_idx, row_in_page, page),
            mbar,
            cta_group=1,
            mcast_mask=cutlass.Int16(0),
        )


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
    seq_q_lens_addr,
    n_q_supers,
    n_qh,
    n_batch,
    qh_per_kh,
    cta_in_pair,
    tma_mcast_mask,
    block_table_tensor=None,
    block_table_v_tensor=None,
    paged_hnd: cutlass.Constexpr[bool] = False,
):
    """TMA-LDG warp: one Q slab per tile, then K/V through the STAGES_KV ring."""
    q_empty_phase = cutlass.Int32(1)
    kv_state = PipelineState.start(phase=1)

    # Q-reload gate: the next tile's Q load waits for the prior tile's O
    # (sharing the slab) to drain — TMA-STG fires mb_q_o_alias, strictly after
    # MMA consumed Q.  Bootstraps consumer-side via q_empty_phase=1.
    mb_q_reload = bars.mb_q_o_alias

    tma_q = GmemTileTma(tma_q_desc)
    if cutlass.const_expr(PAGED_KV and paged_hnd):
        # HND page pools: the row stride is below the head stride, so _host
        # built the descriptors with dims (D, row, H_kv, page). Every load site
        # keeps the (d, head, row, page) vocabulary; the swap lives here.
        _tk, _tv = GmemTileTma(tma_k_desc), GmemTileTma(tma_v_desc)
        tma_k = lambda d, h, r, p: _tk(d, r, h, p)  # noqa: E731
        tma_v = lambda d, h, r, p: _tv(d, r, h, p)  # noqa: E731
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
    # GQA: K/V are indexed by kv-head; with PackGQA the decoded head_idx is the
    # PACKED head (Q head base = head_idx * PACK_G), the KV head is packed head
    # // PACKED_HEADS_PER_KV (== head_idx when the whole group packs) and
    # q_row_base is in TOKEN units (rows // PACK_G).
    q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
    kv_head_idx = cute.arch.make_warp_uniform(
        (head_idx if cutlass.const_expr(PACKED_HEADS_PER_KV == 1) else head_idx // cutlass.Int32(PACKED_HEADS_PER_KV))
        if cutlass.const_expr(CFG.PACK_GQA)
        else head_idx // qh_per_kh
    )
    q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(TOKENS_PER_TILE))

    if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    elif cutlass.const_expr(CFG.MASK_FLAGS == 0):
        kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
    else:
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
        bounds_init = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    # Paged KV: this batch's live page count bounds the block-table walk (MASK_PADDED
    # is mandatory under PAGED_KV, so eff_seqlen_kv is always defined here).
    n_pages_b = cutlass.Int32(0)
    if cutlass.const_expr(PAGED_KV):
        n_pages_b = (eff_seqlen_kv + cutlass.Int32(PAGE_SIZE - 1)) // cutlass.Int32(PAGE_SIZE)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            # TMA-STG advances the Q u O alias gate for EVERY tile, empty ones
            # included (correction still writes O := 0 and STG still drains it).
            # Consume that phase even though no Q reload is needed, or the next
            # live tile's wait passes on stale parity and its Q load races the
            # previous tile's O drain.  mb_qo_slab_free is the return edge.
            mb_q_reload[0].wait(q_empty_phase)
            bars.mb_qo_slab_free[0].arrive()
            q_empty_phase = q_empty_phase ^ 1
        else:
            mb_q_reload[0].wait(q_empty_phase)
            bars.mb_qo_slab_free[0].arrive()
            bars.mb_q_full[0].arrive(n_bytes=qTmaTransactionBytes, pred=nvvm.elect_sync())
            tma_load_tile(
                sQ[0],
                tma_q(cutlass.Int32(0), q_head_idx, q_row_base, batch_idx),
                bars.mb_q_full[0].smem_ptr,
                cta_group=1,
                mcast_mask=tma_mcast_mask,
            )
            q_empty_phase = q_empty_phase ^ 1

            for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
                kv_row_base = kv_loop * CFG.TILE_N

                bars.mb_k_empty[kv_state.idx].wait(kv_state.phase)
                bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=nvvm.elect_sync())
                if cutlass.const_expr(PAGED_KV):
                    _paged_load_tile(
                        sK[kv_state.idx],
                        tma_k,
                        block_table_tensor,
                        batch_idx,
                        kv_head_idx,
                        n_pages_b,
                        kv_loop,
                        bars.mb_k_full[kv_state.idx].smem_ptr,
                        TMA_QK_GRANU_ELEMS,
                    )
                else:
                    tma_load_tile(
                        sK[kv_state.idx],
                        tma_k(cutlass.Int32(0), kv_head_idx, kv_row_base, batch_idx),
                        bars.mb_k_full[kv_state.idx].smem_ptr,
                        cta_group=1,
                        mcast_mask=tma_mcast_mask,
                    )

                bars.mb_v_empty[kv_state.idx].wait(kv_state.phase)
                bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=nvvm.elect_sync())
                if cutlass.const_expr(PAGED_KV):
                    _paged_load_tile(
                        sV[kv_state.idx],
                        tma_v,
                        block_table_v_tensor,
                        batch_idx,
                        kv_head_idx,
                        n_pages_b,
                        kv_loop,
                        bars.mb_v_full[kv_state.idx].smem_ptr,
                        TMA_VO_GRANU_ELEMS,
                    )
                else:
                    tma_load_tile(
                        sV[kv_state.idx],
                        tma_v(cutlass.Int32(0), kv_head_idx, kv_row_base, batch_idx),
                        bars.mb_v_full[kv_state.idx].smem_ptr,
                        cta_group=1,
                        mcast_mask=tma_mcast_mask,
                    )
                kv_state = advance(kv_state, CFG.STAGES_KV)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        nxt_q = cute.arch.make_warp_uniform(nxt_q)
        nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
        nxt_v = cute.arch.make_warp_uniform(nxt_v)
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
        kv_head_idx = cute.arch.make_warp_uniform(
            (head_idx if cutlass.const_expr(PACKED_HEADS_PER_KV == 1) else head_idx // cutlass.Int32(PACKED_HEADS_PER_KV))
            if cutlass.const_expr(CFG.PACK_GQA)
            else head_idx // qh_per_kh
        )
        q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(TOKENS_PER_TILE))
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV > 1):
            kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
        elif cutlass.const_expr(CFG.MASK_FLAGS != 0):
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
            bounds_next = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)
            kv_left = bounds_next.left
            kv_right = bounds_next.right
            if cutlass.const_expr(PAGED_KV):
                n_pages_b = (eff_seqlen_kv + cutlass.Int32(PAGE_SIZE - 1)) // cutlass.Int32(PAGE_SIZE)


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
    qh_per_kh,
    seqlen_kv,
):
    """Persistent O-store warp.  First tile from blockIdx; subsequent tiles
    via the scheduler warp's clusterlaunchcontrol.try_cancel.async.
    """
    o_full_phase = cutlass.Int32(0)  # consumer waits — first-arrive flips 0 → 1
    # Q u O alias return edge (consumer side): LDG arrives mb_qo_slab_free after
    # consuming each alias phase; this warp waits it before its next alias
    # arrive, so it can never lap LDG by two phases (mbarrier parity deadlock).
    slab_free_phase = cutlass.Int32(0)

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

        q_row_base = q_super_idx * cutlass.Int32(TOKENS_PER_TILE)
        q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
        # KV split: partials are stacked split-major on the BATCH axis of the
        # workspace (extent B*SPLIT_KV), so the store needs no new descriptor —
        # only a shifted batch coord.  Folds to batch_idx at SPLIT_KV == 1.
        o_batch = _partial_batch(batch_idx, split_idx, n_batch)

        bars.mb_o_full[0].wait(o_full_phase)
        # fp32 partials wrote the workspace directly, so there is nothing staged
        # to copy.  Skip ONLY the store: the arrive below and the alias handshake
        # after it must still run, or this warp laps the loader and the parity
        # waits deadlock.
        if cutlass.const_expr(not _FP32_PARTIALS):
            tma_store_tile(sO[0], tma_o(cutlass.Int32(0), q_head_idx, q_row_base, o_batch))
            tma_store_commit()
            tma_store_wait(0)

        bars.mb_o_empty[0].arrive()
        # O has drained to GMEM → the shared Q u O slab is free for TMA-LDG to
        # clobber with the next tile's Q.  Throttled by the return edge: at most
        # ONE alias phase ahead of LDG.
        bars.mb_qo_slab_free[0].wait(slab_free_phase)
        bars.mb_q_o_alias[0].arrive()

        o_full_phase = o_full_phase ^ 1
        slab_free_phase = slab_free_phase ^ 1

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
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


# === BMM1 / BMM2 SMEM + idesc constants ===

# Per-tensor swizzle layout enum: Swz128B=2, Swz64B=4, Swz32B=6.
_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
SMEM_LAYOUT_QKO = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_V = _SWZ_ENUM[CFG.V_SWZ_BYTES]

# O SMEM swizzle: third param is the XOR shift offset (=3 across all widths), NOT the B value.
_O_SWZ_B = {128: 3, 64: 2, 32: 1}[CFG.O_SWZ_BYTES]
_O_SMEM_SWIZZLE = cutlass.Swizzle(_O_SWZ_B, 4, 3)

LEADING_BYTE_OFFSET_QK = 0
STRIDE_BYTE_OFFSET_QK = 8 * CFG.Q_SWZ_BYTES

# leading_byte_offset = 0 when TILE_O/8 <= 8 else TILE_N*V_SWZ_BYTES (cga1: full 128 d_v columns per CTA).
_CORE_MATRIX_ROWS = 8
LEADING_BYTE_OFFSET_PV = 0 if (CFG.TILE_O // _CORE_MATRIX_ROWS) <= 8 else CFG.TILE_N * CFG.V_SWZ_BYTES
STRIDE_BYTE_OFFSET_PV = 8 * CFG.V_SWZ_BYTES

NUM_KPHASES_PV = CFG.TILE_N // CFG.TILE_K_HW_BMM2
NUM_KPHASES_PV_PER_CHUNK = NUM_KPHASES_PV // CFG.N_BMM2_CHUNKS


@cute.jit
def _s_slot_ptr(tmem_base_i32, slot_idx, in_slot_off: cutlass.Constexpr[int]):
    """TMEM pointer at column ``slot_idx * S_SLOT_COLS + in_slot_off`` (the S slot
    ring is walked with a runtime index; the address math stays on the Int32
    before the pointer is formed, as the softmax path does)."""
    return nvvm.make_tmem_ptr(tmem_base_i32 + slot_idx * cutlass.Int32(LAYOUT.S_SLOT_COLS) + cutlass.Int32(in_slot_off), cutlass.Int8)


@cute.jit
def _mma_bmm2_tile(bmm2_desc, tmem_base_i32, tmem_raw, desc_V, bars, s_cons, accum_first, mcast_mask):
    """P @ V for one KV tile from S slot ``s_cons.idx``: chunk 0 as soon as
    softmax published it, chunk 1 after the second bmm2_ready arrive; then
    bmm2_done for correction."""
    n_chunks = cutlass.Int32(CFG.N_BMM2_CHUNKS)
    tmem_P = _s_slot_ptr(tmem_base_i32, s_cons.idx, LAYOUT.P_IN_SLOT_OFF)
    bars.mb_bmm2_ready[s_cons.idx * n_chunks].wait(s_cons.phase)
    accum_b2 = accum_first
    for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
        mma_ts_step(bmm2_desc, tmem_P, desc_V, (tmem_raw.subview(LAYOUT.O0_OFF)), local_k, accum_b2)
        accum_b2 = cutlass.Boolean(True)
    if cutlass.const_expr(CFG.N_BMM2_CHUNKS == 2):
        bars.mb_bmm2_ready[s_cons.idx * n_chunks + cutlass.Int32(1)].wait(s_cons.phase)
        for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
            mma_ts_step(
                bmm2_desc,
                tmem_P,
                desc_V,
                (tmem_raw.subview(LAYOUT.O0_OFF)),
                NUM_KPHASES_PV_PER_CHUNK + local_k,
                cutlass.Boolean(True),
            )
    elect_p = nvvm.elect_sync()
    bars.mb_bmm2_done[0].arrive(mcast_mask=mcast_mask, cta_group=1, pred=elect_p)


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
    seq_q_lens_addr,
    n_q_supers,
    n_qh,
    n_batch,
    mcast_mask,
    cta_in_pair,
    qh_per_kh,
):
    """MMA warp (cga1, one Q sub-tile, two S slots): per KV tile i >= 1 issue
    BMM1(i) into slot i % 2 as soon as K(i) has landed, THEN BMM2(i-1) from the
    other slot's P.  BMM1(i) overwrites the slot BMM2(i-2) read as P; that MMA
    was issued in the previous loop iteration and tcgen05 executes in order --
    the prefill kernel's S/P aliasing discipline, one slot further apart.  The
    tensor core therefore runs BMM1(i) while the softmax warpgroup is still on
    S(i-1), instead of idling behind it.
    """
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    nvvm.barrier_cta_arrive(2, 32 * (CFG.CORRECTION_WARPS + 1))

    tmem_base_i32 = tmem_ptr_i32.load()
    tmem_raw = nvvm.make_tmem_ptr(tmem_base_i32, cutlass.Int8)

    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_N,
        m_dim=CFG.TILE_M,
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_O,
        m_dim=CFG.TILE_M,
        b_major=1,
    )
    bmm1_desc = MmaDesc(
        M=CFG.TILE_M,
        N=CFG.TILE_N,
        K=CFG.TILE_K,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM1,
        btranspose=False,
        cta_group=1,
        idesc=idesc_qk,
        kind=MMA_KIND,
    )
    bmm2_desc = MmaDesc(
        M=CFG.TILE_M,
        N=CFG.TILE_O,
        K=CFG.TILE_N,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM2,
        btranspose=True,
        k_subtile=CFG.V_SWZ_BYTES // CFG.BPE,
        cta_group=1,
        idesc=idesc_pv,
        kind=MMA_KIND,
    )

    desc_Q0 = sQ[0].desc()

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
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
            bounds_init = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)
            kv_left = bounds_init.left
            kv_right = bounds_init.right

    q_full_phase = cutlass.Int32(0)
    kv_state = PipelineState.start(phase=0)
    # S slot ring: s_prod is the slot the next BMM1 writes, s_cons the slot the
    # next BMM2 reads P from (one KV tile behind).  Both persist across tiles,
    # in lockstep with the softmax and correction cursors.
    s_prod = PipelineState.start(phase=0)
    s_cons = PipelineState.start(phase=0)
    empty_mainloop_phase = cutlass.Int32(0)
    # Stats-consumed gate (one flip per tile; the stats live at their own
    # column here, so this only keeps the correction epilogue's read ordered
    # before the next tile begins).  Bootstrap phase 1.
    stats_read_phase = cutlass.Int32(1)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            # Empty mainloop: fire bmm2_done so softmax/correction phase trackers stay in lockstep.
            bars.mb_empty_mainloop.wait(empty_mainloop_phase)
            empty_mainloop_phase = empty_mainloop_phase ^ cutlass.Int32(1)
            bars.mb_stats_read[0].wait(stats_read_phase)
            elect_p = nvvm.elect_sync()
            bars.mb_bmm2_done[0].arrive(mcast_mask=mcast_mask, cta_group=1, pred=elect_p)
        else:
            # Prologue: BMM1 for kv = kv_left into slot s_prod.  mb_stats_read
            # keeps the correction epilogue's read of the previous tile's final
            # stats ordered before this tile starts.
            bars.mb_q_full[0].wait(q_full_phase)
            bars.mb_k_full[kv_state.idx].wait(kv_state.phase)
            bars.mb_stats_read[0].wait(stats_read_phase)
            desc_K = sK[kv_state.idx].desc()
            mma_ss(bmm1_desc, desc_Q0, desc_K, _s_slot_ptr(tmem_base_i32, s_prod.idx, 0))
            elect_p = nvvm.elect_sync()
            bars.mb_bmm1_done[s_prod.idx].arrive(mcast_mask=mcast_mask, cta_group=1, pred=elect_p)
            bars.mb_k_empty[kv_state.idx].arrive(mcast_mask=mcast_mask, cta_group=1, pred=elect_p)
            s_prod = advance(s_prod, LAYOUT.S_STAGES)

            q_full_phase = q_full_phase ^ 1

            # Mainloop kv = kv_left+1 .. kv_right-1 (empty when n_kv == 1)
            for kv_loop in cutlass.range(kv_left + cutlass.Int32(1), kv_right, 1, unroll=1):
                old_state = kv_state
                kv_state = advance(kv_state, CFG.STAGES_KV)

                # BMM1 for THIS KV tile into the free slot -- ahead of the
                # previous tile's BMM2, so it overlaps softmax(kv_loop - 1).
                bars.mb_k_full[kv_state.idx].wait(kv_state.phase)
                desc_K = sK[kv_state.idx].desc()
                mma_ss(bmm1_desc, desc_Q0, desc_K, _s_slot_ptr(tmem_base_i32, s_prod.idx, 0))
                elect_p = nvvm.elect_sync()
                bars.mb_bmm1_done[s_prod.idx].arrive(mcast_mask=mcast_mask, cta_group=1, pred=elect_p)
                bars.mb_k_empty[kv_state.idx].arrive(mcast_mask=mcast_mask, cta_group=1, pred=elect_p)
                s_prod = advance(s_prod, LAYOUT.S_STAGES)

                # BMM2 for the PREVIOUS KV tile (P in slot s_cons -> O0), then release its V stage.
                bars.mb_v_full[old_state.idx].wait(old_state.phase)
                desc_V = sV[old_state.idx].desc()
                is_not_first_bmm2 = cutlass.Boolean(kv_loop != (kv_left + cutlass.Int32(1)))
                _mma_bmm2_tile(bmm2_desc, tmem_base_i32, tmem_raw, desc_V, bars, s_cons, is_not_first_bmm2, mcast_mask)
                elect_p = nvvm.elect_sync()
                bars.mb_v_empty[old_state.idx].arrive(mcast_mask=mcast_mask, cta_group=1, pred=elect_p)
                s_cons = advance(s_cons, LAYOUT.S_STAGES)

            # Epilogue: BMM2 for the last kv tile (always runs — n_kv >= 1).  The
            # Q-reload gate is mb_q_o_alias (fired by TMA-STG after O drains), so
            # MMA does NOT fire mb_q_empty.
            bars.mb_v_full[kv_state.idx].wait(kv_state.phase)
            desc_V = sV[kv_state.idx].desc()
            is_not_first_bmm2_epi = cutlass.Boolean((kv_right - kv_left) != cutlass.Int32(1))
            _mma_bmm2_tile(bmm2_desc, tmem_base_i32, tmem_raw, desc_V, bars, s_cons, is_not_first_bmm2_epi, mcast_mask)
            elect_p = nvvm.elect_sync()
            bars.mb_v_empty[kv_state.idx].arrive(mcast_mask=mcast_mask, cta_group=1, pred=elect_p)
            s_cons = advance(s_cons, LAYOUT.S_STAGES)
            kv_state = advance(kv_state, CFG.STAGES_KV)

        # One correction-epilogue arrive per tile — flip once per tile.
        stats_read_phase = stats_read_phase ^ 1

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
            _nq, _nh, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
            is_valid_tile = nxt_v & cutlass.Int32(1)
        else:
            nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
            nxt_q = cute.arch.make_warp_uniform(nxt_q)
            nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
            nxt_v = cute.arch.make_warp_uniform(nxt_v)
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
                eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
                bounds_next = _bounds_for_tile_split(
                    q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE
                )
                kv_left = bounds_next.left
                kv_right = bounds_next.right
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _softmax_kv_body(
    apply_mask: bool,
    kv_loop,
    tmem_ptr_i32,
    bars,
    q_abs,
    eff_seqlen_kv,
    eff_seqlen_q,
    scale_log2,
    total_max,
    total_max_safe,
    total_sum,
    s_idx,
    s_phase,
    stat_empty_phase,
    leader_cta_id,
):
    """Per-iter kv body for the (single) softmax warp group.

    Compile-time apply_mask (Python bool): True runs tcgen05.ld +
    apply_mask_chunk + software row-max; False the unmasked load + row-max.
    total_max runs in scaled (log2) units and starts at -inf; total_max_safe is
    its 0-substituted companion (see row_max_for_exp2).  ``(s_idx, s_phase)`` is
    the S slot ring cursor (a 2-stage PipelineState, unpacked): S is read from
    and P written to slot ``s_idx``; the cursor advances once per KV tile.

    Returns updated (total_max, total_max_safe, total_sum, s_idx, s_phase,
    stat_empty_phase).
    """
    CHUNK = 64
    P_COLS_PER_CHUNK = CHUNK // 2  # fp16/bf16 pack 2 probs per FP32 TMEM cell
    N_CHUNKS = CFG.N_BMM2_CHUNKS
    RESCALE_THRESHOLD = cutlass.Float32(CFG.RESCALE_THRESHOLD)

    bars.mb_bmm1_done[s_idx].wait(s_phase)

    tmem_base = tmem_ptr_i32.load()
    s_addr_base = tmem_base + s_idx * cutlass.Int32(LAYOUT.S_SLOT_COLS)
    p_addr_base = s_addr_base + cutlass.Int32(LAYOUT.P_IN_SLOT_OFF)
    stats_addr = tmem_base + cutlass.Int32(LAYOUT.STATS_OFF)
    n_chunks_i = cutlass.Int32(N_CHUNKS)

    if cutlass.const_expr(apply_mask):
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
                mask_value=float("-inf"),
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

    reg_S = RegTile(reg_S_vec, size=CFG.TILE_N)
    current_max = current_max_unscaled * scale_log2  # -inf when the whole iteration is masked

    # Online softmax with RESCALE_THRESHOLD skip (see prefill_d128_f16.py).
    update_cond = (current_max - total_max) > RESCALE_THRESHOLD
    total_max = cutlass.Float32(
        arith.select(
            update_cond.ir_value(),
            current_max.ir_value(),
            total_max.ir_value(),
        )
    )
    new_total_max_safe = row_max_for_exp2(total_max)
    alpha = cute.math.exp2(
        cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)),
        fastmath=True,
    )
    total_max_safe = new_total_max_safe

    alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
    bars.mb_stat_full[0].arrive()

    reg_S = reg_S * scale_log2 - total_max_safe

    chunk_S_0 = reg_S[0:CHUNK].vec
    chunk_P_0 = cute.math.exp2(chunk_S_0, fastmath=True)
    hoisted_sum = row_reduction_pair(chunk_P_0)
    chunk_P_0_fp16 = chunk_P_0.to(STORAGE_DTYPE)
    nvvm.tcgen05_st(
        "32x32b",
        nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32),
        chunk_P_0_fp16,
    )
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
    bars.mb_bmm2_ready[s_idx * n_chunks_i].arrive(leader_cta_id=leader_cta_id, cta_group=1)

    deferred_P_1 = None
    if cutlass.const_expr(N_CHUNKS == 2):
        chunk_S_1 = reg_S[CHUNK : 2 * CHUNK].vec
        deferred_P_1 = cute.math.exp2(chunk_S_1, fastmath=True)
        chunk_P_1_fp16 = deferred_P_1.to(STORAGE_DTYPE)
        nvvm.tcgen05_st(
            "32x32b",
            nvvm.make_tmem_ptr(
                p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK),
                cutlass.Float32,
            ),
            chunk_P_1_fp16,
        )
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        bars.mb_bmm2_ready[s_idx * n_chunks_i + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=1)

    new_p_sum_pair = hoisted_sum
    if cutlass.const_expr(N_CHUNKS == 2):
        new_p_sum_pair = new_p_sum_pair + row_reduction_pair(deferred_P_1)
    alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
    total_sum = total_sum * alpha_pair + new_p_sum_pair

    bars.mb_stat_empty[0].wait(stat_empty_phase)
    stat_empty_phase = stat_empty_phase ^ 1

    s_next = advance(PipelineState(idx=s_idx, phase=s_phase), LAYOUT.S_STAGES)
    return total_max, total_max_safe, total_sum, s_next.idx, s_next.phase, stat_empty_phase


@cute.jit
def _softmax_warp_group(
    seqlen_q,
    seqlen_kv,
    scale_log2: cutlass.Float32,
    tmem_ptr_i32,
    bars,
    sched,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    n_q_supers,
    n_qh,
    n_batch,
    leader_cta_id,
    cta_in_pair,
    qh_per_kh,
):
    """The softmax warp group: online softmax per kv iter over the 128 x 128 S tile.

    Each lane owns one row; publishes alpha to correction, writes P to the S
    slot tail, fires bmm2_ready[chunk]; end of tile publishes (max, sum).
    """
    # Wait on MMA's TMEM-publish named barrier BEFORE any tmem_ptr_i32.load().
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))

    NEG_INF = cutlass.Float32(float("-inf"))

    # Phase trackers persist across tile boundaries (barriers don't reset).
    # S slot ring cursor (consumer of bmm1_done): idx / phase, 2 stages.
    s_idx = cutlass.Int32(0)
    s_phase = cutlass.Int32(0)
    stat_empty_phase = cutlass.Int32(1)  # bootstrap pre-armed at phase 1 so first wait passes immediately
    epilogue_state = cutlass.Int32(1)

    total_max = NEG_INF
    total_max_safe = NEG_INF
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
    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
    bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)

    tid_in_wg = cute.arch.thread_idx()[0] - cutlass.Int32(CFG.SOFTMAX_WG0_BASE * 32)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        # Top-of-tile mb_o_empty wait: without it softmax can race into the
        # next tile while TMA-STG is still draining the prior tile's O slot.
        bars.mb_o_empty[0].wait(epilogue_state)
        epilogue_state = epilogue_state ^ cutlass.Int32(1)

        total_max = NEG_INF
        total_max_safe = NEG_INF
        total_sum = cutlass.Vector.from_elements(
            (cutlass.Float32(0.0), cutlass.Float32(0.0)),
            cutlass.Float32,
        )
        # PackGQA: q_abs is the row's TOKEN index (row // G).
        q_abs = q_super_idx * cutlass.Int32(TOKENS_PER_TILE) + (tid_in_wg // cutlass.Int32(HEADS_PER_TILE))
        bars.mb_stat_empty[0].wait(stat_empty_phase)
        stat_empty_phase = stat_empty_phase ^ 1
        # 3-segment kv loop: LEFT-masked / fully-unmasked / RIGHT-masked.
        if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
            for kv_loop in cutlass.range(bounds.left, bounds.right, 1, unroll=1):
                total_max, total_max_safe, total_sum, s_idx, s_phase, stat_empty_phase = _softmax_kv_body(
                    False,
                    kv_loop,
                    tmem_ptr_i32,
                    bars,
                    q_abs,
                    eff_seqlen_kv,
                    eff_seqlen_q,
                    scale_log2,
                    total_max,
                    total_max_safe,
                    total_sum,
                    s_idx,
                    s_phase,
                    stat_empty_phase,
                    leader_cta_id,
                )
        else:
            for kv_loop in cutlass.range(bounds.left, bounds.unmasked_lo, 1, unroll=1):
                total_max, total_max_safe, total_sum, s_idx, s_phase, stat_empty_phase = _softmax_kv_body(
                    True,
                    kv_loop,
                    tmem_ptr_i32,
                    bars,
                    q_abs,
                    eff_seqlen_kv,
                    eff_seqlen_q,
                    scale_log2,
                    total_max,
                    total_max_safe,
                    total_sum,
                    s_idx,
                    s_phase,
                    stat_empty_phase,
                    leader_cta_id,
                )
            for kv_loop in cutlass.range(bounds.unmasked_lo, bounds.unmasked_hi, 1, unroll=1):
                total_max, total_max_safe, total_sum, s_idx, s_phase, stat_empty_phase = _softmax_kv_body(
                    False,
                    kv_loop,
                    tmem_ptr_i32,
                    bars,
                    q_abs,
                    eff_seqlen_kv,
                    eff_seqlen_q,
                    scale_log2,
                    total_max,
                    total_max_safe,
                    total_sum,
                    s_idx,
                    s_phase,
                    stat_empty_phase,
                    leader_cta_id,
                )
            for kv_loop in cutlass.range(bounds.unmasked_hi, bounds.right, 1, unroll=1):
                total_max, total_max_safe, total_sum, s_idx, s_phase, stat_empty_phase = _softmax_kv_body(
                    True,
                    kv_loop,
                    tmem_ptr_i32,
                    bars,
                    q_abs,
                    eff_seqlen_kv,
                    eff_seqlen_q,
                    scale_log2,
                    total_max,
                    total_max_safe,
                    total_sum,
                    s_idx,
                    s_phase,
                    stat_empty_phase,
                    leader_cta_id,
                )

        # End-of-kv: publish (total_max, total_sum_final) to TMEM Stats — corr reads it for LSE.
        total_sum_scalar = total_sum[0] + total_sum[1]

        stats_addr_epi = tmem_ptr_i32.load() + cutlass.Int32(LAYOUT.STATS_OFF)
        stats_vec_epi = cutlass.Vector.from_elements((total_max_safe, total_sum_scalar), cutlass.Float32)
        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr_epi, cutlass.Float32), stats_vec_epi)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        bars.mb_stat_full[0].arrive()

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        nxt_q = cute.arch.make_warp_uniform(nxt_q)
        nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
        nxt_v = cute.arch.make_warp_uniform(nxt_v)
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
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
        bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)


@cute.jit
def _correction_warp_group(
    seqlen_q,
    seqlen_kv,
    sO,
    tmem_ptr_i32,
    bars,
    sched,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    n_q_supers,
    n_qh,
    n_batch,
    leader_cta_id,
    cta_in_pair,
    qh_per_kh,
    o_partial_f32=None,
):
    """Correction warp group: 4 warps x 32 lanes = 128 lanes, 1 lane per O row.

    Per kv iter rescales O by alpha (skipped when all_alpha_one warp ballot
    fires); per-tile epilogue normalizes O by 1/total_sum (sink folded), casts
    to the O dtype, swizzled-stores to sO (or writes fp32 partials under a
    split), fires o_full for TMA-STG.
    """
    nvvm.barrier_cta_sync(barrier_id=2, thread_count=32 * (CFG.CORRECTION_WARPS + 1))

    tid_raw = cute.arch.thread_idx()[0]
    tid_in_wg = tid_raw - cutlass.Int32(CFG.CORR_WARP_BASE * 32)

    O_CHUNK = 16
    N_CHUNKS_O = CFG.TILE_O // O_CHUNK
    TMA_O_ITERS = (CFG.TILE_O * CFG.BPE) // CFG.O_SWZ_BYTES
    D_BLOCK_SIZE = CFG.TILE_O // TMA_O_ITERS
    TMA_O_GRANU_ELEMS = CFG.TILE_M * D_BLOCK_SIZE

    stat_full_phase = cutlass.Int32(0)
    bmm2_done_phase = cutlass.Int32(0)
    o_empty_phase = cutlass.Int32(1)  # bootstrap
    # S slot ring cursor: this warpgroup's bmm2_ready arrive (chunk 0 of the
    # tile it just rescaled O for) targets the slot that tile's P lives in.
    s_corr = PipelineState.start(phase=0)
    n_chunks_i = cutlass.Int32(CFG.N_BMM2_CHUNKS)

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
    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
    bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        # Iter-0 lifted out of kv loop: MMA's iter-0 BMM2 uses init_d=False
        # to overwrite O garbage, so alpha-rescale is unnecessary in iter 0.
        if bounds.right > bounds.left:
            bars.mb_bmm2_ready[s_corr.idx * n_chunks_i].arrive(leader_cta_id=leader_cta_id, cta_group=1)
            s_corr = advance(s_corr, LAYOUT.S_STAGES)
            bars.mb_stat_full[0].wait(stat_full_phase)
            bars.mb_stat_empty[0].arrive()
            stat_full_phase = stat_full_phase ^ 1
        else:
            # Empty-mainloop: corr fires; MMA fires bmm2_done.
            bars.mb_empty_mainloop.arrive(leader_cta_id=leader_cta_id, cta_group=1)

        for kv_loop in cutlass.range(bounds.left + cutlass.Int32(1), bounds.right, 1, unroll=1):
            tmem_base_iter = tmem_ptr_i32.load()

            bars.mb_stat_full[0].wait(stat_full_phase)

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

            bars.mb_stat_empty[0].arrive()

            bars.mb_bmm2_done[0].wait(bmm2_done_phase)

            if ~all_alpha_one:
                for chunk_idx in cutlass.range_constexpr(N_CHUNKS_O):
                    o_addr = tmem_base_iter + cutlass.Int32(LAYOUT.O0_OFF + chunk_idx * O_CHUNK)
                    o_chunk = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                        num=O_CHUNK,
                    )
                    o_scaled = vec_scale_pair(o_chunk, alpha, O_CHUNK)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(o_addr, cutlass.Float32), o_scaled)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)

            bars.mb_bmm2_ready[s_corr.idx * n_chunks_i].arrive(leader_cta_id=leader_cta_id, cta_group=1)
            s_corr = advance(s_corr, LAYOUT.S_STAGES)

            stat_full_phase = stat_full_phase ^ 1
            bmm2_done_phase = bmm2_done_phase ^ 1

        # === End-of-kv epilogue ===
        tmem_base_epi = tmem_ptr_i32.load()

        bars.mb_bmm2_done[0].wait(bmm2_done_phase)
        bars.mb_stat_full[0].wait(stat_full_phase)

        stats_addr = tmem_base_epi + cutlass.Int32(LAYOUT.STATS_OFF)
        stats_vec = nvvm.tcgen05_ld(
            "32x32b",
            nvvm.make_tmem_ptr(stats_addr, cutlass.Float32),
            num=2,
        )
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        total_max_scaled = stats_vec[0]  # log2-units (softmax stores it scaled)
        total_sum = stats_vec[1]

        # Fire stat_empty so softmax's NEXT-tile iter 0 wait can pass, and
        # release MMA's NEXT-tile prologue BMM1 into the S_acc slot (the final
        # stats are now safely in registers).
        bars.mb_stat_empty[0].arrive()
        bars.mb_stats_read[0].arrive(leader_cta_id=leader_cta_id, cta_group=1)

        inv_sum = cutlass.Float32(0.0)
        row_dead = cutlass.Float32(0.0) > cutlass.Float32(1.0)
        lse_val = cutlass.Float32(0.0)
        q_row_global = q_super_idx * cutlass.Int32(TOKENS_PER_TILE) + (tid_in_wg // cutlass.Int32(HEADS_PER_TILE))
        row_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE) + (tid_in_wg % cutlass.Int32(HEADS_PER_TILE))
        LN2 = cutlass.Float32(0.6931471805599453)
        total_max_nat = total_max_scaled * LN2
        if cutlass.const_expr(CFG.HAS_SINK):
            sinks_arr = cutlass.make_array_view(sinks_tensor)
            sink_logit = cutlass.Float32(sinks_arr[row_head_idx])
            # A keyless row -- no live key at all: an empty sequence, a row above the
            # bottom-right diagonal, a window past the last key -- holds the sink's
            # mass alone: O := 0, LSE := sink.  The softmax masks with -inf, so such a
            # row publishes total_sum == 0 (an alive row has total_sum >= 1) and the
            # 0-substituted max (row_max_for_exp2).  Computing the fold from those
            # breaks at the sink's far end: new_max = max(0, sink) = 0 and
            # exp(sink - 0) underflows to 0 in fp32 for sink <= -104, so new_sum = 0,
            # inv_sum = 1/0 = inf, O = 0 * inf = NaN, LSE = 0 + log(0) = -inf.  So the
            # two operands are SELECTED for a keyless row: new_max := sink (its exp
            # term is then exactly 1) and scale := 0 (exp(0 - sink) can overflow to
            # inf for a very negative sink, and 0 * inf is NaN); new_sum = 1, lse =
            # sink and inv_sum = 0 follow.  A row with keys takes the fold unchanged.
            # The padded-Q trim below still turns a trimmed row into O = 0 / LSE =
            # -inf, sink or not; row_dead stays False here -- it is the fp32-partial
            # store's flag, and sink + split-KV is declined, so it is never read with
            # a sink.  Same select as the four SM100 f16 prefill tiles (PR #1095).
            kv_empty = total_sum <= cutlass.Float32(0.0)
            new_max = cutlass.Float32(arith.select(kv_empty.ir_value(), sink_logit.ir_value(), cute.math.max(total_max_nat, sink_logit).ir_value()))
            scale = cutlass.Float32(
                arith.select(kv_empty.ir_value(), cutlass.Float32(0.0).ir_value(), cute.math.exp(total_max_nat - new_max, fastmath=True).ir_value())
            )
            new_sum = total_sum * scale + cute.math.exp(sink_logit - new_max, fastmath=True)
            lse_val = new_max + cute.math.log(new_sum, fastmath=True)
            inv_sum = scale / new_sum
        else:
            lse_val = total_max_nat + cute.math.log(cute.math.max(total_sum, cutlass.Float32(1e-30)), fastmath=True)
            inv_sum = cutlass.Float32(1.0) / cute.math.max(total_sum, cutlass.Float32(1e-30))
            # Dead row (no valid KV column at all): O := 0, LSE := -inf.
            row_dead = total_sum <= cutlass.Float32(0.0)
            neg_inf_lse = cutlass.Float32(float("-inf"))
            lse_val = cutlass.Float32(arith.select(row_dead.ir_value(), neg_inf_lse.ir_value(), lse_val.ir_value()))
            inv_sum = cutlass.Float32(arith.select(row_dead.ir_value(), cutlass.Float32(0.0).ir_value(), inv_sum.ir_value()))

        if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT):
            # Dense padded-Q trim: q rows >= seq_len_q[b] write O := 0 / LSE := -inf.
            # Applied AFTER the sink branch on purpose — a trimmed row is dead even with a sink.
            _sq_arr = cute.make_tensor(cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4), cute.make_layout(1 << 24))
            _q_len_b = cutlass.Int32(_sq_arr[batch_idx])
            row_trim = q_row_global >= _q_len_b
            neg_inf_trim = cutlass.Float32(float("-inf"))
            lse_val = cutlass.Float32(arith.select(row_trim.ir_value(), neg_inf_trim.ir_value(), lse_val.ir_value()))
            inv_sum = cutlass.Float32(arith.select(row_trim.ir_value(), cutlass.Float32(0.0).ir_value(), inv_sum.ir_value()))
        # Base-2 Stats (stats_use_log2): natural LSE * log2(e); -inf stays -inf.
        if cutlass.const_expr(CFG.STATS_LOG2):
            lse_val = lse_val * cutlass.Float32(1.4426950408889634)
        if cutlass.const_expr(lse_tensor is None):
            pass  # has_lse=False: the Stats store is compiled out
        else:
            # OOB-row guard: the tile's Q rows can exceed seqlen_q; without it
            # the write aliases the next head's LSE slot.
            if q_row_global < seqlen_q:
                lse_arr = cutlass.make_array_view(lse_tensor)
                lse_batch = _partial_batch(batch_idx, split_idx, n_batch)
                lse_arr[lse_batch, row_head_idx, q_row_global] = lse_val

        sO_sub_base = sO[0].base

        if cutlass.const_expr(_FP32_PARTIALS):
            _store_fp32_partial_tile(
                o_partial_f32,
                tmem_base_epi,
                LAYOUT.O0_OFF,
                inv_sum,
                row_dead,
                q_row_global < seqlen_q,
                _partial_batch(batch_idx, split_idx, n_batch),
                q_row_global,
                row_head_idx,
                CFG.TILE_O,
                O_CHUNK,
            )
            bars.mb_o_empty[0].wait(o_empty_phase)
        else:
            for chunk_idx in cutlass.range_constexpr(N_CHUNKS_O):
                o_fp16 = cutlass.Vector.from_elements(
                    tuple(STORAGE_DTYPE(0.0) for _ in range(O_CHUNK)),
                    STORAGE_DTYPE,
                )
                if cutlass.const_expr(not MAY_BE_EMPTY) or (bounds.right > bounds.left):
                    o_addr = tmem_base_epi + cutlass.Int32(LAYOUT.O0_OFF + chunk_idx * O_CHUNK)
                    o_chunk = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                        num=O_CHUNK,
                    )
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                    o_scaled = o_chunk * inv_sum
                    o_fp16 = o_scaled.to(STORAGE_DTYPE)

                col_offset_const = (chunk_idx * O_CHUNK) % D_BLOCK_SIZE
                block_idx_const = (chunk_idx * O_CHUNK) // D_BLOCK_SIZE
                block_offset_const = block_idx_const * TMA_O_GRANU_ELEMS
                smem_offset = cutlass.Int32(block_offset_const + col_offset_const) + tid_in_wg * cutlass.Int32(D_BLOCK_SIZE)

                smem_ptr = sO_sub_base.subview(smem_offset).data_ptr()
                # mb_o_empty wait gates the FIRST SMEM store (not the earlier
                # TMEM-load loop), keeping TMEM-load/FFMA/cast overlapped with
                # TMA-STG draining the prior persistent tile.
                if chunk_idx == 0:
                    bars.mb_o_empty[0].wait(o_empty_phase)
                smem_ptr.store_swizzled(o_fp16, alignment=64, swizzle=_O_SMEM_SWIZZLE)

            # fence_proxy needed before TMA reads SMEM written by generic-proxy stores.
            nvvm.fence_proxy("async.shared", space="cta")

        bars.mb_o_full[0].arrive()

        stat_full_phase = stat_full_phase ^ 1
        o_empty_phase = o_empty_phase ^ 1
        # Catch-up flip — bmm2_done_phase ^= 1 AFTER the epilogue wait
        # (mainloop flips n_kv-1 times, MMA fires n_kv times; +1 here matches).
        bmm2_done_phase = bmm2_done_phase ^ 1

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
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
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
        bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)

    bars.mb_tmem_dealloc.arrive()


# === Host launcher ===


@cute.jit
def _launch(
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
    seq_q_lens_addr: cutlass.Int64,
    o_partial_f32: Optional[cute.Tensor],
    block_table_tensor: Optional[cute.Tensor],
    block_table_v_tensor: Optional[cute.Tensor],
    paged_hnd: cutlass.Constexpr[bool],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """TMA construction and launch for the explicit pointer host entry."""
    B, QH, KH, SQ, SKV, _ = problem_size
    if cutlass.const_expr(PAGED_KV):
        SKV = block_table_tensor.shape[1] * cutlass.Int32(PAGE_SIZE)

    _O_GRANU_ELEMS = CFG.O_SWZ_BYTES // CFG.BPE
    # Tensors are [B, S, H, D] with stride_order=(3, 2, 1, 0); D is fastest.
    # PackGQA: the Q box is TILE_M/G tokens x G heads (token-major rows).
    qk_box_q = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, TMA_QK_GRANU_ELEMS)
    # Paged KV: K/V are [num_pages, page_size, H_kv, D] views of the page pool
    # and a tile is a stack of KV_BOXES row boxes, so the descriptor box is one
    # box tall.  Dense: the full per-CTA tile.
    qk_box_k = (1, KV_BOX_ROWS, 1, TMA_QK_GRANU_ELEMS)
    vo_box_v = (1, KV_BOX_ROWS, 1, TMA_VO_GRANU_ELEMS)
    vo_box_o = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, _O_GRANU_ELEMS)
    stride_order = (3, 2, 1, 0)
    # Paged KV: HND storage ([num_pages, H_kv, page_size, D]) viewed as
    # [page, row, H_kv, D] has the row stride BELOW the head stride, so its
    # descriptor lists dims innermost-first as (D, row, H_kv, page) and the
    # TMA-LDG warp swaps its (head, row) coords to match; NHD is the dense BSHD
    # order with batch -> page.
    kv_stride_order = (3, 1, 2, 0) if paged_hnd else stride_order

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
        stride_order=kv_stride_order,
        swizzle=_tma_swz(CFG.K_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    tma_v_desc = tmap.create_tensor_map_tiled_from_view(
        v_tensor,
        box_dims=vo_box_v,
        stride_order=kv_stride_order,
        swizzle=_tma_swz(CFG.V_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    _o_box = list(vo_box_o)
    if _FP32_PARTIALS:
        # fp32 is 4 bytes; scale the box by the O element's own width so
        # the inner dimension stays inside the swizzle's byte limit.
        _o_box[-1] = max(1, _o_box[-1] * CFG.BPE_O // 4)
    tma_o_desc = tmap.create_tensor_map_tiled_from_view(
        o_tensor,
        box_dims=tuple(_o_box),
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.O_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )

    # One CTA covers TILE_M Q rows (SQ*PACK_G packed rows per packed head, QH/PACK_G packed heads).
    rows_per_cluster = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA
    q_supers = (SQ * HEADS_PER_TILE + rows_per_cluster - 1) // rows_per_cluster
    # KV split rides the BATCH axis: z = batch + split*B (decoded as b % B, b // B).
    grid_shape = (
        (q_supers, QH // HEADS_PER_TILE, B * SPLIT_KV)
        if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL)
        else (q_supers * (QH // HEADS_PER_TILE) * B * SPLIT_KV, 1, 1)
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
        seq_q_lens_addr,
        o_partial_f32,
        block_table_tensor,
        block_table_v_tensor,
        paged_hnd,
    ).launch(
        grid=grid_shape,
        block=[CFG.THREADS_PER_CTA, 1, 1],
        cluster=(1, 1, 1),
        stream=stream,
    )


@cute.jit
def _host(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    lse_ptr: Optional[cute.Pointer],
    sinks_ptr: cute.Pointer,
    meta_ptr: cute.Pointer,
    o_desc_ptr: cute.Pointer,
    problem_size: Tuple[int, int, int, int, int, int],
    q_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    k_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    v_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_ext: cutlass.Int32,
    scale_softmax_log2: cutlass.Float32,
    n_thd_units: cutlass.Int32,
    seq_q_lens_addr: cutlass.Int64,
    thd_q_lens_ptr: Optional[cute.Pointer],
    thd_kv_lens_ptr: Optional[cute.Pointer],
    thd_lens_form: Optional[cutlass.Int32],
    o_partial_ptr: Optional[cute.Pointer],
    block_table_ptr: Optional[cute.Pointer],
    block_table_v_ptr: Optional[cute.Pointer],
    table_strides: Tuple[cutlass.Int64, cutlass.Int64],
    n_pages: cutlass.Int32,
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    lse_kind: cutlass.Constexpr[str],
    paged_hnd: cutlass.Constexpr[bool],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """Bind the shared explicit SDPA ABI and launch the existing decode kernel."""
    B, QH, KH, SQ, SKV, _ = problem_size
    (
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        sinks_tensor,
        seq_kv_lens_tensor,
        o_desc_words,
        thd_q_lens_tensor,
        thd_kv_lens_tensor,
        o_partial_f32,
        block_table_tensor,
        block_table_v_tensor,
    ) = sdpa_operand_tensors(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        lse_ptr,
        sinks_ptr,
        meta_ptr,
        o_desc_ptr,
        problem_size,
        q_strides,
        k_strides,
        v_strides,
        o_strides,
        lse_strides,
        lse_ext,
        thd_q_lens_ptr,
        thd_kv_lens_ptr,
        thd_lens_form,
        o_partial_ptr,
        d_qk=d_qk,
        d_v=d_v,
        lse_kind=lse_kind,
        thd=CFG.THD_VARLEN,
        split_kv=SPLIT_KV,
        tensor_map_qwords=16,
        paged=PAGED_KV,
        page_size=PAGE_SIZE,
        block_table_ptr=block_table_ptr,
        block_table_v_ptr=block_table_v_ptr,
        table_strides=table_strides,
        n_pages=n_pages,
    )

    _launch(
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        sinks_tensor,
        seq_kv_lens_tensor,
        o_desc_words,
        problem_size,
        scale_softmax_log2,
        seq_q_lens_addr,
        o_partial_f32,
        block_table_tensor,
        block_table_v_tensor,
        paged_hnd,
        stream,
    )


EXPLICIT_ABI = True  # pointer/int host entry; the adapter builds the argument list itself
LSE_KINDS = ("dense",)  # the decode tile never serves THD


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    d_qk: int = CFG.TILE_K,
    d_v: int = CFG.TILE_O,
    has_lse: bool = True,
    lse_kind: str = "dense",
    paged_hnd: bool = False,
) -> Callable:
    """Compile the shared pointer ABI for dense or paged decode. Shapes and strides
    are runtime arguments; head-dim envelopes, Stats presence and pool layout
    are specializations. All stride fakes, including page tables, are Int64."""
    _cache_key = _template_key(globals(), locals(), "compile")
    if not (0 < d_qk <= CFG.TILE_K and 0 < d_v <= CFG.TILE_O):
        raise ValueError(f"d128 envelope: need 0 < d_qk <= {CFG.TILE_K} and 0 < d_v <= {CFG.TILE_O}; got ({d_qk}, {d_v})")
    if (d_qk * CFG.BPE) % 16 != 0 or (d_v * CFG.BPE_O) % 16 != 0:
        raise ValueError(f"d128 envelope: d_qk*BPE and d_v*BPE must be 16-byte multiples (TMA global-stride rule); got ({d_qk}, {d_v}) at BPE={CFG.BPE}")
    if SPLIT_KV > 1 and not has_lse:
        raise ValueError("d128: split_kv > 1 requires has_lse=True (the per-split LSE drives the combine)")
    if lse_kind not in LSE_KINDS:
        raise ValueError(f"lse_kind must be one of {LSE_KINDS}; got {lse_kind!r}")
    if paged_hnd and not PAGED_KV:
        raise ValueError("paged_hnd is a paged-KV specialization")
    gmem = cute.AddressSpace.gmem

    def P(dtype, align=16):
        return cute.runtime.make_ptr(dtype, 16, gmem, assumed_align=align)  # fake: type only

    i32 = cutlass.Int32(0)
    i64_3 = (cutlass.Int64(0),) * 3  # stride slots: Int64 leaves, see _host
    thd = bool(CFG.THD_VARLEN)
    return _compile_cached(
        _host,
        P(STORAGE_DTYPE),
        P(STORAGE_DTYPE),
        P(STORAGE_DTYPE),
        P(cutlass.Float32 if _FP32_PARTIALS else STORAGE_DTYPE),
        P(cutlass.Float32, 4) if has_lse else None,
        P(cutlass.Float32),
        P(cutlass.Int32),
        P(cutlass.Int64),
        (0, 0, 0, 0, 0, 0),
        i64_3,
        i64_3,
        i64_3,
        i64_3,
        i64_3,
        i32,
        cutlass.Float32(0.0),
        i32,
        cutlass.Int64(0),
        P(cutlass.Int32, 4) if thd else None,
        P(cutlass.Int32, 4) if thd else None,
        i32 if thd else None,
        P(cutlass.Float32) if _FP32_PARTIALS else None,
        P(cutlass.Int32, 4) if PAGED_KV else None,
        P(cutlass.Int32, 4) if PAGED_KV else None,
        (cutlass.Int64(0), cutlass.Int64(0)),
        i32,
        d_qk,
        d_v,
        lse_kind,
        paged_hnd,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_fwd_decode_prepared",
    )
