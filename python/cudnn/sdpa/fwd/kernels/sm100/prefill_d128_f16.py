# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
DSL prefill SDPA kernel — llama flavor, d_qk = d_v = 128, FP16/BF16, SM100.

SM100 (Blackwell GB200) llama-class prefill kernel.  Pipeline shape:
TILES_Q=2, two softmax warpgroups, four correction warps, 16 warps / 512
threads, persistent try_cancel scheduler, no Q∪O alias.

SM100 resource layout:
  1. **STAGES_KV=2, both cluster widths at 192 KiB**: at cga2 (CTA_MMA=2,
     the default) the collective MMA halves per-CTA K/V → sQ 64 + sK 32 +
     sV 32 + sO 64.  At cga1 (``cta_mma=1``) there is no collective MMA, so
     K/V double to 64 + 64 and the 64 KiB is bought back by aliasing Q and O
     into one slab (``QO_ALIAS``, mandatory at cga1 and enforced by
     ``_validate_cfg_d128``) → 64(Q∪O) + 64 + 64.  Both land under the
     Blackwell ~228 KiB cap; ``_d128_smem_bytes`` is the checked model.
  2. **TMEM stats** (512-col Blackwell cap): S_acc 0/128 + O 256..511 fill
     all 512 cols, so stats ride the FREE HEAD of each sub-tile's S_acc
     slot — sub-tile 0 → col 0, sub-tile 1 → col 128 (P only aliases the
     tails at 64 / 192).  This matches the C++ SM100 reference layout
     (tmem_Stats = base + softmax_gid*128).  Safe under the existing
     mb_stat_full / mb_stat_empty handshake: correction reads alpha before
     releasing MMA to overwrite that S_acc head next iter.
  3. **Manual row-max** on the MASK_NONE fast path — ``tcgen05_ld`` +
     ``row_max_reduction`` (the masked path's pattern) instead of
     ``tmem_load_max_reduction_tile`` (LDTM.STAT).
  4. SM100 launch: ``cluster=(CTA_MMA, 1, 1)``.

Supported: FP16 / BF16 (``DTYPE_QKV ∈ {2, 3}``); masks none / causal / SWA /
padded and all pairwise combos (causal+swa, causal+padded, swa+padded);
attention sink (per-head logit in the softmax denominator); cga2;
NATURAL / LPT / LPT_L2 schedulers.  Validated on Blackwell (cc 10.0) vs a
torch fp32 reference (atol 0.025) across single-tile, multi-tile GQA, and the
``n_kv = 1`` P14 edge.

THD / varlen (``CFG.THD_VARLEN=1``): packed ``[1,T,H,D]`` Q/K/V + ``cu_seqlens``
coord offset (applied to BOTH Q slabs under TILES_Q=2), per-batch O
TMA-descriptor array (shared ``thd_helpers.py``), packed ragged-Stats LSE
(head-major ``[1,QH,head_stride]`` or token-major ``[T,QH]`` — the epilogue
branches on the LSE tensor's static rank) — via
the shared ``_common_blackwell`` / ``thd_helpers`` mechanism (same as the SM100 qwen
/ dsv4 kernels).  The dense ``[B,S,H,D]`` path is byte-identical (folds out at
``THD_VARLEN=0``).

FP8 (E4M3 / E5M2, incl. output-dtype override) lives in the sibling
``sm100/prefill_d128_fp8.py`` (shares this flavor's config).

KV split (``CFG.SPLIT_KV > 1``): each chunk runs the UNCHANGED mainloop and
epilogue, writing its partial O and LSE into a split-major workspace at batch
coord ``b + s*B`` -- so the O TMA descriptor is untouched, only the batch coord
shifts.  ``sm100/split_combine.py`` reduces over the split axis.

KV split composes with the cluster width, and the pair is what closes the gap
to cuDNN's own SM100 prefill split-K.  cga1 halves both the wasted MMA work at
small S_q (a tile covers 256 Q rows, not 512) and the CTAs per tile, so twice
as many splits fit in one wave.  Measured on B200 at B=1, H=16, S_q=128,
S_kv=32K, d=128, end-to-end incl. combine, max|O-ref| 2e-5 throughout:

    cga2 SPLIT_KV=1   414 us    (32 CTAs — the classic kernel)
    cga2 SPLIT_KV=4   117 us    (128 CTAs)  3.6x
    cga1 SPLIT_KV=8    73 us    (128 CTAs)  5.7x   <- best
    (cuDNN 9.26/9.30's best plan for the same shape: 69 us, cga1 + 8 splits)

On square shapes (S_q == S_kv, 1K/4K/8K) cga1 and cga2 measure within noise —
same CTA count, same work per CTA — so cga1 is a small-S_q lever, not a
regression risk elsewhere.
Gated (config_sm100._validate_params) to SCHED_NATURAL, dense (non-THD), no
sink; requires ``has_lse=True`` since the per-split LSE drives the combine.
At SPLIT_KV == 1 every split helper const-folds away and the traced code is the
classic single-pass kernel.

Paged KV (``CFG.PAGED_KV=1``, issue #920): K/V are page POOLS —
``[num_pages, page_size, H_kv, D]`` (NHD) or ``[num_pages, H_kv, page_size, D]``
(HND, a stride permutation of the same descriptor) — and a per-batch
``block_table`` [B, max_pages] int32 names the pages of each sequence.  Only
the TMA-LDG warp changes: a K/V tile is issued as ``TILE_N / page_size`` row
boxes (one box inside the page when the page is taller than the tile), each
box's page coordinate read from the block table on device.  Boxes past the
batch's ``ceil(seq_kv_len / page_size)`` pages get page coordinate ``-1`` —
TMA-OOB, so they land as exact zeros and still complete their mbarrier bytes
(the same trick issue #624 uses for the THD tail).  Rows inside the last live
page but past ``seq_kv_len`` are loaded and masked (MASK_PADDED is mandatory),
so the cache bytes of an allocated page must be finite — the contract every
paged-attention kernel has.  The per-batch length rides the existing
``seq_kv_lens_tensor`` and ``seqlen_kv`` is the static maximum
``max_pages * page_size``, so nothing is read back to the host (Rule 3).
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

from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d128

# The template loader (api_dsl._load_kernel_module) injects FROST_TEMPLATE_PARAMS
# as a module global before this body executes; a plain import falls back to
# the default TemplateParams so the file stays importable on its own.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
CFG, _TMA = make_cfg_d128(PARAMS)
Cfg = type(CFG)
TMA_QK_ITERS = _TMA.QK_ITERS
TMA_VO_ITERS = _TMA.VO_ITERS
TMA_QK_GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _TMA.VO_GRANU_ELEMS

# O TMA box / store params follow O's swizzle, not V's (under cga2 V may drop to a narrower swizzle while O stays Swz128B).
TMA_O_GRANU_ELEMS_HOST = CFG.O_SWZ_BYTES // CFG.BPE
TMA_O_ITERS_HOST = (CFG.TILE_O * CFG.BPE) // CFG.O_SWZ_BYTES

from typing import NamedTuple

from cudnn.frost.tile_dsl.barrier import (
    PipelineState,
    advance,
    cga_arrive,
    cga_wait,
    # `wait` (free fn) — still used for sched.mb_* (Sched is not in Bars yet;
    # only the per-kernel Bars NamedTuple was migrated to MBarrier in this step).
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
    # SM100: no LDTM.STAT — the MASK_NONE fast path uses manual tcgen05_ld +
    # row_max_reduction (see _softmax_kv_body), so tmem_load_max_reduction_tile
    # is not imported.
    row_reduction_pair,
    row_max_reduction,
    vec_scale_pair,
)
from cudnn.frost.tile_dsl.regtile import RegTile
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

# Storage dtype + MMA kind dispatch — folded at trace time on CFG.DTYPE_QKV.
if CFG.DTYPE_QKV == 2:
    STORAGE_DTYPE = cutlass.BFloat16
    P_STORAGE_DTYPE = cutlass.BFloat16
    MMA_KIND = nvvm.Tcgen05MMAKind.F16
    IS_TF32 = False
elif CFG.DTYPE_QKV == 3:
    STORAGE_DTYPE = cutlass.Float16
    P_STORAGE_DTYPE = cutlass.Float16
    MMA_KIND = nvvm.Tcgen05MMAKind.F16
    IS_TF32 = False
else:
    # SM100: f16/bf16 only.  FP8 needs the dedicated prefill_d128_fp8_sm100
    # kernel.  The config already asserts DTYPE_QKV ∈ {2,3}.
    raise ValueError(f"prefill_sdpa_f16_sm100: DTYPE_QKV={CFG.DTYPE_QKV} not supported (expected 2=BF16 or 3=FP16)")


from cudnn.sdpa.fwd.kernels._common_blackwell import (
    sdpa_operand_tensors,
    Bars,
    make_split_helpers,
    store_fp32_partial_tile as _store_fp32_partial_tile,
    KvLoopBounds,
    make_classic_bars,
    compute_kv_loop_bounds,
    lpt_tile_coords,
    row_max_for_exp2,
    make_sdpa_helpers,
)

CGA_SIZE = CFG.CGA_M * CFG.CGA_N

CTA_GROUP_KIND = nvvm.CTAGroup.CTA_2 if CFG.CTA_MMA == 2 else nvvm.CTAGroup.CTA_1

# Per-CTA buffer element counts + collective TMA transaction byte counts.
# K (seq rows) and V (d_v cols) shrink by CTA_MMA; Q / O are per-CTA full.
# At cga2 leader's expect_tx = per-CTA elems * BPE * CTA_MMA (collective bytes).
qBufferElems = CFG.TILE_M * CFG.TILE_K
kBufferElems = CFG.TILE_N * CFG.TILE_K // CFG.CTA_MMA
vBufferElems = CFG.TILE_O * CFG.TILE_N // CFG.CTA_MMA
oBufferElems = CFG.TILE_M * CFG.TILE_O

# Q∪O alias (dsv3 TF32): O reuses Q's SMEM slab once BMM1 has consumed Q.
# Each of the TILES_Q slabs is sized to max(Q,O) so sQ[qs] and sO[qs] coincide;
# for the classic flavors d_qk >= d_v so the slab is qBufferElems.  TMA-STG
# arrives mb_q_o_alias after O drains; TMA-LDG waits it before reloading Q.
IS_QO_ALIAS = bool(CFG.QO_ALIAS)
QO_SLAB_ELEMS = max(qBufferElems, oBufferElems)

qTmaTransactionBytes = qBufferElems * CFG.BPE * CFG.CTA_MMA
kTmaTransactionBytes = kBufferElems * CFG.BPE * CFG.CTA_MMA
vTmaTransactionBytes = vBufferElems * CFG.BPE * CFG.CTA_MMA


CGA_TILE_M = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA
CTA_MMA = CFG.CTA_MMA
# THD uses a persistent grid + device-bounded claim counter (not the CLC
# envelope). Only this flavor is converted so far.
THD_PERSISTENT = True


# SM100 llama is always cga2 (CTA_MMA=2) → the SCHED_LPT/LPT_L2 reverse-row
# count must be in CGA-tile (cluster) units (q_tiles = n_q_supers // CTA_MMA),
# not per-CTA super-row units; the legacy default over-counts by CTA_MMA and
# scrambles LPT tile assignment (cf. ctm_lpt_decode_qtiles_bug; dsv4/qwen sm100).
_sdpa_h = make_sdpa_helpers(CFG, lpt_q_tiles_in_cga_units=True)
_decode_initial = _sdpa_h.decode_initial
_decode_payload = _sdpa_h.decode_payload
# qtrim variant: collapses the KV loop for CGA tiles entirely past the
# per-batch actual Q length (SEQ_Q_LENS_PRESENT; folds to plain bounds otherwise).
_bounds_for_tile = _sdpa_h.bounds_for_tile_qtrim
_resolve_seqlen_kv = _sdpa_h.resolve_seqlen_kv
_resolve_seqlen_q = _sdpa_h.resolve_seqlen_q

# THD / varlen — flat-grid decode + tma-offset closures (CFG-bound) from the
# factory; O-descriptor builder + TENSOR_MAP_QWORDS from the shared
# kernels/dsl/common/sdpa/thd.py.  Gated by CFG.THD_VARLEN (folds out otherwise).
# Supported at cga1 and cga2 (TILES_Q=2 → two Q slabs / O stores per tile).
# seq_kv_lens overloaded as the THD metadata buffer (int32 len 4B+4):
#   [0..B-1]=seq_kv_lens  [B..2B]=cu_q(B+1)  [2B+1..3B+1]=cu_k(B+1)
#   [3B+2..4B+1]=batch_remap(B)  [4B+2]=live units  [4B+3]=claim counter
from cudnn.sdpa.fwd.kernels.thd_helpers import build_thd_meta_o_descs_kernel as _build_thd_meta_o_descs_kernel, TENSOR_MAP_QWORDS, THD_SETUP_THREADS

_TENSOR_MAP_QWORDS = TENSOR_MAP_QWORDS
# The setup kernel builds the THD metadata buffer DEVICE-side from the
# caller's length tensors and the adapter launches the plan-time envelope
# grid (issue #552) — no length ever reaches the host.
_dispatch_decode_initial = _sdpa_h.dispatch_decode_initial
_dispatch_decode_payload = _sdpa_h.dispatch_decode_payload
_thd_tma_offsets = _sdpa_h.thd_tma_offsets

# === PackGQA ===
#
# HEADS_PER_TILE = CFG.PACK_G heads share one token row-group of the Q tile
# (row r <-> token r // PACK_G, head r % PACK_G): the whole GQA group when it
# divides TILE_M, else its largest divisor that does (partial PackGQA -- 96/8
# packs 4 of its 12 heads, so the grid's head axis holds QH / 4 packed heads
# and PACKED_HEADS_PER_KV = 3 of them read the same KV head).  CFG.QH_PER_KH
# stays the graph's GQA ratio.  The bounds helpers take HEADS_PER_TILE (the
# packed group), never QH_PER_KH.
HEADS_PER_TILE = CFG.PACK_G if CFG.PACK_GQA else 1
TOKENS_PER_TILE = CFG.TILE_M // HEADS_PER_TILE
PACKED_HEADS_PER_KV = CFG.QH_PER_KH // HEADS_PER_TILE if CFG.PACK_GQA else 1

# === KV split ===
#
# The mechanics live in _common_blackwell.make_split_helpers (shared with the other
# SM100 prefill flavors); this is the small-S_q lever for this kernel: a cga2
# cluster covers TILES_Q*TILE_M*CTA_MMA = 512 Q rows, so at S_q = 128 the whole
# problem is ceil(S_q/512) * H * B clusters (32 CTAs at H=16, B=1) on a 148-SM
# part, each walking the entire 32K-token KV loop alone.
_split_h = make_split_helpers(
    CFG,
    bounds_for_tile=_bounds_for_tile,
    dispatch_decode_initial=_dispatch_decode_initial,
    dispatch_decode_payload=_dispatch_decode_payload,
)
SPLIT_KV = _split_h.SPLIT_KV
# A split writes fp32 partials, replacing the SMEM/TMA O path rather than
# widening it; the combine performs the only cast to O's dtype.
_FP32_PARTIALS = SPLIT_KV > 1
MAY_BE_EMPTY = _split_h.MAY_BE_EMPTY
_decode_initial_split = _split_h.decode_initial_split
_decode_payload_split = _split_h.decode_payload_split
_bounds_for_tile_split = _split_h.bounds_for_tile_split
_nomask_range_split = _split_h.nomask_range_split
_partial_batch = _split_h.partial_batch

# === Paged KV ===
#
# A K tile is TILE_N/CTA_MMA rows per CTA (the cga2 pair splits it), a V tile is
# the full TILE_N rows (the pair splits V along d_v instead).  Each is loaded as
# a stack of ``*_BOXES`` row boxes of ``*_BOX_ROWS`` rows, so that no box ever
# straddles a page: box rows = min(page_size, tile rows).  The config validator
# guarantees page_size | 128 or 128 | page_size, so this is exact.
PAGED_KV = bool(CFG.PAGED_KV)
PAGE_SIZE = CFG.PAGE_SIZE if PAGED_KV else 0
_K_TILE_ROWS = CFG.TILE_N // CFG.CTA_MMA
K_BOX_ROWS = min(PAGE_SIZE, _K_TILE_ROWS) if PAGED_KV else _K_TILE_ROWS
V_BOX_ROWS = min(PAGE_SIZE, CFG.TILE_N) if PAGED_KV else CFG.TILE_N
K_BOXES = _K_TILE_ROWS // K_BOX_ROWS
V_BOXES = CFG.TILE_N // V_BOX_ROWS


# P (BMM2 operand) aliases the TAIL of each 128-col S_acc slot since BMM1
# finishes (and softmax loads S into registers) before P is written.
# fp16/bf16 pack 2 probs per FP32 cell → P width = TILE_N/2 ≤ 64 cols → P
# sits at slot offset 64.  TF32 is 1:1 → P width = TILE_N, so P aliases the
# FULL slot (offset 0 / 128).  Mirrors C++ S_bmm2_0 = S_acc_cols - S_bmm2_cols.
_P0_OFF = 0 if IS_TF32 else 64
_P1_OFF = 128 if IS_TF32 else 192


@dataclass(frozen=True)
class KernelTmemLayout:
    """Column offsets for the classic 2-sub-tile SDPA pipeline.

    SM100 512-col TMEM cap.  S_acc[0]=0..127,
    S_acc[1]=128..255, O=256..511 fill all 512 cols, so stats cannot live
    at a dedicated col 544 (off the Blackwell cap).  Instead each sub-tile's
    stats ride the FREE HEAD of its own S_acc slot (col 0 / 128) — P only
    aliases the tails at 64 / 192, and softmax has already read S into
    registers before writing stats.  Matches the C++ SM100 reference layout
    (tmem_Stats = base + softmax_gid*128).  Safe under the existing
    mb_stat_full / mb_stat_empty handshake: correction reads that iter's
    alpha before releasing MMA to overwrite the S_acc head next iter.
    """

    # Blackwell SM10.0 TMEM column cap.
    TOTAL_COLS: int = 512

    S0_OFF: int = 0
    S1_OFF: int = 128

    O0_OFF: int = 256
    O1_OFF: int = 384

    P0_OFF: int = _P0_OFF
    P1_OFF: int = _P1_OFF

    # SM100: stats ride the head of sub-tile qs's S_acc slot (== S{qs}_OFF):
    # sub-tile 0 → col 0, sub-tile 1 → col 128.  Use STATS_OFF + qs*STATS_STRIDE.
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
    # Dense padded-Q trim: separate (B,)-int32 per-batch Q lengths (mirrors
    # cuDNN's SEQLEN_Q pointer / FA's seqused_q). None unless
    # CFG.SEQ_Q_LENS_PRESENT — the DSL specializes on None, so the flag-off
    # ABI is unchanged.
    seq_q_lens_addr: cutlass.Int64 = 0,
    o_partial_f32: Optional[cute.Tensor] = None,
    # Paged KV: [B, max_pages] int32 page ids per batch, one table for K and
    # one for V (the graph contract declares them separately; callers with a
    # shared table bind the same buffer twice). None (folded out of the ABI)
    # unless CFG.PAGED_KV.
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

    # SMEM allocations in natural Q/K/V/O order — Tcgen05SmemDesc.build truncates
    # start_address past ~256 KiB so this order keeps the data buffers in low SMEM.
    # QO_ALIAS: one Q∪O slab (TILES_Q × max(Q,O) elems); sO points into it and
    # strides by QO_SLAB_ELEMS so sO[qs] coincides with sQ[qs].  Else: separate
    # Q and O buffers (the classic layout).
    if cutlass.const_expr(IS_QO_ALIAS):
        sQ_raw = cutlass.Array(STORAGE_DTYPE, CFG.TILES_Q * QO_SLAB_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
        sO_raw = sQ_raw
        _SO_STAGE_ELEMS = QO_SLAB_ELEMS
    else:
        sQ_raw = cutlass.Array(STORAGE_DTYPE, CFG.TILES_Q * qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
        sO_raw = cutlass.Array(STORAGE_DTYPE, CFG.TILES_Q * oBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
        _SO_STAGE_ELEMS = oBufferElems
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
        tma_subtile_stride_elems=(CFG.TILE_N // CFG.CTA_MMA) * TMA_QK_GRANU_ELEMS,
    )
    sV = SmemTile(
        base=sV_raw,
        elems_per_stage=vBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_PV,
        stride_byte_offset=STRIDE_BYTE_OFFSET_PV,
        layout=SMEM_LAYOUT_V,
        # V is split along d_v under cga2 → tma_loads_per_tile = TMA_VO_ITERS / CTA_MMA.
        tma_loads_per_tile=TMA_VO_ITERS // CFG.CTA_MMA,
        tma_granu_elems=TMA_VO_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_N * TMA_VO_GRANU_ELEMS,
    )
    sO = SmemTile(
        base=sO_raw,
        elems_per_stage=_SO_STAGE_ELEMS,
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

    # Scheduler mbar arrive count — stays kernel-local (sched is NOT in Bars).
    READ_TILE_ARRIVERS_TOTAL = ((CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS) + CFG.CORRECTION_WARPS + 1 + 1) * CGA_SIZE + (CFG.CGA_M // CFG.CTA_MMA)

    if warp_idx == 0:
        if nvvm.elect_sync():
            # range_constexpr → Python-int loop variable so bars.mb_X[qs].init()
            # can index the per-stage init_count tuple (mb_bmm2_ready) at trace
            # time.  Loop bounds are small (TILES_Q=2, STAGES_KV=2-4) — unroll
            # is free.
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
                if cutlass.const_expr(IS_QO_ALIAS):
                    bars.mb_q_o_alias[qs].init()
                    bars.mb_qo_slab_free[qs].init()
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

    # P4: cluster fence gates cga2 cross-CTA arrive_on_peer on peer init.
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        cga_arrive()
        cga_wait()

    # The DSL's @cute.kernel stages if/else — use Python ternaries so the chosen
    # expression flows into the trace (variables in branches aren't visible outside).
    cta_id_x = cute.arch.block_idx_in_cluster() if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    cta_in_pair = (cta_id_x & cutlass.Int32(1)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    leader_cta_id = (cta_id_x & cutlass.Int32(~1 & 0xFFFFFFFF)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    mcast_mask = (cutlass.Int32(3) << leader_cta_id) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    # TMA-load self-bit mask: each peer's cta_group::2 TMA targets its own bit;
    # cga2 routing strips bit-24 so bytes land on leader's mbar.
    tma_mcast_mask = (cutlass.Int16(1) << cta_in_pair) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int16(0)
    is_leader = cta_in_pair == cutlass.Int32(0)

    # === Per-warp role dispatch ===
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
            seq_q_lens_addr=seq_q_lens_addr,
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
            tidx=tidx,
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
            cta_id_x=cta_id_x,
            qh_per_kh=qh_per_kh,
            o_partial_f32=o_partial_f32,
        )

    elif warp_idx == CFG.MMA_WARP_ID:
        # Under cga2 the non-leader CTA runs the quiet body (alloc + dealloc only).
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
                    seq_q_lens_addr=seq_q_lens_addr,
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
            o_desc_words=o_desc_words,
            qh_per_kh=qh_per_kh,
            is_leader=is_leader,
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
            o_desc_words=o_desc_words,
            qh_per_kh=qh_per_kh,
            seqlen_kv=seqlen_kv,
        )

    else:  # warp_idx == CFG.SCHED_WARP_ID
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        # try_cancel.multicast::cluster::all — only (0,0,0) CTA issues; at cga1
        # cta_id_x == 0 always, so flag is 1 unconditionally.
        is_cga_first_cta = cta_id_x == cutlass.Int32(0)
        if cutlass.const_expr(CFG.THD_VARLEN):
            # THD: persistent grid + device-bounded claim counter, so no unit
            # past the live total is ever handed out (the CLC path would need
            # the grid to BE the work list, i.e. the plan-time envelope).
            # n_batch is a kernel argument -- do NOT re-derive it from the
            # metadata tensor's layout. seq_kv_lens_tensor is not always a flat
            # 1-D view (the llama THD configs bind a nested one, where shape[0]
            # is itself a tuple), so the arithmetic below has to use the value
            # the caller already threads through every other THD helper.
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
    row_off,
    d_coord,
    mbar,
    tma_mcast_mask,
    box_rows: cutlass.Constexpr[int],
    n_boxes: cutlass.Constexpr[int],
    granu_elems: cutlass.Constexpr[int],
):
    """Issue one paged K or V tile as ``n_boxes`` row boxes through the block table.

    Box ``j`` covers sequence rows ``[kv_tile*TILE_N + row_off + j*box_rows, +box_rows)``
    and lands at that row offset in SMEM (``shifted`` keeps the D-subtile
    iteration of ``tma_load_tile`` intact; 128 B swizzle repeats every 8 rows
    and box_rows is a multiple of 8, so stacked boxes equal one tall box).  A
    box whose page slot is at or past this batch's live page count gets page
    ``-1``: TMA-OOB, zero-filled, bytes still credited to ``mbar``.  The block
    table read itself is clamped into the live range so it never leaves the
    row, whatever the caller left in the padding slots.
    """
    bt = cutlass.make_array_view(block_table_tensor)
    last_live = cute.math.max(n_pages_b - cutlass.Int32(1), cutlass.Int32(0))
    for j in cutlass.range_constexpr(n_boxes):
        g = kv_tile * cutlass.Int32(CFG.TILE_N) + row_off + cutlass.Int32(j * box_rows)
        slot = g // cutlass.Int32(PAGE_SIZE)
        row_in_page = g % cutlass.Int32(PAGE_SIZE)
        page_live = cutlass.Int32(bt[batch_idx, cute.math.min(slot, last_live)])
        in_range = slot < n_pages_b
        page = cutlass.Int32(arith.select(in_range.ir_value(), page_live.ir_value(), cutlass.Int32(-1).ir_value()))
        tma_load_tile(
            smem_tile.shifted(j * box_rows * granu_elems),
            tma(d_coord, kv_head_idx, row_in_page, page),
            mbar,
            cta_group=CFG.CTA_MMA,
            mcast_mask=tma_mcast_mask,
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
    o_desc_words,
    qh_per_kh,
    is_leader,
    cta_in_pair,
    tma_mcast_mask,
    block_table_tensor=None,
    block_table_v_tensor=None,
    paged_hnd: cutlass.Constexpr[bool] = False,
):
    """Unified TMA-LDG warp — cga1 / cga2 x MASK_NONE / PADDED / CAUSAL / SWA.

    Spill-free in CFG.OTHER_REGS via q_row_base pre-multiplied after each
    scheduler decode (gives ptxas a live use of nxt_q before the loop
    back-edge, keeping the LDS.128 result in uniform registers).
    """
    q_empty_phase = cutlass.Int32(1)
    kv_state = PipelineState.start(phase=1)

    # Q-reload gate: under QO_ALIAS the next tile's Q-load must wait for the
    # prior tile's O (sharing the slab) to drain — TMA-STG fires mb_q_o_alias
    # (strictly after MMA consumed Q).  Else gate on mb_q_empty (MMA commit).
    # Both bootstrap consumer-side via q_empty_phase=1.
    mb_q_reload = bars.mb_q_o_alias if cutlass.const_expr(IS_QO_ALIAS) else bars.mb_q_empty

    tma_q = GmemTileTma(tma_q_desc)
    if cutlass.const_expr(CFG.THD_VARLEN and not PAGED_KV):
        # THD: K/V ride the setup kernel's packed-total-clamped runtime
        # descriptors (o_desc_words slots n_batch+1 / n_batch+2), so the last
        # sequence's tile-tail lands as exact zeros instead of reading the
        # buffer's capacity tail (issue #624). Same closure shape as the dense
        # GmemTileTma, so every load site below stays branch-free.
        _k_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(1)) * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        _v_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(2)) * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        tma_k = lambda *coords: tma_slice_runtime_desc(_k_rt_ptr, *coords)  # noqa: E731
        tma_v = lambda *coords: tma_slice_runtime_desc(_v_rt_ptr, *coords)  # noqa: E731
    elif cutlass.const_expr(PAGED_KV and paged_hnd):
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
    # GQA: K/V are indexed by kv-head, not Q-head; with PackGQA the decoded
    # head_idx is the PACKED head (Q head base = head_idx * PACK_G), the KV head
    # is packed head // PACKED_HEADS_PER_KV (== head_idx when the whole group
    # packs) and q_row_base is in TOKEN units (rows // PACK_G).
    q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
    kv_head_idx = cute.arch.make_warp_uniform(
        (head_idx if cutlass.const_expr(PACKED_HEADS_PER_KV == 1) else head_idx // cutlass.Int32(PACKED_HEADS_PER_KV))
        if cutlass.const_expr(CFG.PACK_GQA)
        else head_idx // qh_per_kh
    )
    q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE))
    q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)

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

    # The DSL TMA descriptor is element-typed; contiguous coord is in ELEMENTS not bytes.
    K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_N // CFG.CTA_MMA)
    V_COL_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_O // CFG.CTA_MMA)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            if cutlass.const_expr(IS_QO_ALIAS):
                # TMA-STG advances the Q∪O alias gate for EVERY tile, empty
                # ones included (correction still writes O := 0 and STG still
                # drains it).  Consume that phase even though no Q reload is
                # needed, or the next live tile's wait passes on stale parity
                # and its Q load races the previous tile's O drain — observed
                # as a corrupted long batch whenever a persistent CTA picked up
                # an empty split before it (cga1, multi-wave, mixed lengths).
                # mb_qo_slab_free is the return edge; see make_classic_bars.
                mb_q_reload[0].wait(q_empty_phase)
                bars.mb_qo_slab_free[0].arrive()
                mb_q_reload[1].wait(q_empty_phase)
                bars.mb_qo_slab_free[1].arrive()
                q_empty_phase = q_empty_phase ^ 1
        else:
            # Prologue interleave Q[0] -> K[first] -> Q[1] -> V[first] -> mainloop —
            # K load starts before Q[1] is issued and V before kv mainloop.
            kv_row_base = kv_left * CFG.TILE_N

            mb_q_reload[0].wait(q_empty_phase)
            if cutlass.const_expr(IS_QO_ALIAS):
                bars.mb_qo_slab_free[0].arrive()
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
            if cutlass.const_expr(PAGED_KV):
                _paged_load_tile(
                    sK[kv_state.idx],
                    tma_k,
                    block_table_tensor,
                    batch_idx,
                    kv_head_idx,
                    n_pages_b,
                    kv_left,
                    K_ROW_OFFSET_PEER,
                    cutlass.Int32(0),
                    bars.mb_k_full[kv_state.idx].smem_ptr,
                    tma_mcast_mask,
                    K_BOX_ROWS,
                    K_BOXES,
                    TMA_QK_GRANU_ELEMS,
                )
            else:
                tma_load_tile(
                    sK[kv_state.idx],
                    # THD: the prologue K load MUST apply the per-sequence kv offset
                    # (kv_seq_off = cu_k[batch]) and use the packed batch coord
                    # (tma_batch), exactly like the mainloop K load below and the V
                    # loads.  The old form (`+ K_ROW_OFFSET_PEER, batch_idx`) is
                    # byte-identical for the dense path (kv_seq_off=0,
                    # tma_batch==batch_idx) but reads the WRONG packed location for
                    # THD batch>=1 — corrupting the first (diagonal) KV tile and, via
                    # the online-softmax running max/sum, the whole batch>=1 output.
                    tma_k(cutlass.Int32(0), kv_head_idx, kv_row_base + K_ROW_OFFSET_PEER + kv_seq_off, tma_batch),
                    bars.mb_k_full[kv_state.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                )

            mb_q_reload[1].wait(q_empty_phase)
            if cutlass.const_expr(IS_QO_ALIAS):
                bars.mb_qo_slab_free[1].arrive()
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
            if cutlass.const_expr(PAGED_KV):
                _paged_load_tile(
                    sV[kv_state.idx],
                    tma_v,
                    block_table_v_tensor,
                    batch_idx,
                    kv_head_idx,
                    n_pages_b,
                    kv_left,
                    cutlass.Int32(0),
                    V_COL_OFFSET_PEER,
                    bars.mb_v_full[kv_state.idx].smem_ptr,
                    tma_mcast_mask,
                    V_BOX_ROWS,
                    V_BOXES,
                    TMA_VO_GRANU_ELEMS,
                )
            else:
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
                if cutlass.const_expr(PAGED_KV):
                    _paged_load_tile(
                        sK[kv_state.idx],
                        tma_k,
                        block_table_tensor,
                        batch_idx,
                        kv_head_idx,
                        n_pages_b,
                        kv_loop,
                        K_ROW_OFFSET_PEER,
                        cutlass.Int32(0),
                        bars.mb_k_full[kv_state.idx].smem_ptr,
                        tma_mcast_mask,
                        K_BOX_ROWS,
                        K_BOXES,
                        TMA_QK_GRANU_ELEMS,
                    )
                else:
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
                if cutlass.const_expr(PAGED_KV):
                    _paged_load_tile(
                        sV[kv_state.idx],
                        tma_v,
                        block_table_v_tensor,
                        batch_idx,
                        kv_head_idx,
                        n_pages_b,
                        kv_loop,
                        cutlass.Int32(0),
                        V_COL_OFFSET_PEER,
                        bars.mb_v_full[kv_state.idx].smem_ptr,
                        tma_mcast_mask,
                        V_BOX_ROWS,
                        V_BOXES,
                        TMA_VO_GRANU_ELEMS,
                    )
                else:
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
        # q_row_base compute right after decode drives ptxas R2UR.
        q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE))
        q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)
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

    # cga2: drain trailing empty mbar arrives so SMEM isn't torn down while
    # leader's multicast commits are still in-flight.
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        for _qs in cutlass.range_constexpr(CFG.TILES_Q):
            mb_q_reload[_qs].wait(q_empty_phase)
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
    """Persistent O-store warp.  First tile from blockIdx; subsequent tiles
    via scheduler warp's clusterlaunchcontrol.try_cancel.async.
    """
    o_full_phase = cutlass.Int32(0)  # consumer waits — first-arrive flips 0 → 1
    # Q∪O alias return edge (consumer side): LDG arrives mb_qo_slab_free after
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

        q_row_base = q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE)
        q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
        # KV split: partials are stacked split-major on the BATCH axis of the
        # workspace (extent B*SPLIT_KV), so the store needs no new descriptor —
        # only a shifted batch coord.  Folds to batch_idx at SPLIT_KV == 1.
        o_batch = _partial_batch(batch_idx, split_idx, n_batch)

        for qs in cutlass.range_constexpr(CFG.TILES_Q):
            bars.mb_o_full[qs].wait(o_full_phase)
            # fp32 partials wrote the workspace directly, so there is nothing
            # staged to copy.  Skip ONLY the store: the arrive below and the
            # QO_ALIAS handshake after it must still run, or this warp laps
            # the loader and the parity waits deadlock.
            if cutlass.const_expr(not _FP32_PARTIALS):
                # O TMA params follow O's swizzle (NOT V's) — under gptoss cga2 V drops to Swz64B while O stays Swz128B.
                if cutlass.const_expr(CFG.THD_VARLEN):
                    # THD: store each Q slab through this batch's pre-built descriptor
                    # (base at the sequence's packed row, seq extent = S_q_b → a box
                    # past S_q_b is OOB-clipped).  q_row coord is sequence-local; the
                    # batch coord collapses to 0.  Both slabs share one descriptor.
                    # DEAD unit (batch == n_batch, over-launched grid — issue #552):
                    # no O rows exist and descriptor slot n_batch is never built, so
                    # skip the store; the barrier protocol below still runs.
                    if batch_idx < n_batch:
                        o_desc_ptr = (o_desc_words.iterator.raw_ptr() + batch_idx * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
                        o_slice = tma_slice_runtime_desc(
                            o_desc_ptr,
                            cutlass.Int32(0),
                            q_head_idx,
                            q_row_base + cutlass.Int32(qs * TOKENS_PER_TILE),
                            cutlass.Int32(0),
                        )
                        tma_store_tile(sO[qs], o_slice)
                else:
                    tma_store_tile(
                        sO[qs],
                        tma_o(cutlass.Int32(0), q_head_idx, q_row_base + cutlass.Int32(qs * TOKENS_PER_TILE), o_batch),
                    )

                tma_store_commit()
                tma_store_wait(0)

            bars.mb_o_empty[qs].arrive()
            # QO_ALIAS: O[qs] has drained to GMEM → the shared Q∪O slab is free
            # for TMA-LDG to clobber with the next tile's Q[qs].  Throttled by
            # the return edge: at most ONE alias phase ahead of LDG.
            if cutlass.const_expr(IS_QO_ALIAS):
                bars.mb_qo_slab_free[qs].wait(slab_free_phase)
                bars.mb_q_o_alias[qs].arrive()

        o_full_phase = o_full_phase ^ 1
        if cutlass.const_expr(IS_QO_ALIAS):
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

# Per-tensor swizzle layout enum: SWIZZLE_128B_ATOM_32B=1, Swz128B=2,
# Swz64B=4, Swz32B=6.
_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
_SWZ_ATOM_32B = 1  # required on the transposed BMM2 operand (V) under kind::tf32
SMEM_LAYOUT_Q = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_K = _SWZ_ENUM[CFG.K_SWZ_BYTES]
# V is the B-transposed BMM2 operand → TF32 needs SWIZZLE_128B_ATOM_32B.
# Q/K (BMM1, non-transposed) and O keep standard swizzle for all dtypes.
SMEM_LAYOUT_V = _SWZ_ATOM_32B if IS_TF32 else _SWZ_ENUM[CFG.V_SWZ_BYTES]
SMEM_LAYOUT_O = _SWZ_ENUM[CFG.O_SWZ_BYTES]
SMEM_LAYOUT_QKO = SMEM_LAYOUT_Q

# O SMEM swizzle: third param is the XOR shift offset (=3 across all widths), NOT the B value.
_O_SWZ_B = {128: 3, 64: 2, 32: 1}[CFG.O_SWZ_BYTES]
_O_SMEM_SWIZZLE = cutlass.Swizzle(_O_SWZ_B, 4, 3)

LEADING_BYTE_OFFSET_QK = 0
STRIDE_BYTE_OFFSET_QK = 8 * CFG.Q_SWZ_BYTES

# leading_byte_offset = 0 when (TILE_O/CTA_MMA)/8 <= 8 else TILE_N*V_SWZ_BYTES
# TF32 ATOM32 override (mirrors C++ prefill_sdpa_f16.cu:803-806): the
# SWIZZLE_128B_ATOM_32B mode pins leading = TILE_N*128, stride = 512 (NOT
# the standard 8*V_SWZ).  Standard formulas with kind::tf32 silently zero C.
_CORE_MATRIX_ROWS = 8
_V_PC_COLS = CFG.TILE_O // CFG.CTA_MMA
_LEADING_PV_STD = 0 if (_V_PC_COLS // _CORE_MATRIX_ROWS) <= 8 else CFG.TILE_N * CFG.V_SWZ_BYTES
LEADING_BYTE_OFFSET_PV = (CFG.TILE_N * 128) if IS_TF32 else _LEADING_PV_STD
STRIDE_BYTE_OFFSET_PV = 512 if IS_TF32 else (8 * CFG.V_SWZ_BYTES)

NUM_KPHASES_PV = CFG.TILE_N // CFG.TILE_K_HW_BMM2
NUM_KPHASES_PV_PER_CHUNK = NUM_KPHASES_PV // CFG.N_BMM2_CHUNKS


@cute.jit
def _mma_warp_quiet(tmem_ptr_i32, bars):
    """Minimal quiet body for the non-leader CTA's MMA-warp slot under cga2.

    Non-leader still participates in TMEM alloc / release lock (warp-collective
    .sync.aligned ops) and TMEM-publish named barriers — leader's MMA reads
    through the cluster crossbar from peer's TMEM during cga2 collective MMA.
    """
    # All-lanes warp-collective ops — NO elect_sync gating.
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    # Match lead MMA's named-barrier arrive count (lead is +1 on both publish barriers).
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
    seq_q_lens_addr,
    n_q_supers,
    n_qh,
    n_batch,
    mcast_mask,
    cta_in_pair,
    qh_per_kh,
):
    """Unified MMA warp — cga1 / cga2-leader x MASK_NONE / PADDED / CAUSAL / SWA,
    spill-free in CFG.OTHER_REGS (40).  Non-leader under cga2 uses _mma_warp_quiet.
    """
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    nvvm.barrier_cta_arrive(2, 32 * (CFG.CORRECTION_WARPS + 1))

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    # idesc M is COLLECTIVE (per-CTA M * CTA_MMA): cga2 tcgen05.mma.cta_group::2
    # reads both peers' A and produces 2*LOCAL_M output rows.
    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_N,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_O,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        b_major=1,
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
    # BMM2 V uses non-default K_SUBTILE = V_SWZ_BYTES/BPE (cga2 V Swz64B on dsv3/gptoss).
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
        # Under KV split the MASK_NONE path still has to decode, because the
        # split index (and hence this tile's KV slice) lives in the tile id.
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
    bmm2_ready_phase = cutlass.Int32(0)
    # Initialised unconditionally so type is stable across const_expr branches.
    empty_mainloop_phase = cutlass.Int32(0)
    # Stats-consumed gate (one flip per tile per sub-tile): the prologue BMM1
    # overwrites the S_acc HEAD where the PREVIOUS tile's final
    # (total_max, total_sum) stats live until the correction epilogue has read
    # them.  q_full/k_full alone do NOT order that read before the BMM1 (Q/K
    # can be resident well before correction finishes its epilogue — the
    # window widens by ~a gmem latency under HAS_SINK, corrupting the second
    # sub-tile's LSE/O on causal multi-wave shapes).  Bootstrap phase 1: the
    # first tile has no prior stats to protect, so the wait passes immediately.
    stats_read_phase = cutlass.Int32(1)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            # Empty mainloop: fire both bmm2_done so softmax/correction phase trackers stay in lockstep.
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
            # Prologue: BMM1[sub0], BMM1[sub1] for kv=kv_left.
            # mb_stats_read[qs] gates each BMM1 on the correction epilogue
            # having READ the previous tile's final stats from the S_acc head
            # this BMM1 is about to overwrite (q_full/k_full don't order that).
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

            # Mainloop kv = kv_left+1 .. kv_right-1 (empty when n_kv == 1)
            for kv_loop in cutlass.range(kv_left + cutlass.Int32(1), kv_right, 1, unroll=1):
                old_state = kv_state
                kv_state = advance(kv_state, CFG.STAGES_KV)

                bars.mb_v_full[old_state.idx].wait(old_state.phase)
                desc_V = sV[old_state.idx].desc()
                is_not_first_bmm2 = cutlass.Boolean(kv_loop != (kv_left + cutlass.Int32(1)))

                # BMM2 sub-tile 0
                bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 0].wait(bmm2_ready_phase)
                accum_b2 = is_not_first_bmm2
                for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                    mma_ts_step(bmm2_desc, (tmem_raw.subview(LAYOUT.P0_OFF)), desc_V, (tmem_raw.subview(LAYOUT.O0_OFF)), local_k, accum_b2)
                    accum_b2 = cutlass.Boolean(True)
                # Chunk 1 folds out at TILE_N=64 (N_BMM2_CHUNKS=1).
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

                # BMM1 sub 0 for next kv
                bars.mb_k_full[kv_state.idx].wait(kv_state.phase)
                desc_K = sK[kv_state.idx].desc()
                mma_ss(bmm1_desc, desc_Q0, desc_K, (tmem_raw.subview(LAYOUT.S0_OFF)))
                elect_p = nvvm.elect_sync()
                bars.mb_bmm1_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

                # BMM2 sub-tile 1
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

                # BMM1 sub 1 for next kv
                mma_ss(bmm1_desc, desc_Q1, desc_K, (tmem_raw.subview(LAYOUT.S1_OFF)))
                elect_p = nvvm.elect_sync()
                bars.mb_bmm1_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_k_empty[kv_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

                bmm2_ready_phase = bmm2_ready_phase ^ 1

            # Epilogue: BMM2 for last kv (always runs — n_kv >= 1).
            # Under QO_ALIAS the TMA-LDG Q-reload gate is mb_q_o_alias (fired by
            # TMA-STG after O drains), so MMA does NOT fire mb_q_empty.
            if cutlass.const_expr(not IS_QO_ALIAS):
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
    sub_tile_id: int,
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
    bmm1_phase,
    stat_empty_phase,
    leader_cta_id,
):
    """Per-iter kv body for the softmax warp group.

    Compile-time apply_mask (Python bool) picks the load+max strategy:
    apply_mask=False uses fused tcgen05.ld.red.f32.max (HW row-max);
    apply_mask=True uses tcgen05.ld + apply_mask_chunk + software row-max
    (HW max can't observe NEG_INFINITY written after the load).

    total_max runs in scaled (log2) units and starts at -inf; total_max_safe
    is its 0-substituted companion (see row_max_for_exp2), carried so alpha
    can be exp2(prev_safe - new_safe) with the substitution on both operands.

    Returns updated (total_max, total_max_safe, total_sum, bmm1_phase,
    stat_empty_phase).
    """
    tmem_S_off = LAYOUT.S0_OFF if sub_tile_id == 0 else LAYOUT.S1_OFF
    tmem_P_off = LAYOUT.P0_OFF if sub_tile_id == 0 else LAYOUT.P1_OFF
    stats_off = LAYOUT.STATS_OFF + sub_tile_id * LAYOUT.STATS_STRIDE
    CHUNK = 64
    # fp16/bf16 pack 2 probs per FP32 TMEM cell (CHUNK//2); TF32 is 1:1 (CHUNK).
    P_COLS_PER_CHUNK = CHUNK if IS_TF32 else CHUNK // 2
    N_CHUNKS = CFG.N_BMM2_CHUNKS
    RESCALE_THRESHOLD = cutlass.Float32(CFG.RESCALE_THRESHOLD)

    bars.mb_bmm1_done[sub_tile_id].wait(bmm1_phase)
    bmm1_phase = bmm1_phase ^ 1

    # Hoist all TMEM address math to a single base load — tcgen05.ld/st
    # auto-derives row from warp_id_in_warpgroup; address only needs the column.
    tmem_base = tmem_ptr_i32.load()
    s_addr_base = tmem_base + cutlass.Int32(tmem_S_off)
    p_addr_base = tmem_base + cutlass.Int32(tmem_P_off)
    stats_addr = tmem_base + cutlass.Int32(stats_off)

    # apply_mask is a Python bool — wrap in cutlass.const_expr so the DSL folds
    # at trace time instead of staging cf.if (the two arms produce Vectors
    # built via different MLIR op sequences and the tracer would error).
    if cutlass.const_expr(apply_mask):
        kv_col_base = kv_loop * cutlass.Int32(CFG.TILE_N)
        # Python comprehensions (not for+append) so the tracer sees fully-formed lists.
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
        from cudnn.frost.tile_dsl.regtile import vec_concat

        # vec_concat handles single-element lists too; keeps tracer Vector type uniform across N_CHUNKS.
        reg_S_vec = vec_concat(chunks_S)
        current_max_unscaled = chunks_max[0]
        for m in chunks_max[1:]:
            current_max_unscaled = cute.math.max(current_max_unscaled, m)
    else:
        # SM100: manual row-max (no LDTM.STAT / tmem_load_max_reduction_tile).
        # Load S chunks, reduce per chunk, combine across chunks — the masked
        # path's pattern sans mask.
        from cudnn.frost.tile_dsl.regtile import vec_concat

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

    # Pass size=CFG.TILE_N explicitly — Vector.shape[0] is an MLIR value
    # (not Python int) for vec_concat-built vectors, so auto-detect can't recover the length.
    reg_S = RegTile(reg_S_vec, size=CFG.TILE_N)
    current_max = current_max_unscaled * scale_log2  # -inf when the whole iteration is masked

    # Named-barrier wg0/wg1 sync: sub_tile_id==1 waits at TOP for sub_tile_id==0's bottom arrive.
    if sub_tile_id == 1:
        nvvm.barrier_cta_sync(barrier_id=8, thread_count=256)

    # Online softmax with RESCALE_THRESHOLD skip.  total_max starts at -inf,
    # so the first live iteration always clears the threshold
    # (real - (-inf) = +inf) while a fully-masked one never does
    # (-inf - x = -inf or NaN; ordered > is false for both) — dead
    # iterations can never move the max.
    update_cond = (current_max - total_max) > RESCALE_THRESHOLD
    total_max = cutlass.Float32(
        arith.select(
            update_cond.ir_value(),
            current_max.ir_value(),
            total_max.ir_value(),
        )
    )
    # Canonical 0-substitution at point of use, on BOTH alpha operands:
    # no-update iters give alpha == exp2(0) == 1 exactly (ballot keeps
    # firing); min(., 0) guards the dead->alive transition (safe max drops
    # 0 -> real*scale < 0) where total_sum is still 0 so alpha must merely
    # stay finite.  total_max_safe starts at -inf so iter 0 keeps alpha = 0.
    new_total_max_safe = row_max_for_exp2(total_max)
    alpha = cute.math.exp2(
        cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)),
        fastmath=True,
    )
    total_max_safe = new_total_max_safe

    alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
    bars.mb_stat_full[sub_tile_id].arrive()

    # Rescale full reg_S in one vector op — emits same FFMA2 sequence as explicit half-tile rescales.
    reg_S = reg_S * scale_log2 - total_max_safe

    # Chunk 0 manual unroll — the DSL's @cute.jit tracer makes the loop iter an
    # MLIR value, breaking Python slice.indices() math inside RegTile[].
    chunk_S_0 = reg_S[0:CHUNK].vec
    chunk_P_0 = cute.math.exp2(chunk_S_0, fastmath=True)
    # Hoist chunk 0's sum before the cast to overlap with the cast's FFMA chain.
    hoisted_sum = row_reduction_pair(chunk_P_0)
    chunk_P_0_fp16 = chunk_P_0.to(STORAGE_DTYPE)
    nvvm.tcgen05_st(
        "32x32b",
        nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32),
        chunk_P_0_fp16,
    )
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
    bars.mb_bmm2_ready[sub_tile_id * N_CHUNKS + 0].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

    # Chunk 1 folds out at TILE_N=64 (N_CHUNKS=1).
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
        bars.mb_bmm2_ready[sub_tile_id * N_CHUNKS + 1].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

    # wg0 fires bottom sync to release wg1 spinning at TOP barrier.
    if sub_tile_id == 0:
        nvvm.barrier_cta_sync(barrier_id=8, thread_count=256)

    new_p_sum_pair = hoisted_sum
    if cutlass.const_expr(N_CHUNKS == 2):
        new_p_sum_pair = new_p_sum_pair + row_reduction_pair(deferred_P_1)
    alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
    total_sum = total_sum * alpha_pair + new_p_sum_pair

    bars.mb_stat_empty[sub_tile_id].wait(stat_empty_phase)
    stat_empty_phase = stat_empty_phase ^ 1

    return total_max, total_max_safe, total_sum, bmm1_phase, stat_empty_phase


@cute.jit
def _softmax_warp_group(
    sub_tile_id: int,
    seqlen_q,
    seqlen_kv,
    scale_log2: cutlass.Float32,
    tmem_ptr_i32,
    sQ,
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
    """Softmax warp group (one of two): online softmax per kv iter.

    Each lane owns one row of the 128 x 128 S_acc tile, loads its 128 fp32
    cols with a single tcgen05.ld.red.max, runs online softmax (total_max +
    total_sum tracking with RESCALE_THRESHOLD skip), publishes alpha to
    corr, writes P to TMEM cols at S_acc tail, fires bmm2_ready[sub][chunk].
    """
    # Wait on MMA's TMEM-publish named barrier BEFORE any tmem_ptr_i32.load() —
    # without it softmax can race MMA's tcgen05_alloc and read a stale base.
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))

    tmem_S_off = LAYOUT.S0_OFF if sub_tile_id == 0 else LAYOUT.S1_OFF
    tmem_P_off = LAYOUT.P0_OFF if sub_tile_id == 0 else LAYOUT.P1_OFF

    NEG_INF = cutlass.Float32(float("-inf"))

    CHUNK = 64
    P_COLS_PER_CHUNK = CHUNK // 2
    stats_off = LAYOUT.STATS_OFF + sub_tile_id * LAYOUT.STATS_STRIDE

    # Phase trackers persist across tile boundaries (barriers don't reset).
    bmm1_phase = cutlass.Int32(0)
    stat_empty_phase = cutlass.Int32(1)  # bootstrap pre-armed at phase 1 so first wait passes immediately
    # init phase=1; both softmax wgs wait on slot [0] not [SoftmaxGid].
    epilogue_state = cutlass.Int32(1)

    # total_sum kept as Vector[Float32, 2] (even/odd partials) so per-iter update lowers to packed FMUL2 + FADD2.
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

    softmax_wg_base_const = CFG.SOFTMAX_WG0_BASE if sub_tile_id == 0 else CFG.SOFTMAX_WG1_BASE
    tid_in_wg = cute.arch.thread_idx()[0] - cutlass.Int32(softmax_wg_base_const * 32)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        # Top-of-tile mb_o_empty[0] wait: without it softmax can race into the
        # next tile while TMA-STG is still draining the prior tile's O slot.
        bars.mb_o_empty[0].wait(epilogue_state)
        epilogue_state = epilogue_state ^ cutlass.Int32(1)

        total_max = NEG_INF
        total_max_safe = NEG_INF
        total_sum = cutlass.Vector.from_elements(
            (cutlass.Float32(0.0), cutlass.Float32(0.0)),
            cutlass.Float32,
        )
        # PackGQA: q_abs is the row's TOKEN index (row // PACK_G): every mask
        # predicate downstream is a token-space compare, and all PACK_G rows of
        # one token share it.
        q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE)
        q_abs = q_row_coord + cutlass.Int32(sub_tile_id * TOKENS_PER_TILE) + (tid_in_wg // cutlass.Int32(HEADS_PER_TILE))
        # Bootstrap stat_empty wait BEFORE kv loop — lifts wait off per-iter
        # critical path so alpha publish + stat_full fire happen back-to-back.
        bars.mb_stat_empty[sub_tile_id].wait(stat_empty_phase)
        stat_empty_phase = stat_empty_phase ^ 1
        # 3-segment kv loop: LEFT-masked / fully-unmasked / RIGHT-masked.
        # At MASK_NONE bounds collapse so the two masked sub-loops have empty range and fold out.
        if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
            for kv_loop in cutlass.range(bounds.left, bounds.right, 1, unroll=1):
                total_max, total_max_safe, total_sum, bmm1_phase, stat_empty_phase = _softmax_kv_body(
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
                    total_max_safe,
                    total_sum,
                    bmm1_phase,
                    stat_empty_phase,
                    leader_cta_id,
                )
        else:
            for kv_loop in cutlass.range(bounds.left, bounds.unmasked_lo, 1, unroll=1):
                total_max, total_max_safe, total_sum, bmm1_phase, stat_empty_phase = _softmax_kv_body(
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
                    total_max_safe,
                    total_sum,
                    bmm1_phase,
                    stat_empty_phase,
                    leader_cta_id,
                )
            for kv_loop in cutlass.range(bounds.unmasked_lo, bounds.unmasked_hi, 1, unroll=1):
                total_max, total_max_safe, total_sum, bmm1_phase, stat_empty_phase = _softmax_kv_body(
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
                    total_max_safe,
                    total_sum,
                    bmm1_phase,
                    stat_empty_phase,
                    leader_cta_id,
                )
            for kv_loop in cutlass.range(bounds.unmasked_hi, bounds.right, 1, unroll=1):
                total_max, total_max_safe, total_sum, bmm1_phase, stat_empty_phase = _softmax_kv_body(
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
                    total_max_safe,
                    total_sum,
                    bmm1_phase,
                    stat_empty_phase,
                    leader_cta_id,
                )

        # End-of-kv: publish (total_max, total_sum_final) to TMEM Stats — corr reads it for LSE.
        total_sum_scalar = total_sum[0] + total_sum[1]

        stats_addr_epi = tmem_ptr_i32.load() + cutlass.Int32(stats_off)
        stats_vec_epi = cutlass.Vector.from_elements((total_max_safe, total_sum_scalar), cutlass.Float32)
        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr_epi, cutlass.Float32), stats_vec_epi)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        bars.mb_stat_full[sub_tile_id].arrive()

        # make_warp_uniform on scheduler-payload loads keeps nxt_* in uniform
        # registers across the back-edge so ptxas doesn't STL them.
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
    tidx,
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
    cta_id_x,
    qh_per_kh,
    o_partial_f32=None,
):
    """Correction warp group: 4 warps x 32 lanes = 128 lanes, 1 lane per O row.

    Per kv iter rescales O by alpha (skipped when all_alpha_one warp ballot
    fires); per-tile epilogue normalizes O by 1/total_sum, casts to fp16,
    swizzled-stores to sO, fires o_full for TMA-STG.  Holds the P14
    end-of-tile catch-up flip on bmm2_done_phase (needed at n_kv=1 multi-wave).
    """
    # Wait on MMA's TMEM-publish named barrier BEFORE any tmem_ptr_i32.load() —
    # without it correction can race MMA's tcgen05_alloc and read a stale base.
    nvvm.barrier_cta_sync(barrier_id=2, thread_count=32 * (CFG.CORRECTION_WARPS + 1))

    tid_raw = cute.arch.thread_idx()[0]
    tid_in_wg = tid_raw - cutlass.Int32(CFG.CORR_WARP_BASE * 32)

    # O_CHUNK=16 (halved from 32) shortens the alpha-rescale live range —
    # at 32, DSL regalloc spilled correction-warp regs to the stack.
    O_CHUNK = 16
    N_CHUNKS_O = CFG.TILE_O // O_CHUNK
    # Use O_SWZ_B (NOT V_SWZ_B): under cga2 V is split along d_v and may drop
    # to a narrower swizzle while O is full-row.  V-based stride here silently
    # produces correct TMEM but garbled SMEM → wrong O after TMA-STG.
    TMA_O_ITERS = (CFG.TILE_O * CFG.BPE) // CFG.O_SWZ_BYTES
    D_BLOCK_SIZE = CFG.TILE_O // TMA_O_ITERS
    TMA_O_GRANU_ELEMS = CFG.TILE_M * D_BLOCK_SIZE

    stat_full_phase = cutlass.Int32(0)
    # bmm2_done starts at phase=0; iter 0 is skipped, first wait at kv_loop=1.
    bmm2_done_phase = cutlass.Int32(0)
    o_empty_phase = cutlass.Int32(1)  # bootstrap

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
            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                bars.mb_bmm2_ready[qs * CFG.N_BMM2_CHUNKS + 0].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                bars.mb_stat_full[qs].wait(stat_full_phase)
                bars.mb_stat_empty[qs].arrive()
            stat_full_phase = stat_full_phase ^ 1
        else:
            # Empty-mainloop: corr fires (DSMEM-arrive on leader at cga2); MMA fires bmm2_done[0/1] via multicast.
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

                # all_alpha_one ballot: when every lane has alpha==1.0 the entire rescale loop skips.
                alpha_is_one = alpha == cutlass.Float32(1.0)
                all_alpha_one = vote_sync(0xFFFFFFFF, alpha_is_one, VoteSync.ALL)

                bars.mb_stat_empty[qs].arrive()

                bars.mb_bmm2_done[qs].wait(bmm2_done_phase)

                # vec_scale_pair emits nvvm.mul_packed_f32x2 → FMUL2.  Without it
                # plain o_chunk*alpha lowers to scalar FMUL inside the runtime-if
                # (downstream fp32 tcgen05_st doesn't force packed regs).
                if ~all_alpha_one:
                    for chunk_idx in cutlass.range_constexpr(N_CHUNKS_O):
                        o_addr = tmem_base_iter + cutlass.Int32(tmem_O_off + chunk_idx * O_CHUNK)
                        o_chunk = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                            num=O_CHUNK,
                        )
                        o_scaled = vec_scale_pair(o_chunk, alpha, O_CHUNK)
                        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(o_addr, cutlass.Float32), o_scaled)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)

                bars.mb_bmm2_ready[qs * CFG.N_BMM2_CHUNKS + 0].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            stat_full_phase = stat_full_phase ^ 1
            bmm2_done_phase = bmm2_done_phase ^ 1

        # === End-of-kv epilogue ===
        tmem_base_epi = tmem_ptr_i32.load()
        for qs in cutlass.range_constexpr(CFG.TILES_Q):
            stats_off = LAYOUT.STATS_OFF + qs * LAYOUT.STATS_STRIDE
            tmem_O_off = LAYOUT.O0_OFF if qs == 0 else LAYOUT.O1_OFF

            bars.mb_bmm2_done[qs].wait(bmm2_done_phase)

            # Wait the (total_max, total_sum_final) publish from softmax's epilogue.
            bars.mb_stat_full[qs].wait(stat_full_phase)

            stats_addr = tmem_base_epi + cutlass.Int32(stats_off)
            stats_vec = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(stats_addr, cutlass.Float32),
                num=2,
            )
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            total_max_scaled = stats_vec[0]  # log2-units (softmax stores it scaled)
            total_sum = stats_vec[1]

            # Fire stat_empty so softmax's NEXT-tile iter 0 wait can pass.
            bars.mb_stat_empty[qs].arrive()
            # Release MMA's NEXT-tile prologue BMM1 into this S_acc slot: the
            # final stats ride the slot HEAD and are now safely in registers
            # (tcgen05_wait LOAD above).  Cross-CTA arrive on the leader under
            # cga2 — the collective BMM1 writes both peers' TMEM.
            bars.mb_stats_read[qs].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            inv_sum = cutlass.Float32(0.0)  # pre-declare for DSL if-staging
            # Same reason: row_dead is only bound under some mask/sink configs, but
            # the fp32-partial store reads it unconditionally.  False here means
            # 'not dead'; the branches below override it where it is meaningful.
            row_dead = cutlass.Float32(0.0) > cutlass.Float32(1.0)
            lse_val = cutlass.Float32(0.0)
            q_row_global = (
                q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE) + cutlass.Int32(qs * TOKENS_PER_TILE) + (tid_in_wg // cutlass.Int32(HEADS_PER_TILE))
            )
            row_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE) + (tid_in_wg % cutlass.Int32(HEADS_PER_TILE))
            # With sinks: fold the lift-the-max rescale into threshold_beta so O is scaled by scale/new_sum in one FMUL.
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
                # a sink.  Same idiom as the sm107 f16 / sm100 mxfp8 kernels' _kv_empty.
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
                # Safe inverse: avoid div by 0 on fully-masked rows.
                inv_sum = cutlass.Float32(1.0) / cute.math.max(total_sum, cutlass.Float32(1e-30))
                # Dead row (no valid KV column at all): O := 0, LSE := -inf.
                # total_sum >= 1 for any alive row so this never fires spuriously.
                row_dead = total_sum <= cutlass.Float32(0.0)
                neg_inf_lse = cutlass.Float32(float("-inf"))
                lse_val = cutlass.Float32(arith.select(row_dead.ir_value(), neg_inf_lse.ir_value(), lse_val.ir_value()))
                inv_sum = cutlass.Float32(arith.select(row_dead.ir_value(), cutlass.Float32(0.0).ir_value(), inv_sum.ir_value()))

            # OOB-row guard: under cga2 the cluster's Q rows can exceed seqlen_q;
            # without the guard the write aliases the next head's LSE slot.
            if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT):
                # Dense padded-Q trim (cuDNN >= 9.14): q rows >= seq_len_q[b]
                # write O := 0 / LSE := -inf.  Applied AFTER the sink branch on
                # purpose — a trimmed row is dead even with a sink.  Per-batch
                # q lens come in via the dedicated seq_q_lens_addr parameter.
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
            elif cutlass.const_expr(CFG.THD_VARLEN):
                # THD: q_row_global is sequence-local; the packed ragged-Stats
                # LSE is written in the caller's declared layout — head-major
                # rank-3 [1, QH, head_stride] (index [0, head, cu_q[b] + local])
                # or token-major rank-2 [T, QH] (index [cu_q[b] + local, head])
                # — bound by per-sequence Q len S_q_b.
                _cu = cutlass.make_array_view(seq_kv_lens_tensor)
                _cu_q_b = cutlass.Int32(_cu[n_batch + batch_idx])
                _s_q_b = cutlass.Int32(_cu[n_batch + batch_idx + cutlass.Int32(1)]) - _cu_q_b
                if q_row_global < _s_q_b:
                    lse_arr = cutlass.make_array_view(lse_tensor)
                    if cutlass.const_expr(len(lse_tensor.shape) == 2):
                        # token-major packed (T, H)
                        lse_row = lse_arr[_cu_q_b + q_row_global, :]
                        lse_row[head_idx] = lse_val
                    else:
                        # head-major packed (1, QH, head_stride)
                        if cutlass.const_expr(len(lse_tensor.shape) == 4):
                            # rank-4 = per-batch padded Stats (B, QH, s_max, 1) in the declared strides, no ragged offsets
                            lse_arr[batch_idx, head_idx, q_row_global, 0] = lse_val
                        else:
                            lse_row = lse_arr[cutlass.Int32(0), head_idx, :]
                            lse_row[_cu_q_b + q_row_global] = lse_val
            else:
                if q_row_global < seqlen_q:
                    lse_arr = cutlass.make_array_view(lse_tensor)
                    # KV split: this chunk's LSE goes to its own split-major
                    # slot (batch extent B*SPLIT_KV), matching where TMA-STG put
                    # the chunk's O.  The pair (O_s, lse_s) is everything the
                    # combine needs.  Folds to batch_idx at SPLIT_KV == 1.
                    lse_batch = _partial_batch(batch_idx, split_idx, n_batch)
                    lse_arr[lse_batch, row_head_idx, q_row_global] = lse_val

            sO_sub_base = sO[qs].base

            if cutlass.const_expr(_FP32_PARTIALS):
                _store_fp32_partial_tile(
                    o_partial_f32,
                    tmem_base_epi,
                    tmem_O_off,
                    inv_sum,
                    row_dead,
                    q_row_global < seqlen_q,
                    _partial_batch(batch_idx, split_idx, n_batch),
                    q_row_global,
                    row_head_idx,
                    CFG.TILE_O,
                    O_CHUNK,
                )
                bars.mb_o_empty[qs].wait(o_empty_phase)
            else:
                for chunk_idx in cutlass.range_constexpr(N_CHUNKS_O):
                    o_fp16 = cutlass.Vector.from_elements(
                        tuple(STORAGE_DTYPE(0.0) for _ in range(O_CHUNK)),
                        STORAGE_DTYPE,
                    )
                    if cutlass.const_expr(not MAY_BE_EMPTY) or (bounds.right > bounds.left):
                        o_addr = tmem_base_epi + cutlass.Int32(tmem_O_off + chunk_idx * O_CHUNK)
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
                    # mb_o_empty[qs] wait gates the FIRST SMEM store (not the
                    # earlier TMEM-load loop), keeping TMEM-load/FFMA/cast overlapped
                    # with TMA-STG draining the prior persistent tile.
                    if chunk_idx == 0:
                        bars.mb_o_empty[qs].wait(o_empty_phase)
                    smem_ptr.store_swizzled(o_fp16, alignment=64, swizzle=_O_SMEM_SWIZZLE)

                # fence_proxy needed before TMA reads SMEM written by tcgen05_st (via store_swizzled).
                nvvm.fence_proxy("async.shared", space="cta")

            bars.mb_o_full[qs].arrive()

        stat_full_phase = stat_full_phase ^ 1
        o_empty_phase = o_empty_phase ^ 1
        # P14 catch-up flip — bmm2_done_phase ^= 1 AFTER epilogue wait
        # (mainloop flips n_kv-1 times, MMA fires n_kv times; +1 here matches —
        # without this, n_kv=1 multi-wave deadlocks on the 2nd tile).
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

    # End-of-warp tmem_dealloc: under cga2 each corr lane ALSO DSMEM-arrives
    # on the peer so the peer's local mbar accumulates the full CGA-total count.
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        peer_cta = cta_id_x ^ cutlass.Int32(1)
        bars.mb_tmem_dealloc.arrive_on_peer(peer_cta)
    bars.mb_tmem_dealloc.arrive()


# === Host launcher ===


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
    """Host entry: device pointers, runtime extents and strides in, TMA encodes and launches out.

    Operands are ``[B, S, H, D]`` with the head dim innermost (element stride 1);
    ``*_strides`` carry the (seq, head) element strides, K/V additionally the
    outer stride, which is the page stride of a paged pool and unused otherwise.
    Every stride leaf is Int64 (the ``compile()`` fakes fix the width): a 16-bit
    operand with S * H * D >= 2^27 elements would wrap the Int32 TMA-unit scaling.
    ``problem_size`` = (B, QH, KH, SQ, SKV, 0); under THD SQ/SKV are the packed
    token totals, under paged KV SKV is ``max_pages * PAGE_SIZE``. The batch
    stride of a dense operand is ``S * seq_stride``; a packed THD operand
    has batch extent 1 and binds the seq stride there (never stepped).

    ``lse_kind``: "dense" (B*SPLIT_KV, QH, SQ) in ``lse_strides``; "token" (SQ, QH)
    packed; "head" (1, QH, lse_ext) with lse_ext the head-row stride; "padded"
    (B, QH, lse_ext, 1) in ``lse_strides``. ``lse_ptr`` None compiles the store out."""
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
        tensor_map_qwords=_TENSOR_MAP_QWORDS,
        paged=PAGED_KV,
        page_size=PAGE_SIZE,
        block_table_ptr=block_table_ptr,
        block_table_v_ptr=block_table_v_ptr,
        table_strides=table_strides,
        n_pages=n_pages,
    )

    # Tensors are [B, S, H, D] with stride_order=(3, 2, 1, 0); D is fastest.
    # K is split along seq under cga2 — box rows are per-CTA.  V is split along d_v
    # (plumbed via num_iters//CTA_MMA).  O's TMA box inner dim follows O's swizzle, NOT V's.
    _O_GRANU_ELEMS = CFG.O_SWZ_BYTES // CFG.BPE
    qk_box_q = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, TMA_QK_GRANU_ELEMS)
    # Paged KV: K/V are [num_pages, page_size, H_kv, D] views of the page pool
    # (batch -> page, seq -> row-in-page) and a tile is a stack of K_BOXES /
    # V_BOXES row boxes, so the descriptor box is one box tall.  Dense: the
    # full per-CTA tile.
    qk_box_k = (1, K_BOX_ROWS, 1, TMA_QK_GRANU_ELEMS)
    vo_box_v = (1, V_BOX_ROWS, 1, TMA_VO_GRANU_ELEMS)
    vo_box_o = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, _O_GRANU_ELEMS)
    stride_order = (3, 2, 1, 0)
    # Paged KV: the in-page layout is nothing but the pool's strides.  HND
    # storage ([num_pages, H_kv, page_size, D]) viewed as [page, row, H_kv, D]
    # has the row stride BELOW the head stride, so its descriptor must list dims
    # innermost-first as (D, row, H_kv, page) and the TMA-LDG warp swaps its
    # (head, row) coords to match; NHD is the dense BSHD order with batch -> page.
    if cutlass.const_expr(paged_hnd):
        kv_stride_order = (3, 1, 2, 0)
    else:
        kv_stride_order = stride_order

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
    _v_tma_swz = tmap.TensorMapSwizzle.s128b_atom_32b if IS_TF32 else _tma_swz(CFG.V_SWZ_BYTES)
    tma_v_desc = tmap.create_tensor_map_tiled_from_view(
        v_tensor,
        box_dims=vo_box_v,
        stride_order=kv_stride_order,
        swizzle=_v_tma_swz,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    _o_box = list(vo_box_o)
    if _FP32_PARTIALS:
        _o_box[-1] = max(1, _o_box[-1] * CFG.BPE_O // 4)
    tma_o_desc = tmap.create_tensor_map_tiled_from_view(
        o_tensor,
        box_dims=tuple(_o_box),
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.O_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )

    # Each cluster pair (CTA_MMA CTAs) collectively covers TILE_M*TILES_Q*CTA_MMA Q rows;
    # without the cluster-wide divisor cga2 over-launches and OOB clusters collide in GMEM.
    # PackGQA: SQ*PACK_G packed rows per packed head, and QH/PACK_G packed heads.
    rows_per_cluster = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA
    q_clusters = (SQ * HEADS_PER_TILE + rows_per_cluster - 1) // rows_per_cluster
    grid_q_supers = q_clusters * CFG.CTA_MMA
    q_supers = grid_q_supers
    if cutlass.const_expr(CFG.THD_VARLEN):
        # Setup launch (issue #552): builds the [kv | cu_q | cu_k] metadata
        # buffer device-side from the caller's length tensors, then the
        # per-batch O descriptors and the packed-total-clamped K/V descriptors
        # (o_desc_words slots n_batch+1 / n_batch+2). One elected thread.
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
            cutlass.Int32(QH // HEADS_PER_TILE),
            cutlass.Int32(B),
            cutlass.Int64(o_tensor.stride[1]),
            cutlass.Int32(CGA_TILE_M),
            n_thd_units,
            not PAGED_KV,  # clamp_kv: paged pools have no packed KV total to clamp to
        ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
        grid_shape = (n_thd_units * cutlass.Int32(CFG.CGA_M), cutlass.Int32(1), cutlass.Int32(1))
    else:
        # NATURAL: one cluster per (q super-tile, head tile, batch*split);
        # LPT policies flatten the work list and remap it in the kernel.
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
        seq_q_lens_addr,
        o_partial_f32,
        block_table_tensor,
        block_table_v_tensor,
        paged_hnd,
    ).launch(
        grid=grid_shape,
        block=[CFG.THREADS_PER_CTA, 1, 1],
        cluster=(CFG.CTA_MMA, 1, 1),
        stream=stream,
    )


EXPLICIT_ABI = True  # pointer/int host entry; the adapter builds the argument list itself
LSE_KINDS = ("dense", "token", "head", "padded")


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    d_qk: int = CFG.TILE_K,
    d_v: int = CFG.TILE_O,
    has_lse: bool = True,
    lse_kind: str = "dense",
    paged_hnd: bool = False,
) -> Callable:
    """Compile the host entry for one layout kind.

    Every extent and stride is a runtime argument of the artifact (see ``_host``),
    so the key is only what specializes the traced code: the head-dim ENVELOPE
    (``d_qk`` / ``d_v``: the TMA descriptors carry the real extents while the
    tile box stays the compile-time TILE geometry — box columns past d_qk / d_v
    zero-fill on load, O columns past d_v clip on store), whether the LSE store
    exists, the Stats layout kind, and for paged pools whether the in-page
    layout is HND (row stride below head stride).

    Constraint: every non-innermost TMA global stride must be a 16-byte
    multiple; the compact BSHD H-stride is d * BPE, so d must be a multiple of
    8 at 2 bytes/elem — checked here for the envelope, checked by the adapter
    for declared strides."""
    _cache_key = _template_key(globals(), locals(), "compile")
    if not (0 < d_qk <= CFG.TILE_K and 0 < d_v <= CFG.TILE_O):
        raise ValueError(f"d128 envelope: need 0 < d_qk <= {CFG.TILE_K} and 0 < d_v <= {CFG.TILE_O}; got ({d_qk}, {d_v})")
    if (d_qk * CFG.BPE) % 16 != 0 or (d_v * CFG.BPE_O) % 16 != 0:
        raise ValueError(f"d128 envelope: d_qk*BPE and d_v*BPE must be 16-byte multiples (TMA global-stride rule); got ({d_qk}, {d_v}) at BPE={CFG.BPE}")
    if SPLIT_KV > 1 and not has_lse:
        raise ValueError("d128: split_kv > 1 requires has_lse=True (the per-split LSE drives the combine)")
    if lse_kind not in LSE_KINDS:
        raise ValueError(f"lse_kind must be one of {LSE_KINDS}; got {lse_kind!r}")
    if has_lse and (lse_kind == "dense") == bool(CFG.THD_VARLEN):
        raise ValueError("lse_kind 'dense' is the dense form; 'token' / 'head' / 'padded' are the THD forms")
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
        symbol="frost_sdpa_fwd",
    )
