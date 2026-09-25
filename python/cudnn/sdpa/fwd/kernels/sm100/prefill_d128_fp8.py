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
  5. **exp2 split MUFU / FMA** (``_E2E_*``, **cc 10.0 only**): 32 of the 128 P
     columns of every softmax row are evaluated on the FMA pipe by
     ``exp2_emul_pair`` (blocks of 4, spread over both P chunks), the alpha
     exp2 stays on MUFU -- 97 instead of 129 MUFU.EX2 per row per KV step.
     MEASURED on B200 (llama layer, H=64/8, E4M3, A/B/A x3): +4.5 % at S=8K
     dense, +2.4 % at S=2K, +1.5 % at S=32K, causal +0.3 % (noise).
     ``PARAMS.exp2_fma_split`` (auto-set from the build device) folds it out on
     every other cc, where MUFU.EX2 runs at twice B200's rate and the same
     split loses (``_E2E_ENABLED``).

THD / varlen (``CFG.THD_VARLEN=1``) follows the device-built-metadata +
plan-time-envelope design (``write_thd_meta``, issue #552 / PRs #606, #608)
used by the f16 kernels — FP8 is element-addressed (no block-scale SF), so it
rides the same path: packed ``[1,T,H,D]`` with DYNAMIC token extents (the
packed totals never reach the host), the setup kernel builds the
[kv|cu_q|cu_k] metadata + per-batch O TMA descriptors device-side, and the
launch grid is sized to the MACHINE and bounded by a device-read live unit
count (issue #618; units past that total decode the batch == n_batch sentinel
and drain without loads or stores).
Ragged Stats ride the caller's declared layout (token-major (T, H) or
head-major); the amax_o atomicMax is gated on live rows. Dense path
byte-identical.

Paged KV (``CFG.PAGED_KV=1``, issue #920) is the f16 d128 kernel's specialization
ported hunk-for-hunk (see sm100/prefill_d128_f16.py): K/V are page POOLS —
``[num_pages, page_size, H_kv, D]`` (NHD) or ``[num_pages, H_kv, page_size, D]``
(HND, a stride permutation of the same descriptor) — and a per-batch
``block_table`` [B, max_pages] int32 names the pages of each sequence.  Only
the TMA-LDG warp changes: a K/V tile is issued as ``TILE_N / page_size`` row
boxes (one box inside the page when the page is taller than the tile), each
box's page coordinate read from the block table on device.  Boxes past the
batch's ``ceil(seq_kv_len / page_size)`` pages get page coordinate ``-1`` —
TMA-OOB, exact zeros, mbarrier bytes still credited.  Rows inside the last live
page but past ``seq_kv_len`` are loaded and masked (MASK_PADDED is mandatory),
so the cache bytes of an allocated page must be finite.  Nothing FP8-specific
touches paging: the per-tensor descales are page-invariant scalars folded
in-kernel, P is cast with the same baked bias, and the in-kernel amax runs over
the same live rows; the 128 B swizzle atom is 8 rows at any BPE, so the
page-size rule (multiple of 8 dividing 128, or a multiple of 128) is unchanged.
THD queries over pools are NOT wired here (module-scope guard): the FP8 THD
path clamps its runtime K/V descriptors to a packed KV total a pool does not
have.
"""

from cudnn.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
import os
import sys
from functools import lru_cache
from typing import Callable, Optional, Tuple


from cutlass.base_dsl.typing import Pointer  # was the legacy DSL Pointer pre-DSL-bump
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
# O row bytes: TILE_O storage elements / O_PACK_DIV (E2M1 packs two per byte).
O_ROW_BYTES = CFG.TILE_O * CFG.BPE_O // CFG.O_PACK_DIV
TMA_O_ITERS_HOST = O_ROW_BYTES // CFG.O_SWZ_BYTES

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
    scheduler_warp_loop_persistent,
    read_tile_id_arrive,
    read_clc_payload,
    SCHED_NATURAL,
    SCHED_LPT,
    SCHED_LPT_L2,
)
from cudnn.frost.tile_dsl.pointwise import (
    exp2_mixed,
    # SM100: no LDTM.STAT — the MASK_NONE fast path uses manual tcgen05_ld +
    # row_max_reduction (see _softmax_kv_body); tmem_load_max_reduction_tile
    # is not imported.
    row_reduction_pair,
    row_max_reduction,
    vec_scale_pair,
    fp32_to_fp8_pack,
    fp32_to_e2m1_pack,
    e4m3_scale_rcp,
    amax_to_ue8m0_rp,
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

# exp2 split between MUFU and the FMA pipe.  With 128 exps per row per KV step the
# MUFU pipe (4 lanes/clk/SMSP) is the softmax warps' longest pipe on SM100 (2 x 128
# x 128 exps per SM per step = 2048 clk against ~1024 clk of FP8 MMA), while the
# FP32 pipe has slack; a compile-time subset of the P columns is evaluated by
# ``exp2_emul_pair`` (6 packed instructions per pair) instead.  The ALPHA (rescale)
# exp2 stays on MUFU.  The pattern is the same SHAPE as the cuDNN backend fprop
# kernel's (of every _E2E_FREQ columns the LAST _E2E_RES are emulated, none at or
# past _E2E_LIMIT) with the distribution taken from the MXFP8 sibling
# (``prefill_d128_mxfp8.py``), where it was tuned on B200 (B=1 H=24/8 S=16K dense,
# A/B/A): (16, 4, 128) = 32 columns in blocks of 4 over BOTH chunks, +1.63 % over the
# backend's own (16, 8, 72) and +9.73 % over the all-MUFU kernel; 24 / 40 / 48 / 64
# emulated columns and blocks of 2 / 8 all lost 0.9-8.7 pt.  At TILE_N=128 this
# emulates columns 12..15 of every 16 (8 blocks): MUFU.EX2 per row per step 129 -> 97.
# On THIS kernel MEASURED on B200 (llama layer B=1 H=64/8 E4M3, A/B/A x3, controls
# <= 0.4 %): +4.53 % at S=8K dense, +2.4 % at S=2K (by TFLOPS), +1.50 % at S=32K,
# causal (LPT_L2) +0.26 % = noise.  Derived per chunk as pair indices for exp2_mixed.
#
# ARCH GATE.  The split trades MUFU.EX2 pipe-time for FMA pipe-time, so its sign follows
# the part's MUFU rate -- MEASURED 16 elements/clk/SM on sm_100a (B200) and 32 on
# sm_107a (Rubin, where the same split is -9..-10 %); cc 10.3 (GB300) DOCUMENTS the
# doubled rate too.  ``PARAMS.exp2_fma_split`` is auto-set by the adapter from the
# BUILD device (api_dsl._exp2_fma_split_for: cc == (10, 0) x the kernels that measured
# a win) and folded here at trace time: off, ``_E2E_PAIRS`` are empty and both exp2
# sites trace ``cute.math.exp2`` -- the develop spelling.  Full rationale and the
# measured rates: the ``_E2E_*`` block of ``prefill_d128_mxfp8.py``.
_E2E_FREQ = 16
_E2E_RES = 4
_E2E_LIMIT = CFG.TILE_N
_E2E_CHUNK = CFG.TILE_N // CFG.N_BMM2_CHUNKS
_E2E_ENABLED = bool(PARAMS.exp2_fma_split)
# Emulated columns per row (32 of CFG.TILE_N=128 with the gate on, 0 off): the MUFU.EX2 the softmax
# still issues per row per KV step is the alpha exp2 plus the non-emulated columns (97 on, 129 off).
_E2E_EMULATED_COLS = (_E2E_LIMIT // _E2E_FREQ) * _E2E_RES if _E2E_ENABLED else 0
if not (0 <= _E2E_RES <= _E2E_FREQ and _E2E_FREQ % 2 == 0 and _E2E_RES % 2 == 0 and _E2E_LIMIT % _E2E_FREQ == 0):
    raise ValueError(f"{__name__}: exp2 emulation pattern freq={_E2E_FREQ} res={_E2E_RES} limit={_E2E_LIMIT} must be even with 0 <= res <= freq | limit")
if _E2E_CHUNK * CFG.N_BMM2_CHUNKS != CFG.TILE_N or _E2E_CHUNK % _E2E_FREQ != 0:
    raise ValueError(
        f"{__name__}: softmax chunk ({_E2E_CHUNK}) x N_BMM2_CHUNKS ({CFG.N_BMM2_CHUNKS}) must equal TILE_N ({CFG.TILE_N}) and be a multiple of the emulation period ({_E2E_FREQ})"
    )


def _e2e_pairs(chunk: int, chunk_elems: int) -> frozenset:
    """Pair indices (pair p = columns 2p, 2p+1 of the chunk) that ``exp2_mixed`` emulates."""
    return frozenset(
        p
        for p in range(chunk_elems // 2)
        if ((chunk * chunk_elems + 2 * p) % _E2E_FREQ) >= (_E2E_FREQ - _E2E_RES) and (chunk * chunk_elems + 2 * p) < _E2E_LIMIT
    )


_E2E_PAIRS = tuple(_e2e_pairs(c, _E2E_CHUNK) if _E2E_ENABLED else frozenset() for c in range(CFG.N_BMM2_CHUNKS))
if 2 * sum(len(p) for p in _E2E_PAIRS) != _E2E_EMULATED_COLS:
    raise ValueError(f"{__name__}: exp2 emulation pairs cover {2 * sum(len(p) for p in _E2E_PAIRS)} columns, expected {_E2E_EMULATED_COLS}")

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
elif CFG.DTYPE_O in (4, 5):
    # Block-scaled O: byte containers (E2M1 packed two per byte at 4, E4M3 at
    # 5) staged through the fp8 SMEM/TMA store path; the per-block scale
    # factors go straight to gmem from the correction warps (SF_O).
    OUT_STORAGE_DTYPE = cutlass.Float8E4M3FN
else:
    raise ValueError(f"prefill_sdpa_fp8: DTYPE_O={CFG.DTYPE_O} not supported (expected 0=E4M3 / 1=E5M2 / 2=BF16 / 3=FP16 / 4=NVFP4 / 5=MXFP8)")


from cudnn.sdpa.fwd.kernels._common_blackwell import (
    make_split_helpers,
    store_fp32_partial_tile as _store_fp32_partial_tile,
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
oBufferElems = CFG.TILE_M * CFG.TILE_O // CFG.O_PACK_DIV

qTmaTransactionBytes = qBufferElems * CFG.BPE * CFG.CTA_MMA
kTmaTransactionBytes = kBufferElems * CFG.BPE * CFG.CTA_MMA
vTmaTransactionBytes = vBufferElems * CFG.BPE * CFG.CTA_MMA


CGA_TILE_M = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA
# THD uses a persistent grid + device-bounded claim counter (not the CLC
# envelope). The adapter caps the launch at min(envelope, SMs / CGA_SIZE)
# and the setup kernel publishes the live unit total it stops at.
THD_PERSISTENT = True


# SM100 llama is always cga2 → LPT reverse-row count in CGA-tile units.
_sdpa_h = make_sdpa_helpers(CFG, lpt_q_tiles_in_cga_units=True)
_decode_initial = _sdpa_h.decode_initial
_decode_payload = _sdpa_h.decode_payload
# qtrim variant: collapses the KV loop for CGA tiles entirely past the
# per-batch actual Q length (SEQ_Q_LENS_PRESENT; folds to plain bounds otherwise).
_bounds_for_tile = _sdpa_h.bounds_for_tile_qtrim
_resolve_seqlen_kv = _sdpa_h.resolve_seqlen_kv
_resolve_seqlen_q = _sdpa_h.resolve_seqlen_q

# THD / varlen — shared helpers (FP8 element-addressed like f16, per-tensor
# dequant scales, no block-scale SF).  Gated by CFG.THD_VARLEN (folds out:
# _thd_tma_offsets is (0, 0, batch_idx) dense — TMA coords byte-identical).
# TILES_Q=2: q_seq_off applies to BOTH Q slabs + both O-store slabs.
# seq_kv_lens overloaded as the THD metadata buffer (int32 len 4B+4):
#   [0..B-1]=seq_kv_lens  [B..2B]=cu_q(B+1)  [2B+1..3B+1]=cu_k(B+1)
#   [3B+2..4B+1]=batch_remap(B)  [4B+2]=live units  [4B+3]=claim counter
# The decode walks batch_remap on EVERY THD flavor, so the setup kernel must
# fill it; the trailing two words are read only by the persistent schedulers.
# The setup kernel builds it DEVICE-side from the caller's length tensors
# (issue #552) — no length ever reaches the host — and publishes the live unit
# total + claim counter, so the adapter launches a MACHINE-sized grid rather
# than the plan-time envelope (issue #618).
from cudnn.sdpa.fwd.kernels.thd_helpers import build_thd_meta_o_kv_descs_kernel as _build_thd_meta_o_kv_descs_kernel, TENSOR_MAP_QWORDS

_TENSOR_MAP_QWORDS = TENSOR_MAP_QWORDS
_dispatch_decode_initial = _sdpa_h.dispatch_decode_initial
_dispatch_decode_payload = _sdpa_h.dispatch_decode_payload
_thd_tma_offsets = _sdpa_h.thd_tma_offsets

# === PackGQA ===
HEADS_PER_TILE = CFG.QH_PER_KH if CFG.PACK_GQA else 1
TOKENS_PER_TILE = CFG.TILE_M // HEADS_PER_TILE

# === KV split ===
#
# Mechanics live in _common_blackwell.make_split_helpers, shared with the other
# SM100 prefill flavors: each Q tile's KV loop range is cut into SPLIT_KV
# contiguous chunks, each run as its own persistent tile, and each writing a
# normalized partial O + its own LSE into a split-major workspace that
# sm100/split_combine folds with the exact log-sum-exp identity.  At
# SPLIT_KV == 1 every closure folds away and this is the classic kernel.


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
# guarantees page_size | 128 or 128 | page_size, so this is exact.  Identical to
# the f16 kernel: the box geometry is in ROWS, the 128 B swizzle atom is 8 rows
# at every BPE, and TMA_*_GRANU_ELEMS (128 at BPE=1) rides through unchanged.
PAGED_KV = bool(CFG.PAGED_KV)
PAGE_SIZE = CFG.PAGE_SIZE if PAGED_KV else 0
_K_TILE_ROWS = CFG.TILE_N // CFG.CTA_MMA
K_BOX_ROWS = min(PAGE_SIZE, _K_TILE_ROWS) if PAGED_KV else _K_TILE_ROWS
V_BOX_ROWS = min(PAGE_SIZE, CFG.TILE_N) if PAGED_KV else CFG.TILE_N
K_BOXES = _K_TILE_ROWS // K_BOX_ROWS
V_BOXES = CFG.TILE_N // V_BOX_ROWS
if PAGED_KV and CFG.THD_VARLEN:
    # The FP8 THD leg clamps its runtime K/V descriptors to the packed KV total
    # (_build_thd_meta_o_kv_descs_kernel); a page pool has no such total.  The
    # engine row and the adapter decline THD queries over FP8 pools; this is
    # the backstop that cannot compile the two together.
    raise ValueError("prefill_d128_fp8_sm100: PAGED_KV with THD_VARLEN is not wired (the FP8 THD path clamps runtime K/V descriptors)")


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
    seq_q_lens_addr: cutlass.Int64,
    o_partial_f32: Optional[cute.Tensor],
    # Paged KV: [B, max_pages] int32 page ids per batch, one table for K and
    # one for V (the graph contract declares them separately; callers with a
    # shared table bind the same buffer twice). None (folded out of the ABI)
    # unless CFG.PAGED_KV.
    block_table_tensor: Optional[cute.Tensor] = None,
    block_table_v_tensor: Optional[cute.Tensor] = None,
    # Paged KV: HND pool (row stride below head stride) -> descriptor dims
    # (D, row, H_kv, page); derived by _host from the bound strides.
    paged_hnd: cutlass.Constexpr[bool] = False,
    # Block-scaled O (CFG.O_BLOCK_SCALE > 0): SF_O byte buffer + its 128x4-atom
    # geometry (see _correction_warp_group). None / 0 when the mode is off.
    sf_o_tensor: Optional[cute.Tensor] = None,
    sfo_plane_stride: cutlass.Int32 = 0,
    sfo_row_off_b: cutlass.Int32 = 0,
    sfo_col_off_h: cutlass.Int32 = 0,
    sfo_cols: cutlass.Int32 = 0,
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
            o_scale_fused=o_scale_fused,
            amax_o_tensor=amax_o_tensor,
            o_partial_f32=o_partial_f32,
            qh_per_kh=qh_per_kh,
            sf_o_tensor=sf_o_tensor,
            sfo_plane_stride=sfo_plane_stride,
            sfo_row_off_b=sfo_row_off_b,
            sfo_col_off_h=sfo_col_off_h,
            sfo_cols=sfo_cols,
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
            seq_q_lens_addr=seq_q_lens_addr,
            o_desc_words=o_desc_words,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
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
            seq_q_lens_addr=seq_q_lens_addr,
            o_desc_words=o_desc_words,
            qh_per_kh=qh_per_kh,
            seqlen_kv=seqlen_kv,
        )

    else:  # warp_idx == CFG.SCHED_WARP_ID
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        # try_cancel.multicast::cluster::all — only CGA's (0,0,0) CTA issues.
        is_cga_first_cta = cta_id_x == cutlass.Int32(0)
        if cutlass.const_expr(CFG.THD_VARLEN):
            # THD: persistent grid + device-bounded claim counter, so no unit
            # past the live total is ever handed out (the CLC path would need
            # the grid to BE the work list, i.e. the plan-time envelope).
            # n_batch is a kernel argument -- do NOT re-derive it from the
            # metadata tensor's layout.
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
    row, whatever the caller left in the padding slots.  Same body as the f16
    kernel's (a @cute.jit helper, per python/cudnn/AGENTS.md "CuTeDSL kernel
    bodies"); only ``granu_elems`` differs by BPE and rides in as a parameter.
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
    o_desc_words,
    n_q_supers,
    n_qh,
    n_batch,
    qh_per_kh,
    is_leader,
    cta_in_pair,
    tma_mcast_mask,
    block_table_tensor=None,
    block_table_v_tensor=None,
    paged_hnd: cutlass.Constexpr[bool] = False,
):
    """Unified TMA-LDG warp — cga1/cga2 × MASK_NONE/PADDED/CAUSAL/SWA.

    Spill-free in OTHER_REGS (40) via the q_row_base trick: pre-multiplying
    q_super_idx into q_row_base after each scheduler decode keeps the LDS.128
    result in uniform registers instead of getting STL'd to local stack.
    """
    q_empty_phase = cutlass.Int32(1)
    kv_state = PipelineState.start(phase=1)

    tma_q = GmemTileTma(tma_q_desc)
    # Unlike the f16 kernels, THD here implies dense K/V: PAGED_KV with
    # THD_VARLEN is refused at module scope (the FP8 THD leg clamps runtime
    # K/V descriptors to a packed total), so no ``and not PAGED_KV`` is needed.
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
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
        bounds_init = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, CFG.QH_PER_KH)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    # Paged KV: this batch's live page count bounds the block-table walk (MASK_PADDED
    # is mandatory under PAGED_KV, so eff_seqlen_kv is always defined here).
    n_pages_b = cutlass.Int32(0)
    if cutlass.const_expr(PAGED_KV):
        n_pages_b = (eff_seqlen_kv + cutlass.Int32(PAGE_SIZE - 1)) // cutlass.Int32(PAGE_SIZE)

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
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
            bounds_next = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, CFG.QH_PER_KH)
            kv_left = bounds_next.left
            kv_right = bounds_next.right
            if cutlass.const_expr(PAGED_KV):
                n_pages_b = (eff_seqlen_kv + cutlass.Int32(PAGE_SIZE - 1)) // cutlass.Int32(PAGE_SIZE)

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
    seq_q_lens_addr,
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
            # fp32 partials wrote the workspace directly, so nothing is staged
            # to copy.  Skip ONLY the store: the arrive below and any QO_ALIAS
            # handshake after it must still run, or this warp laps the loader
            # and the parity waits deadlock.
            if cutlass.const_expr(not _FP32_PARTIALS):
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
    seq_q_lens_addr,
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
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
            bounds_init = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, CFG.QH_PER_KH)
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
                    q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, CFG.QH_PER_KH
                )
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
    # exp2 split: the _E2E_PAIRS[chunk] columns on the FMA pipe, the rest on MUFU (see _E2E_FREQ).  Gated per arch
    # at trace time (_E2E_ENABLED): off, both chunks are the plain MUFU exp2 -- the develop spelling.
    if cutlass.const_expr(_E2E_ENABLED):
        chunk_P_0 = exp2_mixed(chunk_S_0, _E2E_PAIRS[0], CHUNK)
    else:
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
        if cutlass.const_expr(_E2E_ENABLED):
            deferred_P_1 = exp2_mixed(chunk_S_1, _E2E_PAIRS[1], CHUNK)
        else:
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
    seq_q_lens_addr,
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

    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
    bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, CFG.QH_PER_KH)

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
        bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, CFG.QH_PER_KH)


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
    o_scale_fused,
    amax_o_tensor,
    qh_per_kh,
    o_partial_f32=None,
    sf_o_tensor=None,
    sfo_plane_stride=0,
    sfo_row_off_b=0,
    sfo_col_off_h=0,
    sfo_cols=0,
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
    TMA_O_ITERS = O_ROW_BYTES // CFG.O_SWZ_BYTES
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

    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
    bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, CFG.QH_PER_KH)

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
            # Same reason: row_dead is only bound under some mask/sink configs, but
            # the fp32-partial store reads it unconditionally.  False here means
            # 'not dead'; the branches below override it where it is meaningful.
            row_dead = cutlass.Float32(0.0) > cutlass.Float32(1.0)
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

            if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT):
                # Dense padded-Q trim: q rows >= seq_len_q[b] write O := 0 / LSE := -inf,
                # applied AFTER the sink branch (a trimmed row is dead even with a sink).
                # Folded into row_dead so the O store's select sanitizes the row even
                # when the padded Q rows held NaN (0 * NaN is NaN).
                _sq_arr = cute.make_tensor(cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4), cute.make_layout(1 << 24))
                _q_len_b = cutlass.Int32(_sq_arr[batch_idx])
                row_trim = q_row_global >= _q_len_b
                neg_inf_trim = cutlass.Float32(float("-inf"))
                lse_val = cutlass.Float32(arith.select(row_trim.ir_value(), neg_inf_trim.ir_value(), lse_val.ir_value()))
                inv_sum = cutlass.Float32(arith.select(row_trim.ir_value(), cutlass.Float32(0.0).ir_value(), inv_sum.ir_value()))
                row_dead = row_dead | row_trim

            # Base-2 Stats (stats_use_log2): natural LSE * log2(e); -inf stays -inf.
            if cutlass.const_expr(CFG.STATS_LOG2):
                lse_val = lse_val * cutlass.Float32(1.4426950408889634)

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
                            if cutlass.const_expr(len(lse_tensor.shape) == 4):
                                # rank-4 = per-batch padded Stats (B, QH, s_max, 1) in the declared strides, no ragged offsets
                                lse_arr[batch_idx, head_idx, q_row_global, 0] = lse_val
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

            if cutlass.const_expr(_FP32_PARTIALS):
                # fp32 partials: the accumulator goes straight to the workspace,
                # so the SMEM O tile and its TMA store are both bypassed.
                _store_fp32_partial_tile(
                    o_partial_f32,
                    tmem_base_epi,
                    tmem_O_off,
                    inv_sum,
                    row_dead,
                    _row_valid,
                    _partial_batch(batch_idx, split_idx, n_batch),
                    q_row_global,
                    row_head_idx,
                    CFG.TILE_O,
                    O_CHUNK,
                )
                # The TMA-store warp group still runs its handshake; release the
                # slot even though nothing was staged.
                bars.mb_o_empty[qs].wait(o_empty_phase)
            elif cutlass.const_expr(CFG.DTYPE_O <= 1):
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
            elif cutlass.const_expr(CFG.O_BLOCK_SCALE > 0):
                # Block-scaled O (DTYPE_O 4 = E2M1 + E4M3 SF per 16 d, 5 = E4M3 +
                # UE8M0 SF per 32 d).  One lane owns a whole O row in 16-column
                # TMEM chunks, so a scale block's amax is thread-local.  Data
                # rides the fp8 SMEM/TMA store (E2M1 packed two per byte, 64 B
                # rows); the SF byte goes straight to gmem in the 128x4 atom:
                #   off(r, c) = plane + (r//128)*128*C + (c//4)*512
                #             + (r%32)*16 + ((r//32)%4)*4 + c%4
                # with the plane / row / column offsets supplied by the adapter
                # (per-(b,h) planes or a token-major [B*S, H*C] matrix).
                _SWZ_BYTES_C = CFG.O_SWZ_BYTES
                _SWZ_SHIFT_C = 3 - _O_SWZ_B
                _SWZ_MASK_C = (1 << _O_SWZ_B) - 1
                _SUBTILE_BYTES_C = CFG.TILE_M * _SWZ_BYTES_C
                _GROUP = CFG.O_BLOCK_SCALE
                _CHUNKS_PER_GROUP = _GROUP // O_CHUNK
                _N_GROUPS = CFG.TILE_O // _GROUP

                row_base_bytes = tid_in_wg * cutlass.Int32(_SWZ_BYTES_C)
                row_xor_field = ((tid_in_wg >> cutlass.Int32(_SWZ_SHIFT_C)) & cutlass.Int32(_SWZ_MASK_C)) << cutlass.Int32(4)

                _sfo_r = q_row_global + batch_idx * sfo_row_off_b
                _sfo_c0 = row_head_idx * sfo_col_off_h
                _sfo_plane = (batch_idx * (n_qh * cutlass.Int32(HEADS_PER_TILE)) + row_head_idx) * sfo_plane_stride
                _sfo_row_part = (
                    _sfo_plane
                    + (_sfo_r >> cutlass.Int32(7)) * (sfo_cols << cutlass.Int32(7))
                    + (_sfo_r & cutlass.Int32(31)) * cutlass.Int32(16)
                    + ((_sfo_r >> cutlass.Int32(5)) & cutlass.Int32(3)) * cutlass.Int32(4)
                )
                _sfo_base = sf_o_tensor.iterator.raw_ptr()
                _zero_i = cutlass.Int32(0)
                _sfo_rows_pad = ((seqlen_q + cutlass.Int32(127)) >> cutlass.Int32(7)) << cutlass.Int32(7)

                def _load_o_chunk_scaled(_chunk: int):
                    """One 16-column TMEM chunk of this lane's O row, normalized and dead-row-sanitized."""
                    o_addr = tmem_base_epi + cutlass.Int32(tmem_O_off + _chunk * O_CHUNK)
                    o_chunk = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                        num=O_CHUNK,
                    )
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                    o_scaled = o_chunk * inv_sum
                    _zero_f = cutlass.Float32(0.0)
                    return [cutlass.Float32(arith.select(row_dead.ir_value(), _zero_f.ir_value(), o_scaled[i].ir_value())) for i in range(O_CHUNK)]

                for g in cutlass.range_constexpr(_N_GROUPS):
                    # Trace-time list build (no loop-carried Python list: the tracer
                    # rejects a list whose length changes across a join).
                    if cutlass.const_expr(_CHUNKS_PER_GROUP == 1):
                        o_elems = _load_o_chunk_scaled(g)
                    else:
                        o_elems = _load_o_chunk_scaled(2 * g) + _load_o_chunk_scaled(2 * g + 1)

                    g_amax = cutlass.Float32(0.0)
                    for _i in cutlass.range_constexpr(_GROUP):
                        _e = o_elems[_i]
                        g_amax = cute.math.max(g_amax, cute.math.max(_e, -_e))
                    _amax_o_local = cute.math.max(_amax_o_local, g_amax)

                    if cutlass.const_expr(CFG.DTYPE_O == 4):
                        # E2M1 max-normal is 6: the E4M3 scale maps the block amax onto it.
                        sf_byte, inv_sf = e4m3_scale_rcp(g_amax * cutlass.Float32(1.0 / 6.0))
                    else:
                        sf_byte, inv_sf = amax_to_ue8m0_rp(g_amax)
                    q_elems = [o_elems[i] * inv_sf for i in range(_GROUP)]

                    if g == 0:
                        bars.mb_o_empty[qs].wait(o_empty_phase)
                    for cj in cutlass.range_constexpr(_CHUNKS_PER_GROUP):
                        _chunk = g * _CHUNKS_PER_GROUP + cj
                        col_elems = _chunk * O_CHUNK
                        block_idx = col_elems // D_BLOCK_SIZE
                        col_in_block_bytes = (col_elems % D_BLOCK_SIZE) * CFG.BPE_O // CFG.O_PACK_DIV
                        block_off_bytes = block_idx * _SUBTILE_BYTES_C
                        swizzled_in_row = row_xor_field ^ cutlass.Int32(col_in_block_bytes)
                        addr_off_bytes = cutlass.Int32(block_off_bytes) + row_base_bytes + swizzled_in_row
                        o_ptr = sO_sub_base.subview(addr_off_bytes).data_ptr()
                        i32_ptr = Pointer(o_ptr, dtype=cutlass.Int32)
                        if cutlass.const_expr(CFG.DTYPE_O == 4):
                            # 16 E2M1 nibbles = 8 bytes = two b32 words.
                            i32_ptr.store(fp32_to_e2m1_pack([q_elems[cj * O_CHUNK + i] for i in range(O_CHUNK)]), alignment=8)
                        else:
                            i32_ptr.store(fp32_to_fp8_pack([q_elems[cj * O_CHUNK + i] for i in range(O_CHUNK)], dtype=cutlass.Float8E4M3FN), alignment=16)

                    # SF byte.  Per-(b,h) planes (sfo_row_off_b == 0): rows past the
                    # sequence in the last tile write 0 so the 128-row-padded atom
                    # never carries garbage.  Token-major ([B*S, H*C], rows of batch
                    # b+1 follow batch b's row S-1 directly): those tail rows belong
                    # to the next batch, so only valid rows are written and the
                    # matrix tail past B*S is the caller's to zero.
                    _sfo_c = _sfo_c0 + cutlass.Int32(g)
                    _sfo_off = _sfo_row_part + (_sfo_c >> cutlass.Int32(2)) * cutlass.Int32(512) + (_sfo_c & cutlass.Int32(3))
                    _sf_ptr = Pointer((_sfo_base + _sfo_off).tospace(cutlass.AddressSpace.generic), dtype=cutlass.Int8)
                    if _row_valid:
                        _sf_ptr.store(cutlass.Int8(sf_byte))
                    else:
                        # Pad rows exist only inside the plane's 128-row-padded
                        # extent; a fully OOB cluster tile (r >= that) owns no bytes.
                        if sfo_row_off_b == cutlass.Int32(0):
                            if _sfo_r < _sfo_rows_pad:
                                _sf_ptr.store(cutlass.Int8(_zero_i))
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
            # sm100/split_combine computes it over the recombined O instead;
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
        bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, CFG.QH_PER_KH)

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
    # Dense padded-Q trim: separate (B,)-int32 lengths. None folds the
    # parameter and all consumers out when the specialization is disabled.
    seq_q_lens_addr: cutlass.Int64 = 0,
    # THD device metadata build (issue #552): the CALLER's Q/KV length
    # tensors — (B,) per-batch lengths or (B+1,) cu prefix sums, per side via
    # thd_lens_form (bit 0: Q is cu, bit 1: KV is cu) — consumed only by the
    # setup kernel, which writes the [kv|cu_q|cu_k] metadata buffer
    # (seq_kv_lens_tensor) device-side. None (folded out of the ABI) for
    # dense graphs.
    thd_q_lens_tensor: Optional[cute.Tensor] = None,
    thd_kv_lens_tensor: Optional[cute.Tensor] = None,
    thd_lens_form: Optional[cutlass.Int32] = None,
    o_partial_f32: Optional[cute.Tensor] = None,
    # Paged KV: [B, max_pages] int32 block tables; None (folded out) unless
    # CFG.PAGED_KV.  The page axis is a DYNAMIC extent and defines the static
    # KV maximum the masks clamp against: seqlen_kv = max_pages * PAGE_SIZE.
    # They sit BEFORE the block-scaled O group: a paged build without it then
    # compiles no dynamic scalar slot that execute() would have to bind (an
    # omitted slot is always a None-specialized one).
    block_table_tensor: Optional[cute.Tensor] = None,
    block_table_v_tensor: Optional[cute.Tensor] = None,
    # Block-scaled O: SF_O byte buffer + 128x4-atom geometry (None / 0 when off).
    sf_o_tensor: Optional[cute.Tensor] = None,
    sfo_plane_stride: cutlass.Int32 = 0,
    sfo_row_off_b: cutlass.Int32 = 0,
    sfo_col_off_h: cutlass.Int32 = 0,
    sfo_cols: cutlass.Int32 = 0,
    stream: _cuda_driver.CUstream = None,
    prepared: cutlass.Constexpr[bool] = False,
) -> None:
    B, QH, KH, SQ, SKV, _ = problem_size
    if cutlass.const_expr(CFG.THD_VARLEN):
        # Packed token totals are runtime values (dynamic extents); the
        # problem_size slots are 0 by contract.
        SQ = q_tensor.shape[1]
        SKV = k_tensor.shape[1]
    if cutlass.const_expr(PAGED_KV):
        SKV = block_table_tensor.shape[1] * cutlass.Int32(PAGE_SIZE)

    # K box rows are per-CTA (TILE_N/CTA_MMA); O box inner must match O's swizzle, not V's.
    # O box sized in BPE_O — O may be written at BF16/FP16 (DTYPE_O != DTYPE_QKV).
    _O_GRANU_ELEMS = CFG.O_SWZ_BYTES // CFG.BPE_O
    if cutlass.const_expr(CFG.PACK_GQA and not prepared):
        # nested: `and` is staged by the DSL, and under dynamic_bhk the head extents
        # are runtime values -- PackGQA is dense-only, so this branch never sees them
        if cutlass.const_expr(q_tensor.shape[2] != k_tensor.shape[2] * CFG.QH_PER_KH):
            raise ValueError(f"CFG.QH_PER_KH ({CFG.QH_PER_KH}) does not match tensor head extents H_q={q_tensor.shape[2]}, H_kv={k_tensor.shape[2]}")
    qk_box_q = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, TMA_QK_GRANU_ELEMS)
    # Paged KV: K/V are [num_pages, page_size, H_kv, D] views of the page pool
    # (batch -> page, seq -> row-in-page) and a tile is a stack of K_BOXES /
    # V_BOXES row boxes, so the descriptor box is one box tall.  Dense: the
    # full per-CTA tile.
    qk_box_k = (1, K_BOX_ROWS, 1, TMA_QK_GRANU_ELEMS)
    vo_box_v = (1, V_BOX_ROWS, 1, TMA_VO_GRANU_ELEMS)
    vo_box_o = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, _O_GRANU_ELEMS)
    stride_order = (3, 2, 1, 0)
    # Paged KV: the in-page layout is nothing but the pool's strides (static —
    # compiled in).  HND storage ([num_pages, H_kv, page_size, D]) viewed as
    # [page, row, H_kv, D] has the row stride BELOW the head stride, so its
    # descriptor must list dims innermost-first as (D, row, H_kv, page) and the
    # TMA-LDG warp swaps its (head, row) coords to match; NHD is the dense BSHD
    # order with batch -> page.
    paged_hnd = bool(PAGED_KV) and k_tensor.stride[1] < k_tensor.stride[2]
    if cutlass.const_expr(PAGED_KV and (v_tensor.stride[1] < v_tensor.stride[2]) != paged_hnd):
        raise ValueError("paged K and V pools must share an in-page layout (both HND or both NHD)")
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
    # V TMA swizzle tracks per-CTA inner bytes.
    tma_v_desc = tmap.create_tensor_map_tiled_from_view(
        v_tensor,
        box_dims=vo_box_v,
        stride_order=kv_stride_order,
        swizzle=_tma_swz(CFG.V_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    # Under fp32 partials the epilogue writes o_tensor directly and this
    # descriptor is never used -- but it still has to BUILD, and an fp32 element
    # doubles the box's inner byte width past what the O swizzle allows.  Halve
    # the box so the descriptor stays legal; nothing reads it.
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
        # PERSISTENT cluster count — the adapter hands down n_thd_units
        # already capped to what the device holds resident, min(plan-time
        # envelope, SMs / CGA_SIZE), NOT the envelope itself (issue #618). It
        # doubles as the claim counter's seed: cluster c runs unit c off its
        # blockIdx, then pulls from the counter, so the grid and the seed must
        # be the same number. Dispatching past the live total stays safe — such
        # a unit decodes the batch == n_batch sentinel and drains without loads
        # or stores. grid_x = n_thd_units * CGA_M.
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
            cutlass.Int32(QH // HEADS_PER_TILE),
            cutlass.Int32(B),
            cutlass.Int64(o_tensor.stride[1]),
            cutlass.Int32(CGA_TILE_M),
            n_thd_units,
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
        seq_q_lens_addr,
        o_partial_f32,
        block_table_tensor,
        block_table_v_tensor,
        paged_hnd,
        sf_o_tensor,
        sfo_plane_stride,
        sfo_row_off_b,
        sfo_col_off_h,
        sfo_cols,
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
    lse_padded_rows: int = 0,
    lse_padded_order: tuple = (3, 2, 1, 0),
    dynamic_bhk: bool = False,
    lse_stride: Optional[tuple[int, int, int]] = None,
    d_qk: int = CFG.TILE_K,
    d_v: int = CFG.TILE_O,
    k_stride: Optional[tuple] = None,
    v_stride: Optional[tuple] = None,
    block_table_stride: Optional[tuple[int, int]] = None,
    block_table_v_stride: Optional[tuple[int, int]] = None,
) -> Callable:
    """Compile with ALL dims concrete — pins TMA strides at compile time.

    PAGED KV (``CFG.PAGED_KV``): ``k_stride`` / ``v_stride`` (REQUIRED) describe the
    page pool viewed as ``[num_pages, page_size, H_kv, D]`` (batch -> page, seq ->
    row-in-page; the strides alone say whether the pool is HND or NHD) and
    ``skv`` is IGNORED — the page count and the block table's page axis are
    runtime extents (``cute.sym_int``), so one artifact serves every cache size
    and every ``max_pages`` (Rule 4); the kernel derives its static KV maximum
    as ``block_table.shape[1] * page_size``.  ``block_table_stride`` /
    ``block_table_v_stride`` are the tables' declared ``(batch, page)`` strides
    (None = row-major compact).  Dense builds take none of these.

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
    _cache_key = _template_key(globals(), locals(), "compile")
    _b0, _qh0, _kh0 = b, qh, kh  # the problem_size fake: runtime scalars, values immaterial
    if dynamic_bhk:
        # Batch and head extents compile DYNAMIC: one artifact per layout class,
        # not per (b, qh, kh) -- serving shapes vary in all three. The kernel
        # already reads B / QH / KH from problem_size at run time; only the fakes
        # pinned them. A packed stride (None) derives from the dynamic extents;
        # a declared stride stays the fixed number it is.
        if not CFG.THD_VARLEN:
            raise ValueError("dynamic_bhk is THD-only (dense shapes still pin the fakes)")
        b = cute.sym_int(divisibility=1)
        qh = cute.sym_int(divisibility=1)
        kh = cute.sym_int(divisibility=1)
        if lse_padded_rows:
            lse_padded_rows = cute.sym_int(divisibility=1)
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
    if lse_padded_rows and not CFG.THD_VARLEN:
        raise ValueError("lse_padded_rows is THD-only (a dense LSE is the compact (B, H, S_q) form)")
    if lse_stride is not None and CFG.THD_VARLEN and not lse_padded_rows:
        raise ValueError("THD LSE is packed (token-major (T, H) or head-major (1, QH, head_stride)); declared strides serve the padded form only")
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
    if PAGED_KV:
        # [num_pages, page_size, H_kv, D] view of the page pool; the declared
        # strides ARE the in-page layout (HND: row stride D below head stride
        # P*D; NHD: BSHD-contiguous) and _host reads the layout off them.
        # Neither depends on num_pages, which therefore compiles dynamic.
        if k_stride is None or v_stride is None:
            raise ValueError("PAGED_KV: k_stride / v_stride (the pools' strides in [num_pages, page_size, H_kv, D] order) are required")

        def _fake_pool(shape, stride):
            if stride[3] != 1:
                raise ValueError(f"declared pool stride {stride}: the head dim must be innermost-contiguous (stride[3] == 1)")
            for axis in (1, 2):  # row/head global strides feed TMA: 16-byte rule (BPE=1 -> the stride itself)
                if (stride[axis] * CFG.BPE) % 16 != 0:
                    raise ValueError(f"declared pool stride {stride} axis {axis} must be a 16-byte multiple at BPE={CFG.BPE} (TMA global-stride rule)")
            return cute.runtime.make_fake_tensor(STORAGE_DTYPE, shape, tuple(stride), assumed_align=16)

        n_pages = cute.sym_int(divisibility=1)
        fake_k = _fake_pool((n_pages, PAGE_SIZE, kh, d_qk), k_stride)
        fake_v = _fake_pool((n_pages, PAGE_SIZE, kh, d_v), v_stride)
        # K and V tables share one dynamic page-axis symbol: the kernel reads
        # its KV maximum from the K table, so both must have the same extent.
        # Their strides are declared (plan-time) and bound as views: a
        # batch-innermost table ((1, B) strides) is as legal as a row-major one.
        _max_pages = cute.sym_int(divisibility=1)

        def _fake_table(stride):
            if stride is None:
                return cute.runtime.make_fake_compact_tensor(cutlass.Int32, (b, _max_pages), stride_order=(1, 0), assumed_align=4)
            return cute.runtime.make_fake_tensor(cutlass.Int32, (b, _max_pages), tuple(stride), assumed_align=4)

        fake_block_table = _fake_table(block_table_stride)
        fake_block_table_v = _fake_table(block_table_v_stride)
    else:
        if k_stride is not None or v_stride is not None or block_table_stride is not None or block_table_v_stride is not None:
            raise ValueError("k_stride / v_stride / block_table_*_stride are PAGED_KV-only (the dense FP8 kernel binds compact K/V)")
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
        fake_block_table = None
        fake_block_table_v = None
    fake_o = cute.runtime.make_fake_compact_tensor(
        # Under fp32 partials o_tensor IS the fp32 split workspace: the epilogue
        # writes it directly and the TMA descriptor built from it goes unused.
        cutlass.Float32 if _FP32_PARTIALS else OUT_STORAGE_DTYPE,
        # E2M1 O is bound as its byte container: d_v / 2 storage elements.
        (_o_batch, sq, qh, d_v // CFG.O_PACK_DIV),
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
        if lse_padded_rows:
            # Per-batch padded Stats without ragged offsets (FlashInfer's (b, s_max, h)
            # buffer): rank-4 (B, QH, s_max, 1) in the caller's strides -- the RANK is
            # what selects the per-batch store, so nothing about the extents or
            # strides has to be static. Rows past a sequence's length are the
            # adapter's to fill (-inf), as the backend does.
            if lse_head_major or lse_head_stride:
                raise ValueError("lse_padded_rows excludes lse_head_major / lse_head_stride")
            fake_lse = (
                cute.runtime.make_fake_tensor(cutlass.Float32, (b, qh, lse_padded_rows, 1), (*lse_stride, 1), assumed_align=4)
                if lse_stride
                else cute.runtime.make_fake_compact_tensor(cutlass.Float32, (b, qh, lse_padded_rows, 1), stride_order=lse_padded_order, assumed_align=4)
            )
        elif lse_head_major:
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
    # overloads it as [seq_kv_lens(B)|cu_q(B+1)|cu_k(B+1)|batch_remap(B)|
    # live|ctr] (len 4B+4).
    _skv_len = cute.sym_int(divisibility=1) if dynamic_bhk else ((4 * b + 4) if CFG.THD_VARLEN else b)
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
    _odesc_len = cute.sym_int(divisibility=1) if dynamic_bhk else (((b + 3) * _TENSOR_MAP_QWORDS) if CFG.THD_VARLEN else 1)
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

    fake_seq_q_lens = cutlass.Int64(0)  # device address of the (B,) int32 Q lengths; 0 (unread) when the flag is off

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
    return _compile_cached(
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
        (_b0, _qh0, _kh0, 0, 0, 0) if CFG.THD_VARLEN else (_b0, _qh0, _kh0, sq, skv, 0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Int32(0),
        _fake_scale(),
        _fake_scale(),
        _fake_scale(),
        _fake_scale(),
        fake_amax_o,
        fake_seq_q_lens,
        fake_thd_q_lens,
        fake_thd_kv_lens,
        fake_thd_lens_form,
        # o_partial_f32 slot: the fp32 partial O under a split, else -- only when a
        # later slot follows it positionally (the paged tables, the block-scaled O
        # group) -- an explicit None; the THD tensors keep their slots when the mode
        # is off (a mid-signature slot shifts them by one).
        *((fake_o,) if _FP32_PARTIALS else ((None,) if (PAGED_KV or CFG.O_BLOCK_SCALE) else ())),
        # Paged KV: the two block tables.  They PRECEDE the block-scaled O group so a
        # paged build without it compiles no dynamic scalar slot execute() would have
        # to bind; a block-scaled dense build None-specializes them (never bound).
        *((fake_block_table, fake_block_table_v) if PAGED_KV else ((None, None) if CFG.O_BLOCK_SCALE else ())),
        # Block-scaled O (never combined with a split): the SF_O buffer (dynamic byte
        # length) + its four geometry scalars.
        *(
            (
                cute.runtime.make_fake_compact_tensor(cutlass.Int8, (cute.sym_int(divisibility=1),), stride_order=(0,), assumed_align=16),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
            )
            if CFG.O_BLOCK_SCALE
            else ()
        ),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_fwd",
    )


# The pointer entry shares the tensor host's descriptor and launch implementation.
# Quantization remains device-side; this replaces the adapter's amax.div_(scale_o).
from cudnn.sdpa.fwd.kernels._common_blackwell import sdpa_operand_tensors

LSE_KINDS = ("dense", "token", "head", "padded")


@cute.kernel
def _unscale_amax_kernel(amax: cute.Pointer, scale: cute.Pointer):
    amax.store(amax.load() / scale.load())


_unscale_amax_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host_prepared(
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
    descale_q_ptr: cute.Pointer,
    descale_k_ptr: cute.Pointer,
    descale_v_ptr: cute.Pointer,
    scale_o_ptr: cute.Pointer,
    amax_o_ptr: cute.Pointer,
    has_amax: cutlass.Constexpr[bool],
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    lse_kind: cutlass.Constexpr[str],
    paged_hnd: cutlass.Constexpr[bool],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """Bind dense or THD pointer views and launch the existing D128 FP8 host.

    Extents and Int64 element strides are runtime slots; the shared binder
    validates their layout and capacity. Scalar scales remain device pointers.
    The legacy kernel reduces amax after scale_o, so a requested amax is
    normalized on the same stream without constructing a framework tensor.
    """
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

    def scalar(ptr):
        return cute.make_tensor(ptr, cute.make_layout((1,), stride=(1,)))

    _host(
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
        cutlass.Float32(1.0),
        n_thd_units,
        scalar(descale_q_ptr),
        scalar(descale_k_ptr),
        scalar(descale_v_ptr),
        scalar(scale_o_ptr),
        scalar(amax_o_ptr),
        seq_q_lens_addr,
        thd_q_lens_tensor,
        thd_kv_lens_tensor,
        thd_lens_form,
        stream=stream,
        prepared=True,
    )
    if cutlass.const_expr(has_amax):
        _unscale_amax_kernel(amax_o_ptr, scale_o_ptr).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


@lru_cache(maxsize=None)
def compile_prepared(  # noqa: A001
    d_qk: int = CFG.TILE_K,
    d_v: int = CFG.TILE_O,
    has_lse: bool = True,
    lse_kind: str = "dense",
    paged_hnd: bool = False,
    has_amax: bool = True,
) -> Callable:
    """Compile an unsplit D128 FP8-to-half pointer entry at plan build.

    The cache key includes the head envelope and optional output layout;
    batch, sequence extents, pointers and Int64 strides bind on each call.
    """
    if PAGED_KV or SPLIT_KV != 1 or CFG.O_BLOCK_SCALE or CFG.BPE_O != 2:
        raise NotImplementedError("prepared FP8 d128 serves unsplit, non-paged half outputs")
    if d_v * CFG.BPE % 16:
        raise ValueError("FP8 V head strides must be 16-byte multiples")
    _cache_key = _template_key(globals(), locals(), "compile_prepared")
    if not (0 < d_qk <= CFG.TILE_K and 0 < d_v <= CFG.TILE_O):
        raise ValueError(f"d128 envelope: need 0 < d_qk <= {CFG.TILE_K} and 0 < d_v <= {CFG.TILE_O}; got ({d_qk}, {d_v})")
    if (d_qk * CFG.BPE) % 16 != 0 or (d_v * CFG.BPE_O) % 16 != 0:
        raise ValueError(f"d128 envelope: d_qk*BPE and d_v*BPE must be 16-byte multiples (TMA global-stride rule); got ({d_qk}, {d_v}) at BPE={CFG.BPE}")
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
        _host_prepared,
        P(STORAGE_DTYPE),
        P(STORAGE_DTYPE),
        P(STORAGE_DTYPE),
        P(OUT_STORAGE_DTYPE),
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
        P(cutlass.Float32, 4),
        P(cutlass.Float32, 4),
        P(cutlass.Float32, 4),
        P(cutlass.Float32, 4),
        P(cutlass.Float32, 4),
        has_amax,
        d_qk,
        d_v,
        lse_kind,
        paged_hnd,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_fwd_prepared",
    )
