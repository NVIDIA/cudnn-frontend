# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
DSL prefill SDPA kernel — classic pipeline, f16/bf16, d=128, SM107 (Rubin).

Ported from the pre-upstream kernel ``prefill_sdpa_f16.py``
by the port script. The kernel BODY is the pre-upstream one,
unchanged — only the DSL surface moved (the pre-upstream ops -> public cutlass/nvvm/cute, and
the pre-upstream tile library -> cudnn.frost.tile_dsl). Two things are NOT mechanical and were done
by hand:

  1. **Config.** The pre-upstream kernels picked a flavor with an env var;
     FROST forbids env-var configuration, so this reads TemplateParams via the
     FROST_TEMPLATE_PARAMS global (engine contract §5/§6).
  2. **TMA descriptors.** ``create_tensor_map_tiled_from_tensor`` became
     ``..._from_view`` — a signature change, not a rename.

Geometry is FROST's ``make_cfg_d128`` as-is for the 128-wide tile: the
pre-upstream helper would give STAGES_KV = 2*CTA_MMA (4 at cga2) where FROST
gives 2; the deeper ring is a PERF lever and this port was for FUNCTIONAL
coverage first, so it is deliberately not taken on d128.  The 32- and 64-wide
tiles below run STAGES_KV = 4 at cga1 (``config_sm107._stages_kv_d128``, the
SM100 d64 precedent -- inherited, not A/B'd).

**Head-dim tile flavors (``TemplateParams.d_flavor`` in {32, 64, 128}).**  One
body, three ``Cfg`` geometries: ``make_cfg_d32`` / ``make_cfg_d64`` /
``make_cfg_d128`` set ``TILE_K = TILE_O`` and everything coupled to it -- the
Q/K/V/O TMA boxes and swizzle atoms (64-byte rows at TILE_K = 32), the BMM1
k-step count (``TILE_K / 16``: 2 / 4 / 8), BMM2's N, the O slab and the TMEM O
slot width (``config_sm107.d128_family_tmem_layout``: S 0 / 128, P in the S
tails, O at 256 + qs * TILE_O, STATS at the top).  Any head dim D <= TILE_K
with D % 8 == 0 rides the width: the TMA descriptors carry the TRUE extents
while the box stays the tile, so the load zero-fills the missing columns
(exact zero terms in QK^T and in the padded V columns of P.V), the
``expect_tx`` byte count is the FULL box (``qBufferElems * BPE * CTA_MMA`` --
an overshooting box delivers every byte, zero-filled), and the O store clips
at d_v.  The softmax is TILE_N-chunked (``CFG.BMM2_CHUNK_SIZE``), unchanged
across the widths: at TILE_K = 32 the MMA work per score element is a quarter
of d128's while the exp / FMA work is the same, so the small tiles run at the
softmax issue rate.  The d128 rendering is byte-identical to the one-geometry
file this used to be (the committed cubin md5 records pin it).

Original pre-upstream header follows.

Port (from the pre-upstream DSL) of the pre-upstream prefill SDPA kernel
(llama flavor: FP16, d_qk = d_v = 128, TILES_Q = 2, two softmax
warpgroups, four correction warps, persistent try_cancel scheduler).

Same kernel shape as the C++ source: pipeline, TMEM/SMEM layout,
barrier inventory + init counts, scheduler, warp dispatch, register
split match.  Canonical reference port; reaches ~0.99x C++ TFLOPS.

THD / varlen is **NOT** ported.  The pre-upstream body carried a packed-varlen
entry point (``CFG.THD_VARLEN=1``: packed ``[1,T,H,D]`` + ``cu_seqlens`` coord
offset, per-batch O TMA-descriptor array, packed ``[1,QH,T]`` LSE), but its
setup-kernel call site still speaks the pre-upstream 7-arg contract against a
14-arg helper and the metadata layout differs (3B+2 vs 4B+4).  ``compile()``
raises on ``CFG.THD_VARLEN`` rather than half-serving it, ``config_sm107``
rejects THD for f16/bf16, and no engine row advertises it.  Only the dense
``[B,S,H,D]`` path is served.
"""

from cudnn.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
import os
import sys
from functools import lru_cache
from typing import Callable, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", ".."))
_PYTHON_DIR = os.path.join(_REPO_DIR, "python")
_CONFIGS_DIR = os.path.normpath(os.path.join(_HERE, "..", "configs"))
# Shared THD / varlen device helpers (build_o_descs_kernel + TENSOR_MAP_QWORDS).
_COMMON_SDPA_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "common", "sdpa"))
for _p in (_PYTHON_DIR, _CONFIGS_DIR, _COMMON_SDPA_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.primitives import vote_sync, VoteSync
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass._mlir.dialects import arith

import cutlass
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver  # noqa: F401  (cute.compile pulls cuda)

from dataclasses import dataclass

# Config comes from the FROST template loader, NOT an env var: the pre-upstream
# kernels picked a flavor with an env var + a sdpa_config_<flavor> module, which the
# FROST engine contract forbids ("no environment variables for configuration" —
# parameters travel as typed dataclasses). The loader injects
# FROST_TEMPLATE_PARAMS before this body runs; the default keeps a direct
# `import` usable as a standalone driver.
from cudnn.sdpa.fwd.config_sm107 import TemplateParams, d128_family_tmem_layout, make_cfg_d32, make_cfg_d64, make_cfg_d128
from cudnn.sdpa.fwd.config_sm107 import KV_MASK_BOOL_WORD, KV_MASK_ADDITIVE_16, KV_MASK_ADDITIVE_F32
from cudnn.sdpa.fwd.config_sm107 import BIAS_MODE_RESIDENT, BIAS_MODE_STREAMED, BIAS_RING_STAGES, bias_smem_tiles

PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
# One body, three head-dim tile geometries (TILE_K = TILE_O = d_flavor); the record field selects the factory, the
# sm100 sibling's `_MAKE_CFG` convention.  128 is the default so a direct import keeps the classic d128 geometry.
_MAKE_CFG = {32: make_cfg_d32, 64: make_cfg_d64, 128: make_cfg_d128}
if PARAMS.d_flavor not in _MAKE_CFG:
    raise ValueError(f"prefill_d128_f16_sm107: d_flavor must be one of {sorted(_MAKE_CFG)}; got {PARAMS.d_flavor}")
CFG, _TMA = _MAKE_CFG[PARAMS.d_flavor](PARAMS)
if PARAMS.paged_kv:
    raise ValueError(
        "prefill_d128_f16_sm107: paged_kv is not wired on this kernel (the PAGED_KV specialization lives in sm100/prefill_d128_f16, sm100/prefill_d256_f16 and sm100/prefill_d128_fp8)"
    )

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
# Retry form of the per-KV-iteration RING waits (k/v/q _full/_empty, bmm1_done / bmm2_ready / bmm2_done, stat_* / xfer_*,
# s_acc / o_empty, empty_mainloop): every such site is spelled ``.wait(..., spin=SPIN_RING_WAITS)``; the waits a warp
# parks in for a whole tile (scheduler payload, tmem_dealloc, the TMA-STG's O-ready wait) and the end-of-kernel drains
# keep the default sleeping form.  ``spin=True`` is the hint-less uniform spin (tile_dsl.barrier.wait); the decision is
# a MEASURED per-kernel fact (Rubin node locked at 2376 MHz, A/B/A x3, control pair, TFLOPS ratios vs the sleeping
# form), so it lives here once, like DESC_VERSION, and never as a literal at a call site
# (test_sm107_ring_waits_take_the_module_spin_constant).  Flipping this constant IS the whole experiment.
# Measured: +4.9 % @ S=2K dense H128 with the ring waits alone (+5.3 % with the predicated credit arrive), +3.6 % @32K,
# THD packed +6.0 %, decode-shaped +4.9..+5.3 %, split-KV-shaped +5.3 %.
# Re-measured on the small-tile bias / kv_mask renderings (bf16 D 32 / 64, a 212-SM cc 10.7 node at the 2376 MHz lock, a
# base / variant / base cross-process A/B/A per cell set, base drift <= 0.2 %): the retry form LOSES on every plain and
# one-batch-entry rendering (d32 plain -3.0 / -3.6 % at S 384 / 512, the one-batch-entry streamed ring -4.6 / -8.3 %, d64 plain
# -6.3 / -6.8 % at S 768 / 1024, the cga2 one-batch-entry ring -3.5 / -3.8 %), is NEUTRAL on the TILE_N 64 pair renderings
# (-0.5..+0.1 %) and WINS on the pair streamed ring at TILE_N 128 only (d32 +1.9..+4.2 % on cells the heuristics do not pick,
# d64 +0.6 / +1.0 % at S 768 / 1024) -- so the constant stays True; a per-rendering spelling is a recorded lever, below the bar
# for a flip from one process pair.  The scheduler credit's lowering (tile_dsl.scheduler.read_tile_id_arrive) keeps the
# library default here, the predicated arrive: the branch form lost on 8 of 9 cga2 cells of the same sweep (-0.6..-3.6 %;
# +0.3 % on one) and is a no-op on every cga1 rendering by construction (the cluster arm folds out at CGA_SIZE 1; the cubins
# are byte-identical), so this kernel carries no constant for it.
SPIN_RING_WAITS: bool = True
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
    tmem_load_max_reduction_tile,
    row_reduction_pair,
    row_max_reduction,
    vec_scale_pair,
)
from cudnn.frost.tile_dsl.regtile import RegTile, vec_concat
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts_step
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_tile, tma_store_commit, tma_store_wait, tma_tensormap_acquire
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, GmemTileTma, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.frost.tile_dsl.mask import (
    apply_mask_chunk,
    compute_kv_loop_bounds as _compute_kv_loop_bounds,
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
elif CFG.DTYPE_QKV == 4:
    # TF32: 4-byte FP32 storage, kind::tf32 MMA (HW rounds operands to a
    # 10-bit mantissa).  Mirrors the C++ prefill_sdpa_f16.cu IS_TF32 path.
    # Three TF32-specific deltas vs BF16/FP16 (all folded on IS_TF32 below):
    #   1. V (the transposed BMM2 operand) SmemDesc + TMA use
    #      SWIZZLE_128B_ATOM_32B (enc=1, leading=TILE_N*128, stride=512).
    #      Standard Swz128B with kind::tf32 + b_trans silently returns all-0.
    #   2. P TMEM packing is 1:1 (one FP32 cell per prob) vs 2:1 for
    #      fp16/bf16, so P aliases the FULL S_acc slot, not just its tail.
    #   3. BPE=4 doubles every SMEM/TMA byte count (sized off CFG.BPE).
    STORAGE_DTYPE = cutlass.Float32
    P_STORAGE_DTYPE = cutlass.Float32
    MMA_KIND = nvvm.Tcgen05MMAKind.TF32
    IS_TF32 = True
else:
    raise ValueError(f"prefill_sdpa_f16: DTYPE_QKV={CFG.DTYPE_QKV} not supported " f"(expected 2=BF16, 3=FP16, or 4=TF32)")


from cudnn.sdpa.fwd.kernels._common_blackwell import (
    sdpa_operand_tensors,
    sdpa_score_tensor,
    sdpa_bias_tensor,
    sdpa_kv_mask_tensor,
    row_max_for_exp2,
    Bars,
    KvLoopBounds,
    make_classic_bars,
    compute_kv_loop_bounds,
    lpt_tile_coords,
    make_sdpa_helpers,
    make_split_helpers,
    store_fp32_partial_tile as _store_fp32_partial_tile,
)
from cudnn.frost.tile_dsl.bias import (
    bias_block_geometry,
    issue_bias_block_load,
    issue_bias_ring_tile,
    load_bias_row_chunk,
    bias_fma_chunk,
    additive_vec_granules,
    additive_vec_fma_chunk,
)
from cudnn.frost.tile_dsl.mask import kv_mask_keep_words

# === Additive bias (resident row-block) + per-key boolean mask: CFG.BIAS_MODE / CFG.KV_MASK_FORM ===
#
# Both arms fold on their Cfg field (TemplateParams.bias_mode / kv_mask_form): 0 / 0 traces the pre-bias instruction stream
# byte-for-byte (the committed cubin records pin it).  Either arm routes EVERY kv tile through the masked softmax body with
# the TRUE -inf mask value and the canonical ``row_max_for_exp2`` substitution (the fused ``tcgen05.ld.red...max`` of the
# unmasked path cannot see a bias, and a key the data mask removes must zero its probability rather than leave the finite
# sentinel's bogus P == 1), so a row whose every key is masked ends the item with ``total_sum == 0`` exactly: the correction
# epilogue folds that into its empty-row SELECT (O = 0, Stats = -inf), never a multiply by zero.
#
# BARRIER TABLE -- the two ITEM-level rows CFG.BIAS_MODE adds (make_classic_bars(bias_resident=True); the sixteen rows of the
# d128 body are unchanged, counted in LANES, so neither arm moves an existing init):
#   mb_bias_full    TMA_LOAD / LOCAL / 1 stage.  Producer: the TMA-LDG warp, ONCE per non-empty work item, after the item's
#                   Q / K / V prologue loads: ``arrive(n_bytes=n_tiles x biasTileBytes, pred=elect_sync())`` -> 1 lane x 1 warp
#                   = 1 == init ONE_LANE; bytes = the FULL boxes of every tile of the item's KV range (``kv_right - kv_left``
#                   tiles x BIAS_BOXES_PER_TILE boxes of BIAS_BLOCK_ROWS x 64 columns; an overshooting box past S_q / S_kv is
#                   zero-filled AND counted).  Consumer: both softmax warpgroups, once per item, before their first bias read
#                   (waits do not count).  Phase: start(0).  EMPTY item (zero kv tiles): no arrive, no wait -- both sides key
#                   on the SAME ``kv_right > kv_left`` their own decode produced.
#   mb_bias_empty   THREAD / LOCAL / 1 stage.  Producer: every softmax warp, ONCE per non-empty item, after the LAST kv step's
#                   bias read: ``if nvvm.elect_sync(): arrive()`` -> 1 lane x 8 warps = 8 == init BIAS_EMPTY_ARRIVERS
#                   (SOFTMAX_WARPGROUPS x SOFTMAX_WG_WARPS, pinned by config_sm107._validate_bias).  Consumer: the TMA-LDG
#                   warp before the NEXT item's fill; consumer phase PRE-ARMED at 1 (the first fill waits on nothing).  EMPTY
#                   item: no arrive, no wait.  No cross-CTA arrive on either row (cga2 CTAs each fill and read their own
#                   block), so P15 adds no drain.
# SMEM BUFFER ROW -- sBias (CFG.BIAS_MODE only), declared LAST (after sQ, sO, sK, sV): STORAGE_DTYPE, BIAS_RESIDENT_TILES x
#   BIAS_BLOCK_ROWS x TILE_N elements (3 x 256 x 128 bf16 = 192 KiB at the 32-wide tile, cga1).  Written by TMA under the
#   128-B swizzle in 64-column boxes (box-major inside a tile, 128 B per row per box); read by the softmax lanes, one q row
#   each, one 128-B row slice per 64-column chunk through ``load_swizzled(Swizzle(3, 4, 3), count=64)``.  Per-lane stride
#   128 B = the bank cycle, so the swizzle does BOTH jobs: it matches the descriptor that wrote it (D) and spreads the 32
#   lanes' 16-B granules over 8 bank groups (L: 4 wavefronts per warp-wide load = ideal; unswizzled = 32-way).  Placed after
#   the descriptor-fed operands so every tcgen05 descriptor root stays under the 256 KiB version-0 window (predicate in
#   config_sm107._validate_bias); the block itself may straddle the line -- no descriptor reads it.
# THE STREAMED RING (CFG.BIAS_MODE == 2, TemplateParams.bias_mode = 2; the fallback above the resident budget): the SAME tile
#   geometry, but sBias is a STAGES_BIAS-deep RING (2 / 3 stages of ONE BIAS_BLOCK_ROWS x TILE_N tile = 32 KiB on the pair
#   item, 64 KiB on the one-batch-entry item) the TMA-LDG warp fills ONE TILE PER KV TILE and the softmax warps release per
#   kv tile -- any S_kv.  The two item-level rows above fold OUT; two RING rows replace them (make_classic_bars(bias_ring_stages)):
#   mb_bias_ring_full[s]   TMA_LOAD / LOCAL / STAGES_BIAS.  Producer: the TMA-LDG warp, once per kv tile of every non-empty
#                          step, in the kv-tile issue order K(a) -> K(b) -> BIAS -> V(a) -> V(b) (the one-batch-entry loop:
#                          Q0 -> K -> Q1 -> BIAS -> V, then K -> BIAS -> V): wait mb_bias_ring_empty[idx] (its state, pre-armed),
#                          ``arrive(n_bytes=biasTileBytes, pred=elect_sync())`` -> 1 lane == init ONE_LANE; bytes = the FULL
#                          boxes of ONE tile (tile_dsl.bias.issue_bias_ring_tile, cta_group=1: every byte on THIS CTA's mbar).
#                          Consumer: both softmax warpgroups, per kv tile, BEFORE the tile's first bias read (ahead of the
#                          BIAS_PRELOAD_CHUNKS pre-load); consumer state PipelineState.start(0), advanced once per kv tile.
#   mb_bias_ring_empty[s]  THREAD / LOCAL / STAGES_BIAS.  Producer: every softmax warp, once per kv tile, after the stage's
#                          LAST bias word is consumed (after the per-chunk FFMA loop): ``if nvvm.elect_sync(): arrive()`` -> 1
#                          lane x 8 warps = 8 == init BIAS_EMPTY_ARRIVERS.  Consumer: the TMA-LDG warp before refilling the
#                          stage; its state PipelineState.start(1) (pre-armed: the first STAGES_BIAS fills wait on nothing).
#   P14: an EMPTY step / item fires nothing, waits nothing and advances nothing on EITHER side (both roles iterate the SAME
#   bounds their own decode produced); a dead sub-tile 1 still runs its kv loop, so both warpgroups wait and release every
#   stage; the ring index runs continuously across steps and items.  No cross-CTA arrive (at cga2 each CTA fills and reads its
#   own ring for its own rows) -> P15 adds no drain.  No accumulator parity is indexed from this ring: STAGES_BIAS is a pure
#   latency knob (2 or 3; config_sm107.BIAS_RING_STAGES).  SMEM: the ring is declared where the resident block is (LAST), the
#   same 128-B swizzle for the same two reasons; the operand region is untouched (DESC_VERSION 0).
# kv_mask (CFG.KV_MASK_FORM) stages nothing in SMEM: per kv tile each softmax warp loads 4 mask bytes per lane and assembles
#   4 keep words by warp ballot (tile_dsl.mask.kv_mask_keep_words), ANDed into the band's keep words inside apply_mask_chunk.
#   A kv_mask-only rendering therefore carries NO resident budget -- any S_kv, so the ring-wrap axis (n_kv = STAGES_KV, +1, two
#   wraps) is reachable on this body and validated there; only CFG.BIAS_MODE bounds S_kv.
#   KV_MASK_FORM 2 / 3 = the ADDITIVE per-key vector (the io dtype's 16-bit pairs / fp32): per 64-column chunk every lane of a
#   softmax warp loads the SAME 128 / 256 bytes of the vector as 16-byte granules (tile_dsl.bias.additive_vec_granules; the LSU
#   broadcasts one sector per instruction, the tail granule is clamped to the last granule-aligned column below the live length)
#   and folds them into the raw scores pre-scale exactly like the bias (additive_vec_fma_chunk: one fma.rn.f32x2 per pair, after
#   the bias add, before the band mask).  Still nothing in SMEM, NO new barrier row and no change to the sixteen d128 rows or the
#   two bias rows: the table above is the table of every kv_mask form.  A finite -1e9 key stays finite (the models' fold: a row
#   whose every key is -1e9 is the uniform average, total_sum > 0), a -inf vector value reproduces the bit word bitwise.
#
# BIAS_PRELOAD_CHUNKS: how many of the step's 64-column bias chunks the softmax lanes load BEFORE the bmm1_done wait (the
# block is resident, so the fragment has no dependency on BMM1): 0 = after the wait, 1 = chunk 0 under BMM1 and chunk 1 under
# chunk 0's math (the register-budget-safe default: 32 packed words live across the wait), 2 = the whole fragment (64 words
# across the wait; read STL / LDL before adopting it).  A per-kernel constant like SPIN_RING_WAITS: flipping it is the whole
# experiment, never a literal at a call site.
BIAS_PRELOAD_CHUNKS: int = 1
_BIAS_ARM = bool(CFG.BIAS_MODE)
# The bias FORM: the resident block (the two item-level rows R1 / R2) or the streamed ring (the two ring rows S1 / S2); every
# form-specific seam folds on one of these two, every shared seam on _BIAS_ARM.
_BIAS_RESIDENT = CFG.BIAS_MODE == BIAS_MODE_RESIDENT
_BIAS_STREAMED = CFG.BIAS_MODE == BIAS_MODE_STREAMED
_KV_MASK_ARM = bool(CFG.KV_MASK_FORM)
_KV_MASK_BOOL = CFG.KV_MASK_FORM == KV_MASK_BOOL_WORD
_KV_MASK_ADD = CFG.KV_MASK_FORM in (KV_MASK_ADDITIVE_16, KV_MASK_ADDITIVE_F32)
# The kv_mask operand's element type at the tvm-ffi boundary: bytes for the bit word, the io dtype / fp32 for the additive vector.
KV_MASK_DTYPE = cutlass.Int8 if _KV_MASK_BOOL else (cutlass.Float32 if CFG.KV_MASK_FORM == KV_MASK_ADDITIVE_F32 else STORAGE_DTYPE)
_INF_MASK_ARM = _BIAS_ARM or _KV_MASK_ARM
# === The (n, n+1) PAIR work item (CFG.N_CHUNK > 0; TemplateParams.n_chunk) ===
#
# A work item is (a TILE_M-row q tile, a head, a CHUNK of up to N_CHUNK consecutive entries nb of the flattened batch axis);
# the two softmax sub-tiles are rows nb / nb + 1 of the SAME q tile, so Q[nb] / Q[nb+1], K[nb] / K[nb+1], V[nb] / V[nb+1] stream
# through the existing slots and rings while the bias row-block (BIAS_BLOCK_ROWS = TILE_M rows x the item's KV range) is
# filled ONCE per item and read by every step of the chunk -- the per-item fixed cost and the bias feed divided by N_CHUNK.
# The n-STEP is the unit of every per-tile barrier protocol below (today's tile body, verbatim, per step); the scheduler
# handshake is per ITEM.  Per step j: nb0 = n0 + 2 j, nb1 = min(nb0 + 1, n_end - 1), live1 = nb0 + 1 < n_end -- a DEAD
# sub-tile 1 (an odd chunk tail) recomputes row nb0 (every load in-bounds, every barrier fired) and skips its O / LSE
# stores only (the THD dead-unit precedent of the TMA-STG warp).  The step's KV range is the UNION of its two rows' top-left
# bands = compute_kv_loop_bounds at the PAIR-MAX of their lengths (monotone in the KV length; the shorter row's extra tiles
# are PADDED-masked at its own length inside the softmax, so they add 0 to its sum), collapsed when both rows are past their
# Q length (the q-trim).  The ITEM's bias block covers compute_kv_loop_bounds at the GLOBAL lengths (S_q, S_kv): a
# predicate both the TMA-LDG and the softmax warps compute from the payload alone -- the fill, the wait and the release
# happen iff item_bounds is non-empty, on both sides, whatever the per-row lengths (a chunk whose steps are all empty
# still fills / waits / releases: bytes, never a correctness term).
#
# BARRIER TABLE deltas (every init count unchanged -- counts are lanes of a role, none depends on the item form):
#   mb_bias_full / mb_bias_empty  once per LIVE ITEM (not per step): the fill after step 0's prologue loads, the wait before
#                                 step 0's kv loop, the release after the LAST step's kv loop; 1 == 1 / 8 == 8 as before.
#   mb_k_* / mb_v_* [STAGES_KV]   the ring carries the PAIR's tiles: per kv tile K[nb0] -> stage s, K[nb1] -> stage s+1 (V
#                                 likewise), kv_state advances TWICE per kv tile on both sides; the MMA commits k_empty[s]
#                                 after BMM1(Q0, K[s]) and k_empty[s+1] after BMM1(Q1, K[s+1]) -- one elected commit per
#                                 stage per phase == KV_EMPTY_ARRIVERS (1), where the one-batch-entry item fired ONE commit
#                                 per stage after both BMM1s.  The ring depth in kv TILES halves.
#   every other ring              per STEP exactly what it was per tile (empty step = today's empty-mainloop protocol).
#   mb_read_tile_id / mb_scheduler per ITEM: all 15 warps credit once per item, as before.
# SMEM: sBias = BIAS_RESIDENT_TILES x TILE_M x TILE_N (32 KiB per tile at 128 x 128 bf16: 6 tiles = 192 KiB at the 32-wide
# tile, 3 = 96 KiB at the 64-wide one), the same 128-B swizzle for the same two reasons (the SMEM BUFFER ROW above);
# both sub-tiles read row tid_in_wg of the block.  The grid is (ceil(S_q / TILE_M), QH, n_chunks): the CLC payload's batch
# field carries the CHUNK index (n_chunks <= 65535, the adapter declines above), `n_batch` is the scheduler's batch extent
# (= n_chunks) and `n_rows_total` the flattened batch (the chunk clamp); cga1 only (config_sm107._validate_bias).
_PAIR_ARM = CFG.N_CHUNK > 0
if BIAS_PRELOAD_CHUNKS not in (0, 1, 2):
    raise ValueError(f"prefill_d128_f16_sm107: BIAS_PRELOAD_CHUNKS must be 0, 1 or 2 (got {BIAS_PRELOAD_CHUNKS})")
if CFG.N_BMM2_CHUNKS not in (1, 2):
    # The softmax body unrolls its P chunks by hand for one or two 64-column chunks (TILE_N 64 / 128); TILE_N = 256 is a
    # validator decline at two sub-tiles and would need a third / fourth chunk here.
    raise ValueError(f"prefill_d128_f16_sm107: the softmax body renders 1 or 2 chunks of 64 (TILE_N 64 / 128); got N_BMM2_CHUNKS={CFG.N_BMM2_CHUNKS}")
if _INF_MASK_ARM and IS_TF32:
    raise ValueError("prefill_d128_f16_sm107: the bias / kv_mask arms are wired for the 16-bit dtypes only")
if _BIAS_STREAMED and CFG.STAGES_BIAS not in BIAS_RING_STAGES:
    # config_sm107._validate_bias pins the same coupling at config-build time; this is the kernel's own backstop (the TMA-LDG
    # side pre-arms STAGES_BIAS fills and the softmax side advances the same count -- both read CFG.STAGES_BIAS).
    raise ValueError(f"prefill_d128_f16_sm107: the streamed bias ring is {sorted(BIAS_RING_STAGES)} stages deep (CFG.STAGES_BIAS={CFG.STAGES_BIAS})")
if _BIAS_ARM and (_BIAS_RESIDENT == _BIAS_STREAMED):
    raise ValueError(f"prefill_d128_f16_sm107: CFG.BIAS_MODE={CFG.BIAS_MODE} is neither the resident block (1) nor the streamed ring (2)")

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
# qtrim variant: collapses the KV loop for CGA tiles entirely past the
# per-batch actual Q length (SEQ_Q_LENS_PRESENT; folds to plain bounds otherwise).
_bounds_for_tile = _sdpa_h.bounds_for_tile_qtrim
# Q-side per-sequence length.  Under THD `seqlen_q` is the PACKED TOTAL (the
# adapter's problem_size seq slots are 0 and _host reads the dynamic extent), so
# every mask/bounds consumer needs the SEQUENCE's own S_q_b instead.  Sequence 0
# starts at cu_q[0] == 0, which makes the packed total accidentally right for it
# and wrong for every later sequence -- the exact "seq 0 exact, seq 1 garbage"
# signature this kernel had.  Folds to the scalar on the dense path.
_resolve_seqlen_q = _sdpa_h.resolve_seqlen_q
_resolve_seqlen_kv = _sdpa_h.resolve_seqlen_kv

# THD / varlen — flat-grid decode + tma-offset closures (CFG-bound) from the
# factory; O-descriptor builder + TENSOR_MAP_QWORDS from the shared
# the shared pre-upstream THD helper.  Gated by CFG.THD_VARLEN (folds out otherwise).
# Supported at cga1 and cga2 (TILES_Q=2 → two Q slabs / O stores per tile).
# seq_kv_lens overloaded as the THD metadata buffer (int32 len 3B+2):
#   [0..B-1]=seq_kv_lens  [B..2B]=cu_q(B+1)  [2B+1..3B+1]=cu_k(B+1)
# THD: FROST's builder is NOT the pre-upstream one, which was 7-arg, O-descriptors-
# only build_o_descs_kernel; FROST does per-execute meta + O (+ K/V base descs)
# in one elected-thread kernel with a 14-arg contract, and ships a dedicated
# f16/bf16 entry point. The import is corrected here; the CALL SITE still speaks
# the pre-upstream contract and is taken from the shipped fp8 SM107 sibling when THD
# is wired (dense path first — see the module header).
from cudnn.sdpa.fwd.kernels.thd_helpers import build_thd_meta_o_descs_kernel as _build_o_descs_kernel, TENSOR_MAP_QWORDS, THD_SETUP_THREADS

_TENSOR_MAP_QWORDS = TENSOR_MAP_QWORDS
_dispatch_decode_initial = _sdpa_h.dispatch_decode_initial
_dispatch_decode_payload = _sdpa_h.dispatch_decode_payload
_thd_tma_offsets = _sdpa_h.thd_tma_offsets

# Split IDs follow the scheduler's batch coordinate; every warp role uses
# the same narrowed KV range. At one split these helpers fold away.
_split_h = make_split_helpers(
    CFG,
    bounds_for_tile=_bounds_for_tile,
    dispatch_decode_initial=_dispatch_decode_initial,
    dispatch_decode_payload=_dispatch_decode_payload,
)
SPLIT_KV = _split_h.SPLIT_KV
_FP32_PARTIALS = SPLIT_KV > 1
if _PAIR_ARM and not (_BIAS_ARM and CFG.CTA_MMA == 1 and CFG.TILES_Q == 2 and not CFG.BOTTOM_RIGHT and not CFG.THD_VARLEN and SPLIT_KV == 1):
    # config_sm107._validate_bias pins the same couplings at config-build time; this is the kernel's own backstop.
    raise ValueError(
        "prefill_d128_f16_sm107: the (n, n+1) pair item (CFG.N_CHUNK > 0) is the resident-bias arm at cga1, two sub-tiles, a top-left band, dense, unsplit"
    )
MAY_BE_EMPTY = _split_h.MAY_BE_EMPTY
_decode_initial_split = _split_h.decode_initial_split
_decode_payload_split = _split_h.decode_payload_split
_bounds_for_tile_split = _split_h.bounds_for_tile_split
_nomask_range_split = _split_h.nomask_range_split
_partial_batch = _split_h.partial_batch


# TMEM column map -- DERIVED from CFG by config_sm107.d128_family_tmem_layout (one formula for the body and the
# validator): S[qs] = qs * TILE_N; P (the packed BMM2 A operand) aliases the TAIL of its own S slot -- BMM1 finishes
# (and the softmax loads S into registers) before P is written; fp16/bf16 pack 2 probs per FP32 cell, so P is TILE_N/2
# wide and sits at +64 at TILE_N = 128 (a 4-byte P would alias the full slot; mirrors C++ S_bmm2_0 = S_acc_cols -
# S_bmm2_cols); O[qs] = TILES_Q * TILE_N + qs * TILE_O (128 / 64 / 32 columns per sub-tile by flavor); STATS at the top of
# the 576-column allocation, outside every S / O range so BMM1's next-iteration write cannot clobber them.  At d128 this is
# the map the kernel always had (0 / 128 | 64 / 192 | 256 / 384 | 544).
_TMEM = d128_family_tmem_layout(CFG)
if IS_TF32 and _TMEM.P_OFF != _TMEM.S_OFF:  # module-level Python, not a traced branch
    raise ValueError("prefill_d128_f16_sm107: a 4-byte P must alias the full S slot (TF32 is not a Rubin flavor)")


@dataclass(frozen=True)
class KernelTmemLayout:
    """Column offsets for the classic 2-sub-tile SDPA pipeline (values from ``d128_family_tmem_layout(CFG)``)."""

    # Rubin SM10.7 cap; requires is_exclusive=True on tcgen05_alloc.  Alloc and dealloc pass this SAME constant.
    TOTAL_COLS: int = _TMEM.TOTAL_COLS

    S0_OFF: int = _TMEM.S_OFF[0]
    S1_OFF: int = _TMEM.S_OFF[1]

    O0_OFF: int = _TMEM.O_OFF[0]
    O1_OFF: int = _TMEM.O_OFF[1]

    P0_OFF: int = _TMEM.P_OFF[0]
    P1_OFF: int = _TMEM.P_OFF[1]

    # qs stride = 8 cols (D128FamilyTmemLayout.STATS_COLS_PER_SUBTILE).
    STATS_OFF: int = _TMEM.STATS_OFF


LAYOUT = KernelTmemLayout()


# === The pair item's row / bounds arithmetic (CFG.N_CHUNK > 0); shared by every warp role so no two roles can drift ===


@cute.jit
def _pair_item_rows(chunk_idx, n_rows_total):
    """The chunk's first row, its end (exclusive) and its step count: rows [n0, n_end) of the flattened batch, two per step."""
    n0 = chunk_idx * cutlass.Int32(CFG.N_CHUNK)
    n_end = cute.math.min(n0 + cutlass.Int32(CFG.N_CHUNK), n_rows_total)
    n_steps = (n_end - n0 + cutlass.Int32(1)) // cutlass.Int32(2)
    return n0, n_end, n_steps


@cute.jit
def _pair_step_rows(n0, n_end, j):
    """Step j's two rows and whether the second is live: a dead sub-tile 1 (the odd tail) CLAMPS to the last live row, so
    every load stays in-bounds and every barrier fires; only its stores are gated by ``live1``."""
    nb0 = n0 + j * cutlass.Int32(2)
    nb1 = cute.math.min(nb0 + cutlass.Int32(1), n_end - cutlass.Int32(1))
    live1 = (nb0 + cutlass.Int32(1)) < n_end
    return nb0, nb1, live1


@cute.jit
def _pair_item_bounds(q_tile_idx, seqlen_q, seqlen_kv):
    """The ITEM's KV range at the GLOBAL lengths: the bias block's tiles [left, right) and the item-live predicate both the
    TMA-LDG and the softmax warps evaluate (the fill / wait / release happen iff right > left on BOTH sides)."""
    return _compute_kv_loop_bounds(
        q_tile_idx * cutlass.Int32(CFG.TILE_M),
        seqlen_q,
        seqlen_kv,
        CFG.WINDOW_LEFT,
        CFG.MASK_FLAGS,
        CFG.TILE_N,
        CFG.TILE_M,
        bottom_right=False,
        window_right=int(CFG.WINDOW_RIGHT),
    )


@cute.jit
def _pair_step_bounds(q_tile_idx, seqlen_q, seqlen_kv, seq_kv_lens_tensor, seq_q_lens_addr, nb0, nb1):
    """The STEP's KV range = the union of its two rows' top-left bands: compute_kv_loop_bounds at the pair-max of the two
    per-row KV lengths (monotone in the length; the SWA lower bound depends on the q row only), collapsed to EMPTY when the
    q tile's base row is at / past BOTH rows' Q lengths (the dense padded-Q trim, pair-wise).  The shorter row's extra
    tiles are PADDED-masked at its OWN length inside the softmax body."""
    kv0 = _resolve_seqlen_kv(seq_kv_lens_tensor, nb0, seqlen_kv)
    kv1 = _resolve_seqlen_kv(seq_kv_lens_tensor, nb1, seqlen_kv)
    kv_pair = cute.math.max(kv0, kv1)
    q_row_coord = q_tile_idx * cutlass.Int32(CFG.TILE_M)
    b = _compute_kv_loop_bounds(
        q_row_coord, seqlen_q, kv_pair, CFG.WINDOW_LEFT, CFG.MASK_FLAGS, CFG.TILE_N, CFG.TILE_M, bottom_right=False, window_right=int(CFG.WINDOW_RIGHT)
    )
    if cutlass.const_expr(int(CFG.SEQ_Q_LENS_PRESENT) == 1):
        arr = cute.make_tensor(cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4), cute.make_layout(1 << 24))
        q_pair = cute.math.max(cutlass.Int32(arr[cutlass.Int32(nb0)]), cutlass.Int32(arr[cutlass.Int32(nb1)]))
        tile_dead = q_row_coord >= q_pair
        dead_lo = cutlass.Int32(arith.select(tile_dead.ir_value(), b.left.ir_value(), b.unmasked_lo.ir_value()))
        dead_hi = cutlass.Int32(arith.select(tile_dead.ir_value(), b.left.ir_value(), b.unmasked_hi.ir_value()))
        dead_right = cutlass.Int32(arith.select(tile_dead.ir_value(), b.left.ir_value(), b.right.ir_value()))
        return KvLoopBounds(left=b.left, unmasked_lo=dead_lo, unmasked_hi=dead_hi, right=dead_right)
    return b


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
    seq_q_lens_addr: cutlass.Int64 = 0,
    o_partial_f32: Optional[cute.Tensor] = None,
    # Score side outputs (the graph's score_max / score_sum_exp ports): fp32 (B, QH, SQ) in the
    # caller's strides -- per Q row the softmax max in NATURAL units (scale * max_j s_j, the sink
    # folded in when present) and the fp32 row-sum of exp(s - max) the LSE is built from, so
    # LSE == score_max + ln(score_sum_exp) to fp32 rounding.  Stored by the correction epilogue
    # next to the LSE under the same row guard; None folds the stores out (the DSL specializes on
    # None), so a plan without them traces byte-identically to before the slots existed.
    score_max_tensor: Optional[cute.Tensor] = None,
    score_sum_exp_tensor: Optional[cute.Tensor] = None,
    # Bias / kv_mask arms (APPENDED, every one None-specialised: a plan without the arm passes None for each and traces
    # byte-identically to before they existed -- the committed cubin records pin it).  The resident bias block's LOAD
    # descriptor; the bias tensor [B_b, H_b, S_q, S_kv] (unread in the body: the descriptor carries the address; present so
    # the bias-on artifact binds it at the tvm-ffi boundary) with its two broadcast extents; the per-key boolean mask as a
    # (B, S_kv) byte view; 1 / softmax_scale for the pre-scale bias add.
    tma_bias_desc: Optional[cutlass.GridConstant[tmap.TensorMap]] = None,
    bias_tensor: Optional[cute.Tensor] = None,
    bias_b_extent: Optional[cutlass.Int32] = None,
    bias_h_extent: Optional[cutlass.Int32] = None,
    kv_mask_tensor: Optional[cute.Tensor] = None,
    inv_scale_softmax: Optional[cutlass.Float32] = None,
    # The (n, n+1) pair item (APPENDED, None-specialised): the flattened batch count (the chunk clamp); `n_batch` then carries
    # the scheduler's batch extent = the chunk count.  None on every other rendering.
    n_rows_total: Optional[cutlass.Int32] = None,
) -> None:

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()
    # Score side outputs requested (trace-time): the softmax tracks the true row max, the epilogue stores the pair.
    _score_stats = (score_max_tensor is not None) or (score_sum_exp_tensor is not None)

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
        # V is split along d_v under cga2 → tma_loads_per_tile = TMA_VO_ITERS / CTA_MMA.
        tma_loads_per_tile=TMA_VO_ITERS // CFG.CTA_MMA,
        tma_granu_elems=TMA_VO_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_N * TMA_VO_GRANU_ELEMS,
        desc_version=DESC_VERSION,
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
        desc_version=DESC_VERSION,
    )
    # The bias block -- LAST in SMEM (the lane-read buffer sits after every descriptor-fed operand; see the SMEM BUFFER ROW at
    # the top of the file).  One stage per resident kv tile (BIAS_MODE 1) or per RING stage (BIAS_MODE 2: STAGES_BIAS stages,
    # one kv tile each); each tile is BIAS_BOXES_PER_TILE 64-column boxes, box-major, 128 B per row per box (the geometry of
    # tile_dsl.bias.bias_block_geometry, all from CFG; n_tiles = the form's SMEM extent, config_sm107.bias_smem_tiles).
    sBias_raw = (
        cutlass.Array(STORAGE_DTYPE, _BIAS_GG.n_tiles * _BIAS_GG.tile_elems, alignment=1024, space=cutlass.AddressSpace.smem)
        if cutlass.const_expr(_BIAS_ARM)
        else None
    )
    sBias = (
        SmemTile(
            base=sBias_raw,
            elems_per_stage=_BIAS_GG.tile_elems,
            stages=_BIAS_GG.n_tiles,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=SMEM_LAYOUT_BIAS,
            tma_loads_per_tile=_BIAS_GG.boxes_per_tile,
            tma_granu_elems=_BIAS_GG.granu_elems,
            tma_subtile_stride_elems=_BIAS_GG.subtile_stride_elems,
            desc_version=DESC_VERSION,
        )
        if cutlass.const_expr(_BIAS_ARM)
        else None
    )

    bars = make_classic_bars(CFG, bias_resident=_BIAS_RESIDENT, bias_ring_stages=CFG.STAGES_BIAS if _BIAS_STREAMED else 0)

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
                bars.mb_o_full[qs].init()
                bars.mb_o_empty[qs].init()
                if cutlass.const_expr(IS_QO_ALIAS):
                    bars.mb_q_o_alias[qs].init()
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
            if cutlass.const_expr(_BIAS_RESIDENT):
                # The two item-level bias rows (one stage each; the same single-warp, single-lane init block).
                bars.mb_bias_full.init()
                bars.mb_bias_empty.init()
            if cutlass.const_expr(_BIAS_STREAMED):
                # The streamed ring's two rows, EVERY stage (``.init()`` initialises stage 0 only; an un-looped init would leave
                # stages 1.. uninitialised and the first wait on them faults the launch).
                for bs in cutlass.range_constexpr(CFG.STAGES_BIAS):
                    bars.mb_bias_ring_full[bs].init()
                    bars.mb_bias_ring_empty[bs].init()

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    # P4: cluster fence gates cga2 cross-CTA arrive_on_peer on peer init.
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        cga_arrive()
        cga_wait()

    # @cute.kernel stages if/else — use Python ternaries so the chosen
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
            track_true_max=_score_stats,
            sBias=sBias,
            kv_mask_tensor=kv_mask_tensor,
            inv_scale=inv_scale_softmax,
            n_rows_total=n_rows_total,
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
            track_true_max=_score_stats,
            sBias=sBias,
            kv_mask_tensor=kv_mask_tensor,
            inv_scale=inv_scale_softmax,
            n_rows_total=n_rows_total,
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
            o_partial_f32=o_partial_f32,
            score_max_tensor=score_max_tensor,
            score_sum_exp_tensor=score_sum_exp_tensor,
            n_rows_total=n_rows_total,
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
                n_rows_total=n_rows_total,
            )

    elif warp_idx == CFG.TMALDG_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        if cutlass.const_expr(_BIAS_ARM):
            nvvm.prefetch_tensormap(tma_bias_desc.get_ptr())
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
            seq_q_lens_addr=seq_q_lens_addr,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            qh_per_kh=qh_per_kh,
            is_leader=is_leader,
            cta_in_pair=cta_in_pair,
            tma_mcast_mask=tma_mcast_mask,
            tma_bias_desc=tma_bias_desc,
            sBias=sBias,
            bias_b_extent=bias_b_extent,
            bias_h_extent=bias_h_extent,
            n_rows_total=n_rows_total,
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
            n_rows_total=n_rows_total,
        )

    else:  # warp_idx == CFG.SCHED_WARP_ID
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        # try_cancel.multicast::cluster::all — only (0,0,0) CTA issues; at cga1
        # cta_id_x == 0 always, so flag is 1 unconditionally.
        is_cga_first_cta = cta_id_x == cutlass.Int32(0)
        if cutlass.const_expr(CFG.THD_VARLEN):
            # THD: PERSISTENT grid + a device-bounded claim counter, so no unit
            # past the live total is ever handed out.  The counter and the live
            # total are the last two slots of the 4B+4 metadata buffer that the
            # setup kernel writes -- nothing consumed them before, which is the
            # other half of why this kernel's THD output was wrong: the dense
            # CLC path hands out tiles against a grid that is the persistent
            # cluster count, not the work list, so the tile -> unit mapping is
            # off for every tile.  n_batch is a kernel ARGUMENT: do not
            # re-derive it from the metadata tensor's layout, which is not
            # always a flat 1-D view.  Identical call to the SM100 f16 sibling
            # and to this file's FP8 sibling.
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
def _bias_ring_fill(sBias, tma_bias, bars, bias_ring_state, kv_tile, q_row, bias_head, bias_batch):
    """TMA-LDG warp, once per kv tile of the STREAMED ring (CFG.BIAS_MODE == 2): wait the stage's release
    (``mb_bias_ring_empty[idx]``, pre-armed for the first STAGES_BIAS fills), arm + queue ONE bias tile into the stage
    (``tile_dsl.bias.issue_bias_ring_tile``: one elected expect_tx of the tile's FULL boxes, cta_group=1), advance.  Called from
    EVERY kv-tile path (the prologue and the mainloop of both item forms) exactly once per kv tile -- the lockstep with the
    softmax side's per-tile wait / release that the ring's P14 statement rests on.  Returns the advanced state."""
    bars.mb_bias_ring_empty[bias_ring_state.idx].wait(bias_ring_state.phase, spin=SPIN_RING_WAITS)
    issue_bias_ring_tile(
        sBias[bias_ring_state.idx],
        tma_bias,
        bars.mb_bias_ring_full[bias_ring_state.idx],
        _BIAS_GG.tile_bytes,
        CFG.TILE_N,
        kv_tile,
        q_row,
        bias_head,
        bias_batch,
    )
    return advance(bias_ring_state, CFG.STAGES_BIAS)


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
    seq_q_lens_addr,
    n_q_supers,
    n_qh,
    n_batch,
    qh_per_kh,
    is_leader,
    cta_in_pair,
    tma_mcast_mask,
    # Resident bias block (appended; the call is by keyword).  None = bias off (CFG.BIAS_MODE == 0).
    tma_bias_desc=None,
    sBias=None,
    bias_b_extent=None,
    bias_h_extent=None,
    n_rows_total=None,
):
    """Unified TMA-LDG warp — cga1 / cga2 x MASK_NONE / PADDED / CAUSAL / SWA.

    Spill-free in CFG.OTHER_REGS via q_row_base pre-multiplied after each
    scheduler decode (gives ptxas a live use of nxt_q before the loop
    back-edge, keeping the LDS.128 result in uniform registers).

    CFG.BIAS_MODE: once per non-empty work item, after the item's Q / K / V
    prologue loads, the warp waits ``mb_bias_empty`` (the softmax warps'
    release of the previous item's block; consumer phase pre-armed at 1), arms
    ``mb_bias_full`` with the FULL boxes of the item's ``kv_right - kv_left``
    bias tiles and queues them (``tile_dsl.bias.issue_bias_block_load``).  The
    bias row is the CTA's 256-row block at ``q_row_base``; the head / batch
    coordinates are clamped to the bias tensor's extents (a broadcast dim of 1
    reads row 0).  An EMPTY item issues nothing and waits on nothing, like the
    softmax side.
    """
    q_empty_phase = cutlass.Int32(1)
    kv_state = PipelineState.start(phase=1)
    # mb_bias_empty consumer phase, PRE-ARMED at 1 (P5 b): the first item's block is untouched, so the first wait must
    # return with no producer arrive.  Loop-carried like q_empty_phase; flips once per NON-empty item on both sides.
    bias_empty_phase = cutlass.Int32(1)
    # The streamed ring's producer-side state over mb_bias_ring_empty (P5 b: PRE-ARMED, the first STAGES_BIAS fills wait on
    # nothing); advanced once per kv tile by _bias_ring_fill.  Loop-carried on the pair item (always a traced state there, a
    # pass-through on the resident pair rendering); None off the ring on the one-batch-entry loop, where it is never assigned.
    bias_ring_state = PipelineState.start(phase=1) if (_BIAS_STREAMED or _PAIR_ARM) else None
    tma_bias = GmemTileTma(tma_bias_desc) if cutlass.const_expr(_BIAS_ARM) else None

    # Q-reload gate: under QO_ALIAS the next tile's Q-load must wait for the
    # prior tile's O (sharing the slab) to drain — TMA-STG fires mb_q_o_alias
    # (strictly after MMA consumed Q).  Else gate on mb_q_empty (MMA commit).
    # Both bootstrap consumer-side via q_empty_phase=1.
    mb_q_reload = bars.mb_q_o_alias if cutlass.const_expr(IS_QO_ALIAS) else bars.mb_q_empty

    tma_q = GmemTileTma(tma_q_desc)
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD: K/V ride the setup kernel's RUNTIME descriptors (o_desc_words
        # slots n_batch+1 / n_batch+2), whose seq extent is CLAMPED to the
        # packed KV total cu_k[B].  The plan-time descriptors describe the
        # buffer's CAPACITY instead, so using them here reads the tail past the
        # packed total: the last sequence's tile steps into it, and a poisoned
        # tail wipes the whole tile through BMM2 (0 * NaN == NaN).  Through the
        # clamped descriptors those rows are TMA-OOB and land as EXACT ZEROS.
        # Same closure shape as GmemTileTma, so every load site stays
        # branch-free.  Mirrors sm107/prefill_d128_fp8.py.
        _k_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(1)) * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        _v_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(2)) * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        # Setup publishes immutable packed-total maps for this launch.
        # Each loader warp acquires both maps before consuming any tile.
        tma_tensormap_acquire(_k_rt_ptr)
        tma_tensormap_acquire(_v_rt_ptr)
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
    )
    # GQA: K/V are indexed by kv-head, not Q-head.
    kv_head_idx = cute.arch.make_warp_uniform(head_idx // qh_per_kh)
    q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M))
    q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)

    if cutlass.const_expr(CFG.MASK_FLAGS == 0):
        kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
    else:
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
        bounds_init = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    # the DSL's TMA descriptors is element-typed; contiguous coord is in ELEMENTS not bytes.
    K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_N // CFG.CTA_MMA)
    V_COL_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_O // CFG.CTA_MMA)

    if cutlass.const_expr(_PAIR_ARM):
        # The (n, n+1) pair item (CFG.N_CHUNK > 0): the same persistent loop over scheduler items, each item a chunk of
        # row pairs (`_*_item_pair`); the per-item decode is the shared scheduler payload (q tile, head, chunk).
        while is_valid_tile > cutlass.Int32(0):
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
            kv_state, q_empty_phase, bias_empty_phase, bias_ring_state = _tmaldg_item_pair(
                tma_q,
                tma_k,
                tma_v,
                sQ,
                sK,
                sV,
                bars,
                mb_q_reload,
                tma_mcast_mask,
                q_super_idx,
                head_idx,
                kv_head_idx,
                batch_idx,
                n_rows_total,
                seqlen_q,
                seqlen_kv,
                seq_kv_lens_tensor,
                seq_q_lens_addr,
                kv_state,
                q_empty_phase,
                bias_empty_phase,
                bias_ring_state,
                tma_bias,
                sBias,
                bias_b_extent,
                bias_h_extent,
            )
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
            )
            is_valid_tile = nxt_v & cutlass.Int32(1)
            sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
            kv_head_idx = cute.arch.make_warp_uniform(head_idx // qh_per_kh)
    else:
        while is_valid_tile > cutlass.Int32(0):
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

            if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
                pass
            else:
                # Prologue interleave Q[0] -> K[first] -> Q[1] -> V[first] -> mainloop —
                # K load starts before Q[1] is issued and V before kv mainloop.
                kv_row_base = kv_left * CFG.TILE_N

                mb_q_reload[0].wait(q_empty_phase, spin=SPIN_RING_WAITS)
                if cutlass.const_expr(CFG.CTA_MMA == 2):
                    bars.mb_q_full[0].arrive(n_bytes=qTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                else:
                    bars.mb_q_full[0].arrive(n_bytes=qTmaTransactionBytes, pred=nvvm.elect_sync())
                tma_load_tile(
                    sQ[0],
                    tma_q(cutlass.Int32(0), head_idx, q_row_base + cutlass.Int32(0 * CFG.TILE_M) + q_seq_off, tma_batch),
                    bars.mb_q_full[0].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                )

                bars.mb_k_empty[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
                if cutlass.const_expr(CFG.CTA_MMA == 2):
                    bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                else:
                    bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=nvvm.elect_sync())
                tma_load_tile(
                    sK[kv_state.idx],
                    # THD: the per-sequence KV offset and the collapsed batch coord are
                    # NOT optional here.  The packed batch extent is 1, so a load at
                    # batch_idx > 0 is OOB and zero-fills -- this prologue K tile was
                    # the one load site in this file still passing batch_idx instead of
                    # tma_batch, and omitting kv_seq_off.  Both fold to the dense values
                    # (0, batch_idx) when THD is off, which is why it looked harmless.
                    tma_k(cutlass.Int32(0), kv_head_idx, kv_row_base + K_ROW_OFFSET_PEER + kv_seq_off, tma_batch),
                    bars.mb_k_full[kv_state.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                    acquire=not CFG.THD_VARLEN,
                )

                mb_q_reload[1].wait(q_empty_phase, spin=SPIN_RING_WAITS)
                if cutlass.const_expr(CFG.CTA_MMA == 2):
                    bars.mb_q_full[1].arrive(n_bytes=qTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                else:
                    bars.mb_q_full[1].arrive(n_bytes=qTmaTransactionBytes, pred=nvvm.elect_sync())
                tma_load_tile(
                    sQ[1],
                    tma_q(cutlass.Int32(0), head_idx, q_row_base + cutlass.Int32(1 * CFG.TILE_M) + q_seq_off, tma_batch),
                    bars.mb_q_full[1].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                )
                q_empty_phase = q_empty_phase ^ 1

                if cutlass.const_expr(_BIAS_STREAMED):
                    # The streamed ring: this step's FIRST kv tile's bias tile, after K (the softmax needs bias and S together) and
                    # before V (needed one stage later, by BMM2).
                    bias_ring_state = _bias_ring_fill(
                        sBias,
                        tma_bias,
                        bars,
                        bias_ring_state,
                        kv_left,
                        q_row_base,
                        cute.math.min(head_idx, bias_h_extent - cutlass.Int32(1)),
                        cute.math.min(batch_idx, bias_b_extent - cutlass.Int32(1)),
                    )

                bars.mb_v_empty[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
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
                    acquire=not CFG.THD_VARLEN,
                )
                kv_state = advance(kv_state, CFG.STAGES_KV)

                if cutlass.const_expr(_BIAS_RESIDENT):
                    # The resident bias block for this item: wait the readers' release of the previous block, then one
                    # expect_tx for the whole block and its tiles' boxes.  Queued AFTER the prologue's Q / K / V so BMM1 of the
                    # first kv tile is not delayed behind 64-192 KiB of bias; the softmax waits the block once per item.
                    bars.mb_bias_empty.wait(bias_empty_phase)
                    bias_empty_phase = bias_empty_phase ^ cutlass.Int32(1)
                    bias_head = cute.math.min(head_idx, bias_h_extent - cutlass.Int32(1))
                    bias_batch = cute.math.min(batch_idx, bias_b_extent - cutlass.Int32(1))
                    issue_bias_block_load(
                        sBias, tma_bias, bars.mb_bias_full, kv_right - kv_left, _BIAS_GG.tile_bytes, CFG.TILE_N, kv_left, q_row_base, bias_head, bias_batch
                    )

                for kv_loop in cutlass.range(kv_left + cutlass.Int32(1), kv_right, 1, unroll=1):
                    kv_row_base = kv_loop * CFG.TILE_N

                    bars.mb_k_empty[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
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
                        acquire=not CFG.THD_VARLEN,
                    )

                    if cutlass.const_expr(_BIAS_STREAMED):
                        # The streamed ring: this kv tile's bias tile (one per kv tile, after K and before V).
                        bias_ring_state = _bias_ring_fill(
                            sBias,
                            tma_bias,
                            bars,
                            bias_ring_state,
                            kv_loop,
                            q_row_base,
                            cute.math.min(head_idx, bias_h_extent - cutlass.Int32(1)),
                            cute.math.min(batch_idx, bias_b_extent - cutlass.Int32(1)),
                        )

                    bars.mb_v_empty[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
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
                        acquire=not CFG.THD_VARLEN,
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
            )
            kv_head_idx = cute.arch.make_warp_uniform(head_idx // qh_per_kh)
            # q_row_base compute right after decode drives ptxas R2UR.
            q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M))
            q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)
            is_valid_tile = nxt_v & cutlass.Int32(1)
            sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
            if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV > 1):
                kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
            elif cutlass.const_expr(CFG.MASK_FLAGS != 0):
                eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
                eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
                bounds_next = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx)
                kv_left = bounds_next.left
                kv_right = bounds_next.right

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


@cute.jit
def _tmaldg_item_pair(
    tma_q,
    tma_k,
    tma_v,
    sQ,
    sK,
    sV,
    bars,
    mb_q_reload,
    tma_mcast_mask,
    q_tile_idx,
    head_idx,
    kv_head_idx,
    chunk_idx,
    n_rows_total,
    seqlen_q,
    seqlen_kv,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    kv_state,
    q_empty_phase,
    bias_empty_phase,
    bias_ring_state,
    tma_bias,
    sBias,
    bias_b_extent,
    bias_h_extent,
):
    """One (n, n+1) pair work item of the TMA-LDG warp (CFG.N_CHUNK > 0; cga1).

    Per step j (rows nb0 / nb1 of the chunk): the prologue Q[nb0] -> K[nb0] (stage a) -> Q[nb1] -> K[nb1] (stage b) -> V[nb0]
    (stage a) -> V[nb1] (stage b), then per further kv tile K / K / V / V through the ring two stages at a time (`kv_state`
    advances twice per kv tile; the MMA consumes stage a for sub-tile 0 and stage b for sub-tile 1).  The bias block is
    filled ONCE per LIVE item, after step 0's prologue loads and before its ring refills (the same placement as the
    one-batch-entry item): wait `mb_bias_empty` (the readers' release of the previous block), one `expect_tx` for the FULL
    boxes of the item's tiles [item.left, item.right) at the GLOBAL lengths, the boxes.  An EMPTY step issues nothing; an
    empty ITEM fills nothing and waits nothing (the softmax side keys on the same global-length predicate).  The STREAMED
    ring (CFG.BIAS_MODE == 2) replaces the once-per-item block by ONE bias tile per kv tile, issued after the step's K pair and
    before its V pair (`_bias_ring_fill`; `bias_ring_state` advances once per kv tile).  Returns the loop-carried (kv_state,
    q_empty_phase, bias_empty_phase, bias_ring_state).
    """
    n0, n_end, n_steps = _pair_item_rows(chunk_idx, n_rows_total)
    q_row = q_tile_idx * cutlass.Int32(CFG.TILE_M)
    item_b = _pair_item_bounds(q_tile_idx, seqlen_q, seqlen_kv)
    item_live = item_b.right > item_b.left
    bias_head = cute.math.min(head_idx, bias_h_extent - cutlass.Int32(1))
    for j in cutlass.range(cutlass.Int32(0), n_steps, 1, unroll=1):
        nb0, nb1, _live1 = _pair_step_rows(n0, n_end, j)
        sb = _pair_step_bounds(q_tile_idx, seqlen_q, seqlen_kv, seq_kv_lens_tensor, seq_q_lens_addr, nb0, nb1)
        kv_left = sb.left
        kv_right = sb.right
        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            pass
        else:
            kv_row_base = kv_left * CFG.TILE_N
            state_a = kv_state
            state_b = advance(kv_state, CFG.STAGES_KV)

            mb_q_reload[0].wait(q_empty_phase, spin=SPIN_RING_WAITS)
            bars.mb_q_full[0].arrive(n_bytes=qTmaTransactionBytes, pred=nvvm.elect_sync())
            tma_load_tile(sQ[0], tma_q(cutlass.Int32(0), head_idx, q_row, nb0), bars.mb_q_full[0].smem_ptr, cta_group=CFG.CTA_MMA, mcast_mask=tma_mcast_mask)

            bars.mb_k_empty[state_a.idx].wait(state_a.phase, spin=SPIN_RING_WAITS)
            bars.mb_k_full[state_a.idx].arrive(n_bytes=kTmaTransactionBytes, pred=nvvm.elect_sync())
            tma_load_tile(
                sK[state_a.idx],
                tma_k(cutlass.Int32(0), kv_head_idx, kv_row_base, nb0),
                bars.mb_k_full[state_a.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )

            mb_q_reload[1].wait(q_empty_phase, spin=SPIN_RING_WAITS)
            bars.mb_q_full[1].arrive(n_bytes=qTmaTransactionBytes, pred=nvvm.elect_sync())
            tma_load_tile(sQ[1], tma_q(cutlass.Int32(0), head_idx, q_row, nb1), bars.mb_q_full[1].smem_ptr, cta_group=CFG.CTA_MMA, mcast_mask=tma_mcast_mask)
            q_empty_phase = q_empty_phase ^ 1

            bars.mb_k_empty[state_b.idx].wait(state_b.phase, spin=SPIN_RING_WAITS)
            bars.mb_k_full[state_b.idx].arrive(n_bytes=kTmaTransactionBytes, pred=nvvm.elect_sync())
            tma_load_tile(
                sK[state_b.idx],
                tma_k(cutlass.Int32(0), kv_head_idx, kv_row_base, nb1),
                bars.mb_k_full[state_b.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )

            if cutlass.const_expr(_BIAS_STREAMED):
                # The streamed ring: the step's first kv tile's bias tile (both sub-tiles read it), after the K pair, before V.
                bias_ring_state = _bias_ring_fill(
                    sBias, tma_bias, bars, bias_ring_state, kv_left, q_row, bias_head, cute.math.min(nb0, bias_b_extent - cutlass.Int32(1))
                )

            bars.mb_v_empty[state_a.idx].wait(state_a.phase, spin=SPIN_RING_WAITS)
            bars.mb_v_full[state_a.idx].arrive(n_bytes=vTmaTransactionBytes, pred=nvvm.elect_sync())
            tma_load_tile(
                sV[state_a.idx],
                tma_v(cutlass.Int32(0), kv_head_idx, kv_row_base, nb0),
                bars.mb_v_full[state_a.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            bars.mb_v_empty[state_b.idx].wait(state_b.phase, spin=SPIN_RING_WAITS)
            bars.mb_v_full[state_b.idx].arrive(n_bytes=vTmaTransactionBytes, pred=nvvm.elect_sync())
            tma_load_tile(
                sV[state_b.idx],
                tma_v(cutlass.Int32(0), kv_head_idx, kv_row_base, nb1),
                bars.mb_v_full[state_b.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            kv_state = advance(state_b, CFG.STAGES_KV)

        if j == cutlass.Int32(0):
            if cutlass.const_expr(_BIAS_RESIDENT):
                if item_live:
                    # The resident block of this ITEM (once; both sub-tiles read row tid_in_wg of it): after step 0's prologue so
                    # BMM1 of the first kv tile is not delayed behind 96-192 KiB of bias, before the ring refills.
                    bars.mb_bias_empty.wait(bias_empty_phase)
                    bias_empty_phase = bias_empty_phase ^ cutlass.Int32(1)
                    bias_batch = cute.math.min(nb0, bias_b_extent - cutlass.Int32(1))
                    issue_bias_block_load(
                        sBias,
                        tma_bias,
                        bars.mb_bias_full,
                        item_b.right - item_b.left,
                        _BIAS_GG.tile_bytes,
                        CFG.TILE_N,
                        item_b.left,
                        q_row,
                        bias_head,
                        bias_batch,
                    )

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            pass
        else:
            for kv_loop in cutlass.range(kv_left + cutlass.Int32(1), kv_right, 1, unroll=1):
                kv_row_base = kv_loop * CFG.TILE_N
                state_a = kv_state
                state_b = advance(kv_state, CFG.STAGES_KV)

                bars.mb_k_empty[state_a.idx].wait(state_a.phase, spin=SPIN_RING_WAITS)
                bars.mb_k_full[state_a.idx].arrive(n_bytes=kTmaTransactionBytes, pred=nvvm.elect_sync())
                tma_load_tile(
                    sK[state_a.idx],
                    tma_k(cutlass.Int32(0), kv_head_idx, kv_row_base, nb0),
                    bars.mb_k_full[state_a.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                )
                bars.mb_k_empty[state_b.idx].wait(state_b.phase, spin=SPIN_RING_WAITS)
                bars.mb_k_full[state_b.idx].arrive(n_bytes=kTmaTransactionBytes, pred=nvvm.elect_sync())
                tma_load_tile(
                    sK[state_b.idx],
                    tma_k(cutlass.Int32(0), kv_head_idx, kv_row_base, nb1),
                    bars.mb_k_full[state_b.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                )

                if cutlass.const_expr(_BIAS_STREAMED):
                    # The streamed ring: this kv tile's bias tile (one per kv tile, after the K pair, before the V pair).
                    bias_ring_state = _bias_ring_fill(
                        sBias, tma_bias, bars, bias_ring_state, kv_loop, q_row, bias_head, cute.math.min(nb0, bias_b_extent - cutlass.Int32(1))
                    )

                bars.mb_v_empty[state_a.idx].wait(state_a.phase, spin=SPIN_RING_WAITS)
                bars.mb_v_full[state_a.idx].arrive(n_bytes=vTmaTransactionBytes, pred=nvvm.elect_sync())
                tma_load_tile(
                    sV[state_a.idx],
                    tma_v(cutlass.Int32(0), kv_head_idx, kv_row_base, nb0),
                    bars.mb_v_full[state_a.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                )
                bars.mb_v_empty[state_b.idx].wait(state_b.phase, spin=SPIN_RING_WAITS)
                bars.mb_v_full[state_b.idx].arrive(n_bytes=vTmaTransactionBytes, pred=nvvm.elect_sync())
                tma_load_tile(
                    sV[state_b.idx],
                    tma_v(cutlass.Int32(0), kv_head_idx, kv_row_base, nb1),
                    bars.mb_v_full[state_b.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                )
                kv_state = advance(state_b, CFG.STAGES_KV)
    return kv_state, q_empty_phase, bias_empty_phase, bias_ring_state


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
    n_rows_total=None,
):
    """Persistent O-store warp.  First tile from blockIdx; subsequent tiles
    via scheduler warp's clusterlaunchcontrol.try_cancel.async.

    CFG.N_CHUNK > 0 (the pair item): per step the two O tiles are rows nb0 / nb1 of the q tile; a DEAD sub-tile 1 (the odd
    chunk tail) skips its GMEM store but still waits `mb_o_full` and arrives `mb_o_empty` (the THD dead-unit precedent below).
    """
    o_full_phase = cutlass.Int32(0)  # consumer waits — first-arrive flips 0 → 1

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
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    # Per-sequence O descriptors stay immutable during this launch.
    last_o_batch = cutlass.Int32(-1)
    if cutlass.const_expr(_PAIR_ARM):
        # The (n, n+1) pair item (CFG.N_CHUNK > 0): the same persistent loop over scheduler items, each item a chunk of
        # row pairs (`_*_item_pair`); the per-item decode is the shared scheduler payload (q tile, head, chunk).
        while is_valid_tile > cutlass.Int32(0):
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
            o_full_phase = _tmastg_item_pair(tma_o, sO, bars, q_super_idx, head_idx, batch_idx, n_rows_total, o_full_phase)
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
            )
            is_valid_tile = nxt_v & cutlass.Int32(1)
            sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
    else:
        while is_valid_tile > cutlass.Int32(0):
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

            q_row_base = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M)

            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                bars.mb_o_full[qs].wait(o_full_phase)

                if cutlass.const_expr(not _FP32_PARTIALS):
                    # O TMA params follow O's swizzle (NOT V's) — under gptoss cga2 V drops to Swz64B while O stays Swz128B.
                    if cutlass.const_expr(CFG.THD_VARLEN):
                        # THD: store each Q slab through this batch's pre-built descriptor
                        # (base at the sequence's packed row, seq extent = S_q_b → a box
                        # past S_q_b is OOB-clipped).  q_row coord is sequence-local; the
                        # batch coord collapses to 0.  Both slabs share one descriptor.
                        #
                        # DEAD unit (batch_idx == n_batch, the over-launched persistent
                        # grid's sentinel): no O rows exist for it and the setup kernel
                        # never builds descriptor slot n_batch, so storing through it
                        # writes a tile to an unbuilt descriptor.  The barrier protocol
                        # below still runs -- only the store is skipped.  This guard is
                        # unnecessary under an exactly-sized grid, which is why the
                        # pre-upstream body did not have it and why it only started
                        # mattering when the scheduler became persistent.
                        if batch_idx < n_batch:
                            o_desc_ptr = (o_desc_words.iterator.raw_ptr() + batch_idx * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
                            o_slice = tma_slice_runtime_desc(
                                o_desc_ptr, cutlass.Int32(0), head_idx, q_row_base + cutlass.Int32(qs * CFG.TILE_M), cutlass.Int32(0)
                            )
                            if batch_idx != last_o_batch:
                                tma_tensormap_acquire(o_slice.desc_ptr)
                                last_o_batch = batch_idx
                            tma_store_tile(sO[qs], o_slice, acquire=False)
                    else:
                        tma_store_tile(
                            sO[qs],
                            tma_o(cutlass.Int32(0), head_idx, q_row_base + cutlass.Int32(qs * CFG.TILE_M), batch_idx),
                        )

                    tma_store_commit()
                    tma_store_wait(0)

                bars.mb_o_empty[qs].arrive()
                # QO_ALIAS: O[qs] has drained to GMEM → the shared Q∪O slab is free
                # for TMA-LDG to clobber with the next tile's Q[qs].
                if cutlass.const_expr(IS_QO_ALIAS):
                    bars.mb_q_o_alias[qs].arrive()

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
            )
            is_valid_tile = nxt_v & cutlass.Int32(1)
            sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


@cute.jit
def _tmastg_item_pair(tma_o, sO, bars, q_tile_idx, head_idx, chunk_idx, n_rows_total, o_full_phase):
    """One (n, n+1) pair work item of the TMA-STG warp: per step, O[nb0] from sO[0] and O[nb1] from sO[1] at the q tile's rows;
    the dead sub-tile 1 of an odd chunk tail stores nothing (its rows belong to row nb0, already stored) but keeps the
    `mb_o_full` wait and the `mb_o_empty` arrive, so the correction's ring never waits on a phase that does not come."""
    n0, n_end, n_steps = _pair_item_rows(chunk_idx, n_rows_total)
    q_row = q_tile_idx * cutlass.Int32(CFG.TILE_M)
    for j in cutlass.range(cutlass.Int32(0), n_steps, 1, unroll=1):
        nb0, nb1, live1 = _pair_step_rows(n0, n_end, j)
        for qs in cutlass.range_constexpr(CFG.TILES_Q):
            bars.mb_o_full[qs].wait(o_full_phase)
            if cutlass.const_expr(qs == 0):
                tma_store_tile(sO[0], tma_o(cutlass.Int32(0), head_idx, q_row, nb0))
                tma_store_commit()
                tma_store_wait(0)
            else:
                if live1:
                    tma_store_tile(sO[1], tma_o(cutlass.Int32(0), head_idx, q_row, nb1))
                    tma_store_commit()
                    tma_store_wait(0)
            bars.mb_o_empty[qs].arrive()
        o_full_phase = o_full_phase ^ 1
    return o_full_phase


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

# Resident bias block geometry (CFG.BIAS_MODE): the TMA box, the box count per tile, the lane read offsets and the 128-B
# swizzle, all from CFG (tile_dsl.bias.bias_block_geometry); None folds every bias seam out.
_BIAS_GG = bias_block_geometry(CFG, swz_enum=_SWZ_ENUM, n_tiles=bias_smem_tiles(CFG)) if _BIAS_ARM else None
SMEM_LAYOUT_BIAS = _BIAS_GG.layout_enum if _BIAS_ARM else SMEM_LAYOUT_QKO

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
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=True)
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
    n_rows_total=None,
):
    """Unified MMA warp — cga1 / cga2-leader x MASK_NONE / PADDED / CAUSAL / SWA,
    spill-free in CFG.OTHER_REGS (40).  Non-leader under cga2 uses _mma_warp_quiet.

    CFG.N_CHUNK > 0 (the pair item): the per-item decode is taken on every mask arm (the pair needs the q tile and the chunk),
    and each step consumes the ring two stages at a time -- stage a = K / V of row nb0 (sub-tile 0), stage b = row nb1
    (sub-tile 1) -- committing `k_empty` / `v_empty` per stage right after the MMA that read it (`_mma_item_pair`).
    """
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=True)
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

    if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1 and not _PAIR_ARM):
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
        )
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
        bounds_init = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    q_full_phase = cutlass.Int32(0)
    kv_state = PipelineState.start(phase=0)
    bmm2_ready_phase = cutlass.Int32(0)
    # Initialised unconditionally so type is stable across const_expr branches.
    empty_mainloop_phase = cutlass.Int32(0)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    if cutlass.const_expr(_PAIR_ARM):
        # The (n, n+1) pair item (CFG.N_CHUNK > 0): the same persistent loop over scheduler items, each item a chunk of
        # row pairs (`_*_item_pair`); the per-item decode is the shared scheduler payload (q tile, head, chunk).
        while is_valid_tile > cutlass.Int32(0):
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
            kv_state, q_full_phase, bmm2_ready_phase, empty_mainloop_phase = _mma_item_pair(
                sK,
                sV,
                tmem_raw,
                bars,
                bmm1_desc,
                bmm2_desc,
                desc_Q0,
                desc_Q1,
                mcast_mask,
                q_super_idx,
                batch_idx,
                n_rows_total,
                seqlen_q,
                seqlen_kv,
                seq_kv_lens_tensor,
                seq_q_lens_addr,
                kv_state,
                q_full_phase,
                bmm2_ready_phase,
                empty_mainloop_phase,
            )
            nvvm.bar_warp_sync(cute.arch.FULL_MASK)
            wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
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
            )
            is_valid_tile = nxt_v & cutlass.Int32(1)
            sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
    else:
        while is_valid_tile > cutlass.Int32(0):
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

            if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
                # Empty mainloop: fire both bmm2_done so softmax/correction phase trackers stay in lockstep.
                bars.mb_empty_mainloop.wait(empty_mainloop_phase, spin=SPIN_RING_WAITS)
                empty_mainloop_phase = empty_mainloop_phase ^ cutlass.Int32(1)
                elect_p = nvvm.elect_sync()
                bars.mb_bmm2_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_bmm2_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            else:
                # Prologue: BMM1[sub0], BMM1[sub1] for kv=kv_left
                bars.mb_q_full[0].wait(q_full_phase, spin=SPIN_RING_WAITS)
                bars.mb_k_full[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
                desc_K = sK[kv_state.idx].desc()
                mma_ss(bmm1_desc, desc_Q0, desc_K, (tmem_raw.subview(LAYOUT.S0_OFF)))
                elect_p = nvvm.elect_sync()
                bars.mb_bmm1_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

                bars.mb_q_full[1].wait(q_full_phase, spin=SPIN_RING_WAITS)
                mma_ss(bmm1_desc, desc_Q1, desc_K, (tmem_raw.subview(LAYOUT.S1_OFF)))
                elect_p = nvvm.elect_sync()
                bars.mb_bmm1_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_k_empty[kv_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

                q_full_phase = q_full_phase ^ 1

                # Mainloop kv = kv_left+1 .. kv_right-1 (empty when n_kv == 1)
                for kv_loop in cutlass.range(kv_left + cutlass.Int32(1), kv_right, 1, unroll=1):
                    old_state = kv_state
                    kv_state = advance(kv_state, CFG.STAGES_KV)

                    bars.mb_v_full[old_state.idx].wait(old_state.phase, spin=SPIN_RING_WAITS)
                    desc_V = sV[old_state.idx].desc()
                    is_not_first_bmm2 = cutlass.Boolean(kv_loop != (kv_left + cutlass.Int32(1)))

                    # BMM2 sub-tile 0
                    bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 0].wait(bmm2_ready_phase, spin=SPIN_RING_WAITS)
                    accum_b2 = is_not_first_bmm2
                    for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                        mma_ts_step(bmm2_desc, (tmem_raw.subview(LAYOUT.P0_OFF)), desc_V, (tmem_raw.subview(LAYOUT.O0_OFF)), local_k, accum_b2)
                        accum_b2 = cutlass.Boolean(True)
                    # Chunk 1 folds out at TILE_N=64 (N_BMM2_CHUNKS=1).
                    if cutlass.const_expr(CFG.N_BMM2_CHUNKS == 2):
                        bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 1].wait(bmm2_ready_phase, spin=SPIN_RING_WAITS)
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
                    bars.mb_k_full[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
                    desc_K = sK[kv_state.idx].desc()
                    mma_ss(bmm1_desc, desc_Q0, desc_K, (tmem_raw.subview(LAYOUT.S0_OFF)))
                    elect_p = nvvm.elect_sync()
                    bars.mb_bmm1_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

                    # BMM2 sub-tile 1
                    bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 0].wait(bmm2_ready_phase, spin=SPIN_RING_WAITS)
                    accum_b2 = is_not_first_bmm2
                    for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                        mma_ts_step(bmm2_desc, (tmem_raw.subview(LAYOUT.P1_OFF)), desc_V, (tmem_raw.subview(LAYOUT.O1_OFF)), local_k, accum_b2)
                        accum_b2 = cutlass.Boolean(True)
                    if cutlass.const_expr(CFG.N_BMM2_CHUNKS == 2):
                        bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 1].wait(bmm2_ready_phase, spin=SPIN_RING_WAITS)
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

                bars.mb_v_full[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
                desc_V = sV[kv_state.idx].desc()
                is_not_first_bmm2_epi = cutlass.Boolean((kv_right - kv_left) != cutlass.Int32(1))

                bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 0].wait(bmm2_ready_phase, spin=SPIN_RING_WAITS)
                accum_b2 = is_not_first_bmm2_epi
                for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                    mma_ts_step(bmm2_desc, (tmem_raw.subview(LAYOUT.P0_OFF)), desc_V, (tmem_raw.subview(LAYOUT.O0_OFF)), local_k, accum_b2)
                    accum_b2 = cutlass.Boolean(True)
                if cutlass.const_expr(CFG.N_BMM2_CHUNKS == 2):
                    bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 1].wait(bmm2_ready_phase, spin=SPIN_RING_WAITS)
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

                bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 0].wait(bmm2_ready_phase, spin=SPIN_RING_WAITS)
                accum_b2 = is_not_first_bmm2_epi
                for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                    mma_ts_step(bmm2_desc, (tmem_raw.subview(LAYOUT.P1_OFF)), desc_V, (tmem_raw.subview(LAYOUT.O1_OFF)), local_k, accum_b2)
                    accum_b2 = cutlass.Boolean(True)
                if cutlass.const_expr(CFG.N_BMM2_CHUNKS == 2):
                    bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 1].wait(bmm2_ready_phase, spin=SPIN_RING_WAITS)
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

            nvvm.bar_warp_sync(cute.arch.FULL_MASK)

            wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
            if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1 and not _PAIR_ARM):
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
                )
                is_valid_tile = nxt_v & cutlass.Int32(1)
                eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
                eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
                bounds_next = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx)
                kv_left = bounds_next.left
                kv_right = bounds_next.right
            sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _mma_item_pair(
    sK,
    sV,
    tmem_raw,
    bars,
    bmm1_desc,
    bmm2_desc,
    desc_Q0,
    desc_Q1,
    mcast_mask,
    q_tile_idx,
    chunk_idx,
    n_rows_total,
    seqlen_q,
    seqlen_kv,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    kv_state,
    q_full_phase,
    bmm2_ready_phase,
    empty_mainloop_phase,
):
    """One (n, n+1) pair work item of the MMA warp (CFG.N_CHUNK > 0; cga1): per step the one-batch-entry pipeline with TWO ring
    stages per kv tile -- stage a holds K[nb0] / V[nb0] (sub-tile 0 = row nb0), stage b K[nb1] / V[nb1] (sub-tile 1 = row nb1).
    `k_empty[s]` is committed right after the ONE BMM1 that read stage s and `v_empty[s]` after the one BMM2 (one elected
    commit per stage per phase == the init count KV_EMPTY_ARRIVERS = 1; the one-batch-entry item fires one commit per stage
    after BOTH sub-tiles' MMAs, which is the same count).  An EMPTY step runs the empty-mainloop protocol (both `bmm2_done`
    fired after `mb_empty_mainloop`, no stage consumed).  Returns (kv_state, q_full_phase, bmm2_ready_phase, empty_mainloop_phase)."""
    n0, n_end, n_steps = _pair_item_rows(chunk_idx, n_rows_total)
    for j in cutlass.range(cutlass.Int32(0), n_steps, 1, unroll=1):
        nb0, nb1, _live1 = _pair_step_rows(n0, n_end, j)
        sb = _pair_step_bounds(q_tile_idx, seqlen_q, seqlen_kv, seq_kv_lens_tensor, seq_q_lens_addr, nb0, nb1)
        kv_left = sb.left
        kv_right = sb.right
        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            bars.mb_empty_mainloop.wait(empty_mainloop_phase, spin=SPIN_RING_WAITS)
            empty_mainloop_phase = empty_mainloop_phase ^ cutlass.Int32(1)
            elect_p = nvvm.elect_sync()
            bars.mb_bmm2_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_bmm2_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        else:
            state_a = kv_state
            state_b = advance(kv_state, CFG.STAGES_KV)

            # Prologue: BMM1[sub0] on K[nb0] (stage a), BMM1[sub1] on K[nb1] (stage b) for kv = kv_left
            bars.mb_q_full[0].wait(q_full_phase, spin=SPIN_RING_WAITS)
            bars.mb_k_full[state_a.idx].wait(state_a.phase, spin=SPIN_RING_WAITS)
            desc_K = sK[state_a.idx].desc()
            mma_ss(bmm1_desc, desc_Q0, desc_K, (tmem_raw.subview(LAYOUT.S0_OFF)))
            elect_p = nvvm.elect_sync()
            bars.mb_bmm1_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_k_empty[state_a.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

            bars.mb_q_full[1].wait(q_full_phase, spin=SPIN_RING_WAITS)
            bars.mb_k_full[state_b.idx].wait(state_b.phase, spin=SPIN_RING_WAITS)
            desc_K = sK[state_b.idx].desc()
            mma_ss(bmm1_desc, desc_Q1, desc_K, (tmem_raw.subview(LAYOUT.S1_OFF)))
            elect_p = nvvm.elect_sync()
            bars.mb_bmm1_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_k_empty[state_b.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

            q_full_phase = q_full_phase ^ 1

            # Mainloop kv = kv_left+1 .. kv_right-1 (empty when n_kv == 1)
            for kv_loop in cutlass.range(kv_left + cutlass.Int32(1), kv_right, 1, unroll=1):
                old_a = state_a
                old_b = state_b
                state_a = advance(state_b, CFG.STAGES_KV)
                state_b = advance(state_a, CFG.STAGES_KV)
                is_not_first_bmm2 = cutlass.Boolean(kv_loop != (kv_left + cutlass.Int32(1)))

                # BMM2 sub-tile 0 on V[nb0] (stage old_a)
                bars.mb_v_full[old_a.idx].wait(old_a.phase, spin=SPIN_RING_WAITS)
                desc_V = sV[old_a.idx].desc()
                bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 0].wait(bmm2_ready_phase, spin=SPIN_RING_WAITS)
                accum_b2 = is_not_first_bmm2
                for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                    mma_ts_step(bmm2_desc, (tmem_raw.subview(LAYOUT.P0_OFF)), desc_V, (tmem_raw.subview(LAYOUT.O0_OFF)), local_k, accum_b2)
                    accum_b2 = cutlass.Boolean(True)
                if cutlass.const_expr(CFG.N_BMM2_CHUNKS == 2):
                    bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 1].wait(bmm2_ready_phase, spin=SPIN_RING_WAITS)
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
                bars.mb_v_empty[old_a.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

                # BMM1 sub 0 for the next kv tile on K[nb0] (stage a)
                bars.mb_k_full[state_a.idx].wait(state_a.phase, spin=SPIN_RING_WAITS)
                desc_K = sK[state_a.idx].desc()
                mma_ss(bmm1_desc, desc_Q0, desc_K, (tmem_raw.subview(LAYOUT.S0_OFF)))
                elect_p = nvvm.elect_sync()
                bars.mb_bmm1_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_k_empty[state_a.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

                # BMM2 sub-tile 1 on V[nb1] (stage old_b)
                bars.mb_v_full[old_b.idx].wait(old_b.phase, spin=SPIN_RING_WAITS)
                desc_V = sV[old_b.idx].desc()
                bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 0].wait(bmm2_ready_phase, spin=SPIN_RING_WAITS)
                accum_b2 = is_not_first_bmm2
                for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                    mma_ts_step(bmm2_desc, (tmem_raw.subview(LAYOUT.P1_OFF)), desc_V, (tmem_raw.subview(LAYOUT.O1_OFF)), local_k, accum_b2)
                    accum_b2 = cutlass.Boolean(True)
                if cutlass.const_expr(CFG.N_BMM2_CHUNKS == 2):
                    bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 1].wait(bmm2_ready_phase, spin=SPIN_RING_WAITS)
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
                bars.mb_v_empty[old_b.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

                # BMM1 sub 1 for the next kv tile on K[nb1] (stage b)
                bars.mb_k_full[state_b.idx].wait(state_b.phase, spin=SPIN_RING_WAITS)
                desc_K = sK[state_b.idx].desc()
                mma_ss(bmm1_desc, desc_Q1, desc_K, (tmem_raw.subview(LAYOUT.S1_OFF)))
                elect_p = nvvm.elect_sync()
                bars.mb_bmm1_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_k_empty[state_b.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

                bmm2_ready_phase = bmm2_ready_phase ^ 1

            # Epilogue: BMM2 of the last kv tile for both sub-tiles (always runs -- n_kv >= 1); the Q slots are free.
            if cutlass.const_expr(not IS_QO_ALIAS):
                elect_p = nvvm.elect_sync()
                for qs in cutlass.range_constexpr(CFG.TILES_Q):
                    bars.mb_q_empty[qs].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

            is_not_first_bmm2_epi = cutlass.Boolean((kv_right - kv_left) != cutlass.Int32(1))

            bars.mb_v_full[state_a.idx].wait(state_a.phase, spin=SPIN_RING_WAITS)
            desc_V = sV[state_a.idx].desc()
            bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 0].wait(bmm2_ready_phase, spin=SPIN_RING_WAITS)
            accum_b2 = is_not_first_bmm2_epi
            for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                mma_ts_step(bmm2_desc, (tmem_raw.subview(LAYOUT.P0_OFF)), desc_V, (tmem_raw.subview(LAYOUT.O0_OFF)), local_k, accum_b2)
                accum_b2 = cutlass.Boolean(True)
            if cutlass.const_expr(CFG.N_BMM2_CHUNKS == 2):
                bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 1].wait(bmm2_ready_phase, spin=SPIN_RING_WAITS)
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
            bars.mb_v_empty[state_a.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

            bars.mb_v_full[state_b.idx].wait(state_b.phase, spin=SPIN_RING_WAITS)
            desc_V = sV[state_b.idx].desc()
            bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 0].wait(bmm2_ready_phase, spin=SPIN_RING_WAITS)
            accum_b2 = is_not_first_bmm2_epi
            for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                mma_ts_step(bmm2_desc, (tmem_raw.subview(LAYOUT.P1_OFF)), desc_V, (tmem_raw.subview(LAYOUT.O1_OFF)), local_k, accum_b2)
                accum_b2 = cutlass.Boolean(True)
            if cutlass.const_expr(CFG.N_BMM2_CHUNKS == 2):
                bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 1].wait(bmm2_ready_phase, spin=SPIN_RING_WAITS)
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
            bars.mb_v_empty[state_b.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

            bmm2_ready_phase = bmm2_ready_phase ^ 1
            kv_state = advance(state_b, CFG.STAGES_KV)
    return kv_state, q_full_phase, bmm2_ready_phase, empty_mainloop_phase


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
    total_sum,
    bmm1_phase,
    stat_empty_phase,
    leader_cta_id,
    true_max,
    track_true_max: bool = False,
):
    """Per-iter kv body for the softmax warp group.

    Compile-time apply_mask (Python bool) picks the load+max strategy:
    apply_mask=False uses fused tcgen05.ld.red.f32.max (HW row-max);
    apply_mask=True uses tcgen05.ld + apply_mask_chunk + software row-max
    (HW max can't observe NEG_INFINITY written after the load).

    ``true_max`` is the row's TRUE running max (scaled units), carried beside the
    threshold-skipped reference ``total_max`` and updated only when
    ``track_true_max`` (the score side outputs are requested): the reference may
    lag the true max by up to RESCALE_THRESHOLD, which the LSE tolerates but a
    published ``score_max`` may not.  Untracked it is a loop-invariant the
    compiler folds away.

    Returns updated (total_max, total_sum, bmm1_phase, stat_empty_phase, true_max).
    """
    tmem_S_off = LAYOUT.S0_OFF if sub_tile_id == 0 else LAYOUT.S1_OFF
    tmem_P_off = LAYOUT.P0_OFF if sub_tile_id == 0 else LAYOUT.P1_OFF
    stats_off = LAYOUT.STATS_OFF + sub_tile_id * 8
    # The softmax chunk is a TILE_N (KV-axis) quantity -- the Cfg's BMM2_CHUNK_SIZE, never TILE_O's (the head-dim tile does
    # not change it; config_sm107._validate_head_dim_tile pins N_BMM2_CHUNKS x BMM2_CHUNK_SIZE == TILE_N).
    CHUNK = CFG.BMM2_CHUNK_SIZE
    # fp16/bf16 pack 2 probs per FP32 TMEM cell (CHUNK//2); TF32 is 1:1 (CHUNK).
    P_COLS_PER_CHUNK = CHUNK if IS_TF32 else CHUNK // 2
    N_CHUNKS = CFG.N_BMM2_CHUNKS
    NEG_INF = cutlass.Float32(-3.4028235e38)
    RESCALE_THRESHOLD = cutlass.Float32(CFG.RESCALE_THRESHOLD)

    bars.mb_bmm1_done[sub_tile_id].wait(bmm1_phase, spin=SPIN_RING_WAITS)
    bmm1_phase = bmm1_phase ^ 1

    # Hoist all TMEM address math to a single base load — Rubin tcgen05.ld/st
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
                window_right=CFG.WINDOW_RIGHT,
            )
            for c in range(N_CHUNKS)
        ]
        chunks_max = [row_max_reduction(chunks_S[c]) for c in range(N_CHUNKS)]
        # vec_concat handles single-element lists too; keeps tracer Vector type uniform across N_CHUNKS.
        reg_S_vec = vec_concat(chunks_S)
        current_max_unscaled = chunks_max[0]
        for m in chunks_max[1:]:
            current_max_unscaled = cute.math.max(current_max_unscaled, m)
    else:
        reg_S_tile, current_max_unscaled = tmem_load_max_reduction_tile(
            s_addr_base,
            num_elems=CFG.TILE_N,
        )
        reg_S_vec = reg_S_tile.vec

    # Pass size=CFG.TILE_N explicitly — Vector.shape[0] is an MLIR value
    # (not Python int) for vec_concat-built vectors, so auto-detect can't recover the length.
    reg_S = RegTile(reg_S_vec, size=CFG.TILE_N)
    current_max = current_max_unscaled * scale_log2

    # Named-barrier wg0/wg1 sync: sub_tile_id==1 waits at TOP for sub_tile_id==0's bottom arrive.
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
    if cutlass.const_expr(track_true_max):
        true_max = cute.math.max(true_max, current_max, ftz=True)

    alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
    bars.mb_stat_full[sub_tile_id].arrive()

    # Rescale full reg_S in one vector op — emits same FFMA2 sequence as explicit half-tile rescales.
    reg_S = reg_S * scale_log2 - new_total_max

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

    bars.mb_stat_empty[sub_tile_id].wait(stat_empty_phase, spin=SPIN_RING_WAITS)
    stat_empty_phase = stat_empty_phase ^ 1

    return total_max, total_sum, bmm1_phase, stat_empty_phase, true_max


@cute.jit
def _softmax_kv_body_inf(
    sub_tile_id: int,
    kv_loop,
    kv_tile_rel,
    tmem_ptr_i32,
    bars,
    q_abs,
    block_row,
    batch_idx,
    eff_seqlen_kv,
    eff_seqlen_q,
    scale_log2,
    inv_scale,
    total_max,
    total_max_safe,
    total_sum,
    bmm1_phase,
    stat_empty_phase,
    leader_cta_id,
    sBias,
    kv_mask_tensor,
    bias_ring_state=None,
):
    """The per-iter kv body of the BIAS / KV_MASK arms (CFG.BIAS_MODE or CFG.KV_MASK_FORM): every kv tile takes this path.

    Differences from ``_softmax_kv_body(apply_mask=True)``, in pipeline order:

    1. BEFORE the ``bmm1_done`` wait (no dependency on BMM1): the kv_mask keep words of this tile (4 byte loads + 4
       ballots per warp; ``tile_dsl.mask.kv_mask_keep_words``) and the first ``BIAS_PRELOAD_CHUNKS`` 64-column bias chunks
       of this lane's row from the resident block (one swizzled 128-B load each, kept as 32 packed 16-bit pairs).
    2. Per 64-column chunk after the raw S load: ``s = bias * inv_scale + s`` as one ``fma.rn.f32x2`` per pair (the bias in
       NATURAL units added pre-scale, so the exp2 argument ``s * scale_log2 - m`` is unchanged in form), then the band mask
       AND the kv_mask word in ONE bit-word pass with the TRUE -inf mask value.
    3. The online softmax of the -inf form: ``total_max`` starts at -inf and only a live column can move it; the max is
       0-substituted at the point of use (``row_max_for_exp2``) on BOTH alpha operands, so a dead iteration ships
       ``P == 0`` and alpha == 1 exactly, and a row whose every key is masked ends the item with ``total_sum == 0``
       (the correction epilogue's empty-row SELECT reads that).

    ``kv_tile_rel`` = ``kv_loop - <the item's first tile>``, the tile's index inside the resident block; ``block_row`` the
    lane's row inside the block (``sub_tile_id * TILE_M + tid_in_wg`` on the one-batch-entry item, whose block is both
    sub-tiles' 256 rows; ``tid_in_wg`` on the (n, n+1) pair item, whose 128-row block both sub-tiles read).  On the STREAMED
    ring (CFG.BIAS_MODE == 2) ``bias_ring_state`` is the consumer state: the tile is ring stage ``bias_ring_state.idx``, waited
    (``mb_bias_ring_full``) before the first read and released (``mb_bias_ring_empty``, one elected lane per warp) after the last
    word is consumed; the CALLER advances the state once per kv tile.  Returns (total_max, total_max_safe, total_sum,
    bmm1_phase, stat_empty_phase).
    """
    tmem_S_off = LAYOUT.S0_OFF if sub_tile_id == 0 else LAYOUT.S1_OFF
    tmem_P_off = LAYOUT.P0_OFF if sub_tile_id == 0 else LAYOUT.P1_OFF
    stats_off = LAYOUT.STATS_OFF + sub_tile_id * 8
    CHUNK = CFG.BMM2_CHUNK_SIZE
    P_COLS_PER_CHUNK = CHUNK // 2
    N_CHUNKS = CFG.N_BMM2_CHUNKS
    WORDS_PER_CHUNK = CHUNK // 32
    RESCALE_THRESHOLD = cutlass.Float32(CFG.RESCALE_THRESHOLD)
    kv_col_base = kv_loop * cutlass.Int32(CFG.TILE_N)

    # 1. Loads with no BMM1 dependency go ahead of the wait.
    kv_words = None
    if cutlass.const_expr(_KV_MASK_BOOL):
        kv_words = kv_mask_keep_words(kv_mask_tensor, batch_idx, kv_col_base, eff_seqlen_kv, CFG.TILE_N)
    if cutlass.const_expr(_BIAS_STREAMED):
        # The streamed ring (S1): this kv tile's bias stage must have landed before the first read -- ahead of the pre-load.
        bars.mb_bias_ring_full[bias_ring_state.idx].wait(bias_ring_state.phase, spin=SPIN_RING_WAITS)
    # The tile's index inside sBias: the ring stage (streamed) or the item-relative kv tile (resident) -- a Python ternary.
    bias_tile = bias_ring_state.idx if _BIAS_STREAMED else kv_tile_rel
    # Python comprehension (not a staged loop / container mutation): the pre-loaded chunks' packed words.
    bias_pre = [load_bias_row_chunk(sBias.base, _BIAS_GG, bias_tile, c, block_row) for c in range(min(BIAS_PRELOAD_CHUNKS, N_CHUNKS))] if _BIAS_ARM else []

    bars.mb_bmm1_done[sub_tile_id].wait(bmm1_phase, spin=SPIN_RING_WAITS)
    bmm1_phase = bmm1_phase ^ 1

    tmem_base = tmem_ptr_i32.load()
    s_addr_base = tmem_base + cutlass.Int32(tmem_S_off)
    p_addr_base = tmem_base + cutlass.Int32(tmem_P_off)
    stats_addr = tmem_base + cutlass.Int32(stats_off)

    # 2. Raw S chunk -> + bias (pre-scale) -> band mask AND kv_mask word (-inf) -> software row max.
    causal_diag = eff_seqlen_kv - eff_seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else None
    chunks_S = []
    for c in cutlass.range_constexpr(N_CHUNKS):
        raw = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * CHUNK), cutlass.Float32), num=CHUNK)
        if cutlass.const_expr(_BIAS_ARM):
            # Chunk c's words: pre-loaded ahead of the wait, or loaded here (a Python ternary on a trace-time bool).
            bias_words = bias_pre[c] if c < BIAS_PRELOAD_CHUNKS else load_bias_row_chunk(sBias.base, _BIAS_GG, bias_tile, c, block_row)
            s_chunk = bias_fma_chunk(raw, bias_words, inv_scale, CHUNK, dtype=STORAGE_DTYPE)
        else:
            s_chunk = raw
        if cutlass.const_expr(_KV_MASK_ADD):
            # The additive per-key vector: this chunk's 64 values (warp-uniform 16-byte granules, loaded right here so at most
            # one granule is live beside the S chunk) folded pre-scale like the bias.
            s_chunk = additive_vec_fma_chunk(
                s_chunk,
                additive_vec_granules(kv_mask_tensor, batch_idx, kv_col_base + cutlass.Int32(c * CHUNK), eff_seqlen_kv, CHUNK, dtype=KV_MASK_DTYPE),
                inv_scale,
                CHUNK,
                dtype=KV_MASK_DTYPE,
            )
        keep = tuple(kv_words[c * WORDS_PER_CHUNK : (c + 1) * WORDS_PER_CHUNK]) if cutlass.const_expr(_KV_MASK_BOOL) else None
        chunks_S.append(
            apply_mask_chunk(
                s_chunk,
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
                keep_words=keep,
            )
        )
    if cutlass.const_expr(_BIAS_STREAMED):
        # The stage's LAST bias word is consumed (S2): release it to the TMA-LDG warp -- one elected lane per warp (8 == init;
        # THREAD arrives have no pred= path, so the elect is a branch).
        if nvvm.elect_sync():
            bars.mb_bias_ring_empty[bias_ring_state.idx].arrive()
    chunks_max = []
    for c in cutlass.range_constexpr(N_CHUNKS):
        chunks_max.append(row_max_reduction(chunks_S[c]))
    reg_S_vec = vec_concat(chunks_S)
    current_max_unscaled = chunks_max[0]
    for m in chunks_max[1:]:
        current_max_unscaled = cute.math.max(current_max_unscaled, m)
    reg_S = RegTile(reg_S_vec, size=CFG.TILE_N)
    current_max = current_max_unscaled * scale_log2  # -inf when every column of this iteration is masked

    if sub_tile_id == 1:
        nvvm.barrier_cta_sync(barrier_id=8, thread_count=256)

    # 3. Online softmax, -inf form.  total_max starts at -inf: the first live iteration always clears the threshold
    # (real - (-inf) = +inf), a dead one never does (-inf - x is -inf or NaN; ordered > is false for both).
    update_cond = (current_max - total_max) > RESCALE_THRESHOLD
    total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
    # 0-substitution at the point of use on BOTH alpha operands: a no-update iteration gives alpha == exp2(0) == 1 exactly
    # (the correction's all_alpha_one ballot keeps firing); min(., 0) guards the dead -> alive transition (the safe max
    # drops 0 -> real * scale < 0) where total_sum is still 0, so alpha need only stay finite.
    new_total_max_safe = row_max_for_exp2(total_max)
    alpha = cute.math.exp2(cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)), fastmath=True)
    total_max_safe = new_total_max_safe

    alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
    bars.mb_stat_full[sub_tile_id].arrive()

    reg_S = reg_S * scale_log2 - total_max_safe

    chunk_S_0 = reg_S[0:CHUNK].vec
    chunk_P_0 = cute.math.exp2(chunk_S_0, fastmath=True)
    hoisted_sum = row_reduction_pair(chunk_P_0)
    chunk_P_0_fp16 = chunk_P_0.to(STORAGE_DTYPE)
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32), chunk_P_0_fp16)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
    bars.mb_bmm2_ready[sub_tile_id * N_CHUNKS + 0].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

    deferred_P_1 = None
    if cutlass.const_expr(N_CHUNKS == 2):
        chunk_S_1 = reg_S[CHUNK : 2 * CHUNK].vec
        deferred_P_1 = cute.math.exp2(chunk_S_1, fastmath=True)
        chunk_P_1_fp16 = deferred_P_1.to(STORAGE_DTYPE)
        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32), chunk_P_1_fp16)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        bars.mb_bmm2_ready[sub_tile_id * N_CHUNKS + 1].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

    if sub_tile_id == 0:
        nvvm.barrier_cta_sync(barrier_id=8, thread_count=256)

    new_p_sum_pair = hoisted_sum
    if cutlass.const_expr(N_CHUNKS == 2):
        new_p_sum_pair = new_p_sum_pair + row_reduction_pair(deferred_P_1)
    alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
    total_sum = total_sum * alpha_pair + new_p_sum_pair

    bars.mb_stat_empty[sub_tile_id].wait(stat_empty_phase, spin=SPIN_RING_WAITS)
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
    track_true_max: bool = False,
    # Bias / kv_mask arms (appended; the call is by keyword).  None = the arm is off.
    sBias=None,
    kv_mask_tensor=None,
    inv_scale=None,
    n_rows_total=None,
):
    """Softmax warp group (one of two): online softmax per kv iter.

    Each lane owns one row of the 128 x 128 S_acc tile, loads its 128 fp32
    cols with a single tcgen05.ld.red.max, runs online softmax (total_max +
    total_sum tracking with RESCALE_THRESHOLD skip), publishes alpha to
    corr, writes P to TMEM cols at S_acc tail, fires bmm2_ready[sub][chunk].

    CFG.BIAS_MODE / CFG.KV_MASK_FORM: every kv tile runs ``_softmax_kv_body_inf``
    (the -inf-masked software-max body with the bias add and the kv_mask word);
    with the bias, each warp waits ``mb_bias_full`` once per non-empty item
    before its first read and releases ``mb_bias_empty`` once per non-empty item
    after its last (``if elect_sync():`` -> one lane per warp; 8 == init).
    """
    # Wait on MMA's TMEM-publish named barrier BEFORE any tmem_ptr_i32.load() —
    # without it softmax can race MMA's tcgen05_alloc and read a stale base.
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))

    tmem_S_off = LAYOUT.S0_OFF if sub_tile_id == 0 else LAYOUT.S1_OFF
    tmem_P_off = LAYOUT.P0_OFF if sub_tile_id == 0 else LAYOUT.P1_OFF

    NEG_INF = cutlass.Float32(-3.4028235e38)

    CHUNK = CFG.BMM2_CHUNK_SIZE  # the TILE_N-derived softmax chunk (see _softmax_kv_body)
    P_COLS_PER_CHUNK = CHUNK // 2
    stats_off = LAYOUT.STATS_OFF + sub_tile_id * 8

    # Phase trackers persist across tile boundaries (barriers don't reset).
    bmm1_phase = cutlass.Int32(0)
    stat_empty_phase = cutlass.Int32(1)  # bootstrap pre-armed at phase 1 so first wait passes immediately
    # init phase=1; both softmax wgs wait on slot [0] not [SoftmaxGid].
    epilogue_state = cutlass.Int32(1)
    # mb_bias_full consumer phase (wait-then-arrive, start(0)); flips once per NON-empty item, like the producer's.
    bias_full_phase = cutlass.Int32(0)
    # The streamed ring's consumer-side state over mb_bias_ring_full (start(0): wait-then-arrive), advanced once per kv tile after
    # the body's wait / release.  Loop-carried on the pair item (always a traced state there, a pass-through on the resident pair
    # rendering); None off the ring on the one-batch-entry loop, where it is never assigned.
    bias_ring_state = PipelineState.start(phase=0) if (_BIAS_STREAMED or _PAIR_ARM) else None
    NEG_INF_TRUE = cutlass.Float32(float("-inf"))
    total_max_safe = NEG_INF_TRUE

    # total_sum kept as Vector[Float32, 2] (even/odd partials) so per-iter update lowers to packed FMUL2 + FADD2.
    total_max = NEG_INF
    true_max = NEG_INF  # the score side outputs' TRUE row max (see _softmax_kv_body); loop-invariant when untracked
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
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
    bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx)

    softmax_wg_base_const = CFG.SOFTMAX_WG0_BASE if sub_tile_id == 0 else CFG.SOFTMAX_WG1_BASE
    tid_in_wg = cute.arch.thread_idx()[0] - cutlass.Int32(softmax_wg_base_const * 32)

    if cutlass.const_expr(_PAIR_ARM):
        # The (n, n+1) pair item (CFG.N_CHUNK > 0): the same persistent loop over scheduler items, each item a chunk of
        # row pairs (`_*_item_pair`); the per-item decode is the shared scheduler payload (q tile, head, chunk).
        while is_valid_tile > cutlass.Int32(0):
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
            bmm1_phase, stat_empty_phase, epilogue_state, bias_full_phase, bias_ring_state = _softmax_item_pair(
                sub_tile_id,
                tmem_ptr_i32,
                bars,
                q_super_idx,
                batch_idx,
                n_rows_total,
                seqlen_q,
                seqlen_kv,
                seq_kv_lens_tensor,
                seq_q_lens_addr,
                scale_log2,
                inv_scale,
                tid_in_wg,
                leader_cta_id,
                sBias,
                kv_mask_tensor,
                bmm1_phase,
                stat_empty_phase,
                epilogue_state,
                bias_full_phase,
                bias_ring_state,
            )
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
            )
            is_valid_tile = nxt_v & cutlass.Int32(1)
            sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
    else:
        while is_valid_tile > cutlass.Int32(0):
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

            # Top-of-tile mb_o_empty[0] wait: without it softmax can race into the
            # next tile while TMA-STG is still draining the prior tile's O slot.
            bars.mb_o_empty[0].wait(epilogue_state, spin=SPIN_RING_WAITS)
            epilogue_state = epilogue_state ^ cutlass.Int32(1)

            total_max = NEG_INF
            true_max = NEG_INF
            total_sum = cutlass.Vector.from_elements(
                (cutlass.Float32(0.0), cutlass.Float32(0.0)),
                cutlass.Float32,
            )
            q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M)
            q_abs = q_row_coord + cutlass.Int32(sub_tile_id * CFG.TILE_M) + tid_in_wg
            # Bootstrap stat_empty wait BEFORE kv loop — lifts wait off per-iter
            # critical path so alpha publish + stat_full fire happen back-to-back.
            bars.mb_stat_empty[sub_tile_id].wait(stat_empty_phase, spin=SPIN_RING_WAITS)
            stat_empty_phase = stat_empty_phase ^ 1
            # 3-segment kv loop: LEFT-masked / fully-unmasked / RIGHT-masked.
            # At MASK_NONE bounds collapse so the two masked sub-loops have empty range and fold out.
            if cutlass.const_expr(_INF_MASK_ARM):
                # Bias / kv_mask: ONE loop, every tile through the -inf masked body.  The resident block is waited once
                # per non-empty item (the TMA-LDG warp arms it once; the predicate is the SAME kv range both decodes
                # produced, so an empty item touches neither bias barrier on either side).
                item_live = bounds.right > bounds.left
                if cutlass.const_expr(_BIAS_RESIDENT):
                    if item_live:
                        bars.mb_bias_full.wait(bias_full_phase)
                        bias_full_phase = bias_full_phase ^ cutlass.Int32(1)
                total_max = NEG_INF_TRUE
                total_max_safe = NEG_INF_TRUE
                for kv_loop in cutlass.range(bounds.left, bounds.right, 1, unroll=1):
                    total_max, total_max_safe, total_sum, bmm1_phase, stat_empty_phase = _softmax_kv_body_inf(
                        sub_tile_id,
                        kv_loop,
                        kv_loop - bounds.left,
                        tmem_ptr_i32,
                        bars,
                        q_abs,
                        cutlass.Int32(sub_tile_id * CFG.TILE_M) + tid_in_wg,
                        batch_idx,
                        eff_seqlen_kv,
                        eff_seqlen_q,
                        scale_log2,
                        inv_scale,
                        total_max,
                        total_max_safe,
                        total_sum,
                        bmm1_phase,
                        stat_empty_phase,
                        leader_cta_id,
                        sBias,
                        kv_mask_tensor,
                        bias_ring_state,
                    )
                    if cutlass.const_expr(_BIAS_STREAMED):
                        # One ring stage consumed per kv tile (the body waited and released it): advance in lockstep with the fill.
                        bias_ring_state = advance(bias_ring_state, CFG.STAGES_BIAS)
                if cutlass.const_expr(_BIAS_RESIDENT):
                    if item_live:
                        # The block's LAST read of this item is behind us: release it to the TMA-LDG warp (one lane per
                        # warp; THREAD arrives have no pred= path, so the elect is a branch).
                        if nvvm.elect_sync():
                            bars.mb_bias_empty.arrive()
            elif cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
                for kv_loop in cutlass.range(bounds.left, bounds.right, 1, unroll=1):
                    total_max, total_sum, bmm1_phase, stat_empty_phase, true_max = _softmax_kv_body(
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
                        bmm1_phase,
                        stat_empty_phase,
                        leader_cta_id,
                        true_max,
                        track_true_max,
                    )
            else:
                for kv_loop in cutlass.range(bounds.left, bounds.unmasked_lo, 1, unroll=1):
                    total_max, total_sum, bmm1_phase, stat_empty_phase, true_max = _softmax_kv_body(
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
                        bmm1_phase,
                        stat_empty_phase,
                        leader_cta_id,
                        true_max,
                        track_true_max,
                    )
                for kv_loop in cutlass.range(bounds.unmasked_lo, bounds.unmasked_hi, 1, unroll=1):
                    total_max, total_sum, bmm1_phase, stat_empty_phase, true_max = _softmax_kv_body(
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
                        bmm1_phase,
                        stat_empty_phase,
                        leader_cta_id,
                        true_max,
                        track_true_max,
                    )
                for kv_loop in cutlass.range(bounds.unmasked_hi, bounds.right, 1, unroll=1):
                    total_max, total_sum, bmm1_phase, stat_empty_phase, true_max = _softmax_kv_body(
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
                        bmm1_phase,
                        stat_empty_phase,
                        leader_cta_id,
                        true_max,
                        track_true_max,
                    )

            # End-of-kv: publish (total_max, total_sum_final) to TMEM Stats — corr reads it for LSE.
            total_sum_scalar = total_sum[0] + total_sum[1]

            stats_addr_epi = tmem_ptr_i32.load() + cutlass.Int32(stats_off)
            stats_vec_epi = cutlass.Vector.from_elements((total_max, total_sum_scalar), cutlass.Float32)
            if cutlass.const_expr(track_true_max):
                # Score side outputs: the TRUE row max rides the stats slot's free columns 2-3 (8 per sub-tile;
                # the correction's load widens to 4) -- stored twice so no constant reaches the store.
                stats_vec_epi = cutlass.Vector.from_elements((total_max, total_sum_scalar, true_max, true_max), cutlass.Float32)
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
            )
            is_valid_tile = nxt_v & cutlass.Int32(1)
            sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
            bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx)


@cute.jit
def _softmax_item_pair(
    sub_tile_id: int,
    tmem_ptr_i32,
    bars,
    q_tile_idx,
    chunk_idx,
    n_rows_total,
    seqlen_q,
    seqlen_kv,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    scale_log2,
    inv_scale,
    tid_in_wg,
    leader_cta_id,
    sBias,
    kv_mask_tensor,
    bmm1_phase,
    stat_empty_phase,
    epilogue_state,
    bias_full_phase,
    bias_ring_state,
):
    """One (n, n+1) pair work item of a softmax warp group (CFG.N_CHUNK > 0): this sub-tile is row nb0 (WG0) / nb1 (WG1) of every
    step; the resident block is waited ONCE per live item before step 0 and released ONCE after the last step (the SAME
    global-length predicate the TMA-LDG warp fills on); per step the one-batch-entry tile body -- the top-of-step `mb_o_empty[0]`
    wait, the stat_empty bootstrap, the -inf masked kv body at THIS row's lengths (`_softmax_kv_body_inf` with block row
    `tid_in_wg`: both sub-tiles read the 128-row block) over the PAIR's union of kv tiles, the stats publish.  On the STREAMED
    ring the per-item wait / release fold out and the body waits / releases ONE ring stage per kv tile (`bias_ring_state`,
    advanced here per kv tile).  Returns the loop-carried (bmm1_phase, stat_empty_phase, epilogue_state, bias_full_phase,
    bias_ring_state)."""
    NEG_INF_TRUE = cutlass.Float32(float("-inf"))
    stats_off = LAYOUT.STATS_OFF + sub_tile_id * 8
    n0, n_end, n_steps = _pair_item_rows(chunk_idx, n_rows_total)
    item_b = _pair_item_bounds(q_tile_idx, seqlen_q, seqlen_kv)
    item_live = item_b.right > item_b.left
    q_abs = q_tile_idx * cutlass.Int32(CFG.TILE_M) + tid_in_wg
    if cutlass.const_expr(_BIAS_RESIDENT):
        if item_live:
            bars.mb_bias_full.wait(bias_full_phase)
            bias_full_phase = bias_full_phase ^ cutlass.Int32(1)
    for j in cutlass.range(cutlass.Int32(0), n_steps, 1, unroll=1):
        nb0, nb1, _live1 = _pair_step_rows(n0, n_end, j)
        nb = nb0 if sub_tile_id == 0 else nb1  # a trace-time select: sub_tile_id is the warp group's Python int
        sb = _pair_step_bounds(q_tile_idx, seqlen_q, seqlen_kv, seq_kv_lens_tensor, seq_q_lens_addr, nb0, nb1)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, nb, seqlen_kv)
        # Top-of-step mb_o_empty[0] wait (the TMA-STG drains the previous step's O slot before this one is reused).
        bars.mb_o_empty[0].wait(epilogue_state, spin=SPIN_RING_WAITS)
        epilogue_state = epilogue_state ^ cutlass.Int32(1)
        total_max = NEG_INF_TRUE
        total_max_safe = NEG_INF_TRUE
        total_sum = cutlass.Vector.from_elements(
            (cutlass.Float32(0.0), cutlass.Float32(0.0)),
            cutlass.Float32,
        )
        bars.mb_stat_empty[sub_tile_id].wait(stat_empty_phase, spin=SPIN_RING_WAITS)
        stat_empty_phase = stat_empty_phase ^ 1
        for kv_loop in cutlass.range(sb.left, sb.right, 1, unroll=1):
            total_max, total_max_safe, total_sum, bmm1_phase, stat_empty_phase = _softmax_kv_body_inf(
                sub_tile_id,
                kv_loop,
                kv_loop - item_b.left,
                tmem_ptr_i32,
                bars,
                q_abs,
                tid_in_wg,
                nb,
                eff_seqlen_kv,
                seqlen_q,
                scale_log2,
                inv_scale,
                total_max,
                total_max_safe,
                total_sum,
                bmm1_phase,
                stat_empty_phase,
                leader_cta_id,
                sBias,
                kv_mask_tensor,
                bias_ring_state,
            )
            if cutlass.const_expr(_BIAS_STREAMED):
                # One ring stage consumed per kv tile (the body waited and released it): advance in lockstep with the fill.
                bias_ring_state = advance(bias_ring_state, CFG.STAGES_BIAS)
        # End-of-step: publish (total_max, total_sum_final) to the TMEM Stats slot -- the correction reads it for the LSE.
        total_sum_scalar = total_sum[0] + total_sum[1]
        stats_addr_epi = tmem_ptr_i32.load() + cutlass.Int32(stats_off)
        stats_vec_epi = cutlass.Vector.from_elements((total_max, total_sum_scalar), cutlass.Float32)
        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr_epi, cutlass.Float32), stats_vec_epi)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        bars.mb_stat_full[sub_tile_id].arrive()
    if cutlass.const_expr(_BIAS_RESIDENT):
        if item_live:
            # The block's LAST read of this item is behind us: release it to the TMA-LDG warp (one lane per warp).
            if nvvm.elect_sync():
                bars.mb_bias_empty.arrive()
    return bmm1_phase, stat_empty_phase, epilogue_state, bias_full_phase, bias_ring_state


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
    o_partial_f32: Optional[cute.Tensor],
    score_max_tensor: Optional[cute.Tensor] = None,
    score_sum_exp_tensor: Optional[cute.Tensor] = None,
    n_rows_total=None,
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

    # Read tidx directly from PTX sreg — cute.arch.thread_idx() in @cute.jit may yield warp-uniform via folding.
    tid_raw = cute.arch.thread_idx()[0]
    tid_in_wg = tid_raw - cutlass.Int32(CFG.CORR_WARP_BASE * 32)

    # O_CHUNK=16 (halved from 32) shortens the alpha-rescale live range —
    # at 32, the DSL's register allocator spilled correction-warp regs to the stack.
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
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
    bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx)

    if cutlass.const_expr(_PAIR_ARM):
        # The (n, n+1) pair item (CFG.N_CHUNK > 0): the same persistent loop over scheduler items, each item a chunk of
        # row pairs (`_*_item_pair`); the per-item decode is the shared scheduler payload (q tile, head, chunk).
        while is_valid_tile > cutlass.Int32(0):
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
            stat_full_phase, bmm2_done_phase, o_empty_phase = _corr_item_pair(
                sO,
                tmem_ptr_i32,
                tid_in_wg,
                bars,
                lse_tensor,
                sinks_tensor,
                seq_kv_lens_tensor,
                seq_q_lens_addr,
                q_super_idx,
                head_idx,
                batch_idx,
                n_rows_total,
                seqlen_q,
                seqlen_kv,
                leader_cta_id,
                stat_full_phase,
                bmm2_done_phase,
                o_empty_phase,
            )
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
            )
            is_valid_tile = nxt_v & cutlass.Int32(1)
            sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
    else:
        while is_valid_tile > cutlass.Int32(0):
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
            # Iter-0 lifted out of kv loop: MMA's iter-0 BMM2 uses init_d=False
            # to overwrite O garbage, so alpha-rescale is unnecessary in iter 0.
            if bounds.right > bounds.left:
                for qs in cutlass.range_constexpr(CFG.TILES_Q):
                    bars.mb_bmm2_ready[qs * CFG.N_BMM2_CHUNKS + 0].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
                for qs in cutlass.range_constexpr(CFG.TILES_Q):
                    bars.mb_stat_full[qs].wait(stat_full_phase, spin=SPIN_RING_WAITS)
                    bars.mb_stat_empty[qs].arrive()
                stat_full_phase = stat_full_phase ^ 1
            else:
                # Empty-mainloop: corr fires (DSMEM-arrive on leader at cga2); MMA fires bmm2_done[0/1] via multicast.
                bars.mb_empty_mainloop.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            for kv_loop in cutlass.range(bounds.left + cutlass.Int32(1), bounds.right, 1, unroll=1):
                tmem_base_iter = tmem_ptr_i32.load()
                for qs in cutlass.range_constexpr(CFG.TILES_Q):
                    stats_off = LAYOUT.STATS_OFF + qs * 8
                    tmem_O_off = LAYOUT.O0_OFF if qs == 0 else LAYOUT.O1_OFF

                    bars.mb_stat_full[qs].wait(stat_full_phase, spin=SPIN_RING_WAITS)

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

                    bars.mb_bmm2_done[qs].wait(bmm2_done_phase, spin=SPIN_RING_WAITS)

                    # vec_scale_pair emits nvvm.mul_packed_f32x2 → FMUL2.  Without it
                    # plain o_chunk*alpha lowers to scalar FMUL inside the runtime-if
                    # (downstream fp32 tcgen05_st doesn't force packed regs).
                    # Per-chunk cfence keeps ptxas from sinking the next chunk's
                    # tcgen05_ld ahead of the prior tcgen05_st (TMEM-col anti-dep).
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
                stats_off = LAYOUT.STATS_OFF + qs * 8
                tmem_O_off = LAYOUT.O0_OFF if qs == 0 else LAYOUT.O1_OFF

                bars.mb_bmm2_done[qs].wait(bmm2_done_phase, spin=SPIN_RING_WAITS)

                # Wait the (total_max, total_sum_final) publish from softmax's epilogue.
                bars.mb_stat_full[qs].wait(stat_full_phase, spin=SPIN_RING_WAITS)

                stats_addr = tmem_base_epi + cutlass.Int32(stats_off)
                _score_stats = (score_max_tensor is not None) or (score_sum_exp_tensor is not None)
                if cutlass.const_expr(_score_stats):
                    # The score arm also reads the TRUE row max the softmax published in column 2.
                    stats_vec = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(stats_addr, cutlass.Float32),
                        num=4,
                    )
                else:
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

                inv_sum = cutlass.Float32(0.0)  # pre-declare for DSL if-staging
                lse_val = cutlass.Float32(0.0)
                score_max_val = cutlass.Float32(0.0)  # pre-declare for DSL if-staging (score side outputs)
                score_sum_val = cutlass.Float32(0.0)
                # With sinks: fold the lift-the-max rescale into threshold_beta so O is scaled by scale/new_sum in one FMUL.
                LN2 = cutlass.Float32(0.6931471805599453)
                total_max_nat = total_max_scaled * LN2
                if cutlass.const_expr(CFG.HAS_SINK):
                    sinks_arr = cutlass.make_array_view(sinks_tensor)
                    sink_logit = sinks_arr[head_idx]
                    new_max = cute.math.max(total_max_nat, sink_logit)
                    scale = cute.math.exp(total_max_nat - new_max, fastmath=True)
                    new_sum = total_sum * scale + cute.math.exp(sink_logit - new_max, fastmath=True)
                    lse_val = new_max + cute.math.log(new_sum, fastmath=True)
                    inv_sum = scale / new_sum
                else:
                    lse_val = total_max_nat + cute.math.log(total_sum, fastmath=True)
                    # Safe inverse: avoid div by 0 on fully-masked rows.
                    inv_sum = cutlass.Float32(1.0) / cute.math.max(total_sum, cutlass.Float32(1e-30))
                if cutlass.const_expr(_score_stats):
                    # Score side outputs (score_max / score_sum_exp) from the TRUE row max (stats column 2): the LSE
                    # above is built on the threshold-skipped REFERENCE max and stays bitwise what it was; the sum is
                    # re-normalised onto the true max -- sum_true = sum_ref * exp2(m_ref - m_true), a factor in (0, 1]
                    # because the reference lags the true max.  A sink is one more column of the published pair.  The
                    # keyless-row SELECTs below override both.
                    true_max_nat = stats_vec[2] * LN2
                    sum_true = total_sum * cute.math.exp2(total_max_scaled - stats_vec[2], fastmath=True)
                    if cutlass.const_expr(CFG.HAS_SINK):
                        smax_sink = cute.math.max(true_max_nat, sink_logit)
                        score_max_val = smax_sink
                        score_sum_val = sum_true * cute.math.exp(true_max_nat - smax_sink, fastmath=True) + cute.math.exp(sink_logit - smax_sink, fastmath=True)
                    else:
                        score_max_val = true_max_nat
                        score_sum_val = sum_true
                # --- empty KV range (fully-masked / zero-length KV for this tile) ---
                # Same guard as prefill_d256_{f16,fp8,mxfp8}_sm107.py.  With
                # bounds.right <= bounds.left the kv loop ran ZERO iterations, so
                # BMM2 never overwrote the O accumulator (mma_ss overwrites only on
                # its first k-step) and TMEM still holds the PREVIOUS persistent
                # tile's O -- residue, possibly a NaN bit pattern.
                #
                # The softmax warp still publishes (total_max = -inf, total_sum = 0)
                # for an empty range, so LSE is already right on both arms: no-sink
                # gives -inf + log(0) = -inf (this kernel puts NO 1e-30 floor on the
                # log -- if one is ever added, the select below must cover lse_val
                # too, or it leaks log(1e-30) = -69.08), and the sink arm folds to
                # exactly sink_logit.  inv_sum, however, is 1/max(0, 1e-30) = 1e30
                # (no sink) or 0 (sink), so O = residue * 1e30 -> +-inf/NaN, or
                # residue * 0 -> NaN.  Zero O with a SELECT, never a multiply.
                # OOB-row guard: under cga2 the cluster's Q rows can exceed seqlen_q;
                # without the guard the write aliases the next head's LSE slot.
                q_row_global = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M) + cutlass.Int32(qs * CFG.TILE_M) + tid_in_wg
                _kv_empty = bounds.right <= bounds.left
                # A row with no live key inside a live tile -- bottom-right rows above the diagonal, a left band
                # past the last key, a zero KV length -- read off the mask geometry: these kernels mask with a
                # finite sentinel, so a keyless row's softmax sum is N, not 0, and cannot tell itself apart.
                if cutlass.const_expr(CFG.MASK_FLAGS != 0):
                    _diag = (eff_seqlen_kv - eff_seqlen_q) if cutlass.const_expr(CFG.BOTTOM_RIGHT) else cutlass.Int32(0)
                    _last_k = cute.math.min(eff_seqlen_kv, bounds.right * cutlass.Int32(CFG.TILE_N)) - cutlass.Int32(1)
                    if cutlass.const_expr(CFG.MASK_FLAGS & MASK_CAUSAL):
                        _last_k = cute.math.min(_last_k, q_row_global + _diag + cutlass.Int32(CFG.WINDOW_RIGHT))
                    _first_k = bounds.left * cutlass.Int32(CFG.TILE_N)
                    if cutlass.const_expr(CFG.MASK_FLAGS & MASK_SWA):
                        _first_k = cute.math.max(_first_k, q_row_global + _diag - cutlass.Int32(CFG.WINDOW_LEFT))
                    _kv_empty = _kv_empty | (_first_k > _last_k)
                if cutlass.const_expr(_INF_MASK_ARM):
                    # The bias / kv_mask arms mask with the TRUE -inf and the 0-substituted max, so a row whose every key
                    # the DATA mask removed (or a bias of -inf) -- invisible to the band geometry above -- ends the item
                    # with total_sum == 0 EXACTLY (a live row has sum >= 2^-RESCALE_THRESHOLD): fold it into the empty-row
                    # predicate, so O is SELECTed to 0 and the LSE to -inf (sink: sink_logit) like a geometric empty row.
                    _kv_empty = _kv_empty | (total_sum == cutlass.Float32(0.0))
                if cutlass.const_expr(not CFG.HAS_SINK):
                    # A sink leaves real mass and the branch above already yields
                    # LSE = sink_logit there; without one an empty row is -inf.
                    lse_val = cutlass.Float32(arith.select(_kv_empty.ir_value(), cutlass.Float32(float("-inf")).ir_value(), lse_val.ir_value()))
                    if cutlass.const_expr(_score_stats):
                        # A keyless row: max = -inf, sum_exp = 0 -- SELECTED (the published max of an empty
                        # range is the sentinel the softmax never cleared, the sum a 0 the LSE log maps to -inf).
                        score_max_val = cutlass.Float32(arith.select(_kv_empty.ir_value(), cutlass.Float32(float("-inf")).ir_value(), score_max_val.ir_value()))
                        score_sum_val = cutlass.Float32(arith.select(_kv_empty.ir_value(), cutlass.Float32(0.0).ir_value(), score_sum_val.ir_value()))
                else:
                    # A keyless row with a sink holds the sink's mass alone: LSE = sink_logit. The finite mask
                    # sentinel, scaled, can overflow to -inf and NaN the sink fold, so the value is selected, not
                    # computed; O is zeroed by the same _kv_empty select below.
                    lse_val = cutlass.Float32(arith.select(_kv_empty.ir_value(), cutlass.Float32(sink_logit).ir_value(), lse_val.ir_value()))
                    if cutlass.const_expr(_score_stats):
                        # The sink alone: max = sink_logit, sum_exp = exp(sink - sink) = 1 (LSE = sink_logit + ln 1).
                        score_max_val = cutlass.Float32(arith.select(_kv_empty.ir_value(), cutlass.Float32(sink_logit).ir_value(), score_max_val.ir_value()))
                        score_sum_val = cutlass.Float32(arith.select(_kv_empty.ir_value(), cutlass.Float32(1.0).ir_value(), score_sum_val.ir_value()))
                if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT):
                    # Dense padded-Q trim: q rows >= seq_len_q[b] write O := 0 / LSE := -inf
                    # (after the sink branch: a trimmed row is dead even with a sink); folded
                    # into _kv_empty so the O cast's select zeroes the row, NaN residue or not.
                    _sq_arr = cute.make_tensor(
                        cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4), cute.make_layout(1 << 24)
                    )
                    row_trim = q_row_global >= cutlass.Int32(_sq_arr[cutlass.Int32(batch_idx)])
                    lse_val = cutlass.Float32(arith.select(row_trim.ir_value(), cutlass.Float32(float("-inf")).ir_value(), lse_val.ir_value()))
                    inv_sum = cutlass.Float32(arith.select(row_trim.ir_value(), cutlass.Float32(0.0).ir_value(), inv_sum.ir_value()))
                    _kv_empty = _kv_empty | row_trim
                    if cutlass.const_expr(_score_stats):
                        # A trimmed row is dead even with a sink: max = -inf, sum_exp = 0, like its LSE.
                        score_max_val = cutlass.Float32(arith.select(row_trim.ir_value(), cutlass.Float32(float("-inf")).ir_value(), score_max_val.ir_value()))
                        score_sum_val = cutlass.Float32(arith.select(row_trim.ir_value(), cutlass.Float32(0.0).ir_value(), score_sum_val.ir_value()))
                # Convert only the final Stats; -inf padding stays -inf.
                if cutlass.const_expr(CFG.STATS_LOG2 and SPLIT_KV == 1):
                    lse_val = lse_val * cutlass.Float32(1.4426950408889634)
                if cutlass.const_expr(CFG.THD_VARLEN):
                    # THD: q_row_global is sequence-local; LSE is packed [1,QH,T] →
                    # index [0, head, cu_q[b] + local], bound by per-sequence Q len S_q_b.
                    _cu = cutlass.make_array_view(seq_kv_lens_tensor)
                    _cu_q_b = cutlass.Int32(_cu[n_batch + batch_idx])
                    _s_q_b = cutlass.Int32(_cu[n_batch + batch_idx + cutlass.Int32(1)]) - _cu_q_b
                    if cutlass.const_expr(lse_tensor is not None):
                        if q_row_global < _s_q_b:
                            lse_arr = cutlass.make_array_view(lse_tensor)
                            # The packed ragged Stats buffer is written in the
                            # CALLER's declared layout, selected by the STATIC rank
                            # compile() baked in: token-major rank-2 [T, QH] (the
                            # DEFAULT -- _thd_compile_kwargs only sets head-major
                            # when the caller asks) or head-major rank-3
                            # [1, QH, head_stride].  Serving only the rank-3 arm
                            # would silently transpose every LSE on the common path.
                            if cutlass.const_expr(len(lse_tensor.shape) == 2):
                                lse_arr[_cu_q_b + q_row_global, head_idx] = lse_val
                            else:
                                if cutlass.const_expr(len(lse_tensor.shape) == 4):
                                    # rank-4 = per-batch padded Stats (B, QH, s_max, 1) in the declared strides, no ragged offsets
                                    lse_arr[batch_idx, head_idx, q_row_global, 0] = lse_val
                                else:
                                    lse_arr[cutlass.Int32(0), head_idx, _cu_q_b + q_row_global] = lse_val
                else:
                    if cutlass.const_expr(lse_tensor is not None):
                        if q_row_global < seqlen_q:
                            lse_arr = cutlass.make_array_view(lse_tensor)
                            lse_arr[_partial_batch(batch_idx, split_idx, n_batch), head_idx, q_row_global] = lse_val
                # Score side outputs: the DENSE, unsplit row (compile() rejects the THD and split builds), the
                # same OOB-row guard as the dense LSE store, natural units whatever STATS_LOG2 says (the base-2
                # convention is the Stats tensor's alone).  Each store folds out independently on None.
                if cutlass.const_expr(score_max_tensor is not None):
                    if q_row_global < seqlen_q:
                        _smax_arr = cutlass.make_array_view(score_max_tensor)
                        _smax_arr[batch_idx, head_idx, q_row_global] = score_max_val
                if cutlass.const_expr(score_sum_exp_tensor is not None):
                    if q_row_global < seqlen_q:
                        _ssum_arr = cutlass.make_array_view(score_sum_exp_tensor)
                        _ssum_arr[batch_idx, head_idx, q_row_global] = score_sum_val

                sO_sub_base = sO[qs].base

                if cutlass.const_expr(_FP32_PARTIALS):
                    _store_fp32_partial_tile(
                        o_partial_f32,
                        tmem_base_epi,
                        tmem_O_off,
                        inv_sum,
                        _kv_empty,
                        q_row_global < seqlen_q,
                        _partial_batch(batch_idx, split_idx, n_batch),
                        q_row_global,
                        head_idx,
                        CFG.TILE_O,
                        O_CHUNK,
                    )
                    bars.mb_o_empty[qs].wait(o_empty_phase, spin=SPIN_RING_WAITS)
                else:
                    for chunk_idx in cutlass.range_constexpr(N_CHUNKS_O):
                        o_addr = tmem_base_epi + cutlass.Int32(tmem_O_off + chunk_idx * O_CHUNK)
                        o_chunk = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                            num=O_CHUNK,
                        )
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                        o_scaled = o_chunk * inv_sum
                        # SELECT the zero for an empty KV range -- see _kv_empty above.
                        # NB: a plain `for` statement, not a comprehension -- the DSL
                        # preprocessor only rewrites statement-level range_constexpr
                        # loops ("range_constexpr should be preprocessed by preprocessor").
                        _o_elems = []
                        for _i in cutlass.range_constexpr(O_CHUNK):
                            _o_elems.append(cutlass.Float32(arith.select(_kv_empty.ir_value(), cutlass.Float32(0.0).ir_value(), o_scaled[_i].ir_value())))
                        o_scaled = cutlass.Vector.from_elements(tuple(_o_elems), cutlass.Float32)
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
                            bars.mb_o_empty[qs].wait(o_empty_phase, spin=SPIN_RING_WAITS)
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
            )
            is_valid_tile = nxt_v & cutlass.Int32(1)
            sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
            bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx)

    # End-of-warp tmem_dealloc: under cga2 each corr lane ALSO DSMEM-arrives
    # on the peer so the peer's local mbar accumulates the full CGA-total count.
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        peer_cta = cta_id_x ^ cutlass.Int32(1)
        bars.mb_tmem_dealloc.arrive_on_peer(peer_cta)
    bars.mb_tmem_dealloc.arrive()


@cute.jit
def _corr_item_pair(
    sO,
    tmem_ptr_i32,
    tid_in_wg,
    bars,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    q_tile_idx,
    head_idx,
    chunk_idx,
    n_rows_total,
    seqlen_q,
    seqlen_kv,
    leader_cta_id,
    stat_full_phase,
    bmm2_done_phase,
    o_empty_phase,
):
    """One (n, n+1) pair work item of the correction group (CFG.N_CHUNK > 0; dense, unsplit, no score side outputs -- the arm's
    gates): per step the one-batch-entry tile body -- the iter-0 lift / the empty-mainloop arrive, the per-kv alpha rescale,
    then the epilogue per sub-tile with THAT ROW's batch entry: the empty-row predicate from its own KV length and the pair's
    union bounds (`| total_sum == 0` for a data-masked row), the padded-Q trim from its own Q length, the LSE store at
    [nb, head, q_row] gated by the row's liveness (a dead sub-tile 1 stores nothing), O SELECTed to 0 on an empty row and
    staged into sO[qs] for the TMA-STG warp.  Returns the loop-carried (stat_full_phase, bmm2_done_phase, o_empty_phase)."""
    O_CHUNK = 16
    N_CHUNKS_O = CFG.TILE_O // O_CHUNK
    TMA_O_ITERS = (CFG.TILE_O * CFG.BPE) // CFG.O_SWZ_BYTES
    D_BLOCK_SIZE = CFG.TILE_O // TMA_O_ITERS
    TMA_O_GRANU_ELEMS = CFG.TILE_M * D_BLOCK_SIZE
    LN2 = cutlass.Float32(0.6931471805599453)
    n0, n_end, n_steps = _pair_item_rows(chunk_idx, n_rows_total)
    q_row_global = q_tile_idx * cutlass.Int32(CFG.TILE_M) + tid_in_wg
    for j in cutlass.range(cutlass.Int32(0), n_steps, 1, unroll=1):
        nb0, nb1, live1 = _pair_step_rows(n0, n_end, j)
        bounds = _pair_step_bounds(q_tile_idx, seqlen_q, seqlen_kv, seq_kv_lens_tensor, seq_q_lens_addr, nb0, nb1)
        # Iter-0 lifted out of the kv loop: MMA's iter-0 BMM2 uses init_d=False, so the alpha rescale is unnecessary there.
        if bounds.right > bounds.left:
            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                bars.mb_bmm2_ready[qs * CFG.N_BMM2_CHUNKS + 0].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                bars.mb_stat_full[qs].wait(stat_full_phase, spin=SPIN_RING_WAITS)
                bars.mb_stat_empty[qs].arrive()
            stat_full_phase = stat_full_phase ^ 1
        else:
            # Empty step: corr fires mb_empty_mainloop; the MMA fires bmm2_done[0/1] after waiting it.
            bars.mb_empty_mainloop.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

        for kv_loop in cutlass.range(bounds.left + cutlass.Int32(1), bounds.right, 1, unroll=1):
            tmem_base_iter = tmem_ptr_i32.load()
            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                stats_off = LAYOUT.STATS_OFF + qs * 8
                tmem_O_off = LAYOUT.O0_OFF if qs == 0 else LAYOUT.O1_OFF

                bars.mb_stat_full[qs].wait(stat_full_phase, spin=SPIN_RING_WAITS)

                stats_addr = tmem_base_iter + cutlass.Int32(stats_off)
                stats_vec = nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(stats_addr, cutlass.Float32),
                    num=2,
                )
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                alpha = stats_vec[0]

                alpha_is_one = alpha == cutlass.Float32(1.0)
                all_alpha_one = vote_sync(0xFFFFFFFF, alpha_is_one, VoteSync.ALL)

                bars.mb_stat_empty[qs].arrive()

                bars.mb_bmm2_done[qs].wait(bmm2_done_phase, spin=SPIN_RING_WAITS)

                if ~all_alpha_one:
                    for chunk_idx_o in cutlass.range_constexpr(N_CHUNKS_O):
                        o_addr = tmem_base_iter + cutlass.Int32(tmem_O_off + chunk_idx_o * O_CHUNK)
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

        # === End-of-step epilogue, per sub-tile = per ROW of the pair ===
        tmem_base_epi = tmem_ptr_i32.load()
        for qs in cutlass.range_constexpr(CFG.TILES_Q):
            stats_off = LAYOUT.STATS_OFF + qs * 8
            tmem_O_off = LAYOUT.O0_OFF if qs == 0 else LAYOUT.O1_OFF
            nb = nb0 if qs == 0 else nb1  # trace-time select on the unrolled qs
            # The row's own lengths (the pair's union bounds were derived at the max; a row shorter than its partner has
            # its tail tiles PADDED-masked in the softmax, so its total_sum is exact).
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, nb, seqlen_kv)

            bars.mb_bmm2_done[qs].wait(bmm2_done_phase, spin=SPIN_RING_WAITS)
            bars.mb_stat_full[qs].wait(stat_full_phase, spin=SPIN_RING_WAITS)

            stats_addr = tmem_base_epi + cutlass.Int32(stats_off)
            stats_vec = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(stats_addr, cutlass.Float32),
                num=2,
            )
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            total_max_scaled = stats_vec[0]  # log2-units (softmax stores it scaled)
            total_sum = stats_vec[1]

            bars.mb_stat_empty[qs].arrive()

            inv_sum = cutlass.Float32(0.0)
            lse_val = cutlass.Float32(0.0)
            total_max_nat = total_max_scaled * LN2
            if cutlass.const_expr(CFG.HAS_SINK):
                sinks_arr = cutlass.make_array_view(sinks_tensor)
                sink_logit = sinks_arr[head_idx]
                new_max = cute.math.max(total_max_nat, sink_logit)
                scale = cute.math.exp(total_max_nat - new_max, fastmath=True)
                new_sum = total_sum * scale + cute.math.exp(sink_logit - new_max, fastmath=True)
                lse_val = new_max + cute.math.log(new_sum, fastmath=True)
                inv_sum = scale / new_sum
            else:
                lse_val = total_max_nat + cute.math.log(total_sum, fastmath=True)
                # Safe inverse: avoid div by 0 on fully-masked rows (the SELECT below keeps it out of O).
                inv_sum = cutlass.Float32(1.0) / cute.math.max(total_sum, cutlass.Float32(1e-30))
            # Empty-row predicate: the step's union range empty, the row's own band geometry keyless, or (the -inf masked
            # body) a DATA-masked / shorter row whose sum ended at EXACTLY 0.  SELECT, never a multiply by zero.
            _kv_empty = bounds.right <= bounds.left
            if cutlass.const_expr(CFG.MASK_FLAGS != 0):
                _last_k = cute.math.min(eff_seqlen_kv, bounds.right * cutlass.Int32(CFG.TILE_N)) - cutlass.Int32(1)
                if cutlass.const_expr(CFG.MASK_FLAGS & MASK_CAUSAL):
                    _last_k = cute.math.min(_last_k, q_row_global + cutlass.Int32(CFG.WINDOW_RIGHT))
                _first_k = bounds.left * cutlass.Int32(CFG.TILE_N)
                if cutlass.const_expr(CFG.MASK_FLAGS & MASK_SWA):
                    _first_k = cute.math.max(_first_k, q_row_global - cutlass.Int32(CFG.WINDOW_LEFT))
                _kv_empty = _kv_empty | (_first_k > _last_k)
            _kv_empty = _kv_empty | (total_sum == cutlass.Float32(0.0))
            if cutlass.const_expr(not CFG.HAS_SINK):
                lse_val = cutlass.Float32(arith.select(_kv_empty.ir_value(), cutlass.Float32(float("-inf")).ir_value(), lse_val.ir_value()))
            else:
                lse_val = cutlass.Float32(arith.select(_kv_empty.ir_value(), cutlass.Float32(sink_logit).ir_value(), lse_val.ir_value()))
            if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT):
                # Dense padded-Q trim at THIS row's Q length: q rows >= seq_len_q[nb] write O := 0 / LSE := -inf.
                _sq_arr = cute.make_tensor(cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4), cute.make_layout(1 << 24))
                row_trim = q_row_global >= cutlass.Int32(_sq_arr[cutlass.Int32(nb)])
                lse_val = cutlass.Float32(arith.select(row_trim.ir_value(), cutlass.Float32(float("-inf")).ir_value(), lse_val.ir_value()))
                inv_sum = cutlass.Float32(arith.select(row_trim.ir_value(), cutlass.Float32(0.0).ir_value(), inv_sum.ir_value()))
                _kv_empty = _kv_empty | row_trim
            if cutlass.const_expr(CFG.STATS_LOG2):
                lse_val = lse_val * cutlass.Float32(1.4426950408889634)
            if cutlass.const_expr(lse_tensor is not None):
                # The LSE row of THIS batch entry; a dead sub-tile 1 (an odd chunk tail recomputing row nb0) stores nothing.
                if cutlass.const_expr(qs == 0):
                    if q_row_global < seqlen_q:
                        lse_arr = cutlass.make_array_view(lse_tensor)
                        lse_arr[nb, head_idx, q_row_global] = lse_val
                else:
                    if live1 & (q_row_global < seqlen_q):
                        lse_arr = cutlass.make_array_view(lse_tensor)
                        lse_arr[nb, head_idx, q_row_global] = lse_val

            sO_sub_base = sO[qs].base
            for chunk_idx_o in cutlass.range_constexpr(N_CHUNKS_O):
                o_addr = tmem_base_epi + cutlass.Int32(tmem_O_off + chunk_idx_o * O_CHUNK)
                o_chunk = nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                    num=O_CHUNK,
                )
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                o_scaled = o_chunk * inv_sum
                _o_elems = []
                for _i in cutlass.range_constexpr(O_CHUNK):
                    _o_elems.append(cutlass.Float32(arith.select(_kv_empty.ir_value(), cutlass.Float32(0.0).ir_value(), o_scaled[_i].ir_value())))
                o_scaled = cutlass.Vector.from_elements(tuple(_o_elems), cutlass.Float32)
                o_fp16 = o_scaled.to(STORAGE_DTYPE)

                col_offset_const = (chunk_idx_o * O_CHUNK) % D_BLOCK_SIZE
                block_idx_const = (chunk_idx_o * O_CHUNK) // D_BLOCK_SIZE
                block_offset_const = block_idx_const * TMA_O_GRANU_ELEMS
                smem_offset = cutlass.Int32(block_offset_const + col_offset_const) + tid_in_wg * cutlass.Int32(D_BLOCK_SIZE)

                smem_ptr = sO_sub_base.subview(smem_offset).data_ptr()
                if chunk_idx_o == 0:
                    bars.mb_o_empty[qs].wait(o_empty_phase, spin=SPIN_RING_WAITS)
                smem_ptr.store_swizzled(o_fp16, alignment=64, swizzle=_O_SMEM_SWIZZLE)

            # fence_proxy before the TMA store reads SMEM written by the generic proxy.
            nvvm.fence_proxy("async.shared", space="cta")

            bars.mb_o_full[qs].arrive()

        stat_full_phase = stat_full_phase ^ 1
        o_empty_phase = o_empty_phase ^ 1
        # P14 catch-up flip (the mainloop flips n_kv - 1 times, the MMA fires n_kv times per step).
        bmm2_done_phase = bmm2_done_phase ^ 1
    return stat_full_phase, bmm2_done_phase, o_empty_phase


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
    lse_ext: cutlass.Int64,
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
    # Score side outputs (APPENDED; the prepared binder fills them from the frame template): fp32
    # (B, QH, SQ) buffers in their declared (batch, head, row) element strides, None = not requested
    # (the store folds out of the kernel; the stride slots are then unread).
    score_max_ptr: Optional[cute.Pointer] = None,
    score_sum_exp_ptr: Optional[cute.Pointer] = None,
    score_max_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64] = (0, 0, 0),
    score_sum_exp_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64] = (0, 0, 0),
    # Additive bias [B_b, H_b, S_q, S_kv] in the io dtype (APPENDED; the prepared binder fills them): pointer, its
    # (batch, head, q-row) element strides and its two broadcast extents (B_b in {1, B}, H_b in {1, QH}); the per-key
    # boolean mask [B_m, S_kv] as bytes with its batch element stride; 1 / softmax_scale for the pre-scale bias add.
    # None / zeros on a plan without the arm (CFG.BIAS_MODE / KV_MASK_FORM fold the paths out; compile() backstops).
    bias_ptr: Optional[cute.Pointer] = None,
    bias_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64] = (0, 0, 0),
    bias_extents: Tuple[int, int] = (1, 1),
    kv_mask_ptr: Optional[cute.Pointer] = None,
    kv_mask_batch_stride: cutlass.Int64 = 0,
    inv_scale_softmax: cutlass.Float32 = 1.0,
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

    ``lse_kind``: "dense" (B, QH, SQ) in ``lse_strides``; "token" (SQ, QH)
    packed; "head" (1, QH, lse_ext) with lse_ext the head-row stride; "padded"
    (B, QH, lse_ext, 1) in ``lse_strides``. ``lse_ptr`` None compiles the store out.
    Unused slots (no THD, no paged KV, no split) are passed as None / zeros."""
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
    )
    score_max_tensor = sdpa_score_tensor(score_max_ptr, problem_size, score_max_strides)
    score_sum_exp_tensor = sdpa_score_tensor(score_sum_exp_ptr, problem_size, score_sum_exp_strides)
    bias_tensor = sdpa_bias_tensor(bias_ptr, problem_size, bias_strides, bias_extents)
    kv_mask_tensor = sdpa_kv_mask_tensor(kv_mask_ptr, problem_size, kv_mask_batch_stride)
    stride_order = (3, 2, 1, 0)
    kv_stride_order = stride_order
    _O_GRANU_ELEMS = CFG.O_SWZ_BYTES // CFG.BPE
    qk_box_q = (1, CFG.TILE_M, 1, TMA_QK_GRANU_ELEMS)
    qk_box_k = (1, CFG.TILE_N // CFG.CTA_MMA, 1, TMA_QK_GRANU_ELEMS)
    vo_box_v = (1, CFG.TILE_N, 1, TMA_VO_GRANU_ELEMS)
    vo_box_o = (1, CFG.TILE_M, 1, _O_GRANU_ELEMS)

    # Per-tensor TMA swizzle tracks per-CTA inner bytes (derived from CFG.*_SWZ_BYTES).
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
    # V TMA: TF32 needs SWIZZLE_128B_ATOM_32B (transposed BMM2 operand) — same
    # 128-B swizzle line / box geometry as standard Swz128B, just the atom mode.
    # Standard Swz128B with kind::tf32 + b_trans silently returns all-zero O.
    _v_tma_swz = tmap.TensorMapSwizzle.s128b_atom_32b if IS_TF32 else _tma_swz(CFG.V_SWZ_BYTES)
    tma_v_desc = tmap.create_tensor_map_tiled_from_view(
        v_tensor,
        box_dims=vo_box_v,
        stride_order=stride_order,
        swizzle=_v_tma_swz,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    # The unused split-output descriptor still needs a legal FP32 box.
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
    # Resident bias block LOAD descriptor over the bias tensor's (B_b, H_b, S_q, S_kv) dims: one box = BIAS_BLOCK_ROWS rows
    # x 64 columns under the 128-B swizzle (tile_dsl.bias.bias_block_geometry).  None when the arm is off: every bias /
    # kv_mask kernel argument is None-specialised so the plain rendering keeps its parameter list (and its cubin).
    tma_bias_desc = (
        tmap.create_tensor_map_tiled_from_view(
            bias_tensor,
            box_dims=_BIAS_GG.box_dims,
            stride_order=stride_order,
            swizzle=_tma_swz(_BIAS_GG.swz_bytes),
            l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
        )
        if cutlass.const_expr(bias_tensor is not None)
        else None
    )
    bias_b_ext = cutlass.Int32(bias_extents[0]) if cutlass.const_expr(bias_tensor is not None) else None
    bias_h_ext = cutlass.Int32(bias_extents[1]) if cutlass.const_expr(bias_tensor is not None) else None
    inv_scale_arg = inv_scale_softmax if cutlass.const_expr(_INF_MASK_ARM) else None

    # Each cluster pair (CTA_MMA CTAs) collectively covers TILE_M*TILES_Q*CTA_MMA Q rows;
    # without the cluster-wide divisor cga2 over-launches and OOB clusters collide in GMEM.
    rows_per_cluster = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA
    q_clusters = (SQ + rows_per_cluster - 1) // rows_per_cluster
    # The (n, n+1) pair item: a q TILE of TILE_M rows per CTA (both sub-tiles are rows of it), the CHUNK index on the grid's
    # batch axis (ceil(B / N_CHUNK) chunks; `n_batch` = that count for the scheduler, `n_rows_total` = B for the chunk clamp).
    grid_q_supers = ((SQ + CFG.TILE_M - 1) // CFG.TILE_M) if cutlass.const_expr(_PAIR_ARM) else (q_clusters * CFG.CTA_MMA)
    q_supers = grid_q_supers
    sched_batch = ((B + CFG.N_CHUNK - 1) // CFG.N_CHUNK) if cutlass.const_expr(_PAIR_ARM) else B
    n_rows_total_arg = cutlass.Int32(B) if cutlass.const_expr(_PAIR_ARM) else None
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD: build the per-batch O descriptor array (reuse tma_o_desc over the
        # packed [1,T,QH,D_v] O as base), then launch the exact flat
        # batch-outermost grid (n_thd_units = Σ_b ceil(S_q_b/CGA_TILE_M)*QH,
        # host-computed); grid_x = n_thd_units * CGA_M.  Works at cga1 (CGA_M=1).
        # Builds the [seq_kv|cu_q|cu_k|batch_remap|live|ctr] metadata buffer
        # (4B+4) device-side from the caller's length tensors -- no host cumsum,
        # no H2D -- then the per-batch O descriptor array.  The 4B+4 layout is
        # NOT optional here: this kernel's decode is the shared
        # _common_blackwell._thd_decode, which reads batch_remap at 3*n_batch+2
        # and relies on the dead-unit sentinel.  It was already using that
        # decode while allocating the pre-upstream 3B+2 buffer, which is why
        # compile() used to raise rather than half-serve THD.
        #
        # Per-token element stride of packed O is o_tensor.stride[1] (= QH*d_v),
        # NOT CFG.TILE_O -- those coincide only at QH == 1.
        _build_o_descs_kernel(
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
            cutlass.Int64(o_tensor.stride[1]),
            cutlass.Int32(CGA_TILE_M),
            n_thd_units,
        ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
        grid_shape = (n_thd_units * cutlass.Int32(CFG.CGA_M), cutlass.Int32(1), cutlass.Int32(1))
    else:
        # Grid Python-folds on Cfg constant (avoids DSL if-staging).
        grid_shape = (
            (grid_q_supers, QH, sched_batch * SPLIT_KV)
            if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL)
            else (grid_q_supers * QH * sched_batch * SPLIT_KV, 1, 1)
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
        cutlass.Int32(QH),
        cutlass.Int32(sched_batch),
        cutlass.Int32(QH // KH),
        scale_softmax_log2,
        seq_q_lens_addr,
        o_partial_f32,
        score_max_tensor,
        score_sum_exp_tensor,
        tma_bias_desc,
        bias_tensor,
        bias_b_ext,
        bias_h_ext,
        kv_mask_tensor,
        inv_scale_arg,
        n_rows_total_arg,
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
    has_score_max: bool = False,
    has_score_sum_exp: bool = False,
    has_bias: bool = False,
    has_kv_mask: bool = False,
) -> Callable:
    """Compile the host entry for one layout kind.

    ``has_bias`` / ``has_kv_mask`` (APPENDED) bind the additive bias (pointer + strides + broadcast extents) and the
    per-key boolean mask (pointer + batch stride) at the tvm-ffi boundary; each must agree with the module's
    ``CFG.BIAS_MODE`` / ``CFG.KV_MASK_FORM`` (the TemplateParams decided the traced arm, this key only shapes the
    argument list) -- a disagreement is a lowering bug and raises here.

    Every extent and stride is a runtime argument of the artifact (see ``_host``),
    so the key is only what specializes the traced code: the head-dim ENVELOPE
    (``d_qk`` / ``d_v``: the TMA descriptors carry the real extents while the
    tile box stays the compile-time TILE geometry — box columns past d_qk / d_v
    zero-fill on load, O columns past d_v clip on store), whether the LSE store
    exists, the Stats layout kind, and for paged pools whether the in-page
    layout is HND (row stride below head stride).

    Constraint: every non-innermost TMA global stride must be a 16-byte
    multiple; the compact BSHD H-stride is d * BPE, so d must be a multiple of
    8 at 2 bytes/elem — checked here for the envelope, by the adapter for
    declared strides."""
    _cache_key = _template_key(globals(), locals(), "compile")
    if not (0 < d_qk <= CFG.TILE_K and 0 < d_v <= CFG.TILE_O):
        raise ValueError(f"d{CFG.TILE_K} envelope: need 0 < d_qk <= {CFG.TILE_K} and 0 < d_v <= {CFG.TILE_O}; got ({d_qk}, {d_v})")
    if (d_qk * CFG.BPE) % 16 != 0 or (d_v * CFG.BPE_O) % 16 != 0:
        raise ValueError(
            f"d{CFG.TILE_K} envelope: d_qk*BPE and d_v*BPE must be 16-byte multiples (TMA global-stride rule); got ({d_qk}, {d_v}) at BPE={CFG.BPE}"
        )
    if SPLIT_KV > 1 and not has_lse:
        raise ValueError("split_kv > 1 requires partial LSE for the combine")
    if lse_kind not in LSE_KINDS:
        raise ValueError(f"lse_kind must be one of {LSE_KINDS}; got {lse_kind!r}")
    if has_lse and (lse_kind == "dense") == bool(CFG.THD_VARLEN):
        raise ValueError("lse_kind 'dense' is the dense form; 'token' / 'head' / 'padded' are the THD forms")
    if paged_hnd:
        raise ValueError("paged_hnd is a paged-KV specialization")
    if (has_score_max or has_score_sum_exp) and (SPLIT_KV > 1 or CFG.THD_VARLEN):
        # The stores index the dense (B, QH, SQ) row; a split's partials and the packed THD rows have no
        # such slot (the combine writes no score outputs).  Declined by the engine row first; a backstop here.
        raise ValueError("score_max / score_sum_exp are served on dense, unsplit builds only (no THD, no split_kv > 1)")
    if has_bias != _BIAS_ARM or has_kv_mask != _KV_MASK_ARM:
        raise ValueError(
            f"compile(has_bias={has_bias}, has_kv_mask={has_kv_mask}) disagrees with the module's CFG.BIAS_MODE={CFG.BIAS_MODE} / "
            f"CFG.KV_MASK_FORM={CFG.KV_MASK_FORM}: the TemplateParams decide the traced arm, the key must bind the operands it reads"
        )
    if _INF_MASK_ARM and (has_score_max or has_score_sum_exp):
        # The score side outputs ride the threshold-skipped reference max of the legacy body; the -inf body does not
        # publish the true max.  Declined by the engine row (the bias / kv_mask rows select no score flavor); a backstop.
        raise ValueError("score_max / score_sum_exp are not wired on the bias / kv_mask arms")
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
        cutlass.Int64(0),
        cutlass.Float32(0.0),
        i32,
        cutlass.Int64(0),
        P(cutlass.Int32, 4) if thd else None,
        P(cutlass.Int32, 4) if thd else None,
        i32 if thd else None,
        P(cutlass.Float32) if _FP32_PARTIALS else None,
        None,
        None,
        (cutlass.Int64(0), cutlass.Int64(0)),
        i32,
        d_qk,
        d_v,
        lse_kind,
        paged_hnd,
        P(cutlass.Float32, 4) if has_score_max else None,
        P(cutlass.Float32, 4) if has_score_sum_exp else None,
        i64_3,
        i64_3,
        P(STORAGE_DTYPE) if has_bias else None,
        i64_3,
        (1, 1),
        P(KV_MASK_DTYPE, 1 if _KV_MASK_BOOL else 16) if has_kv_mask else None,
        cutlass.Int64(0),
        cutlass.Float32(1.0),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_fwd",
    )
