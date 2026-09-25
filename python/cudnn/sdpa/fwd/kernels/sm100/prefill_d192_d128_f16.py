# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
DSL prefill SDPA kernel — DSv3 MLA flavor, d_qk = 192, d_v = 128, FP16/BF16, SM100.

SM100 (Blackwell GB200) DSv3-class prefill kernel.  Pipeline shape:
TILES_Q=2, two softmax warpgroups, four correction warps, 16 warps / 512
threads, persistent try_cancel scheduler, Q∪O alias.

SM100 resource layout:
  1. **cga2-only, STAGES_KV=2**: per-CTA K/V halved by the collective MMA →
     SMEM 192 KiB (sQ 64 + sK 32 + sV 32 + sO 64), under the Blackwell
     ~228 KiB cap.
  2. **SMEM stats**: alpha and final (max,sum) use a dedicated 2 KiB mailbox.
     This leaves all 512 TMEM cols to S_acc 0/128 + O 256..511 and removes
     the cross-tile dependency that otherwise delays the next BMM1 until
     correction has consumed final stats from an aliased S_acc head.
  3. **Manual row-max** on the MASK_NONE fast path — ``tcgen05_ld`` +
     ``row_max_reduction`` (the masked path's pattern) instead of
     ``tmem_load_max_reduction_tile`` (LDTM.STAT).
  4. SM100 launch: ``cluster=(CTA_MMA, 1, 1)``.
  5. **exp2 split MUFU / FMA** (``_E2E_*``, **cc 10.0 only**): 32 of the 128 P
     columns of every softmax row are evaluated on the FMA pipe by
     ``exp2_emul_pair`` (blocks of 4, spread over both P chunks) on EVERY mask
     arm, the alpha exp2 stays on MUFU -- 97 instead of 129 MUFU.EX2 per row
     per KV step.  MEASURED on B200 (DSv3 layer, H=128/128 bf16, A/B/A x3):
     +1.9 % at S=8K dense, +2.6 % at S=2K, +3.1 % at S=32K, +1.75 % causal.
     ``PARAMS.exp2_fma_split`` (auto-set from the build device) folds it out on
     every other cc, where MUFU.EX2 runs at twice B200's rate and the same
     split loses (``_E2E_ENABLED``); there the dense top-left causal arm keeps
     its pre-existing tail emulation (``_exp2_mixed_late``).

Supported: FP16 / BF16 (``DTYPE_QKV in {2, 3}``); masks none / causal / SWA /
padded and all pairwise combos (causal+swa, causal+padded, swa+padded);
attention sink (per-head logit in the softmax denominator); cga2; fixed LPT
scheduler.  Validated on Blackwell (cc 10.0) vs a torch fp32 reference.

THD / varlen (``CFG.THD_VARLEN=1``): packed ``[1,T,H,D]`` Q/K/V + ``cu_seqlens``
coord offset (applied to BOTH Q slabs under TILES_Q=2), per-batch O
TMA-descriptor array (shared ``thd_helpers.py``), packed ``[1,QH,T]`` LSE — via
the shared ``_common_blackwell`` / ``thd_helpers`` mechanism (same as the SM100 qwen
/ dsv4 kernels).  The dense ``[B,S,H,D]`` path is byte-identical (folds out at
``THD_VARLEN=0``).

Paged KV (``CFG.PAGED_KV=1``, issue #920): the d128 flavor's specialization,
transplanted verbatim — K/V are page POOLS ``[num_pages, page_size, H_kv, D]``
(HND or NHD by strides) and a ``[B, max_pages]`` int32 block table per pool
names each sequence's pages; the TMA-LDG warp issues every K/V tile as a stack
of page-sized row boxes with the page id read on device, boxes past a batch's
live pages TMA-OOB zero-fill, and the per-batch ``seq_kv_lens`` (mandatory)
bounds both the block-table walk and the softmax mask.  The two pools differ in
row width here (K 192, V 128) and already have separate descriptors, so the
mixed widths are the dense contract plus the page indirection.  Rows inside the
last live page but past ``seq_kv_len`` are LOADED and masked, so an allocated
page's bytes must be finite (the paged caller contract).

"""

from cudnn.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
from functools import lru_cache
from typing import Callable, NamedTuple, Optional, Tuple

from cutlass.experimental import primitives as nvvm
from cutlass.experimental.primitives import vote_sync, VoteSync
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer

import cutlass
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver  # noqa: F401  (cute.compile pulls cuda)

from dataclasses import dataclass

from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d192

# The template loader (api_dsl._load_kernel_module) injects FROST_TEMPLATE_PARAMS
# as a module global before this body executes; a plain import falls back to
# the default TemplateParams so the file stays importable on its own.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
CFG, _TMA = make_cfg_d192(PARAMS)
Cfg = type(CFG)
TMA_QK_ITERS = _TMA.QK_ITERS
TMA_VO_ITERS = _TMA.VO_ITERS
TMA_QK_GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _TMA.VO_GRANU_ELEMS

# O TMA box / store params follow O's swizzle, not V's (under cga2 V may drop to a narrower swizzle while O stays Swz128B).
TMA_O_GRANU_ELEMS_HOST = CFG.O_SWZ_BYTES // CFG.BPE
TMA_O_ITERS_HOST = (CFG.TILE_O * CFG.BPE) // CFG.O_SWZ_BYTES

from cudnn.frost.tile_dsl.barrier import (
    PipelineState,
    advance,
    arrive_expect_tx,
    cga_arrive,
    cga_wait,
    wait,
)
from cudnn.frost.tile_dsl.scheduler import (
    Sched,
    scheduler_warp_loop,
    scheduler_warp_loop_persistent,
    read_tile_id_arrive,
    read_clc_payload,
    SCHED_NATURAL,
)
from cudnn.frost.tile_dsl.pointwise import (
    # SM100: no LDTM.STAT — the MASK_NONE fast path uses manual tcgen05_ld +
    # row_max_reduction (see _softmax_kv_body), so tmem_load_max_reduction_tile
    # is not imported.
    exp2_mixed,
    row_reduction_pair,
    row_max_reduction,
    vec_scale_pair,
)
from cudnn.frost.tile_dsl.regtile import RegTile
from cudnn.frost.tile_dsl.mma import desc_opaque, mma_ss, mma_ts_step
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

from cudnn.block_sparse_attention.csrc.utils.kernel_utils import ex2_emulation_2

_PADDED_CAUSAL = CFG.MASK_FLAGS == (MASK_CAUSAL | MASK_PADDED) and CFG.WINDOW_RIGHT == 0
# Dense top-left causal MHA: BMM1/BMM2 issue tweaks (an opaque K descriptor, one BMM2 issue election per
# sub-tile, chunk 1's row-sum after its publish) plus -- with the cc 10.0 exp2 split OFF -- the pre-existing
# causal-only tail emulation ``_exp2_mixed_late`` (the last 34 columns of chunk 1 on ex2_emulation_2's
# degree-2 polynomial, PR #841).  With the split ON (``_E2E_ENABLED`` below) the module-wide ``_E2E_*``
# pattern replaces that tail emulation on EVERY mask arm -- the exact form MEASURED on B200.  The tail
# emulation itself is untouched and still traces wherever the split is off; it runs ungated on cc 10.3
# today and its own arch gate is a separate follow-up.
_DENSE_CAUSAL_PUBLIC_EXP2_MIX = not CFG.THD_VARLEN and CFG.MASK_FLAGS == MASK_CAUSAL and CFG.WINDOW_RIGHT == 0 and not CFG.BOTTOM_RIGHT and not CFG.HAS_SINK
_REUSE_BMM2_ISSUE_ELECTION = CFG.THD_VARLEN or _DENSE_CAUSAL_PUBLIC_EXP2_MIX

# exp2 split between MUFU and the FMA pipe.  With 128 exps per row per KV step the MUFU pipe
# (4 lanes/clk/SMSP) is the softmax warps' longest pipe on SM100 while the FP32 pipe has slack;
# a compile-time subset of the P columns is evaluated by ``exp2_emul_pair`` (6 packed FP32/INT
# instructions per pair) instead of ``ex2.approx``.  Same shape as the cuDNN backend fprop
# kernel's pattern (of every _E2E_FREQ columns the LAST _E2E_RES are emulated, none at or past
# _E2E_LIMIT); the distribution is the one the sm100 d128 MXFP8 prefill measured best on B200
# (B=1 H=24/8 S=16K dense, A/B/A): 32 of 128 columns in blocks of 4 spread over BOTH 64-wide
# chunks, +1.63 % over the backend's own (16, 8, 72) and +9.73 % over the all-MUFU kernel there;
# 24/40/48/64 emulated columns and blocks of 2 or 8 all lost.  Per row per KV step MUFU.EX2
# 129 -> 97 (the alpha exp2 stays on MUFU).  On THIS kernel MEASURED on B200 (DSv3 layer B=1
# H=128/128 bf16, A/B/A x3, controls <= 0.76 %): +1.90 % at S=8K dense, +2.64 % at S=2K,
# +3.07 % at S=32K, +1.75 % at S=8K causal (LPT_L2).  Derived per softmax chunk as pair indices
# for ``exp2_mixed``; sign and pattern are per kernel -- re-measure before changing the constants.
#
# ARCH GATE.  The split trades MUFU.EX2 pipe-time for FMA pipe-time, so its sign follows the part's
# MUFU rate -- MEASURED 16 elements/clk/SM on sm_100a (B200) and 32 on sm_107a (Rubin, where the same
# split is -9..-10 %); cc 10.3 (GB300) DOCUMENTS the doubled rate too.  ``PARAMS.exp2_fma_split`` is
# auto-set by the adapter from the BUILD device (api_dsl._exp2_fma_split_for: cc == (10, 0) x the
# kernels that measured a win) and folded here at trace time: off, ``_E2E_PAIRS`` are empty and both
# exp2 sites trace the develop spelling.  Full rationale and the measured rates: the ``_E2E_*`` block
# of ``prefill_d128_mxfp8.py``.
_E2E_FREQ = 16
_E2E_RES = 4
_E2E_LIMIT = CFG.TILE_N
_SOFTMAX_CHUNK = 64  # tcgen05_ld width of one softmax chunk (see _softmax_kv_body CHUNK)
_E2E_ENABLED = bool(PARAMS.exp2_fma_split)
# Emulated columns per row (32 of CFG.TILE_N=128 with the gate on, 0 off): the MUFU.EX2 the softmax still
# issues per row per KV step is the alpha exp2 plus the non-emulated columns (97 on, 129 off).
_E2E_EMULATED_COLS = (_E2E_LIMIT // _E2E_FREQ) * _E2E_RES if _E2E_ENABLED else 0
if not (0 <= _E2E_RES <= _E2E_FREQ and _E2E_FREQ % 2 == 0 and _E2E_RES % 2 == 0 and _E2E_LIMIT % _E2E_FREQ == 0):
    raise ValueError(f"{__name__}: exp2 emulation pattern freq={_E2E_FREQ} res={_E2E_RES} limit={_E2E_LIMIT} must be even with 0 <= res <= freq | limit")
if CFG.N_BMM2_CHUNKS * _SOFTMAX_CHUNK != CFG.TILE_N:
    raise ValueError(f"{__name__}: N_BMM2_CHUNKS ({CFG.N_BMM2_CHUNKS}) x softmax chunk ({_SOFTMAX_CHUNK}) must equal TILE_N ({CFG.TILE_N})")


def _e2e_pairs(chunk: int, chunk_elems: int = _SOFTMAX_CHUNK) -> frozenset:
    """Pair indices (pair p = columns 2p, 2p+1 of the chunk) that ``exp2_mixed`` emulates."""
    return frozenset(
        p
        for p in range(chunk_elems // 2)
        if ((chunk * chunk_elems + 2 * p) % _E2E_FREQ) >= (_E2E_FREQ - _E2E_RES) and (chunk * chunk_elems + 2 * p) < _E2E_LIMIT
    )


_E2E_PAIRS = tuple(_e2e_pairs(c) if _E2E_ENABLED else frozenset() for c in range(CFG.N_BMM2_CHUNKS))
if 2 * sum(len(p) for p in _E2E_PAIRS) != _E2E_EMULATED_COLS:
    raise ValueError(f"{__name__}: exp2 emulation pairs cover {2 * sum(len(p) for p in _E2E_PAIRS)} columns, expected {_E2E_EMULATED_COLS}")


def _exp2_mixed_late(vec):
    values = []
    for i in range(0, int(vec.shape[0]), 2):
        if i >= 30:
            x, y = ex2_emulation_2(vec[i], vec[i + 1], poly_degree=2)
        else:
            x = cute.math.exp2(vec[i], fastmath=True)
            y = cute.math.exp2(vec[i + 1], fastmath=True)
        values.extend((x, y))
    return cutlass.Vector.from_elements(tuple(values), cutlass.Float32)


@cute.jit
def _wait_ptr(mb, phase):
    phase = cutlass.Int32(phase)
    while not nvvm.mbarrier_wait_parity(mb, phase, nvvm.MBarrierWait.TRY):
        pass


@cute.jit
def _wait_mbarrier(mb, phase):
    _wait_ptr(mb.smem_ptr, phase)


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
    # SM100 d192/d128: f16/bf16 only. The config already asserts
    # DTYPE_QKV in {2, 3}.
    raise ValueError(f"prefill_sdpa_d192_d128_f16_sm100: DTYPE_QKV={CFG.DTYPE_QKV} not supported (expected 2=BF16 or 3=FP16)")


from cudnn.sdpa.fwd.kernels._common_blackwell import (
    sdpa_operand_tensors,
    KvLoopBounds,
    make_split_helpers,
    store_fp32_partial_tile as _store_fp32_partial_tile,
    make_classic_bars,
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
# THD uses a persistent grid + device-bounded claim counter (not the CLC
# envelope). The adapter caps the launch at min(envelope, SMs / CGA_SIZE)
# and the setup kernel publishes the live unit total it stops at.
THD_PERSISTENT = True


# LPT q-tile accounting is expressed in CGA-tile units; make_cfg_d192 has
# already applied this specialization's effective L2 grouping budget.
_sdpa_h = make_sdpa_helpers(CFG, lpt_q_tiles_in_cga_units=True)
_decode_initial = _sdpa_h.decode_initial
_decode_payload = _sdpa_h.decode_payload


@cute.jit
def _bounds_for_tile(
    q_super_idx,
    seqlen_q,
    seqlen_kv,
    cta_in_pair,
    seq_q_lens_addr,
    batch_idx,
    qh_per_kh: int = 1,
):
    # Keep the KV pipeline active for padded-Q tail tiles; the epilogue trims
    # rows beyond the per-batch Q length before storing O and LSE.
    return _sdpa_h.bounds_for_tile(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, qh_per_kh)


_resolve_seqlen_kv = _sdpa_h.resolve_seqlen_kv
_resolve_seqlen_q = _sdpa_h.resolve_seqlen_q
CAN_HAVE_EMPTY_KV = (CFG.MASK_FLAGS & (MASK_PADDED | MASK_SWA)) != 0 or CFG.BOTTOM_RIGHT

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
# caller's length tensors (issue #552) — no length ever reaches the host — and
# also publishes the live unit total + claim counter the persistent scheduler
# reads, so the adapter launches a MACHINE-sized grid rather than the plan-time
# envelope (issue #618).
_dispatch_decode_initial = _sdpa_h.dispatch_decode_initial
_dispatch_decode_payload = _sdpa_h.dispatch_decode_payload
_thd_tma_offsets = _sdpa_h.thd_tma_offsets

# === PackGQA ===
HEADS_PER_TILE = CFG.QH_PER_KH if CFG.PACK_GQA else 1
TOKENS_PER_TILE = CFG.TILE_M // HEADS_PER_TILE

# A sufficiently wide causal+SWA band cannot touch its lower and upper
# boundaries in the same TILE_N-wide score tile.  The four-way partition below
# assigns a partial KV-tail tile to the lower side when those ranges meet.
_MASK_TOKENS_PER_CGA = CFG.TILES_Q * TOKENS_PER_TILE * CFG.CTA_MMA
_SWA_ONE_SIDED_GEOMETRY = bool(
    (CFG.MASK_FLAGS & MASK_SWA) and (CFG.MASK_FLAGS & MASK_CAUSAL) and CFG.WINDOW_LEFT + CFG.WINDOW_RIGHT >= _MASK_TOKENS_PER_CGA + CFG.TILE_N - 2
)

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
# Same specialization as the d128 / d256 flavors (issue #920): K/V are page
# POOLS viewed as [num_pages, page_size, H_kv, D] and a [B, max_pages] int32
# block table names each sequence's pages.  A K tile is TILE_N/CTA_MMA rows per
# CTA (the cga2 pair splits it), a V tile is the full TILE_N rows (the pair
# splits V along d_v instead).  Each is loaded as a stack of ``*_BOXES`` row
# boxes of ``*_BOX_ROWS`` rows, so that no box ever straddles a page: box rows =
# min(page_size, tile rows).  The config validator guarantees page_size | 128 or
# 128 | page_size, so this is exact.  The two pools differ in row width here
# (K: TILE_K=192 -> TMA_QK_ITERS=3 D-subtiles per box, V: TILE_O=128 -> 2) and
# already have separate descriptors, so the mixed widths cost nothing extra.
PAGED_KV = bool(CFG.PAGED_KV)
PAGE_SIZE = CFG.PAGE_SIZE if PAGED_KV else 0
_K_TILE_ROWS = CFG.TILE_N // CFG.CTA_MMA
K_BOX_ROWS = min(PAGE_SIZE, _K_TILE_ROWS) if PAGED_KV else _K_TILE_ROWS
V_BOX_ROWS = min(PAGE_SIZE, CFG.TILE_N) if PAGED_KV else CFG.TILE_N
K_BOXES = _K_TILE_ROWS // K_BOX_ROWS
V_BOXES = CFG.TILE_N // V_BOX_ROWS

# This flavor carries its OWN empty-KV predicate (defined above), narrower than
# MAY_BE_EMPTY: it counts only PADDED / SWA / bottom-right, the masks that can
# empty a tile.  KV split makes an empty range reachable without any of them — a
# split past the end of a short range gets zero tiles — so the nine sites gated
# on CAN_HAVE_EMPTY_KV (the empty-tile handshake, the QO_ALIAS drain, and the
# epilogue's O-store guards) have to compile in under split as well.  Leaving it
# const-folded to False is what desynchronises the warp groups.
CAN_HAVE_EMPTY_KV = CAN_HAVE_EMPTY_KV or SPLIT_KV > 1

# The predecoded THD+SWA scheduler hands the TMA-LDG warp its next tile's KV
# bounds from decoded SMEM and never re-resolves eff_seqlen_kv on that
# back-edge; the paged loader needs the per-batch live page count
# (ceil(eff_seqlen_kv / PAGE_SIZE)) for every tile, so PAGED_KV takes the plain
# decode path (the same one the d128 / d256 flavors use) and folds this off.
_PREDECODE_THD_SWA_SEGMENTS = bool(CFG.THD_VARLEN and SPLIT_KV == 1 and not CFG.BOTTOM_RIGHT and _SWA_ONE_SIDED_GEOMETRY and not PAGED_KV)


@cute.jit
def _swa_segment_bounds(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx):
    bounds = _bounds_for_tile_split(
        q_super_idx,
        eff_seqlen_q,
        eff_seqlen_kv,
        cta_in_pair,
        seq_q_lens_addr,
        batch_idx,
        cutlass.Int32(0),
        CFG.QH_PER_KH,
    )
    cga_q_row_coord = (q_super_idx - cta_in_pair) * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE)
    causal_diag = eff_seqlen_kv - eff_seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else cutlass.Int32(0)
    lower_anchor = cga_q_row_coord + causal_diag + cutlass.Int32(_MASK_TOKENS_PER_CGA - 1 - CFG.WINDOW_LEFT)
    lower_end_if_positive = (lower_anchor + cutlass.Int32(CFG.TILE_N - 1)) // cutlass.Int32(CFG.TILE_N)
    lower_end_raw = cutlass.Int32(
        arith.select(
            (lower_anchor > cutlass.Int32(0)).ir_value(),
            lower_end_if_positive.ir_value(),
            cutlass.Int32(0).ir_value(),
        )
    )
    swa_left_end = cute.math.min(cute.math.max(lower_end_raw, bounds.left), bounds.right)
    pad_start = eff_seqlen_kv // cutlass.Int32(CFG.TILE_N) if cutlass.const_expr(CFG.MASK_FLAGS & MASK_PADDED) else bounds.right
    swa_left_pad_start = cute.math.min(cute.math.max(pad_start, bounds.left), swa_left_end)
    causal_start = (cga_q_row_coord + causal_diag + cutlass.Int32(CFG.WINDOW_RIGHT)) // cutlass.Int32(CFG.TILE_N)
    right_start_raw = cute.math.min(causal_start, pad_start)
    swa_right_start = cute.math.min(
        cute.math.max(cute.math.max(right_start_raw, swa_left_end), bounds.left),
        bounds.right,
    )
    return bounds.left, swa_left_pad_start, swa_left_end, swa_right_start, bounds.right


class _PredecodedSched(NamedTuple):
    mb_scheduler: object
    mb_read_tile_id: object
    mb_decoded: object
    tile_id_smem: object
    decoded_smem: object
    bidx_init: object
    bidy_init: object
    bidz_init: object


@cute.jit
def _scheduler_warp_loop_predecode(
    sched,
    is_cga_first_cta,
    cta_in_pair,
    n_q_supers,
    n_qh,
    n_batch,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    seqlen_q,
    seqlen_kv,
    qh_per_kh,
):
    meta = cutlass.make_array_view(seq_kv_lens_tensor)
    ctr_off = cutlass.Int32(4) * n_batch + cutlass.Int32(3)
    live_off = cutlass.Int32(4) * n_batch + cutlass.Int32(2)
    ctr_ptr = Pointer(seq_kv_lens_tensor.iterator.raw_ptr(), dtype=cutlass.Int32) + ctr_off
    state = PipelineState.start()
    is_valid = cutlass.Int32(1)
    while is_valid > cutlass.Int32(0):
        wait(sched.mb_read_tile_id.subview(state.idx), state.phase)
        if nvvm.elect_sync():
            arrive_expect_tx(sched.mb_scheduler.subview(state.idx), 16)
        if nvvm.elect_sync() and is_cga_first_cta:
            uid = cutlass.Int32(nvvm.atomicrmw(nvvm.AtomicOp.ADD, ctr_ptr, cutlass.Int32(1)))
            live = cutlass.Int32(meta[live_off])
            valid = cutlass.Int32(
                arith.select(
                    (uid < live).ir_value(),
                    cutlass.Int32(1).ir_value(),
                    cutlass.Int32(0).ir_value(),
                )
            )
            linear = uid * cutlass.Int32(CFG.CGA_M)
            tile_ptr = cute.make_ptr(
                cutlass.Int32,
                sched.tile_id_smem.subview(state.idx * cutlass.Int32(8)).data_ptr().toint(cutlass.Int32),
                cutlass.AddressSpace.smem,
                assumed_align=16,
            )
            mbar_ptr = cute.make_ptr(
                cutlass.Int64,
                sched.mb_scheduler.subview(state.idx).data_ptr().toint(cutlass.Int32),
                cutlass.AddressSpace.smem,
                assumed_align=8,
            )
            payload = (linear, cutlass.Int32(0), valid, cutlass.Int32(0))
            for i in cutlass.range_constexpr(CGA_SIZE):
                for word in cutlass.range_constexpr(4):
                    cute.arch.store_async_dsmem(tile_ptr + word, payload[word], mbar_ptr, i)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(state.idx), state.phase)
        payload_base = state.idx * cutlass.Int32(8)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, payload_base)
        nxt_q = cute.arch.make_warp_uniform(nxt_q)
        nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
        nxt_v = cute.arch.make_warp_uniform(nxt_v)
        q_super_idx, head_idx, batch_idx, _ = _decode_payload_split(
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
        is_valid = nxt_v & cutlass.Int32(1)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
        segments = _swa_segment_bounds(
            q_super_idx,
            eff_seqlen_q,
            eff_seqlen_kv,
            cta_in_pair,
            seq_q_lens_addr,
            batch_idx,
        )
        if nvvm.elect_sync():
            decoded_base = state.idx * cutlass.Int32(11)
            sched.decoded_smem.subview(decoded_base).store(q_super_idx)
            sched.decoded_smem.subview(decoded_base + cutlass.Int32(1)).store(head_idx)
            sched.decoded_smem.subview(decoded_base + cutlass.Int32(2)).store(batch_idx)
            sched.decoded_smem.subview(decoded_base + cutlass.Int32(3)).store(is_valid)
            sched.decoded_smem.subview(decoded_base + cutlass.Int32(4)).store(eff_seqlen_kv)
            sched.decoded_smem.subview(decoded_base + cutlass.Int32(5)).store(eff_seqlen_q)
            for i in cutlass.range_constexpr(5):
                sched.decoded_smem.subview(decoded_base + cutlass.Int32(6 + i)).store(segments[i])
            nvvm.mbarrier_arrive(sched.mb_decoded.subview(state.idx))
        state = advance(state, CFG.SCHEDULER_STAGES)


@cute.jit
def _softmax_next_payload(
    sched,
    sched_state,
    cta_in_pair,
    n_q_supers,
    n_qh,
    n_batch,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    seqlen_q,
    seqlen_kv,
    qh_per_kh,
):
    if cutlass.const_expr(_PREDECODE_THD_SWA_SEGMENTS):
        wait(sched.mb_decoded.subview(sched_state.idx), sched_state.phase)
        base = sched_state.idx * cutlass.Int32(11)
        payload = tuple(cute.arch.make_warp_uniform(sched.decoded_smem.subview(base + cutlass.Int32(i)).load()) for i in range(11))
        bounds = KvLoopBounds(
            left=payload[6],
            unmasked_lo=payload[8],
            unmasked_hi=payload[9],
            right=payload[10],
        )
        return (
            payload[0],
            payload[1],
            payload[2],
            cutlass.Int32(0),
            payload[3],
            payload[4],
            payload[5],
            bounds,
            payload[7],
            payload[8],
            payload[9],
        )

    _wait_ptr(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
    raw_base = sched_state.idx * cutlass.Int32(8)
    nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, raw_base)
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
    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
    bounds = _bounds_for_tile_split(
        q_super_idx,
        eff_seqlen_q,
        eff_seqlen_kv,
        cta_in_pair,
        seq_q_lens_addr,
        batch_idx,
        split_idx,
        CFG.QH_PER_KH,
    )
    segments = (bounds.left, bounds.unmasked_lo, bounds.unmasked_lo, bounds.unmasked_hi, bounds.right)
    return (
        q_super_idx,
        head_idx,
        batch_idx,
        split_idx,
        nxt_v & cutlass.Int32(1),
        eff_seqlen_kv,
        eff_seqlen_q,
        bounds,
        segments[1],
        segments[2],
        segments[3],
    )


@cute.jit
def _apply_padding_mask_if_needed(reg_S, kv_col_base, eff_seqlen_kv):
    """Apply the per-element padding predicate only to a partial KV chunk."""
    result = reg_S
    if kv_col_base + cutlass.Int32(int(reg_S.shape[0])) > eff_seqlen_kv:
        result = apply_mask_chunk(
            reg_S,
            cutlass.Int32(0),
            kv_col_base,
            eff_seqlen_kv,
            0,
            MASK_PADDED,
            N=int(reg_S.shape[0]),
            mask_value=float("-inf"),
        )
    return result


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

    SM100 512-col TMEM cap. S_acc[0]=0..127, S_acc[1]=128..255, and
    O=256..511 fill all 512 cols. P aliases the tails at 64 / 192 after
    softmax has loaded S; row stats use a dedicated SMEM mailbox.
    """

    # Blackwell SM10.0 cap; requires is_exclusive=True on tcgen05_alloc.
    TOTAL_COLS: int = 512

    S0_OFF: int = 0
    S1_OFF: int = 128

    O0_OFF: int = 256
    O1_OFF: int = 384

    P0_OFF: int = _P0_OFF
    P1_OFF: int = _P1_OFF


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
    # Keep V on a distinct SMEM address region from K; this avoids an unfavorable
    # K/V placement for the BMM2 handoff without changing occupancy.
    _kv_smem_pad = cutlass.Array(cutlass.Int8, 8192, alignment=1024, space=cutlass.AddressSpace.smem)
    sV_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * vBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sStats_raw = cutlass.Array(
        cutlass.Float32,
        CFG.TILES_Q * 2 * CFG.TILE_M,
        alignment=128,
        space=cutlass.AddressSpace.smem,
    )

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
    sched = (
        _PredecodedSched(
            mb_scheduler=cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            mb_read_tile_id=cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            mb_decoded=cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            tile_id_smem=cutlass.Array(cutlass.Int32, CFG.SCHEDULER_STAGES * 8, alignment=16, space=cutlass.AddressSpace.smem),
            decoded_smem=cutlass.Array(cutlass.Int32, CFG.SCHEDULER_STAGES * 11, alignment=16, space=cutlass.AddressSpace.smem),
            bidx_init=bidx,
            bidy_init=bidy,
            bidz_init=bidz,
        )
        if cutlass.const_expr(_PREDECODE_THD_SWA_SEGMENTS)
        else Sched(
            mb_scheduler=cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            mb_read_tile_id=cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            tile_id_smem=cutlass.Array(cutlass.Int32, CFG.SCHEDULER_STAGES * 8, alignment=16, space=cutlass.AddressSpace.smem),
            bidx_init=bidx,
            bidy_init=bidy,
            bidz_init=bidz,
        )
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
                if cutlass.const_expr(_PREDECODE_THD_SWA_SEGMENTS):
                    nvvm.mbarrier_init(sched.mb_decoded.subview(s), CFG.ONE_LANE)
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
            sStats_raw=sStats_raw,
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
            sStats_raw=sStats_raw,
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
            sStats_raw=sStats_raw,
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
        if cutlass.const_expr(_PREDECODE_THD_SWA_SEGMENTS):
            _scheduler_warp_loop_predecode(
                sched,
                is_cga_first_cta,
                cta_in_pair,
                n_q_supers,
                n_qh,
                n_batch,
                seq_kv_lens_tensor,
                seq_q_lens_addr,
                seqlen_q,
                seqlen_kv,
                qh_per_kh,
            )
        elif cutlass.const_expr(CFG.THD_VARLEN):
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
    row, whatever the caller left in the padding slots.  (Duplicated per flavor
    file on purpose — python/cudnn/AGENTS.md, CuTeDSL kernel bodies.)
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

    # The DSL TMA descriptor is element-typed; contiguous coord is in ELEMENTS not bytes.
    K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_N // CFG.CTA_MMA)
    V_COL_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_O // CFG.CTA_MMA)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            if cutlass.const_expr(CAN_HAVE_EMPTY_KV and IS_QO_ALIAS):
                # TMA-STG advances the Q/O alias gate for every tile, including
                # empty ones. Consume that transaction even though no Q reload
                # is needed, or the next nonempty tile observes stale parity.
                # mb_qo_slab_free is the RETURN edge: without it, runs of
                # empty tiles let STG (whose empty-tile O store depends only
                # on correction, never on this warp) race >= 2 alias phases
                # ahead of a delayed LDG, and mbarrier parity waits deadlock
                # at lead 2 — the observed zero-KV cluster hang.
                _wait_mbarrier(mb_q_reload[0], q_empty_phase)
                bars.mb_qo_slab_free[0].arrive()
                _wait_mbarrier(mb_q_reload[1], q_empty_phase)
                bars.mb_qo_slab_free[1].arrive()
                q_empty_phase = q_empty_phase ^ 1
        else:
            # Prologue interleave Q[0] -> K[first] -> Q[1] -> V[first] -> mainloop —
            # K load starts before Q[1] is issued and V before kv mainloop.
            kv_row_base = kv_left * CFG.TILE_N

            _wait_mbarrier(mb_q_reload[0], q_empty_phase)
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

            _wait_mbarrier(bars.mb_k_empty[kv_state.idx], kv_state.phase)
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

            _wait_mbarrier(mb_q_reload[1], q_empty_phase)
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

            _wait_mbarrier(bars.mb_v_empty[kv_state.idx], kv_state.phase)
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

            for kv_loop in cutlass.range(kv_left + cutlass.Int32(1), kv_right, 1, unroll=6):
                kv_row_base = kv_loop * CFG.TILE_N

                _wait_mbarrier(bars.mb_k_empty[kv_state.idx], kv_state.phase)
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

                _wait_mbarrier(bars.mb_v_empty[kv_state.idx], kv_state.phase)
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

        if cutlass.const_expr(_PREDECODE_THD_SWA_SEGMENTS):
            wait(sched.mb_decoded.subview(sched_state.idx), sched_state.phase)
            decoded_base = sched_state.idx * cutlass.Int32(11)
            q_super_idx = cute.arch.make_warp_uniform(sched.decoded_smem.subview(decoded_base).load())
            head_idx = cute.arch.make_warp_uniform(sched.decoded_smem.subview(decoded_base + cutlass.Int32(1)).load())
            batch_idx = cute.arch.make_warp_uniform(sched.decoded_smem.subview(decoded_base + cutlass.Int32(2)).load())
            is_valid_tile = cute.arch.make_warp_uniform(sched.decoded_smem.subview(decoded_base + cutlass.Int32(3)).load())
            split_idx = cutlass.Int32(0)
            kv_left = cute.arch.make_warp_uniform(sched.decoded_smem.subview(decoded_base + cutlass.Int32(6)).load())
            kv_right = cute.arch.make_warp_uniform(sched.decoded_smem.subview(decoded_base + cutlass.Int32(10)).load())
        else:
            # Raw scheduler payload loads avoid extra uniformization on this back-edge.
            _wait_ptr(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
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
        q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
        kv_head_idx = cute.arch.make_warp_uniform(head_idx if cutlass.const_expr(CFG.PACK_GQA) else head_idx // qh_per_kh)
        # q_row_base compute right after decode drives ptxas R2UR.
        q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE))
        q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV > 1):
            kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
        elif cutlass.const_expr(CFG.MASK_FLAGS != 0 and not _PREDECODE_THD_SWA_SEGMENTS):
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
            bounds_next = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, CFG.QH_PER_KH)
            kv_left = bounds_next.left
            kv_right = bounds_next.right
            if cutlass.const_expr(PAGED_KV):
                # The next tile may belong to another batch: refresh its live
                # page count with its bounds (PAGED_KV folds the predecoded
                # THD+SWA path off, so this branch is the one taken).
                n_pages_b = (eff_seqlen_kv + cutlass.Int32(PAGE_SIZE - 1)) // cutlass.Int32(PAGE_SIZE)

    # cga2: drain trailing empty mbar arrives so SMEM isn't torn down while
    # leader's multicast commits are still in-flight.
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        for _qs in cutlass.range_constexpr(CFG.TILES_Q):
            _wait_mbarrier(mb_q_reload[_qs], q_empty_phase)
        q_empty_phase = q_empty_phase ^ cutlass.Int32(1)
        for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
            _wait_mbarrier(bars.mb_k_empty[kv_state.idx], kv_state.phase)
            _wait_mbarrier(bars.mb_v_empty[kv_state.idx], kv_state.phase)
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
    # Alias-gate return edge (see the arrive site below): starts 0 — the
    # first wait consumes LDG's first real slab_free arrive.
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
        # KV split: partials are stacked split-major on the workspace BATCH axis
        # (extent B*SPLIT_KV), so the store needs no new descriptor — only a
        # shifted batch coord.  Folds to batch_idx at SPLIT_KV == 1.
        o_batch = _partial_batch(batch_idx, split_idx, n_batch)

        for qs in cutlass.range_constexpr(CFG.TILES_Q):
            _wait_mbarrier(bars.mb_o_full[qs], o_full_phase)
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
                    # DEAD unit (batch == n_batch, envelope grid — issue #552): no O
                    # rows exist and descriptor slot n_batch is never built, so skip
                    # the store; the barrier protocol below still runs.
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
            # for TMA-LDG to clobber with the next tile's Q[qs].  The
            # mb_qo_slab_free wait (return edge) throttles this arrive to at
            # most ONE phase ahead of LDG's alias-gate consumption — without
            # it, empty-tile runs (zero-KV varlen) let this warp lap a
            # delayed LDG by 2+ phases and mbarrier parity waits deadlock.
            # Consumer-side bootstrap: slab_free_phase starts 0, so the first
            # wait consumes LDG's first REAL arrive (LDG cannot be preceded —
            # its arrive follows its own alias wait, which this warp's tile-1
            # arrive has not yet advanced).
            if cutlass.const_expr(IS_QO_ALIAS):
                _wait_mbarrier(bars.mb_qo_slab_free[qs], slab_free_phase)
                bars.mb_q_o_alias[qs].arrive()

        o_full_phase = o_full_phase ^ 1
        if cutlass.const_expr(IS_QO_ALIAS):
            slab_free_phase = slab_free_phase ^ 1

        _wait_ptr(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
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

    _wait_mbarrier(bars.mb_tmem_dealloc, cutlass.Int32(0))
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
    # Only specializations that retain the legacy empty handshake use this.
    empty_mainloop_phase = cutlass.Int32(0)
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            if cutlass.const_expr(not CAN_HAVE_EMPTY_KV):
                _wait_mbarrier(bars.mb_empty_mainloop, empty_mainloop_phase)
                empty_mainloop_phase = empty_mainloop_phase ^ cutlass.Int32(1)
                elect_p = nvvm.elect_sync()
                bars.mb_bmm2_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_bmm2_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        else:
            # Prologue: BMM1[sub0], BMM1[sub1] for kv=kv_left.
            _wait_mbarrier(bars.mb_q_full[0], q_full_phase)
            _wait_mbarrier(bars.mb_k_full[kv_state.idx], kv_state.phase)
            desc_K = sK[kv_state.idx].desc()
            if cutlass.const_expr(_DENSE_CAUSAL_PUBLIC_EXP2_MIX):
                desc_K = desc_opaque(desc_K)
            mma_ss(bmm1_desc, desc_Q0, desc_K, (tmem_raw.subview(LAYOUT.S0_OFF)), elect_once=CFG.THD_VARLEN)
            elect_p = nvvm.elect_sync()
            bars.mb_bmm1_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

            _wait_mbarrier(bars.mb_q_full[1], q_full_phase)
            mma_ss(bmm1_desc, desc_Q1, desc_K, (tmem_raw.subview(LAYOUT.S1_OFF)), elect_once=CFG.THD_VARLEN)
            elect_p = nvvm.elect_sync()
            bars.mb_bmm1_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_k_empty[kv_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

            q_full_phase = q_full_phase ^ 1

            # Mainloop kv = kv_left+1 .. kv_right-1 (empty when n_kv == 1)
            for kv_loop in cutlass.range(kv_left + cutlass.Int32(1), kv_right, 1, unroll=1):
                old_state = kv_state
                kv_state = advance(kv_state, CFG.STAGES_KV)

                _wait_mbarrier(bars.mb_v_full[old_state.idx], old_state.phase)
                desc_V = sV[old_state.idx].desc()
                is_not_first_bmm2 = cutlass.Boolean(kv_loop != (kv_left + cutlass.Int32(1)))

                # BMM2 sub-tile 0
                bmm2_issue = nvvm.elect_sync() if cutlass.const_expr(_REUSE_BMM2_ISSUE_ELECTION) else None
                _wait_mbarrier(bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 0], bmm2_ready_phase)
                accum_b2 = is_not_first_bmm2
                for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                    mma_ts_step(
                        bmm2_desc,
                        (tmem_raw.subview(LAYOUT.P0_OFF)),
                        desc_V,
                        (tmem_raw.subview(LAYOUT.O0_OFF)),
                        local_k,
                        accum_b2,
                        issue_mma=bmm2_issue,
                    )
                    accum_b2 = cutlass.Boolean(True)
                if cutlass.const_expr(CFG.N_BMM2_CHUNKS == 2):
                    _wait_mbarrier(bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 1], bmm2_ready_phase)
                    for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                        mma_ts_step(
                            bmm2_desc,
                            (tmem_raw.subview(LAYOUT.P0_OFF)),
                            desc_V,
                            (tmem_raw.subview(LAYOUT.O0_OFF)),
                            NUM_KPHASES_PV_PER_CHUNK + local_k,
                            cutlass.Boolean(True),
                            issue_mma=bmm2_issue,
                        )
                elect_p = nvvm.elect_sync()
                bars.mb_bmm2_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

                # BMM1 sub 0 for next kv
                _wait_mbarrier(bars.mb_k_full[kv_state.idx], kv_state.phase)
                desc_K = sK[kv_state.idx].desc()
                if cutlass.const_expr(_DENSE_CAUSAL_PUBLIC_EXP2_MIX):
                    desc_K = desc_opaque(desc_K)
                mma_ss(bmm1_desc, desc_Q0, desc_K, (tmem_raw.subview(LAYOUT.S0_OFF)), elect_once=CFG.THD_VARLEN)
                elect_p = nvvm.elect_sync()
                bars.mb_bmm1_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

                # BMM2 sub-tile 1
                bmm2_issue = nvvm.elect_sync() if cutlass.const_expr(_REUSE_BMM2_ISSUE_ELECTION) else None
                _wait_mbarrier(bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 0], bmm2_ready_phase)
                accum_b2 = is_not_first_bmm2
                for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                    mma_ts_step(
                        bmm2_desc,
                        (tmem_raw.subview(LAYOUT.P1_OFF)),
                        desc_V,
                        (tmem_raw.subview(LAYOUT.O1_OFF)),
                        local_k,
                        accum_b2,
                        issue_mma=bmm2_issue,
                    )
                    accum_b2 = cutlass.Boolean(True)
                if cutlass.const_expr(CFG.N_BMM2_CHUNKS == 2):
                    _wait_mbarrier(bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 1], bmm2_ready_phase)
                    for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                        mma_ts_step(
                            bmm2_desc,
                            (tmem_raw.subview(LAYOUT.P1_OFF)),
                            desc_V,
                            (tmem_raw.subview(LAYOUT.O1_OFF)),
                            NUM_KPHASES_PV_PER_CHUNK + local_k,
                            cutlass.Boolean(True),
                            issue_mma=bmm2_issue,
                        )
                elect_p = nvvm.elect_sync()
                bars.mb_bmm2_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_v_empty[old_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

                # BMM1 sub 1 for next kv
                mma_ss(bmm1_desc, desc_Q1, desc_K, (tmem_raw.subview(LAYOUT.S1_OFF)), elect_once=CFG.THD_VARLEN)
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

            _wait_mbarrier(bars.mb_v_full[kv_state.idx], kv_state.phase)
            desc_V = sV[kv_state.idx].desc()
            is_not_first_bmm2_epi = cutlass.Boolean((kv_right - kv_left) != cutlass.Int32(1))

            bmm2_issue = nvvm.elect_sync() if cutlass.const_expr(_REUSE_BMM2_ISSUE_ELECTION) else None
            _wait_mbarrier(bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 0], bmm2_ready_phase)
            accum_b2 = is_not_first_bmm2_epi
            for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                mma_ts_step(
                    bmm2_desc,
                    (tmem_raw.subview(LAYOUT.P0_OFF)),
                    desc_V,
                    (tmem_raw.subview(LAYOUT.O0_OFF)),
                    local_k,
                    accum_b2,
                    issue_mma=bmm2_issue,
                )
                accum_b2 = cutlass.Boolean(True)
            if cutlass.const_expr(CFG.N_BMM2_CHUNKS == 2):
                _wait_mbarrier(bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 1], bmm2_ready_phase)
                for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                    mma_ts_step(
                        bmm2_desc,
                        (tmem_raw.subview(LAYOUT.P0_OFF)),
                        desc_V,
                        (tmem_raw.subview(LAYOUT.O0_OFF)),
                        NUM_KPHASES_PV_PER_CHUNK + local_k,
                        cutlass.Boolean(True),
                        issue_mma=bmm2_issue,
                    )
            elect_p = nvvm.elect_sync()
            bars.mb_bmm2_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

            bmm2_issue = nvvm.elect_sync() if cutlass.const_expr(_REUSE_BMM2_ISSUE_ELECTION) else None
            _wait_mbarrier(bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 0], bmm2_ready_phase)
            accum_b2 = is_not_first_bmm2_epi
            for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                mma_ts_step(
                    bmm2_desc,
                    (tmem_raw.subview(LAYOUT.P1_OFF)),
                    desc_V,
                    (tmem_raw.subview(LAYOUT.O1_OFF)),
                    local_k,
                    accum_b2,
                    issue_mma=bmm2_issue,
                )
                accum_b2 = cutlass.Boolean(True)
            if cutlass.const_expr(CFG.N_BMM2_CHUNKS == 2):
                _wait_mbarrier(bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 1], bmm2_ready_phase)
                for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                    mma_ts_step(
                        bmm2_desc,
                        (tmem_raw.subview(LAYOUT.P1_OFF)),
                        desc_V,
                        (tmem_raw.subview(LAYOUT.O1_OFF)),
                        NUM_KPHASES_PV_PER_CHUNK + local_k,
                        cutlass.Boolean(True),
                        issue_mma=bmm2_issue,
                    )
            elect_p = nvvm.elect_sync()
            bars.mb_bmm2_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_v_empty[kv_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)

            bmm2_ready_phase = bmm2_ready_phase ^ 1
            kv_state = advance(kv_state, CFG.STAGES_KV)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        _wait_ptr(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
            _nq, _nh, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
            is_valid_tile = nxt_v & cutlass.Int32(1)
        else:
            nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
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

    _wait_mbarrier(bars.mb_tmem_dealloc, cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _softmax_kv_body(
    apply_mask: bool,
    may_need_padding: bool,
    body_mask_flags: int,
    sub_tile_id: int,
    kv_loop,
    s_addr_base,
    p_addr_base,
    sStats_raw,
    tid_in_wg,
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
    apply_mask=False uses manual tcgen05.ld + software row-max;
    apply_mask=True uses tcgen05.ld + apply_mask_chunk + software row-max
    (HW max can't observe NEG_INFINITY written after the load).

    total_max runs in scaled (log2) units and starts at -inf; total_max_safe
    is its 0-substituted companion (see row_max_for_exp2), carried so alpha
    can be exp2(prev_safe - new_safe) with the substitution on both operands.

    Returns updated (total_max, total_max_safe, total_sum, bmm1_phase,
    stat_empty_phase).
    """
    CHUNK = _SOFTMAX_CHUNK
    # fp16/bf16 pack 2 probs per FP32 TMEM cell (CHUNK//2); TF32 is 1:1 (CHUNK).
    P_COLS_PER_CHUNK = CHUNK if IS_TF32 else CHUNK // 2
    N_CHUNKS = CFG.N_BMM2_CHUNKS
    RESCALE_THRESHOLD = cutlass.Float32(CFG.RESCALE_THRESHOLD)

    if cutlass.const_expr(apply_mask):
        kv_col_base = kv_loop * cutlass.Int32(CFG.TILE_N)
        # Bottom-right causal: runtime SKV-SQ diagonal offset (folds out when
        # CFG.BOTTOM_RIGHT is 0 - top-left masking is unchanged).
        causal_diag = eff_seqlen_kv - eff_seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else None

    _wait_mbarrier(bars.mb_bmm1_done[sub_tile_id], bmm1_phase)
    bmm1_phase = bmm1_phase ^ 1

    # apply_mask is a Python bool — wrap in cutlass.const_expr so the DSL folds
    # at trace time instead of staging cf.if (the two arms produce Vectors
    # built via different MLIR op sequences and the tracer would error).
    if cutlass.const_expr(apply_mask):
        # Python comprehensions (not for+append) so the tracer sees fully-formed lists.
        raw_chunks = [
            nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * CHUNK), cutlass.Float32),
                num=CHUNK,
            )
            for c in range(N_CHUNKS)
        ]
        if cutlass.const_expr(_PADDED_CAUSAL):
            mask_q_abs = q_abs
            if cutlass.const_expr(CFG.BOTTOM_RIGHT):
                mask_q_abs = mask_q_abs + causal_diag
            mask_q_abs = cute.math.min(mask_q_abs, eff_seqlen_kv - cutlass.Int32(1))
            # Causal + padded as ONE causal edge: kv > min(q (+ diag), seq_kv - 1) masks
            # exactly the causal OR padded set, so the chunk sees a top-left causal mask
            # anchored at mask_q_abs (WINDOW_RIGHT == 0 in this arm by _PADDED_CAUSAL).
            chunks_S = [
                apply_mask_chunk(
                    raw_chunks[c],
                    mask_q_abs,
                    kv_col_base + cutlass.Int32(c * CHUNK),
                    eff_seqlen_kv,
                    0,
                    MASK_CAUSAL,
                    N=CHUNK,
                    mask_value=float("-inf"),
                    window_right=CFG.WINDOW_RIGHT,
                )
                for c in range(N_CHUNKS)
            ]
        elif cutlass.const_expr(CFG.MASK_FLAGS == MASK_CAUSAL and CFG.BOTTOM_RIGHT == 0):
            # Top-left causal (+ the compile-time right band, cuDNN diagonal_band_right_bound).
            chunks_S = [
                apply_mask_chunk(
                    raw_chunks[c],
                    q_abs,
                    kv_col_base + cutlass.Int32(c * CHUNK),
                    eff_seqlen_kv,
                    0,
                    MASK_CAUSAL,
                    N=CHUNK,
                    mask_value=float("-inf"),
                    window_right=CFG.WINDOW_RIGHT,
                )
                for c in range(N_CHUNKS)
            ]
        elif cutlass.const_expr(CFG.MASK_FLAGS == MASK_CAUSAL and CFG.BOTTOM_RIGHT != 0):
            # Bottom-right causal: the diagonal sits causal_diag = S_kv - S_q columns right of top-left.
            chunks_S = [
                apply_mask_chunk(
                    raw_chunks[c],
                    q_abs,
                    kv_col_base + cutlass.Int32(c * CHUNK),
                    eff_seqlen_kv,
                    0,
                    MASK_CAUSAL,
                    N=CHUNK,
                    bottom_right=CFG.BOTTOM_RIGHT,
                    causal_diag=causal_diag,
                    mask_value=float("-inf"),
                    window_right=CFG.WINDOW_RIGHT,
                )
                for c in range(N_CHUNKS)
            ]
        else:
            chunk_mask_flags = body_mask_flags & ~MASK_PADDED if CFG.MASK_FLAGS & MASK_SWA else body_mask_flags
            chunks_S = [
                apply_mask_chunk(
                    raw_chunks[c],
                    q_abs,
                    kv_col_base + cutlass.Int32(c * CHUNK),
                    eff_seqlen_kv,
                    CFG.WINDOW_LEFT,
                    chunk_mask_flags,
                    N=CHUNK,
                    bottom_right=CFG.BOTTOM_RIGHT,
                    causal_diag=causal_diag,
                    mask_value=float("-inf"),
                    window_right=CFG.WINDOW_RIGHT,
                )
                for c in range(N_CHUNKS)
            ]
        if cutlass.const_expr(may_need_padding and (CFG.MASK_FLAGS & MASK_PADDED) and (CFG.MASK_FLAGS & MASK_SWA)):
            chunks_S = [
                _apply_padding_mask_if_needed(
                    chunks_S[c],
                    kv_col_base + cutlass.Int32(c * CHUNK),
                    eff_seqlen_kv,
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
        # Pure MASK_NONE avoids the x32 lowering, which spills under DSL 4.7.
        # Causal kernels retain x32 for their unmasked middle segment.
        from cudnn.frost.tile_dsl.regtile import vec_concat

        if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
            raw_chunks = [
                nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * CHUNK), cutlass.Float32),
                    num=CHUNK,
                )
                for c in range(N_CHUNKS)
            ]
        else:
            LOAD_CHUNK = 32
            raw_chunks = [
                nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * LOAD_CHUNK), cutlass.Float32),
                    num=LOAD_CHUNK,
                )
                for c in range(CFG.TILE_N // LOAD_CHUNK)
            ]
        reg_S_vec = vec_concat(raw_chunks)
        current_max_unscaled = row_max_reduction(reg_S_vec)

    # Pass size=CFG.TILE_N explicitly — Vector.shape[0] is an MLIR value
    # (not Python int) for vec_concat-built vectors, so auto-detect can't recover the length.
    reg_S = RegTile(reg_S_vec, size=CFG.TILE_N)
    current_max = current_max_unscaled * scale_log2  # -inf when the whole iteration is masked

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

    _wait_mbarrier(bars.mb_stat_empty[sub_tile_id], stat_empty_phase)
    stat_empty_phase = stat_empty_phase ^ 1
    stats_base = cutlass.Int32(sub_tile_id * 2 * CFG.TILE_M)
    sStats_raw.subview(stats_base + tid_in_wg).store(alpha)
    bars.mb_stat_full[sub_tile_id].arrive()

    # Rescale full reg_S in one vector op — emits same FFMA2 sequence as explicit half-tile rescales.
    reg_S = reg_S * scale_log2 - total_max_safe

    # Chunk 0 manual unroll — the DSL's @cute.jit tracer makes the loop iter an
    # MLIR value, breaking Python slice.indices() math inside RegTile[].
    chunk_S_0 = reg_S[0:CHUNK].vec
    # exp2 split: the _E2E_PAIRS[chunk] columns on the FMA pipe, the rest on MUFU (see _E2E_FREQ).  Gated per arch
    # at trace time (_E2E_ENABLED): off, this chunk is the plain MUFU exp2 -- the develop spelling.
    if cutlass.const_expr(_E2E_ENABLED):
        chunk_P_0 = exp2_mixed(chunk_S_0, _E2E_PAIRS[0], CHUNK)
    else:
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
    p1_sum_pair = None
    if cutlass.const_expr(N_CHUNKS == 2):
        chunk_S_1 = reg_S[CHUNK : 2 * CHUNK].vec
        # Split ON: the module-wide pattern on every mask arm.  Split OFF: develop's spelling -- the dense
        # top-left causal arm's tail emulation (_exp2_mixed_late), the plain MUFU exp2 everywhere else.
        if cutlass.const_expr(_E2E_ENABLED):
            deferred_P_1 = exp2_mixed(chunk_S_1, _E2E_PAIRS[1], CHUNK)
        else:
            if cutlass.const_expr(_DENSE_CAUSAL_PUBLIC_EXP2_MIX):
                deferred_P_1 = _exp2_mixed_late(chunk_S_1)
            else:
                deferred_P_1 = cute.math.exp2(chunk_S_1, fastmath=True)
        if cutlass.const_expr(not _DENSE_CAUSAL_PUBLIC_EXP2_MIX):
            p1_sum_pair = row_reduction_pair(deferred_P_1)
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
        if cutlass.const_expr(_DENSE_CAUSAL_PUBLIC_EXP2_MIX):
            p1_sum_pair = row_reduction_pair(deferred_P_1)

    new_p_sum_pair = hoisted_sum
    if cutlass.const_expr(N_CHUNKS == 2):
        new_p_sum_pair = new_p_sum_pair + p1_sum_pair
    sum_lo, sum_hi = cute.arch.fma_packed_f32x2(
        (total_sum[0], total_sum[1]),
        (alpha, alpha),
        (new_p_sum_pair[0], new_p_sum_pair[1]),
    )
    total_sum = cutlass.Vector.from_elements((sum_lo, sum_hi), cutlass.Float32)

    return total_max, total_max_safe, total_sum, bmm1_phase, stat_empty_phase


class _SoftmaxKvContext(NamedTuple):
    sub_tile_id: object
    s_addr_base: object
    p_addr_base: object
    sStats_raw: object
    tid_in_wg: object
    bars: object
    q_abs: object
    eff_seqlen_kv: object
    eff_seqlen_q: object
    scale_log2: object
    leader_cta_id: object


@cute.jit
def _softmax_kv_range(may_need_mask: bool, may_need_padding: bool, mask_flags: int, begin, end, context, state):
    total_max, total_max_safe, total_sum, bmm1_phase, stat_empty_phase = state
    for kv_loop in cutlass.range(begin, end, 1, unroll=1):
        total_max, total_max_safe, total_sum, bmm1_phase, stat_empty_phase = _softmax_kv_body(
            may_need_mask,
            may_need_padding,
            mask_flags,
            context.sub_tile_id,
            kv_loop,
            context.s_addr_base,
            context.p_addr_base,
            context.sStats_raw,
            context.tid_in_wg,
            context.bars,
            context.q_abs,
            context.eff_seqlen_kv,
            context.eff_seqlen_q,
            context.scale_log2,
            total_max,
            total_max_safe,
            total_sum,
            bmm1_phase,
            stat_empty_phase,
            context.leader_cta_id,
        )
    return total_max, total_max_safe, total_sum, bmm1_phase, stat_empty_phase


@cute.jit
def _softmax_masked_kv_loops(
    one_sided_swa: bool,
    bounds,
    swa_left_pad_start,
    swa_left_end,
    swa_right_start,
    context,
    state,
):
    if cutlass.const_expr(one_sided_swa):
        regions = (
            (True, False, MASK_SWA, bounds.left, swa_left_pad_start),
            (True, True, MASK_SWA, swa_left_pad_start, swa_left_end),
            (False, False, CFG.MASK_FLAGS, swa_left_end, swa_right_start),
            (True, True, MASK_CAUSAL, swa_right_start, bounds.right),
        )
    else:
        regions = (
            (True, False, CFG.MASK_FLAGS, bounds.left, bounds.unmasked_lo),
            (False, False, CFG.MASK_FLAGS, bounds.unmasked_lo, bounds.unmasked_hi),
            (True, True, CFG.MASK_FLAGS, bounds.unmasked_hi, bounds.right),
        )
    for may_need_mask, may_need_padding, mask_flags, begin, end in regions:
        state = _softmax_kv_range(
            may_need_mask,
            may_need_padding,
            mask_flags,
            begin,
            end,
            context,
            state,
        )
    return state


@cute.jit
def _softmax_warp_group(
    sub_tile_id: int,
    seqlen_q,
    seqlen_kv,
    scale_log2: cutlass.Float32,
    tmem_ptr_i32,
    sQ,
    sStats_raw,
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

    # Phase trackers persist across tile boundaries (barriers don't reset).
    bmm1_phase = cutlass.Int32(0)
    stat_empty_phase = cutlass.Int32(1)  # bootstrap pre-armed at phase 1 so first wait passes immediately
    # total_sum kept as Vector[Float32, 2] (even/odd partials) so per-iter update lowers to packed FMUL2 + FADD2.
    total_max = NEG_INF
    total_max_safe = NEG_INF
    total_sum = cutlass.Vector.from_elements(
        (cutlass.Float32(0.0), cutlass.Float32(0.0)),
        cutlass.Float32,
    )

    q_super_idx, _head_idx, batch_idx, split_idx = _decode_initial_split(
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
    initial_segments = (
        _swa_segment_bounds(
            q_super_idx,
            eff_seqlen_q,
            eff_seqlen_kv,
            cta_in_pair,
            seq_q_lens_addr,
            batch_idx,
        )
        if cutlass.const_expr(_PREDECODE_THD_SWA_SEGMENTS)
        else (bounds.left, bounds.unmasked_lo, bounds.unmasked_lo, bounds.unmasked_hi, bounds.right)
    )
    swa_left_pad_start = initial_segments[1]
    swa_left_end = initial_segments[2]
    swa_right_start = initial_segments[3]

    softmax_wg_base_const = CFG.SOFTMAX_WG0_BASE if sub_tile_id == 0 else CFG.SOFTMAX_WG1_BASE
    tid_in_wg = cute.arch.thread_idx()[0] - cutlass.Int32(softmax_wg_base_const * 32)
    tmem_base_softmax = tmem_ptr_i32.load()
    s_addr_base_softmax = tmem_base_softmax + cutlass.Int32(tmem_S_off)
    p_addr_base_softmax = tmem_base_softmax + cutlass.Int32(tmem_P_off)
    stats_base = cutlass.Int32(sub_tile_id * 2 * CFG.TILE_M)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        total_max = NEG_INF
        total_max_safe = NEG_INF
        total_sum = cutlass.Vector.from_elements(
            (cutlass.Float32(0.0), cutlass.Float32(0.0)),
            cutlass.Float32,
        )
        # PackGQA: q_abs is the row's TOKEN index (row // G): every mask
        # predicate downstream is a token-space compare, and all G rows of one
        # token share it.
        q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE)
        q_abs = q_row_coord + cutlass.Int32(sub_tile_id * TOKENS_PER_TILE) + (tid_in_wg // cutlass.Int32(HEADS_PER_TILE))
        # 3-segment kv loop: LEFT-masked / fully-unmasked / RIGHT-masked.
        # At MASK_NONE bounds collapse so the two masked sub-loops have empty range and fold out.
        if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
            for kv_loop in cutlass.range(bounds.left, bounds.right, 1, unroll=1):
                total_max, total_max_safe, total_sum, bmm1_phase, stat_empty_phase = _softmax_kv_body(
                    False,
                    False,
                    CFG.MASK_FLAGS,
                    sub_tile_id,
                    kv_loop,
                    s_addr_base_softmax,
                    p_addr_base_softmax,
                    sStats_raw,
                    tid_in_wg,
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
            context = _SoftmaxKvContext(
                sub_tile_id,
                s_addr_base_softmax,
                p_addr_base_softmax,
                sStats_raw,
                tid_in_wg,
                bars,
                q_abs,
                eff_seqlen_kv,
                eff_seqlen_q,
                scale_log2,
                leader_cta_id,
            )
            state = (total_max, total_max_safe, total_sum, bmm1_phase, stat_empty_phase)
            if cutlass.const_expr(_SWA_ONE_SIDED_GEOMETRY):
                if cutlass.const_expr(not _PREDECODE_THD_SWA_SEGMENTS):
                    # Keep dense segment arithmetic in this warp: extracting it
                    # changes CuTeDSL lowering on the hot dense path.
                    cga_q_row_coord = (q_super_idx - cta_in_pair) * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE)
                    causal_diag = eff_seqlen_kv - eff_seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else cutlass.Int32(0)
                    lower_anchor = cga_q_row_coord + causal_diag + cutlass.Int32(_MASK_TOKENS_PER_CGA - 1 - CFG.WINDOW_LEFT)
                    lower_end_if_positive = (lower_anchor + cutlass.Int32(CFG.TILE_N - 1)) // cutlass.Int32(CFG.TILE_N)
                    lower_end_raw = cutlass.Int32(
                        arith.select(
                            (lower_anchor > cutlass.Int32(0)).ir_value(),
                            lower_end_if_positive.ir_value(),
                            cutlass.Int32(0).ir_value(),
                        )
                    )
                    left_end = cute.math.min(cute.math.max(lower_end_raw, bounds.left), bounds.right)
                    pad_start = eff_seqlen_kv // cutlass.Int32(CFG.TILE_N) if cutlass.const_expr(CFG.MASK_FLAGS & MASK_PADDED) else bounds.right
                    left_pad_start = cute.math.min(cute.math.max(pad_start, bounds.left), left_end)
                    causal_start = (cga_q_row_coord + causal_diag + cutlass.Int32(CFG.WINDOW_RIGHT)) // cutlass.Int32(CFG.TILE_N)
                    right_start_raw = cute.math.min(causal_start, pad_start)
                    right_start = cute.math.min(
                        cute.math.max(cute.math.max(right_start_raw, left_end), bounds.left),
                        bounds.right,
                    )
                    swa_left_pad_start = left_pad_start
                    swa_left_end = left_end
                    swa_right_start = right_start
            else:
                swa_left_pad_start = bounds.unmasked_lo
                swa_left_end = bounds.unmasked_lo
                swa_right_start = bounds.unmasked_hi

            total_max, total_max_safe, total_sum, bmm1_phase, stat_empty_phase = _softmax_masked_kv_loops(
                _SWA_ONE_SIDED_GEOMETRY,
                bounds,
                swa_left_pad_start,
                swa_left_end,
                swa_right_start,
                context,
                state,
            )

        if cutlass.const_expr(not CAN_HAVE_EMPTY_KV) or (bounds.right > bounds.left):
            # Only real mainloops own a stats transaction. Empty tiles bypass
            # this phase entirely and are materialized by the correction warp.
            total_sum_scalar = total_sum[0] + total_sum[1]
            _wait_mbarrier(bars.mb_stat_empty[sub_tile_id], stat_empty_phase)
            stat_empty_phase = stat_empty_phase ^ 1
            sStats_raw.subview(stats_base + tid_in_wg).store(total_max_safe)
            sStats_raw.subview(stats_base + cutlass.Int32(CFG.TILE_M) + tid_in_wg).store(total_sum_scalar)
            bars.mb_stat_full[sub_tile_id].arrive()

        (
            q_super_idx,
            _head_idx,
            batch_idx,
            split_idx,
            is_valid_tile,
            eff_seqlen_kv,
            eff_seqlen_q,
            bounds,
            swa_left_pad_start,
            swa_left_end,
            swa_right_start,
        ) = _softmax_next_payload(
            sched,
            sched_state,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
            seq_q_lens_addr,
            seqlen_q,
            seqlen_kv,
            qh_per_kh,
        )
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


@cute.jit
def _correction_warp_group(
    seqlen_q,
    seqlen_kv,
    sO,
    sStats_raw,
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
    O_CHUNK = 8
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
    bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, CFG.QH_PER_KH)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        # Iter-0 lifted out of kv loop: MMA's iter-0 BMM2 uses init_d=False
        # to overwrite O garbage, so alpha-rescale is unnecessary in iter 0.
        if bounds.right > bounds.left:
            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                bars.mb_bmm2_ready[qs * CFG.N_BMM2_CHUNKS + 0].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                _wait_mbarrier(bars.mb_stat_full[qs], stat_full_phase)
                bars.mb_stat_empty[qs].arrive()
            stat_full_phase = stat_full_phase ^ 1
        else:
            if cutlass.const_expr(not CAN_HAVE_EMPTY_KV):
                bars.mb_empty_mainloop.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

        for kv_loop in cutlass.range(bounds.left + cutlass.Int32(1), bounds.right, 1, unroll=1):
            tmem_base_iter = tmem_ptr_i32.load()
            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                tmem_O_off = LAYOUT.O0_OFF if qs == 0 else LAYOUT.O1_OFF

                _wait_mbarrier(bars.mb_stat_full[qs], stat_full_phase)

                stats_base = cutlass.Int32(qs * 2 * CFG.TILE_M)
                alpha = sStats_raw.subview(stats_base + tid_in_wg).load()

                # all_alpha_one ballot: when every lane has alpha==1.0 the entire rescale loop skips.
                alpha_is_one = alpha == cutlass.Float32(1.0)
                all_alpha_one = vote_sync(0xFFFFFFFF, alpha_is_one, VoteSync.ALL)

                bars.mb_stat_empty[qs].arrive()

                _wait_mbarrier(bars.mb_bmm2_done[qs], bmm2_done_phase)

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
            tmem_O_off = LAYOUT.O0_OFF if qs == 0 else LAYOUT.O1_OFF

            total_max_scaled = cutlass.Float32(float("-inf"))
            total_sum = cutlass.Float32(0.0)
            if cutlass.const_expr(not CAN_HAVE_EMPTY_KV) or (bounds.right > bounds.left):
                # Wait the (total_max, total_sum_final) publish from softmax's epilogue.
                _wait_mbarrier(bars.mb_stat_full[qs], stat_full_phase)

                stats_base = cutlass.Int32(qs * 2 * CFG.TILE_M)
                total_max_scaled = sStats_raw.subview(stats_base + tid_in_wg).load()
                total_sum = sStats_raw.subview(stats_base + cutlass.Int32(CFG.TILE_M) + tid_in_wg).load()

                bars.mb_stat_empty[qs].arrive()

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
                # Keyless row (total_sum == 0; the softmax masks with -inf and publishes a
                # 0-substituted max): the sink is the row's whole mass, O := 0 / LSE := sink.
                # Select new_max := sink and scale := 0 for it instead of computing the fold
                # (exp(sink - 0) underflows to a zero denominator for sink <= -104 -> O = NaN,
                # LSE = -inf; exp(0 - sink) can overflow, 0 * inf is NaN) -- the arithmetic is
                # spelled out at the d128 kernel's sink fold.  Rows with keys are unchanged.
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
                # Top-left causal and no-mask dense tiles always have at least
                # one valid KV per live row, so fold this select out there.
                if cutlass.const_expr((CFG.MASK_FLAGS & (MASK_PADDED | MASK_SWA)) != 0 or CFG.BOTTOM_RIGHT):
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
                # rank-3 [1, QH, head_stride] or token-major rank-2 [T, QH] —
                # bound by per-sequence Q len S_q_b.
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
                    # This chunk's LSE goes to its own split-major slot, matching
                    # where TMA-STG put the chunk's O.  The pair (O_s, lse_s) is
                    # everything the combine needs.
                    lse_batch = _partial_batch(batch_idx, split_idx, n_batch)
                    lse_arr[lse_batch, row_head_idx, q_row_global] = lse_val

            sO_sub_base = sO[qs].base

            if cutlass.const_expr(_FP32_PARTIALS):
                # fp32 partials: the accumulator goes straight to the workspace,
                # bypassing the SMEM O tile and its TMA store.
                # The O TMEM loads need BMM2 retired -- the staged branch waits for
                # exactly this before its own loads.
                # An empty split chunk never ran BMM2, so nothing will arrive on
                # mb_bmm2_done and waiting on it hangs.  Its partial is never read
                # either: the chunk's lse is -inf, which the combine skips.
                if cutlass.const_expr(not CAN_HAVE_EMPTY_KV) or (bounds.right > bounds.left):
                    _wait_mbarrier(bars.mb_bmm2_done[qs], bmm2_done_phase)
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
                if cutlass.const_expr(not CAN_HAVE_EMPTY_KV) or (bounds.right > bounds.left):
                    # The final stats live in the S_acc slot and can be consumed
                    # before BMM2 retires. Only the O TMEM loads need this wait.
                    _wait_mbarrier(bars.mb_bmm2_done[qs], bmm2_done_phase)

                for chunk_idx in cutlass.range_constexpr(N_CHUNKS_O):
                    o_fp16 = cutlass.Vector.from_elements(
                        tuple(STORAGE_DTYPE(0.0) for _ in range(O_CHUNK)),
                        STORAGE_DTYPE,
                    )
                    if cutlass.const_expr(not CAN_HAVE_EMPTY_KV) or (bounds.right > bounds.left):
                        o_addr = tmem_base_epi + cutlass.Int32(tmem_O_off + chunk_idx * O_CHUNK)
                        o_chunk = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                            num=O_CHUNK,
                        )
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                        o_scaled = vec_scale_pair(o_chunk, inv_sum, O_CHUNK)
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
                        _wait_mbarrier(bars.mb_o_empty[qs], o_empty_phase)
                    smem_ptr.store_swizzled(o_fp16, alignment=64, swizzle=_O_SMEM_SWIZZLE)

                # fence_proxy needed before TMA reads SMEM written by tcgen05_st (via store_swizzled).
                nvvm.fence_proxy("async.shared", space="cta")

            bars.mb_o_full[qs].arrive()

        if cutlass.const_expr(not CAN_HAVE_EMPTY_KV) or (bounds.right > bounds.left):
            stat_full_phase = stat_full_phase ^ 1
            bmm2_done_phase = bmm2_done_phase ^ 1
        o_empty_phase = o_empty_phase ^ 1

        _wait_ptr(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
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
        paged=PAGED_KV,
        page_size=PAGE_SIZE,
        block_table_ptr=block_table_ptr,
        block_table_v_ptr=block_table_v_ptr,
        table_strides=table_strides,
        n_pages=n_pages,
    )
    # Paged KV: HND storage viewed as [page, row, H_kv, D] has the row stride
    # BELOW the head stride, so its descriptor lists dims innermost-first as
    # (D, row, H_kv, page) and the TMA-LDG warp swaps its (head, row) coords;
    # NHD is the dense BSHD order with batch -> page.
    stride_order = (3, 2, 1, 0)
    if cutlass.const_expr(paged_hnd):
        kv_stride_order = (3, 1, 2, 0)
    else:
        kv_stride_order = stride_order
    _O_GRANU_ELEMS = CFG.O_SWZ_BYTES // CFG.BPE
    qk_box_q = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, TMA_QK_GRANU_ELEMS)
    qk_box_k = (1, K_BOX_ROWS, 1, TMA_QK_GRANU_ELEMS)
    vo_box_v = (1, V_BOX_ROWS, 1, TMA_VO_GRANU_ELEMS)
    vo_box_o = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, _O_GRANU_ELEMS)

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
        stride_order=kv_stride_order,
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
        stride_order=kv_stride_order,
        swizzle=_v_tma_swz,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    # Unused under fp32 partials, but it must still BUILD: an fp32 element
    # doubles the box's inner byte width past what the O swizzle allows.
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

    # Each cluster pair (CTA_MMA CTAs) collectively covers TILE_M*TILES_Q*CTA_MMA Q rows;
    # without the cluster-wide divisor cga2 over-launches and OOB clusters collide in GMEM.
    # PackGQA: SQ*G packed rows per packed head, and QH/G packed heads.
    rows_per_cluster = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA
    q_clusters = (SQ * HEADS_PER_TILE + rows_per_cluster - 1) // rows_per_cluster
    grid_q_supers = q_clusters * CFG.CTA_MMA
    q_supers = grid_q_supers
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD: build the [kv|cu_q|cu_k] metadata + per-batch O descriptor
        # array DEVICE-side (reuse tma_o_desc over the packed [1,T,QH,D_v] O
        # as base), then launch the PERSISTENT grid: the adapter hands down
        # n_thd_units already capped to what the device holds resident,
        # min(plan-time envelope, SMs / CGA_SIZE), NOT the envelope itself
        # (issue #618). It doubles as the claim counter's seed: cluster c runs
        # unit c off its blockIdx, then pulls from the counter, so the grid and
        # the seed must be the same number. Dispatching past the live total
        # stays safe — such a unit decodes the batch == n_batch sentinel and
        # drains without loads or stores. grid_x = n_thd_units * CGA_M.
        # Works at cga1 (CGA_M=1).
        # ENVELOPE: the packed-O row stride is QH * ACTUAL d_v (o_tensor's
        # static inner extent), not QH * TILE_O — the per-batch descriptor
        # bases must step in real rows or every batch >= 1 lands OOB.
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
            cutlass.Int32(o_tensor.stride[1]),
            cutlass.Int32(CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA),
            n_thd_units,  # persistent cluster count; also seeds the claim counter
            not PAGED_KV,  # clamp_kv: paged pools have no packed KV total to clamp to
        ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
        grid_shape = (n_thd_units * cutlass.Int32(CFG.CGA_M), cutlass.Int32(1), cutlass.Int32(1))
    else:
        # Grid Python-folds on Cfg constant (avoids DSL if staging).
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
    8 at 2 bytes/elem — checked here for the envelope, by the adapter for
    declared strides."""
    _cache_key = _template_key(globals(), locals(), "compile")
    if not (0 < d_qk <= CFG.TILE_K and 0 < d_v <= CFG.TILE_O):
        raise ValueError(f"d192 envelope: need 0 < d_qk <= {CFG.TILE_K} and 0 < d_v <= {CFG.TILE_O}; got ({d_qk}, {d_v})")
    if (d_qk * CFG.BPE) % 16 != 0 or (d_v * CFG.BPE_O) % 16 != 0:
        raise ValueError(f"d192 envelope: d_qk*BPE and d_v*BPE must be 16-byte multiples (TMA global-stride rule); got ({d_qk}, {d_v}) at BPE={CFG.BPE}")
    if SPLIT_KV > 1 and not has_lse:
        raise ValueError("split_kv > 1 requires has_lse=True (the per-split LSE drives the combine)")
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
