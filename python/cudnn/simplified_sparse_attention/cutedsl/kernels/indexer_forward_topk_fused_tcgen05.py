# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SSA fused indexer: score + top-k coarse histogram in one kernel (Stage 1).

Fusion knob ``SSA_INDEXER_FUSION=hist`` (or ``fusion="hist"`` on the tcgen05
``run()``). Rect/THD path only; the existing unfused kernels are untouched.

The filtered top-k reads the score workspace S twice per row: the p1
coarse-histogram scan and the p3 filter/collect scan (plus refine). This
kernel folds p1 into the score epilogue: while each 128x256 MMA tile's values
are still in registers it bins them (fp32 -> fp16 top-byte ordered key, the
exact ``to_coarse_key`` transform) into an SMEM per-row histogram, and flushes
per-chunk totals to a gmem ``hist[query_len, 256]`` int32 buffer. The top-k
then runs with ``external_hist=hist`` and skips its p1 scan entirely -- one
full S read (8GB at Tk=256K) never happens.

Kernel shape: grid (n_chunks, q_nblk); each CTA owns a 128-row x 16K-col chunk
(C_TILES=64 tiles of 128x256), loads Q once, and loops load-K -> MMA ->
TMA-store epilogue + histogram. SMEM: 96KB Q/K (the fp32 staging half aliases
the dead K region -- Q must survive the loop, unlike the unfused kernel which
aliases both) + 64.25KB histogram (row-pair packed u16 halves in u32 words so
plain u32 SMEM atomics work; stride 257 spreads banks since 257 % 32 = 1;
per-row-per-chunk counts <= 16384 < 65536 so halves cannot carry) + 1KB row
bounds. ~162KB -> 1 CTA/SM.
"""

import functools
import json
import logging
import math
import os
import time
from contextlib import contextmanager
from threading import Lock

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op

from ._cuda_preload import preload_tvm_ffi_cuda as _preload_tvm_ffi_cuda
from ._stream_lock import serialize_workspace, stream_lock
from .cute_dsl_filter_topk.filtered_top_k_varlen_util import (
    _SSA_LOG2E,
    _ssa_block_reduce_max,
    _ssa_block_reduce_sum,
    atomic_add,
    float_as_uint32,
    half_as_ushort,
    uint32_as_float,
    ushort_as_half,
)


@dsl_user_op
def _red_smem_add_u32(dst_ptr, val, *, loc=None, ip=None):
    """No-return SMEM atomic add: red.shared.add.u32.

    IKET measured the histogram atomic chain at 52-81% of the fused kernel.
    cute.arch.atomic_add emits an atom with a result register (and sys
    scope), whose destination dependency throttles the unrolled per-element
    chain; red has no result, so the 32 adds per thread issue back-to-back.
    """
    llvm.inline_asm(
        None,
        [dst_ptr.toint(loc=loc, ip=ip).ir_value(), val.ir_value(loc=loc, ip=ip)],
        "red.shared.add.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_smem_or_u32(dst_ptr, val, *, loc=None, ip=None):
    """No-return SMEM atomic OR: red.shared.or.b32 (bitmap bit set)."""
    llvm.inline_asm(
        None,
        [dst_ptr.toint(loc=loc, ip=ip).ir_value(), val.ir_value(loc=loc, ip=ip)],
        "red.shared.or.b32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _f32_skey(x):
    """fp32 -> signed int32 ordered key: k(a) < k(b) <=> a < b for reals.
    Positive floats keep their bits; negative floats flip the magnitude bits,
    so more-negative values sort lower (k1 sampler)."""
    bits = I32(float_as_uint32(x))
    return cutlass.select_(bits >= 0, bits, bits ^ I32(0x7FFFFFFF))


def _skey_f32(key):
    """Inverse of _f32_skey."""
    bits = cutlass.select_(key >= 0, key, key ^ I32(0x7FFFFFFF))
    return F32(uint32_as_float(cutlass.Uint32(bits)))


@cute.jit
def _k1_keep_insert(sKeep, sKC, sKG, sKI, tidx, v, keep_b):
    """Insert v into thread tidx's private smem keep of keep_b slots (k1
    sampler). Only thread tidx touches its slots, so nothing synchronizes.
    While filling, v takes the next free slot. Once full, a slot-monotone
    tournament overwrites the round-robin slot only if v beats it, so a slot
    only ever rises and the gate sKG ratchets to the largest evicted value.
    A dropped value can only lower the row's extracted cut (over-collect);
    exactness never depends on the keep."""
    kc = I32(sKC[tidx])
    keep_slot = I32(0)
    keep_old = F32(0.0)
    keep_gate = F32(0.0)
    if kc < I32(keep_b):
        sKeep[kc, tidx] = v
        sKC[tidx] = kc + I32(1)
    else:
        keep_slot = I32(sKI[tidx])
        keep_old = F32(sKeep[keep_slot, tidx])
        if v > keep_old:
            sKeep[keep_slot, tidx] = v
            keep_gate = F32(sKG[tidx])
            sKG[tidx] = cutlass.select_(keep_old > keep_gate, keep_old, keep_gate)
        keep_slot = keep_slot + I32(1)
        keep_slot = cutlass.select_(keep_slot >= I32(keep_b), I32(0), keep_slot)
        sKI[tidx] = keep_slot


# Dev-only IKET phase timing (no-op unless FUSED_IKET=1; compile with
# CUTE_DSL_COMPILER_OPT=iket and run under `run-iket ... profile`).
_FUSED_IKET = os.environ.get("FUSED_IKET", "0") == "1"
if _FUSED_IKET:
    import cutlass.cute.experimental.iket as _iket


def _ik_b(name):
    if cutlass.const_expr(_FUSED_IKET):
        return _iket.range_start(name)
    return None


def _ik_e(tok):
    if cutlass.const_expr(_FUSED_IKET):
        _iket.range_end(tok)


def _derive_topk(topk_indices, query_len):
    topk = topk_indices.numel() // query_len
    if topk not in (512, 1024, 2048):
        raise ValueError(f"SSA indexer top_k must be 512/1024/2048, got {topk}")
    return topk


BF16 = cutlass.BFloat16
F32 = cutlass.Float32
I32 = cutlass.Int32

D = 128
TOPK = 512
BLK = 128
NT = 256  # MMA tile: 128 x 256 (the dense_gemm-sweep winner for this shape)
# 512 threads = 4 x 128-lane groups. The epilogue is the serial wall (measured
# 19.5us/tile at 128 threads: 256 cvt+twiddle+atomic chains per thread); the
# tmem-copy tv layout spans 128 lanes (window follows warp_id % 4, the spmm v4
# pattern), so group g = tidx // 128 drains slice sl == g of each 128x128 half
# in parallel -- per-thread epilogue work / 4.
THREADS = 512
C_TILES = 64  # k-tiles per CTA chunk: 64 * 256 = 16384 cols
# Smaller filtered-pass chunks smooth the large work imbalance induced by
# ragged per-row [start, end) bounds.  A GB300 interleaved sweep found 14 to be
# the robust point across 64K--256K and K=512--2048; keep an override for
# production bisects without changing the other score/histogram paths.
FILTER_C_TILES = int(os.environ.get("SSA_FUSED_FILTER_C_TILES", "14"))
if FILTER_C_TILES <= 0:
    raise ValueError("SSA_FUSED_FILTER_C_TILES must be positive")
HIST_STRIDE = 257  # 257 % 32 == 1 -> row-pairs spread across SMEM banks
SCALE = 0.08838834764831845
# The filtered path's candidate capacity is a fixed multiple of top-K, never
# of key length.  The cursor is an exact certificate; a row that does not fit
# is repaired on-device in fixed 128-row chunks.  Therefore a larger key
# extent can increase repair work, but can never grow the persistent
# candidate allocation into an O(Tq*Tk) score surrogate.
#
# For the supported K={512,1024,2048}, capacity is 4*K.  Candidate storage is
# two 4-byte arrays, hence 32*Tq*K bytes.  The exact repair
# workspace is [128, ceil(Tk/256)*256] fp32, so the complete score-side
# workspace is O(Tq*K + Tk), not O(Tq*Tk).
#
# SSA_FUSED_TIE_CAP remains a validation/diagnostic override (for forcing
# overflow), but is rejected above the same 16*K hard bound so an environment
# knob cannot silently defeat the production memory contract.
_TIE_CAP_OVERRIDE = os.environ.get("SSA_FUSED_TIE_CAP")
TIE_CAP = int(_TIE_CAP_OVERRIDE) if _TIE_CAP_OVERRIDE else 0
_CAP_PER_TOPK = 4
_CAP_HARD_MULTIPLE = 16

_logger = logging.getLogger(__name__)
# Fallback observability. The seeded filter is exact via the row fallback,
# but a firing means the seed was too tight for the live score distribution;
# tracking the rate lets production decide whether to raise
# SSA_FUSED_TIE_HEADROOM. These are process-wide cumulative counters a
# training script can read directly (fused_fallback_stats()); each firing
# also logs at WARNING (rate-limited to once per SSA_FUSED_LOG_EVERY
# firings, default 1 -> every firing; set higher to sample). Zero overhead
# when nothing fires.
_FALLBACK_CALLS = 0  # filtered invocations that fell back
_FALLBACK_ROWS = 0  # cumulative flagged rows across all firings
_TOTAL_CALLS = 0  # filtered invocations total
_LOG_EVERY = max(1, int(os.environ.get("SSA_FUSED_LOG_EVERY", "1")))


def fused_fallback_stats():
    """(total_filtered_calls, fallback_calls, cumulative_flagged_rows) so a
    training loop can monitor the seeded-filter fallback rate."""
    graph_calls = 0
    graph_rows = 0
    for st in _state_filtered.values():
        stats = st.get("fallback_stats")
        if stats is not None:
            vals = stats.tolist()
            graph_calls += int(vals[0])
            graph_rows += int(vals[1])
    return (_TOTAL_CALLS, _FALLBACK_CALLS + graph_calls, _FALLBACK_ROWS + graph_rows)


def reset_fused_fallback_stats():
    global _FALLBACK_CALLS, _FALLBACK_ROWS, _TOTAL_CALLS
    _FALLBACK_CALLS = _FALLBACK_ROWS = _TOTAL_CALLS = 0
    for st in _state_filtered.values():
        for name in ("fallback_stats", "flag_kinds"):
            stats = st.get(name)
            if stats is not None:
                stats.zero_()


def fused_fallback_kinds():
    """Cumulative (seed_too_tight, buffer_overflow) certificate-failure rows
    of the compact selector since the last reset; reading synchronizes.
    (None, None) unless SSA_FUSED_FALLBACK_KINDS=1 and the compact selector."""
    if not _FALLBACK_KINDS or os.environ.get("SSA_FILTERED_SELECT", "compact") != "compact":
        return (None, None)
    tight = overflow = 0
    for st in _state_filtered.values():
        kinds = st.get("flag_kinds")
        if kinds is not None:
            vals = kinds.tolist()
            tight += int(vals[0])
            overflow += int(vals[1])
    return (tight, overflow)


def fused_fallback_report(reset=False, printer=print):
    """Print and return the cumulative fallback counts, split by cause when
    SSA_FUSED_FALLBACK_KINDS=1. Reading synchronizes the device."""
    total, fb_calls, fb_rows = fused_fallback_stats()
    tight, overflow = fused_fallback_kinds()
    lines = [
        "=== SSA filtered-fusion fallback report ===",
        f"  filtered calls : {total}",
        f"  fell back      : {fb_calls} calls, {fb_rows} rows",
    ]
    if tight is None:
        lines.append("  by cause       : off (SSA_FUSED_FALLBACK_KINDS=1, compact select)")
    else:
        lines += [f"  seed_too_tight : {tight} rows (fewer than top_k collected)", f"  buffer_overflow: {overflow} rows (more than capacity collected)"]
    printer("\n".join(lines))
    if reset:
        reset_fused_fallback_stats()
    return dict(total_calls=total, fallback_calls=fb_calls, fallback_rows=fb_rows, seed_too_tight_rows=tight, buffer_overflow_rows=overflow)


def set_fused_scr_observer(callback):
    """Observe each filtered call's main-path buffers; None disables.

    The callback runs on the caller stream after the certificate, before the
    exact repair, with a dict: cursors[:, 0] = C, cursors[:, 1] = S, thr
    (crossing bin, remaining sample need), bnd (collect boundary), flags,
    starts/ends, the call's indexer inputs q/k (for an offline replay),
    sample_rank R, sample_scores [4, positions] (NaN where not sampled or
    out of bounds) and sample_rows. The buffers are reused by
    later calls: copy them on the current stream before returning. It never
    runs under graph capture; an exception disables it instead of failing
    the call. Requires SSA_FUSED_SCR_COUNTS=1 at import.
    """
    global _SCR_OBSERVER
    if callback is not None and not _SCR_COUNTS:
        raise RuntimeError("set SSA_FUSED_SCR_COUNTS=1 before importing the kernel")
    if callback is not None and not callable(callback):
        raise TypeError("SCR observer must be callable or None")
    _SCR_OBSERVER = callback


def _observe_fused_scr(view):
    global _SCR_OBSERVER
    callback = _SCR_OBSERVER
    if callback is None or torch.cuda.is_current_stream_capturing():
        return
    try:
        callback(view)
    except Exception:
        # Optional diagnostics must not fail a training call.
        _SCR_OBSERVER = None
        _logger.exception("SSA filtered SCR observer failed; disabling it")


def _fallback_topk_impl(key_len):
    """Use only the adversarially qualified exact batch-repair backend.

    The batch path is a validation/bisect oracle, not the production sparse
    repair path. Keep it on REREAD_ALWAYS at every key length: the GVR
    implementation in this combined lineage is not qualified for all-equal
    overflow and therefore cannot be selected by an environment override.
    """
    del key_len
    return "filtered"


# fusion="filtered" (seeded single pass): sample 1/SAMPLE_STRIDE of each
# row's keys, widen the estimated threshold bin by SAMPLE_MARGIN. The
# margin trades candidate-buffer volume against seed-miss (fallback) rate;
# misses are exactly detected, never silent.
#
# SSA_FUSED_SAMPLE_STRIDE: sampling stride (coverage = 1/stride), read at
# import. The rank budget (topk // stride), the sample-pass grid and the
# sampled positions all follow it. It must divide 32: token mode then reads
# only K rows inside the 16384-row aligned key capacity, and a 32-column
# collect span holds a whole number of sampled columns.
SAMPLE_STRIDE = int(os.environ.get("SSA_FUSED_SAMPLE_STRIDE", "16"))
if SAMPLE_STRIDE not in (1, 2, 4, 8, 16, 32):
    raise ValueError(f"SSA_FUSED_SAMPLE_STRIDE must divide 32, got {SAMPLE_STRIDE}")
SAMPLE_CTILES = 8
# Sampling strategy of the sample (and refine) pass; coverage is
# 1/SAMPLE_STRIDE either way, so the rank-budget math is identical:
#   token (default) -- every SAMPLE_STRIDE'th single key (uniform scatter),
#     loaded by the same per-row cp.async with a multiplied row index.
#   tile -- every SAMPLE_STRIDE'th 256-key tile (contiguous blocks at fixed
#     positions). Score mass clustered between the sampled blocks is never
#     seen, and every row of a call shares the same blocks, so a layout the
#     blocks misrepresent misplaces every row's cut together.
# Exactness is unaffected either way (cursor certificate + per-row repair).
SAMPLE_MODE = os.environ.get("SSA_FUSED_SAMPLE_MODE", "token")
if SAMPLE_MODE not in ("tile", "token"):
    raise ValueError(f"SSA_FUSED_SAMPLE_MODE must be 'tile' or 'token', got {SAMPLE_MODE!r}")
# Filtered sampler: how the per-row collect boundary is chosen before the
# single full-GEMM collect pass.
#   k1 (default) -- One CTA per 128-row q block (two when
#     that still fits one wave of SMs, plus a merge kernel) streams the
#     block's sampled K rows: one pseudo-random key per K1_STRIDE-wide
#     stratum, the phase hashed per q block, so score structure aligned with
#     the stride cannot alias. Each row's top samples stay in per-thread smem
#     keeps; a warp then extracts the row's budget-th largest sample exactly
#     (4-pass byte radix) and the boundary is one fp32 notch below it. The
#     budget is evaluated per row as the exact Poisson tail at K1_RISK.
#   refine -- the dev12-dev15 two-pass histogram cut (SAMPLE_STRIDE token or
#     tile samples, crossing bin, then the fp16 sub-code of that bin); kept
#     for bisects and the SCR diagnostics.
# Collect, select, certificate and repair are the same for both; exactness
# never depends on the cut.
SAMPLER = os.environ.get("SSA_FUSED_SAMPLER", "k1")
if SAMPLER not in ("k1", "refine"):
    raise ValueError(f"SSA_FUSED_SAMPLER must be 'k1' or 'refine', got {SAMPLER!r}")
# k1 geometry, fixed per top_k (see _k1_geometry): base stride 64, doubled
# until the per-thread keep fits smem (K=512/1024: 64, K=2048: 128). It does
# not depend on the key length, so the candidate capacity does not either and
# one compiled state serves every Tk.
K1_STRIDE = 64
K1_KEEP_MIN = 24  # minimum keep slots per thread (extra slots tighten the cut)
K1_RISK = 1e-6  # per-row probability that the budget-th sample misses the top-k
K1M_THREADS = 256  # merge kernel CTA (one warp per row)
# Optional failure diagnostics, off by default; outputs are identical either
# way and nothing extra is compiled or allocated when off.
#   SSA_FUSED_FALLBACK_KINDS=1: cumulative certificate-failure rows by cause
#     (seed_too_tight: fewer than top_k collected; buffer_overflow: more than
#     the candidate capacity), read by fused_fallback_kinds().
#   SSA_FUSED_SCR_COUNTS=1: per row, C = collected candidates (cursor slot 0)
#     and S = collected candidates at sampled key positions (the unused
#     cursor slot 1 of the single-list collect); R is the sampled rank
#     budget. Also keeps the raw fp32 sample scores of four rows. An observer
#     registered with set_fused_scr_observer() sees each call's buffers.
_FALLBACK_KINDS = os.environ.get("SSA_FUSED_FALLBACK_KINDS", "0") == "1"
_SCR_COUNTS = os.environ.get("SSA_FUSED_SCR_COUNTS", "0") == "1"
if _SCR_COUNTS and SAMPLER != "refine":
    # S counts collected candidates at the refine pass's sampled positions.
    # An optional diagnostic must not stop training: it is off under k1.
    _logger.warning("SSA_FUSED_SCR_COUNTS=1 diagnoses the refine sampler and is ignored " "with SSA_FUSED_SAMPLER=%s", SAMPLER)
    _SCR_COUNTS = False
_SCR_OBSERVER = None
# Token-mode sampled columns within a 32-aligned 32-column span, as an i32.
_SCR_TOKEN_BITS = sum(1 << j for j in range(0, 32, SAMPLE_STRIDE))
if _SCR_TOKEN_BITS >= 1 << 31:
    _SCR_TOKEN_BITS -= 1 << 32
# margin 0: the 2x-oversampled seed scan alone is sufficient (all gates
# zero-flag, incl. shortrows); the extra bin was pure candidate-volume waste
# (measured -9..-12% end to end). Misses stay exactly detectable via the
# cursor certificate.
SAMPLE_MARGIN = int(os.environ.get("SSA_FUSED_MARGIN", "0"))
# Fine within-bin fp32 quantile boundary. The coarse rule collects the WHOLE
# byte bin that crosses the sampled rank budget; the fine rule cuts INSIDE
# that bin at an fp16 sub-code and emits the exact fp32 boundary of that
# sub-code, so the rest of the bin is never collected. That roughly halves
# the collected candidate volume (measured -54.7% candidate store sectors,
# -40.6% cursor-atomic sectors).
#
# The sub-code is MEASURED, not interpolated: a second sample pass recomputes
# the same sampled scores and histograms the low fp16 byte of those that fall
# in the crossing bin, and the cut is the highest sub-code that keeps the
# sampled rank budget. The boundary is therefore the exact fp16-resolution
# quantile of the sample. A bin is a quarter octave wide; when the row's score
# spread near the cut is narrow relative to the score magnitude, its samples
# pile up at the bin's low edge, and the former uniform-within-bin estimate
# cut far above them (Ultra 1M: 27 of 288 budgeted samples kept, half the
# rows repaired). Correctness never depends on the boundary: the cursor
# certificate detects every miss and the exact per-row repair fixes the
# flagged rows.
#
# Sampled rank budget = ceil(RANK_NUM * (top_k / SAMPLE_STRIDE) / RANK_DEN),
# i.e. at K=512 (top_k/stride = 32): 2/1 -> 64 (the coarse budget), 9/4 -> 72.
# A bigger budget is more conservative -- it pushes the boundary down,
# collecting more but making a miss less likely. The fine rule pairs with the
# larger 9/4 budget because the extra rank absorbs the within-bin estimation
# error; that combination measured ZERO misses and ZERO overflows over
# 491,520 audited rows, while every budget tighter than it (fine ranks 64,
# 56, 48, 40) was measured net-neutral or net-slower once its repair cost was
# charged, and is closed.
#
# Defaults are paired so that SSA_FUSED_FINE_BOUNDARY=0 is an exact bisect
# back to the previous coarse rank-64 path; an explicit RANK_NUM/RANK_DEN
# still overrides either way. The fine cut (and its refine pass) is off when
# SAMPLE_MARGIN > 0, which deliberately collects whole shifted bins.
FINE_BOUNDARY = os.environ.get("SSA_FUSED_FINE_BOUNDARY", "1") == "1"
SAMPLE_RANK_NUM = int(os.environ.get("SSA_FUSED_SAMPLE_RANK_NUM", "9" if FINE_BOUNDARY else "2"))
SAMPLE_RANK_DEN = int(os.environ.get("SSA_FUSED_SAMPLE_RANK_DEN", "4" if FINE_BOUNDARY else "1"))
FINE_REFINE = FINE_BOUNDARY and SAMPLE_MARGIN == 0
MASK_WALK = os.environ.get("SSA_FUSED_MASK_WALK", "1") == "1"
# Instruction-footprint knobs for the filtered classify loop, which NCU puts
# at ~48% of all executed instructions in the collect pass.
# FAST_CNT: build the hit mask by selecting the constant bit (1 << j) and
# recover the count with one popc -- bit-identical, drops 2 ALU ops/element.
# RAW_CMP: compare the raw TMEM value against the boundary pulled back
# through SCALE, so the 32 per-element multiplies collapse to one multiply
# per surviving candidate. Stored candidate values stay bit-identical.
# Both default ON after candidate-set containment, interleaved A/B, and the
# full validate_thd_indexer gate on GB300.  The knobs retain an explicit
# baseline path for future profiling and bisects.
FAST_CNT = os.environ.get("SSA_FUSED_FAST_CNT", "1") == "1"
RAW_CMP = os.environ.get("SSA_FUSED_RAW_CMP", "1") == "1"
# SPAN_SKIP: clamp each chunk CTA's tile interval to the row-block's live
# column range [min(start), max(end)) before the Q load and the tile loop.
# The grid partitions [0, k_tiles) uniformly regardless of the per-row
# [start, end) bounds, so on packed-THD / multi-document extents (start != 0)
# a chunk can lie entirely outside every one of its 128 rows' spans and still
# run its full K stream, MMA and classify to append nothing. Exact by
# construction: a tile with no in-span column cannot produce a candidate.
# Costs two CTA barriers and a 128-lane smem min/max once per CTA.
#
# Measured with interleaved runs of 40 iterations,
# candidate SETS identical per row and no candidate outside its span on every
# leg): packed-THD extents 2.56x-8.34x on the filter pass (256K/K512 2-doc
# 2.3307 -> 0.7648 ms, 8-doc 2.6973 -> 0.3235), and no cost where there is
# nothing to skip -- 256K/K512 rect 0.998x, within its 0.031 ms sd, with the
# taper shapes 1.012-1.020x. Default ON; SSA_FUSED_SPAN_SKIP=0 bisects back.
SPAN_SKIP = os.environ.get("SSA_FUSED_SPAN_SKIP", "1") == "1"
# The same dead-chunk waste is in the filtered pipeline's `ps` sample pass:
# its grid partitions the key range uniformly too, so on packed THD its CTAs
# are dead in the same proportion. It samples every SAMPLE_STRIDE'th tile, so
# the clamp there is rounded UP to the sampling grid (see the tile_stride
# branch in the span clamp) and the sampled histogram stays bit-identical.
# Separately bisectable from the collect pass; defaults to it.
SAMPLE_SPAN_SKIP = os.environ.get("SSA_FUSED_SAMPLE_SPAN_SKIP", "1" if SPAN_SKIP else "0") == "1"
# Relative slack on the pulled-back boundary. The round trip through fp32
# 1/SCALE costs at most ~4.3e-7 relative; 4e-6 is ~10x that, and widening
# the boundary can only ever ADD candidates, never drop one.
RAW_CMP_SLACK = 4e-6
# K-major filter grid (one CTA per K tile, Q L2-resident) remains unqualified
# for the production packed-THD contract. Keep the implementation for future
# experiments, but fail closed if its old environment knob is requested.
KMAJOR = False
SEL_THR = 512  # compact-select CTA size; top_k // SEL_THR winner slots/thread
# winner-emission bitmap: 1 bit per LOCAL column of the row. Rows longer than
# BMAP_BITS fall back to the bitonic sort (Tk 512K+ tiers).
BMAP_WORDS = 8192  # 32KB SMEM -> covers row length <= 256K
BMAP_WPT = BMAP_WORDS // SEL_THR  # words per thread in the popcount scan


def _poisson_budget_host(mu, delta):
    """min b with P(Poisson(mu) >= b) <= delta (host reference of the k1
    kernel's per-row budget; mu > 48 uses the Bernstein bound like the
    kernel)."""
    if mu <= 0.0:
        return 1
    if mu > 48.0:
        ln_d = math.log(1.0 / delta)
        return math.ceil(mu + ln_d / 3.0 + math.sqrt(2.0 * mu * ln_d))
    lnmu = math.log(mu)
    terms = [math.exp(i * lnmu - mu - math.lgamma(i + 1)) for i in range(160)]
    tail = 0.0
    b = 160
    for i in range(159, 0, -1):
        tail += terms[i]
        if tail > delta:
            break
        b = i
    return max(1, b)


@functools.lru_cache(maxsize=None)
def _k1_geometry(topk):
    """(stride, keep_b, k_db, b_max) of the k1 sampler at this top_k.

    Rows longer than the candidate capacity (>= 4 * top_k keys) are sampled:
    a span of L keys meets at most L / stride + 2 strata, so the row's mean
    sampled hit count mu = top_k * m / L stays below top_k / stride + 0.5,
    and keep_b must hold that mu's budget (+1 for the kernel's fp32 tail).
    The in-kernel budget is clamped to keep_b, so an undersized keep would
    tighten the cut silently instead of failing. When the keep outgrows smem
    the stride doubles."""
    stride = K1_STRIDE
    while True:
        b_max = _poisson_budget_host(topk / stride + 0.5, K1_RISK) + 1
        keep_b = max(K1_KEEP_MIN, (b_max + 3) // 4 * 4)
        keep_bytes = 4 * THREADS * keep_b
        # smem: Q 32 KB + K 64 KB (x2 double-buffered) + keeps + ~10 KB state
        k_db = (32 + 128 + 10) * 1024 + keep_bytes <= 227 * 1024
        if (32 + 64 + 10) * 1024 + keep_bytes <= 227 * 1024:
            return stride, keep_b, k_db, b_max
        if stride >= 256:
            raise ValueError(f"k1 sampler: keep_b={keep_b} does not fit smem at top_k={topk}")
        stride *= 2


@functools.lru_cache(maxsize=None)
def _gamma_upper_quantile(b, delta):
    """x with P(Gamma(b, 1) > x) = delta, by bisection on the Poisson cdf
    (P(Gamma(b) > x) = P(Poisson(x) <= b - 1))."""

    def pois_cdf(x):
        return sum(math.exp(-x + i * math.log(x) - math.lgamma(i + 1)) for i in range(b))

    lo, hi = float(b), 4.0 * b + 64.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if pois_cdf(mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


@functools.lru_cache(maxsize=None)
def _k1_cand_cap(topk):
    """Upper K1_RISK tail of the k1 per-row candidate count. The budget-th
    largest of m sampled scores sits at row fraction ~ Gamma(budget) / m, so
    about stride * Gamma(budget) keys lie above it, independent of the row
    length, with relative spread 1 / sqrt(budget). 5% margin."""
    stride, _, _, b_max = _k1_geometry(topk)
    return int(math.ceil(1.05 * stride * _gamma_upper_quantile(b_max, K1_RISK)))


def _tie_cap(top_k: int, key_len: int = 262144) -> int:
    """Return a key-length-independent per-row candidate capacity.

    ``key_len`` remains in the signature for compatibility with existing
    callers and tests.  It deliberately does not participate in the result.
    """
    del key_len
    hard_cap = _CAP_HARD_MULTIPLE * top_k
    if _TIE_CAP_OVERRIDE:
        if not top_k <= TIE_CAP <= hard_cap:
            raise ValueError("SSA_FUSED_TIE_CAP must be in " f"[{top_k}, {hard_cap}] for top_k={top_k}, got {TIE_CAP}")
        return TIE_CAP
    if SAMPLER == "k1":
        # The k1 candidate count's upper K1_RISK tail (1.29x at K=1024/2048,
        # 1.99x at K=512), never below the refine path's capacity.
        return min(hard_cap, max(_CAP_PER_TOPK * top_k, _k1_cand_cap(top_k)))
    return _CAP_PER_TOPK * top_k


@cute.jit
def _coarse8(x):
    """fp32 -> 8-bit descending-order coarse key; must match the top-k's
    ``to_coarse_key`` bit-for-bit (fp32 -> fp16 RN convert, sign-flip twiddle,
    top byte) or the preloaded histogram would disagree with p3's classify."""
    h = x.to(cutlass.Float16)
    bits = half_as_ushort(h)
    key = cutlass.Uint16(0)
    if bits & 0x8000:
        key = cutlass.Uint16(bits)
    else:
        key = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
    return I32((key >> 8) & 0xFF)


@cute.jit
def _key16(x):
    """fp32 -> full 16-bit descending-order fp16 key, the same convert and
    twiddle as ``_coarse8``: bits [15:8] are the coarse bin, bits [7:0] the
    sub-code inside it (the codes ``_f16_key_boundary`` cuts at)."""
    h = x.to(cutlass.Float16)
    bits = half_as_ushort(h)
    key = cutlass.Uint16(0)
    if bits & 0x8000:
        key = cutlass.Uint16(bits)
    else:
        key = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)
    return I32(key) & I32(0xFFFF)


@cute.jit
def _f16_key_boundary(key16):
    """Exact fp32 boundary B for the fp16-derived coarse key (verbatim port
    of the boundary_classify helper): for real fp32 x,
    key(cvt_rn_f16(x)) < key16  <=>  x > B. IKET measured the per-element
    cvt+twiddle classify at 88% of the filter pass; one precomputed fp32
    boundary per row turns it into a single float compare."""
    k_hi = cutlass.Int32(key16)
    k_up = k_hi - cutlass.Int32(1)  # next code = next LARGER value
    bits_hi = cutlass.select_(k_hi >= cutlass.Int32(0x8000), k_hi, (~k_hi) & cutlass.Int32(0x7FFF))
    bits_up = cutlass.select_(k_up >= cutlass.Int32(0x8000), k_up, (~k_up) & cutlass.Int32(0x7FFF))
    v_hi = cutlass.Float32(ushort_as_half(cutlass.Uint16(bits_hi)))
    v_up = cutlass.Float32(ushort_as_half(cutlass.Uint16(bits_up)))
    big = cutlass.Float32(65536.0)
    v_hi = cutlass.select_(v_hi > big, big, cutlass.select_(v_hi < -big, -big, v_hi))
    v_up = cutlass.select_(v_up > big, big, cutlass.select_(v_up < -big, -big, v_up))
    m = (v_hi + v_up) * cutlass.Float32(0.5)
    odd = (bits_hi & cutlass.Int32(1)) == cutlass.Int32(1)
    u = cutlass.Uint32(float_as_uint32(m))
    u_prev = cutlass.select_(
        m > cutlass.Float32(0.0), u - cutlass.Uint32(1), cutlass.select_(m < cutlass.Float32(0.0), u + cutlass.Uint32(1), cutlass.Uint32(0x80000001))
    )
    return uint32_as_float(cutlass.select_(odd, u_prev, u))


@cute.jit
def _ordered32(x):
    """fp32 -> 32-bit descending-order key; must match the ``to_ordered``
    fp32 branch bit-for-bit (larger value -> smaller key)."""
    bits = float_as_uint32(x)
    key = cutlass.Uint32(0)
    if bits & 0x80000000:
        key = cutlass.Uint32(bits)
    else:
        key = (bits ^ cutlass.Uint32(0xFFFFFFFF)) & cutlass.Uint32(0x7FFFFFFF)
    return cutlass.Uint32(key)


# Sparse exact-fallback transport.  A CUDA conditional graph executes these
# only for 128-row chunks that contain at least one certificate failure.  The
# fixed-size staging shape lets the vetted exact scorer/top-k be captured once;
# invalid lanes in the last chunk are initialized as harmless [0, 1) rows and
# are never scattered back.
@cute.kernel
def _fallback_gather_rows_kernel(
    mFlags: cute.Tensor,
    mQ: cute.Tensor,
    mStarts: cute.Tensor,
    mEnds: cute.Tensor,
    mFbQ: cute.Tensor,
    mFbStarts: cute.Tensor,
    mFbEnds: cute.Tensor,
    chunk_offset: I32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    linear = I32(bidx * 512 + tidx)
    slot = linear // I32(D)
    d = linear - slot * I32(D)
    nbad = I32(mFlags[0])
    valid = chunk_offset + slot < nbad
    row = I32(0)
    if valid:
        row = I32(mFlags[chunk_offset + slot + I32(1)])
        mFbQ[slot, d] = mQ[row, d]
    else:
        mFbQ[slot, d] = BF16(0.0)
    if d == 0:
        if valid:
            mFbStarts[slot] = mStarts[row]
            mFbEnds[slot] = mEnds[row]
        else:
            mFbStarts[slot] = I32(0)
            mFbEnds[slot] = I32(1)


@cute.kernel
def _fallback_scatter_rows_kernel(
    mFlags: cute.Tensor,
    mFbIdx: cute.Tensor,
    mFbProb: cute.Tensor,
    mOutIdx: cute.Tensor,
    mOutProb: cute.Tensor,
    mStats: cute.Tensor,
    chunk_offset: I32,
    top_k: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    slot, _, _ = cute.arch.block_idx()
    nbad = I32(mFlags[0])
    if chunk_offset + I32(slot) < nbad:
        row = I32(mFlags[chunk_offset + I32(slot) + I32(1)])
        for e in cutlass.range_constexpr(top_k // 512):
            i = I32(e * 512 + tidx)
            mOutIdx[row, i] = mFbIdx[slot, i]
            mOutProb[row, i] = mFbProb[slot, i]
        # Exactly one thread accounts for the whole device-produced list.
        if slot == 0 and tidx == 0:
            if chunk_offset == 0:
                mStats[0] = I32(mStats[0]) + I32(1)
                mStats[1] = I32(mStats[1]) + nbad


class _SimplifiedSparseFallbackGatherRows:
    @cute.jit
    def __call__(
        self,
        flags: cute.Tensor,
        q: cute.Tensor,
        starts: cute.Tensor,
        ends: cute.Tensor,
        fb_q: cute.Tensor,
        fb_starts: cute.Tensor,
        fb_ends: cute.Tensor,
        chunk_offset: I32,
        stream,
    ):
        _fallback_gather_rows_kernel(
            flags,
            q,
            starts,
            ends,
            fb_q,
            fb_starts,
            fb_ends,
            chunk_offset,
        ).launch(grid=(32, 1, 1), block=(512, 1, 1), stream=stream)


class _SimplifiedSparseFallbackScatterRows:
    def __init__(self, top_k: int):
        self.top_k = top_k

    @cute.jit
    def __call__(
        self,
        flags: cute.Tensor,
        fb_idx: cute.Tensor,
        fb_prob: cute.Tensor,
        out_idx: cute.Tensor,
        out_prob: cute.Tensor,
        stats: cute.Tensor,
        chunk_offset: I32,
        stream,
    ):
        _fallback_scatter_rows_kernel(
            flags,
            fb_idx,
            fb_prob,
            out_idx,
            out_prob,
            stats,
            chunk_offset,
            top_k=self.top_k,
        ).launch(grid=(BLK, 1, 1), block=(512, 1, 1), stream=stream)


@cute.kernel
def _score_hist_kernel_tcgen05(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mHist: cute.Tensor,
    mStarts: cute.Tensor,
    mEnds: cute.Tensor,
    S_mma: cute.TiledMma,
    lQs: cute.ComposedLayout,
    lKs: cute.ComposedLayout,
    lFlat: cute.ComposedLayout,
    lFlatK: cute.ComposedLayout,
    S_mma_h: cute.TiledMma,
    q_rows: I32,
    k_tiles_total: I32,
    write_s: cutlass.Constexpr = True,
    mC_tma: cute.Tensor = None,
    lEpi: cute.ComposedLayout = None,
    tma_c: cute.CopyAtom = None,
    collect: cutlass.Constexpr = False,
    mThr: cute.Tensor = None,
    mCursors: cute.Tensor = None,
    mDefIdx: cute.Tensor = None,
    mDefVal: cute.Tensor = None,
    mTieIdx: cute.Tensor = None,
    mTieVal: cute.Tensor = None,
    tie_cap: cutlass.Constexpr = 0,
    tile_stride: cutlass.Constexpr = 1,
    c_tiles: cutlass.Constexpr = C_TILES,
    filtered: cutlass.Constexpr = False,
    mBnd: cute.Tensor = None,
    k_db: cutlass.Constexpr = True,
    top_k: cutlass.Constexpr = TOPK,
    n_tile: cutlass.Constexpr = NT,
    span_skip: cutlass.Constexpr = False,
    mask_walk: cutlass.Constexpr = False,
    fast_cnt: cutlass.Constexpr = False,
    raw_cmp: cutlass.Constexpr = False,
    refine: cutlass.Constexpr = False,
    col_scale: cutlass.Constexpr = 1,
    scr_counts: cutlass.Constexpr = False,
    mScrSampleScores: cute.Tensor = None,
):
    # refine=True (sample pass geometry): the seeded filter's second sample
    # pass. Same samples and scores as the first; mHist[r, s] counts the
    # samples whose 16-bit key is (mThr[r, 0] << 8) + s, i.e. the sub-code
    # histogram of the row's crossing bin.
    # col_scale > 1 (token sampling, tile_stride == 1): the kernel's K "rows"
    # are sample positions p, loaded from K row p * col_scale, and every
    # column id (bounds, span clamp) is the global key id p * col_scale.
    # k_tiles_total then counts sample-space tiles of n_tile positions.
    # scr_counts (filtered collect only): also count, in cursor slot 1, the
    # collected candidates at sampled key positions (the SCR diagnostic's S).
    # mScrSampleScores (sample pass only): raw fp32 sample scores of four
    # fixed rows, indexed by sample position.
    # k_db=False (filtered mode): single K buffer. The double buffer measured
    # zero benefit (loads already hidden) and its 64KB is exactly what keeps
    # the kernel at 1 CTA/SM; NCU shows the filtered pass latency-bound at
    # 25% warp occupancy, so 2 CTAs/SM is worth far more than the prefetch.
    # tile_stride > 1: sample pass -- process every tile_stride'th 256-col
    # tile (the seeded-filter scheme's 1/stride sampled histogram).
    # filtered=True: the "filtered single pass" -- like collect, but ONE
    # candidate list per row (mTieIdx/mTieVal, cursor slot 0) taking every
    # element with coarse key <= the row's seeded threshold mThr[r, 0]; no
    # histogram, no definite/tie split. Exactness is verified downstream by
    # the cursor count (>= top_k proves the true top-k is inside).
    # write_s=False: no fp32 staging / TMA store / acquire serialization; S is
    # never written (Stage-2 passes and the hist-only probe).
    # collect=True (Stage-2 pass-2): mHist is unused; the epilogue classifies
    # each value against the row's coarse threshold bin mThr[r, 0] (computed
    # by _threshold_kernel from the pass-1 histogram) and appends
    #   key <  tb -> (idx, val) to the definite list  (mDefIdx/mDefVal)
    #   key == tb -> (idx, val) to the tie list       (mTieIdx/mTieVal, capped)
    # via per-row gmem cursors (matches are sparse: ~TOPK + tie-bin per row
    # over the whole row, so per-element cursor atomics are cheap).
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    cc, bi, _ = cute.arch.block_idx()

    smem = utils.SmemAllocator()
    # Q tile (32KB) stays live across the whole chunk; the K tile is DOUBLE
    # buffered (2 x 64KB) so tile jt+1's cp.async streams under tile jt's
    # MMA + epilogue. Each K buffer doubles as the 128x128 fp32 staging
    # buffer (64KB, exact fit) for the tile that consumed it -- the TMA store
    # of tile jt drains from buffer p while K(jt+1) loads into buffer p^1.
    # Total: 32 + 128 (K x2) + 64.25 (hist) + 1 (bounds) KB ~= 225.6KB.
    pQK = smem.allocate_array(BF16, BLK * D + (2 * n_tile * D if k_db else n_tile * D), byte_alignment=1024)
    if cutlass.const_expr(not (collect or filtered)):
        pHist = smem.allocate_array(I32, (BLK // 2) * HIST_STRIDE, byte_alignment=128)
    # bounds (2*BLK) + per-row threshold bin (BLK, collect/filtered modes)
    # + a 32-word tail (span_skip: the block's min-start / max-end scalars;
    # padded to the 128B alignment quantum so smem/CTA is unchanged in
    # practice and the 2 CTA/SM cap is untouched).
    se_words = cutlass.const_expr((3 * BLK if (collect or filtered or refine) else 2 * BLK) + (32 if span_skip else 0))
    pSE = smem.allocate_array(I32, se_words, byte_alignment=128)
    if cutlass.const_expr(filtered):
        pBnd = smem.allocate_array(F32, BLK, byte_alignment=128)
    mbar = smem.allocate_array(cutlass.Int64, 1, byte_alignment=16)

    sQ = cute.make_tensor(cute.recast_ptr(pQK, lQs.inner), lQs.outer)
    sK0 = cute.make_tensor(cute.recast_ptr(pQK + BLK * D, lKs.inner), lKs.outer)
    sQ2 = cute.make_tensor(cute.recast_ptr(pQK, lFlat.inner), lFlat.outer)
    sK2_0 = cute.make_tensor(cute.recast_ptr(pQK + BLK * D, lFlatK.inner), lFlatK.outer)
    if cutlass.const_expr(k_db):
        sK1 = cute.make_tensor(cute.recast_ptr(pQK + BLK * D + n_tile * D, lKs.inner), lKs.outer)
        sK2_1 = cute.make_tensor(cute.recast_ptr(pQK + BLK * D + n_tile * D, lFlatK.inner), lFlatK.outer)
    if cutlass.const_expr(write_s):
        sOut0 = cute.make_tensor(cute.recast_ptr(pQK + BLK * D, lEpi.inner, dtype=F32), lEpi.outer)
        sOut1 = cute.make_tensor(cute.recast_ptr(pQK + BLK * D + n_tile * D, lEpi.inner, dtype=F32), lEpi.outer)
    if cutlass.const_expr(not (collect or filtered)):
        sHist = cute.make_tensor(pHist, cute.make_layout(((BLK // 2) * HIST_STRIDE,)))
    sSE = cute.make_tensor(pSE, cute.make_layout((se_words,)))
    if cutlass.const_expr(filtered):
        sBnd = cute.make_tensor(pBnd, cute.make_layout((BLK,)))
        if cutlass.const_expr(mask_walk):
            # Once MMA completes, the single K buffer is dead until the next
            # tile load. Alias its exact 64 KiB as [j, thread] fp32 staging:
            # each fixed-j store is bank/coalescing friendly, while a thread
            # can dynamically load only its set-bit survivors afterward.
            sMaskVal = cute.make_tensor(
                cute.recast_ptr(pQK + BLK * D, dtype=F32),
                cute.make_layout((32, THREADS), stride=(THREADS, 1)),
            )

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar + 0, 1)
    cute.arch.mbarrier_init_fence()

    # ---- zero the histogram, stage this row-block's [start, end) bounds
    # (and, in collect mode, the per-row coarse threshold bin) ----
    if cutlass.const_expr(not (collect or filtered)):
        for i in cutlass.range_constexpr(((BLK // 2) * HIST_STRIDE + THREADS - 1) // THREADS):
            hi = i * THREADS + tidx
            if hi < (BLK // 2) * HIST_STRIDE:
                sHist[hi] = 0
    if tidx < BLK:
        rg0 = bi * BLK + tidx
        st0 = I32(0)
        en0 = I32(0)
        if rg0 < q_rows:
            st0 = I32(mStarts[rg0])
            en0 = I32(mEnds[rg0])
        sSE[tidx] = st0
        sSE[BLK + tidx] = en0
        if cutlass.const_expr(collect or filtered or refine):
            tb0 = I32(-1)  # padded rows: no key is < or == -1 -> no appends
            if rg0 < q_rows:
                tb0 = I32(mThr[rg0, 0])
            sSE[2 * BLK + tidx] = tb0
        if cutlass.const_expr(filtered):
            bn0 = cutlass.Float32.inf  # padded rows collect nothing
            if rg0 < q_rows:
                bn0 = F32(mBnd[rg0])
            sBnd[tidx] = bn0

    # ---- this CTA's tile interval, clamped to the block's live columns ----
    # The grid partitions [0, k_tiles_total) uniformly, with no reference to
    # the per-row [start, end) bounds. Under span_skip the interval is first
    # intersected with [min start, max end) over the block's 128 rows, so a
    # chunk that no row of this block can reach retires without loading Q,
    # streaming K, or issuing a single MMA. Exactness is structural: every
    # tile dropped here has no column inside any of those rows' spans, and
    # the epilogue's per-element bounds test would have rejected all 256 of
    # its columns anyway.
    jb = cc * I32(c_tiles * tile_stride)
    je = jb + I32(c_tiles * tile_stride)
    if je > I32(k_tiles_total):
        je = I32(k_tiles_total)
    if cutlass.const_expr(span_skip):
        sk = cutlass.const_expr(se_words - 32)
        if tidx == 0:
            # identities: no start exceeds the key extent, no end precedes 0
            sSE[sk] = I32(k_tiles_total) * I32(n_tile * col_scale)
            sSE[sk + 1] = I32(0)
        cute.arch.barrier()
        if tidx < BLK:
            # padded rows abstain: they contribute no live column
            if (bi * BLK + tidx) < q_rows:
                cute.arch.atomic_min(sSE.iterator + sk, sSE[tidx], sem="relaxed", scope="cta")
                cute.arch.atomic_max(sSE.iterator + (sk + 1), sSE[BLK + tidx], sem="relaxed", scope="cta")
        cute.arch.barrier()
        # col_scale > 1: one sample-space tile spans n_tile * col_scale global
        # columns; flooring lo_t keeps the clamp a superset of the live tiles.
        lo_t = sSE[sk] // I32(n_tile * col_scale)  # first tile holding a start
        hi_t = (sSE[sk + 1] + I32(n_tile * col_scale - 1)) // I32(n_tile * col_scale)  # one past last
        if cutlass.const_expr(tile_stride > 1):
            # Sample pass: jb starts at cc*c_tiles*tile_stride and steps by
            # tile_stride, so only tiles congruent to 0 mod tile_stride are
            # ever visited. Rounding lo_t UP to that grid keeps the visited
            # set a SUBSET of the unclamped one -- no sampled tile is shifted
            # into or out of phase -- so the histogram is bit-identical: every
            # tile dropped lies outside the block's live range and would have
            # contributed zero counts through the epilogue's bounds test.
            lo_t = ((lo_t + I32(tile_stride - 1)) // I32(tile_stride)) * I32(tile_stride)
        if jb < lo_t:
            jb = lo_t
        if je > hi_t:
            je = hi_t
        if jb > je:  # chunk lies wholly outside the live range
            jb = je

    # ---- TMEM ----
    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(barrier_for_retrieve=tmem_bar)
    tmem.allocate(n_tile)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(F32)
    tmem.relinquish_alloc_permit()

    tS_f = S_mma.make_fragment_C(S_mma.partition_shape_C((BLK, n_tile)))
    tS = cute.make_tensor(tmem_ptr, tS_f.layout)
    fQ = S_mma.make_fragment_A(sQ)
    fK0 = S_mma.make_fragment_B(sK0)
    if cutlass.const_expr(k_db):
        fK1 = S_mma.make_fragment_B(sK1)

    g2s_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        BF16,
        num_bits_per_copy=128,
    )
    tiled_g2s = cute.make_tiled_copy_tv(g2s_atom, cute.make_layout((1, 16)), cute.make_layout((1, 8)))
    half = lane // 16
    thr_g2s = tiled_g2s.get_slice(lane % 16)
    qbase = mQ.iterator.toint()
    kbase = mK.iterator.toint()
    nwarp = cutlass.const_expr(THREADS // 32)

    # ---- stage Q once (survives the chunk loop) ----
    # Skipped for an out-of-range chunk: Q is 32 KB per CTA and is the bulk of
    # the per-CTA prologue that the c_tiles ladder prices at ~1.1e-5 ms/CTA.
    if jb < je:
        for i in cutlass.range_constexpr(BLK // (nwarp * 2)):
            r = i * (nwarp * 2) + warp_idx * 2 + half
            cute.copy(
                tiled_g2s,
                thr_g2s.partition_S(
                    cute.make_tensor(cute.make_ptr(BF16, qbase + (bi * BLK + r) * (D * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, D)))
                ),
                thr_g2s.partition_D(cute.local_tile(sQ2, (1, D), (r, 0))),
            )
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier()

    # ---- epilogue partitions (constant across tiles) ----
    ld32 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), F32)
    tSh_f = S_mma_h.make_fragment_C(S_mma_h.partition_shape_C((BLK, BLK)))
    cp_h = tcgen05.make_tmem_copy(ld32, cute.make_tensor(tmem_ptr, tSh_f.layout)[(None, None), 0, 0])
    ltid = tidx % 128
    grp = tidx // 128
    thr_h = cp_h.get_slice(ltid)
    idC = cute.make_identity_tensor((BLK, BLK))
    tr_cO = thr_h.partition_D(idC)
    nslh = cutlass.const_expr(cute.size(tr_cO, mode=[2]))
    rSh = cute.make_rmem_tensor(tr_cO[None, None, 0].shape, F32)
    if cutlass.const_expr(write_s):
        tr_sO0 = thr_h.partition_D(sOut0)
        tr_sO1 = thr_h.partition_D(sOut1)
        gC_epi = cute.flat_divide(mC_tma, (BLK, BLK))
        bSG_s0, bSG_g = cpasync.tma_partition(
            tma_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sOut0, 0, 2),
            cute.group_modes(gC_epi, 0, 2),
        )
        bSG_s1, _ = cpasync.tma_partition(
            tma_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sOut1, 0, 2),
            cute.group_modes(gC_epi, 0, 2),
        )
        c_group = pipeline.CooperativeGroup(pipeline.Agent.ThreadBlock)
        c_pipe = pipeline.PipelineTmaStore.create(num_stages=1, producer_group=c_group)

    nk = cutlass.const_expr(cute.size(fQ, mode=[2]))
    # jb/je were fixed before the Q stage so the span clamp could gate it.

    # ---- prologue: K(jb) into buffer 0 ----
    # Guarded: with span_skip an empty interval leaves jb == je, which may sit
    # at k_tiles_total, so the unconditional prologue would read K past its
    # last tile.
    if jb < je:
        for i in cutlass.range_constexpr(n_tile // (nwarp * 2)):
            r = i * (nwarp * 2) + warp_idx * 2 + half
            # Token sampling: sample position -> K row is a multiply. The
            # rows stay inside the key capacity (SAMPLE_STRIDE divides 32).
            _kr = jb * n_tile + r
            if cutlass.const_expr(col_scale != 1):
                _kr = _kr * I32(col_scale)
            cute.copy(
                tiled_g2s,
                thr_g2s.partition_S(
                    cute.make_tensor(cute.make_ptr(BF16, kbase + _kr * (D * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, D)))
                ),
                thr_g2s.partition_D(cute.local_tile(sK2_0, (1, D), (r, 0))),
            )
    cute.arch.cp_async_commit_group()

    ph = I32(0)
    st = I32(0)
    jt = jb
    while jt < je:
        # ---- prefetch K(jt+1) into buffer st^1: streams under this tile's
        # MMA + epilogue. Buffer st^1's previous staging store was acquired
        # inline at the end of iteration jt-1, so the region is free. ----
        _ikp = _ik_b("k_prefetch_issue")
        if cutlass.const_expr(not k_db):
            # single K buffer: (re)load K(jt) in place; the first tile came
            # from the prologue. No prefetch overlap -- measured worthless,
            # traded for 2 CTAs/SM of occupancy.
            if jt > jb:
                for i in cutlass.range_constexpr(n_tile // (nwarp * 2)):
                    r = i * (nwarp * 2) + warp_idx * 2 + half
                    _kr = jt * n_tile + r
                    if cutlass.const_expr(col_scale != 1):
                        _kr = _kr * I32(col_scale)
                    cute.copy(
                        tiled_g2s,
                        thr_g2s.partition_S(
                            cute.make_tensor(cute.make_ptr(BF16, kbase + _kr * (D * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, D)))
                        ),
                        thr_g2s.partition_D(cute.local_tile(sK2_0, (1, D), (r, 0))),
                    )
                cute.arch.cp_async_commit_group()
        elif jt + I32(tile_stride) < je:
            if st == 0:
                for i in cutlass.range_constexpr(n_tile // (nwarp * 2)):
                    r = i * (nwarp * 2) + warp_idx * 2 + half
                    _kr = (jt + I32(tile_stride)) * n_tile + r
                    if cutlass.const_expr(col_scale != 1):
                        _kr = _kr * I32(col_scale)
                    cute.copy(
                        tiled_g2s,
                        thr_g2s.partition_S(
                            cute.make_tensor(cute.make_ptr(BF16, kbase + _kr * (D * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, D)))
                        ),
                        thr_g2s.partition_D(cute.local_tile(sK2_1, (1, D), (r, 0))),
                    )
            else:
                for i in cutlass.range_constexpr(n_tile // (nwarp * 2)):
                    r = i * (nwarp * 2) + warp_idx * 2 + half
                    _kr = (jt + I32(tile_stride)) * n_tile + r
                    if cutlass.const_expr(col_scale != 1):
                        _kr = _kr * I32(col_scale)
                    cute.copy(
                        tiled_g2s,
                        thr_g2s.partition_S(
                            cute.make_tensor(cute.make_ptr(BF16, kbase + _kr * (D * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, D)))
                        ),
                        thr_g2s.partition_D(cute.local_tile(sK2_0, (1, D), (r, 0))),
                    )
            cute.arch.cp_async_commit_group()
        _ik_e(_ikp)
        _ikw = _ik_b("k_wait")
        if cutlass.const_expr(not k_db):
            cute.arch.cp_async_wait_group(0)
        elif jt + I32(tile_stride) < je:
            # complete K(jt), keep the newest prefetch in flight
            cute.arch.cp_async_wait_group(1)
        else:
            # last tile: nothing was committed this iteration, drain fully
            cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()
        _ik_e(_ikw)

        # ---- UMMA: S_tile = Q @ K(st)^T ----
        _ikm = _ik_b("mma_issue")
        if warp_idx == 0:
            S_mma.set(tcgen05.Field.ACCUMULATE, False)
            if cutlass.const_expr(not k_db):
                for kb in cutlass.range_constexpr(nk):
                    cute.gemm(S_mma, tS, fQ[None, None, kb, 0], fK0[None, None, kb, 0], tS)
                    S_mma.set(tcgen05.Field.ACCUMULATE, True)
            elif st == 0:
                for kb in cutlass.range_constexpr(nk):
                    cute.gemm(S_mma, tS, fQ[None, None, kb, 0], fK0[None, None, kb, 0], tS)
                    S_mma.set(tcgen05.Field.ACCUMULATE, True)
            else:
                for kb in cutlass.range_constexpr(nk):
                    cute.gemm(S_mma, tS, fQ[None, None, kb, 0], fK1[None, None, kb, 0], tS)
                    S_mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(mbar + 0)
        _ik_e(_ikm)
        _ikmw = _ik_b("mma_wait")
        cute.arch.mbarrier_wait(mbar + 0, ph)
        _ik_e(_ikmw)
        ph = ph ^ 1

        # ---- epilogue per 128x128 half: TMEM -> regs -> {histogram, smem
        # staging} -> TMA bulk store ----
        for h in cutlass.range_constexpr(n_tile // BLK):
            tS_h = cute.make_tensor(tmem_ptr + h * BLK, tSh_f.layout)[(None, None), 0, 0]
            tr_Sh = thr_h.partition_S(tS_h)
            col_base = jt * I32(n_tile) + I32(h * BLK)
            # Sample position of this half's first column (SCR sample scores).
            # Tile sampling visits every tile_stride'th tile; token sampling
            # visits every sample-space tile.
            spos_base = (jt // I32(tile_stride)) * I32(n_tile) + I32(h * BLK)
            if cutlass.const_expr(col_scale != 1):
                # token sampling: global column id = col_scale * position
                # (the hist branch below scales per-element offsets to match;
                # only the sample passes use col_scale != 1).
                col_base = col_base * I32(col_scale)
            # slice sl is drained by warp-group sl (grp == tidx // 128); the
            # four slices of this half proceed in parallel.
            for sl in cutlass.range_constexpr(nslh):
                if sl == grp:
                    _ikd = _ik_b("epi_tmem_drain")
                    cute.copy(cp_h, tr_Sh[None, None, sl], rSh)
                    _ik_e(_ikd)
                    _iks = _ik_b("epi_stage")
                    if cutlass.const_expr(write_s):
                        if st == 0:
                            tr_sO0[None, None, sl].store(rSh.load() * F32(SCALE))
                        else:
                            tr_sO1[None, None, sl].store(rSh.load() * F32(SCALE))
                    _ik_e(_iks)
                    # Ld32x32b rep32 hands each thread 32 consecutive columns
                    # of ONE tile row; bin from the registers while hot.
                    _ikh = _ik_b("epi_hist")
                    crd0 = tr_cO[None, None, sl][0]
                    crdL = tr_cO[None, None, sl][cutlass.const_expr(cute.size(rSh.shape) - 1)]
                    r0 = crd0[0]
                    my_st = sSE[r0]
                    my_en = sSE[BLK + r0]
                    if cutlass.const_expr(collect or filtered):
                        # Stage-2 pass-2: classify vs the row's threshold bin
                        # and append compact (idx, val) candidates. Matches
                        # are sparse (~TOPK + tie-bin per whole row), so the
                        # per-element gmem cursor atomics are cheap.
                        rg1 = bi * BLK + r0
                        tb = sSE[2 * BLK + r0]
                        if cutlass.const_expr(filtered):
                            # Span fast path (the IKET 145-clk/elem lesson):
                            # in-span tiles run branch-free except the
                            # rare-taken append compare; only per-row
                            # boundary tiles pay the per-element bounds.
                            c_lo = col_base + crd0[1]
                            c_hi = col_base + crdL[1]
                            curp = mCursors.iterator + rg1 * 2
                            bnd = F32(sBnd[r0])
                            # CuTe lowers this runtime branch through an SSA
                            # join. Seed every branch-local scalar with its DSL
                            # type so the fast and boundary paths join stably.
                            filtered_mask = I32(0)
                            filtered_count = I32(0)
                            filtered_hit = I32(0)
                            filtered_base = I32(0)
                            filtered_offset = I32(0)
                            filtered_position = I32(0)
                            filtered_col = I32(0)
                            filtered_value = F32(0.0)
                            filtered_compare = F32(0.0)
                            scr_hits = I32(0)
                            # cbnd is what the classify loop compares against.
                            cbnd = F32(bnd)
                            if cutlass.const_expr(raw_cmp):
                                cb = bnd * F32(1.0 / SCALE)
                                cbnd = cb * cutlass.select_(cb > F32(0.0), F32(1.0 - RAW_CMP_SLACK), F32(1.0 + RAW_CMP_SLACK))
                            if c_lo >= my_st and c_hi < my_en:
                                # Two-phase branchless collect: at ~4%/lane
                                # hit rate a per-element append branch makes
                                # ~3/4 of the warp-steps take the atomic path
                                # (warp convergence). Phase 1 is pure ALU
                                # (hit mask + count, fully pipelined); phase
                                # 2 reserves ONE cursor range per thread and
                                # writes only the hits.
                                _ikm1 = _ik_b("flt_mask")
                                for j in cutlass.range_constexpr(cute.size(rSh.shape)):
                                    filtered_value = F32(rSh[j])
                                    filtered_compare = filtered_value
                                    if cutlass.const_expr(not raw_cmp):
                                        filtered_compare = filtered_compare * F32(SCALE)
                                    if cutlass.const_expr(mask_walk):
                                        sMaskVal[j, tidx] = filtered_compare
                                    if cutlass.const_expr(fast_cnt):
                                        # Selecting the constant bit folds the
                                        # per-j shift into the SEL; the count
                                        # comes from one popc after the loop
                                        # instead of 32 running adds.
                                        bit = (1 << j) - (1 << 32) if j == 31 else (1 << j)
                                        filtered_mask = filtered_mask | cutlass.select_(filtered_compare > cbnd, I32(bit), I32(0))
                                    else:
                                        filtered_hit = cutlass.select_(filtered_compare > cbnd, I32(1), I32(0))
                                        filtered_mask = filtered_mask | (filtered_hit << j)
                                        filtered_count = filtered_count + filtered_hit
                                if cutlass.const_expr(fast_cnt):
                                    filtered_count = I32(cute.arch.popc(filtered_mask))
                                if cutlass.const_expr(scr_counts):
                                    # SCR S: hits at sampled key positions. A
                                    # thread's 32 columns start at a multiple
                                    # of 32, so token positions are fixed bits.
                                    if cutlass.const_expr(SAMPLE_MODE == "token"):
                                        scr_hits = I32(cute.arch.popc(filtered_mask & I32(_SCR_TOKEN_BITS)))
                                    else:
                                        scr_hits = cutlass.select_(jt % I32(SAMPLE_STRIDE) == 0, filtered_count, I32(0))
                                    if scr_hits > 0:
                                        atomic_add(curp + I32(1), scr_hits)
                                _ik_e(_ikm1)
                                _ikm2 = _ik_b("flt_append")
                                if filtered_count > 0:
                                    filtered_base = atomic_add(curp, filtered_count)
                                    if cutlass.const_expr(mask_walk):
                                        scan = filtered_mask
                                        while scan != 0:
                                            low = scan & (I32(0) - scan)
                                            j = I32(cute.arch.popc(low - I32(1)))
                                            filtered_position = filtered_base + filtered_offset
                                            filtered_offset = filtered_offset + 1
                                            if filtered_position < I32(tie_cap):
                                                mTieIdx[rg1, filtered_position] = col_base + crd0[1] + j
                                                filtered_value = sMaskVal[j, tidx]
                                                if cutlass.const_expr(raw_cmp):
                                                    filtered_value = filtered_value * F32(SCALE)
                                                mTieVal[rg1, filtered_position] = filtered_value
                                            scan = scan & (scan - I32(1))
                                    else:
                                        for j in cutlass.range_constexpr(cute.size(rSh.shape)):
                                            if ((filtered_mask >> j) & 1) == 1:
                                                filtered_position = filtered_base + filtered_offset
                                                filtered_offset = filtered_offset + 1
                                                if filtered_position < I32(tie_cap):
                                                    crdj = tr_cO[None, None, sl][j]
                                                    mTieIdx[rg1, filtered_position] = col_base + crdj[1]
                                                    mTieVal[rg1, filtered_position] = F32(rSh[j]) * F32(SCALE)
                                _ik_e(_ikm2)
                            else:
                                for j in cutlass.range_constexpr(cute.size(rSh.shape)):
                                    crdj = tr_cO[None, None, sl][j]
                                    filtered_col = col_base + crdj[1]
                                    if filtered_col >= my_st and filtered_col < my_en:
                                        filtered_value = F32(rSh[j])
                                        filtered_compare = filtered_value
                                        if cutlass.const_expr(not raw_cmp):
                                            filtered_compare = filtered_compare * F32(SCALE)
                                        if filtered_compare > cbnd:
                                            filtered_position = atomic_add(curp, I32(1))
                                            if filtered_position < I32(tie_cap):
                                                mTieIdx[rg1, filtered_position] = filtered_col
                                                if cutlass.const_expr(raw_cmp):
                                                    filtered_compare = filtered_value * F32(SCALE)
                                                mTieVal[rg1, filtered_position] = filtered_compare
                                            if cutlass.const_expr(scr_counts):
                                                if cutlass.const_expr(SAMPLE_MODE == "token"):
                                                    if filtered_col % I32(SAMPLE_STRIDE) == 0:
                                                        atomic_add(curp + I32(1), I32(1))
                                                else:
                                                    if jt % I32(SAMPLE_STRIDE) == 0:
                                                        atomic_add(curp + I32(1), I32(1))
                            _ik_e(_ikh)
                        else:
                            for j in cutlass.range_constexpr(cute.size(rSh.shape)):
                                col = col_base + tr_cO[None, None, sl][j][1]
                                if col >= my_st and col < my_en:
                                    v = F32(rSh[j]) * F32(SCALE)
                                    key = _coarse8(v)
                                    if key < tb:
                                        pos = atomic_add(mCursors.iterator + rg1 * 2, I32(1))
                                        if pos < I32(top_k):
                                            mDefIdx[rg1, pos] = col
                                            mDefVal[rg1, pos] = v
                                    elif key == tb:
                                        pos = atomic_add(mCursors.iterator + rg1 * 2 + 1, I32(1))
                                        if pos < I32(tie_cap):
                                            mTieIdx[rg1, pos] = col
                                            mTieVal[rg1, pos] = v
                            _ik_e(_ikh)
                    else:
                        hbase = (r0 >> 1) * I32(HIST_STRIDE)
                        hval = I32(1) << ((r0 & 1) * 16)
                        # Span fast path: when the thread's whole column span
                        # is in-bounds (every tile except the per-row boundary
                        # ones), drop the per-element branch -- the branchy
                        # loop measured 145 clk/element (no pipelining).
                        # col_scale != 1 (token sampling): global ids are
                        # col_scale * sample position, monotone in j, so the
                        # span test is exact unchanged.
                        if cutlass.const_expr(col_scale != 1):
                            c_lo = col_base + crd0[1] * I32(col_scale)
                            c_hi = col_base + crdL[1] * I32(col_scale)
                        else:
                            c_lo = col_base + crd0[1]
                            c_hi = col_base + crdL[1]
                        # Keep runtime span-branch values type-stable at the
                        # CuTe SSA join; these initializers are not data inputs.
                        hist_col = I32(0)
                        hist_key = I32(0)
                        if cutlass.const_expr(refine):
                            # Only samples in the row's crossing bin count;
                            # they land on the sub-code (low key byte).
                            refine_bin = sSE[2 * BLK + r0]
                            if c_lo >= my_st and c_hi < my_en:
                                for j in cutlass.range_constexpr(cute.size(rSh.shape)):
                                    hist_key = _key16(F32(rSh[j]) * F32(SCALE))
                                    if (hist_key >> 8) == refine_bin:
                                        _red_smem_add_u32(sHist.iterator + hbase + (hist_key & I32(0xFF)), hval)
                            else:
                                for j in cutlass.range_constexpr(cute.size(rSh.shape)):
                                    if cutlass.const_expr(col_scale != 1):
                                        hist_col = col_base + tr_cO[None, None, sl][j][1] * I32(col_scale)
                                    else:
                                        hist_col = col_base + tr_cO[None, None, sl][j][1]
                                    if hist_col >= my_st and hist_col < my_en:
                                        hist_key = _key16(F32(rSh[j]) * F32(SCALE))
                                        if (hist_key >> 8) == refine_bin:
                                            _red_smem_add_u32(sHist.iterator + hbase + (hist_key & I32(0xFF)), hval)
                        else:
                            if c_lo >= my_st and c_hi < my_en:
                                for j in cutlass.range_constexpr(cute.size(rSh.shape)):
                                    hist_key = _coarse8(F32(rSh[j]) * F32(SCALE))
                                    _red_smem_add_u32(sHist.iterator + hbase + hist_key, hval)
                            else:
                                for j in cutlass.range_constexpr(cute.size(rSh.shape)):
                                    if cutlass.const_expr(col_scale != 1):
                                        hist_col = col_base + tr_cO[None, None, sl][j][1] * I32(col_scale)
                                    else:
                                        hist_col = col_base + tr_cO[None, None, sl][j][1]
                                    if hist_col >= my_st and hist_col < my_en:
                                        hist_key = _coarse8(F32(rSh[j]) * F32(SCALE))
                                        _red_smem_add_u32(sHist.iterator + hbase + hist_key, hval)
                            if cutlass.const_expr(mScrSampleScores is not None):
                                # SCR: four fixed rows' raw fp32 sample MMA
                                # values by sample position; no rescoring.
                                scr_row = bi * I32(BLK) + r0
                                scr_slot = I32(-1)
                                if scr_row == 0:
                                    scr_slot = I32(0)
                                if scr_row == q_rows // I32(3):
                                    scr_slot = I32(1)
                                if scr_row == (I32(2) * q_rows) // I32(3):
                                    scr_slot = I32(2)
                                if scr_row == q_rows - I32(1):
                                    scr_slot = I32(3)
                                if scr_slot >= 0:
                                    for j in cutlass.range_constexpr(cute.size(rSh.shape)):
                                        crdj = tr_cO[None, None, sl][j]
                                        if cutlass.const_expr(col_scale != 1):
                                            hist_col = col_base + crdj[1] * I32(col_scale)
                                        else:
                                            hist_col = col_base + crdj[1]
                                        if hist_col >= my_st and hist_col < my_en:
                                            mScrSampleScores[scr_slot, spos_base + crdj[1]] = F32(rSh[j])
                        _ik_e(_ikh)
            _ikt = _ik_b("epi_tma_store")
            cute.arch.fence_view_async_tmem_load()
            if cutlass.const_expr(write_s):
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier()
                if warp_idx == 0:
                    if st == 0:
                        cute.copy(tma_c, bSG_s0[(None,)], bSG_g[(None, bi, jt * (n_tile // BLK) + h)])
                    else:
                        cute.copy(tma_c, bSG_s1[(None,)], bSG_g[(None, bi, jt * (n_tile // BLK) + h)])
                    c_pipe.producer_commit()
                    c_pipe.producer_acquire()
            cute.arch.barrier()
            _ik_e(_ikt)
        st = st ^ 1
        jt = jt + I32(tile_stride)

    if cutlass.const_expr(write_s):
        c_pipe.producer_tail()

    # ---- flush the chunk's per-row histogram halves to gmem ----
    # thread handles row ltid, bin quarter [grp*64, grp*64+64).
    if cutlass.const_expr(not (collect or filtered)):
        _ikf = _ik_b("hist_flush")
        cute.arch.barrier()
        rg = bi * BLK + ltid
        if rg < q_rows:
            hbase = (ltid >> 1) * I32(HIST_STRIDE)
            sh = (ltid & 1) * 16
            for bq in cutlass.range_constexpr(64):
                b = grp * I32(64) + I32(bq)
                cnt = (sHist[hbase + b] >> sh) & I32(0xFFFF)
                if cnt > 0:
                    cute.arch.atomic_add(mHist.iterator + rg * I32(256) + b, cnt, scope="gpu")
        _ik_e(_ikf)

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


class _SimplifiedSparseScoreHistFused:
    """Launcher: rect dense (q_pad, k_pad) fp32 workspace + per-row coarse
    histogram, chunked grid (n_chunks, q_nblk)."""

    def __init__(self, query_len: int, query_pad: int, key_pad: int, write_s: bool = True):
        self.q_nblk = query_pad // BLK
        self.k_tiles = key_pad // NT
        self.n_chunks = (self.k_tiles + C_TILES - 1) // C_TILES
        self.q_rows = query_len
        self.write_s = write_s

    @cute.jit
    def __call__(self, q: cute.Tensor, k: cute.Tensor, s: cute.Tensor, hist: cute.Tensor, starts: cute.Tensor, ends: cute.Tensor):
        S_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, tcgen05.CtaGroup.ONE, (BLK, BLK))
        S_mma_w = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, tcgen05.CtaGroup.ONE, (BLK, NT))
        lQs = sm100_utils.make_smem_layout_a(S_mma, (BLK, BLK, D), BF16, 1)
        lKs_w = sm100_utils.make_smem_layout_b(S_mma_w, (BLK, NT, D), BF16, 1)
        a128 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, BF16)
        lFlat = cute.tile_to_shape(a128, (BLK, D), order=(0, 1))
        lFlatK = cute.tile_to_shape(a128, (NT, D), order=(0, 1))
        # Swizzled staging layout (the dense_gemm C-store recipe): IKET
        # measured the plain row-major staging at 30% of the kernel -- the
        # tmem copy hands each lane one ROW, so at any instant the 32 lanes
        # store 512B apart = one bank. The epi swizzle spreads them.
        epi_staged = sm100_utils.make_smem_layout_epi(F32, utils.LayoutEnum.from_tensor(s), (BLK, BLK), 1)
        lEpi = cute.slice_(epi_staged, (None, None, 0))
        tma_c, mC_tma = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), s, lEpi, (BLK, BLK))
        _score_hist_kernel_tcgen05(
            q,
            k,
            hist,
            starts,
            ends,
            S_mma_w,
            lQs,
            lKs_w,
            lFlat,
            lFlatK,
            S_mma,
            q_rows=self.q_rows,
            k_tiles_total=self.k_tiles,
            write_s=self.write_s,
            mC_tma=mC_tma,
            lEpi=lEpi,
            tma_c=tma_c,
        ).launch(
            grid=(self.n_chunks, self.q_nblk, 1),
            block=(THREADS, 1, 1),
        )


@cute.kernel
def _threshold_kernel(
    mHist: cute.Tensor,
    mStarts: cute.Tensor,
    mEnds: cute.Tensor,
    mThr: cute.Tensor,
    q_rows: I32,
    top_k: cutlass.Constexpr,
    trivial_k: cutlass.Constexpr = 0,
    margin: cutlass.Constexpr = 0,
    mBnd: cute.Tensor = None,
    fine: cutlass.Constexpr = 0,
):
    """Per row: first coarse bin with inclusive-prefix count > top_k, plus
    need = top_k - count(strictly above bins). Rows with length <= trivial_k
    (default: top_k) take everything: tb = 256 (every key classifies in).

    Seeded-filter mode: the histogram holds a 1/stride column SAMPLE, top_k
    is TOPK/stride, trivial_k stays TOPK (real row length decides), and
    margin widens the estimated bin conservatively; correctness never depends
    on the estimate (the collect cursor count is the exact verifier)."""
    _triv = cutlass.const_expr(trivial_k if trivial_k > 0 else top_k)
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    r = bidx * 256 + tidx
    if r < q_rows:
        length = I32(mEnds[r]) - I32(mStarts[r])
        tb = I32(256)
        need = I32(0)
        if length > I32(_triv):
            cum = I32(0)
            above = I32(0)
            run = I32(1)
            tb = I32(255)
            for b in cutlass.range(256):
                h = I32(mHist[r, b])
                if run == 1 and cum + h > I32(top_k):
                    tb = I32(b)
                    above = cum
                    run = I32(0)
                cum = cum + h
            need = I32(top_k) - above
            if cutlass.const_expr(margin > 0):
                tb = tb + I32(margin)
                if tb > 255:
                    tb = I32(255)
        mThr[r, 0] = tb
        mThr[r, 1] = need
        if cutlass.const_expr(mBnd is not None):
            # fp32 collect boundary: v > bnd  <=>  ordered key < the cut code.
            # tb >= 255 (incl. the collect-all 256) -> -inf (take all reals);
            # tb == 127 would straddle signed zero (float compares cannot
            # separate the +-0 codes) -> widen one bin to 128.
            #
            # Coarse cut = (tb << 8) + 256, i.e. the WHOLE bin tb is
            # collected. The fine cut keeps only the top slice of bin tb: the
            # bin holds h samples spread over its 256 sub-codes, so cutting at
            # sub-code
            #     f = ceil(256 * need / h)
            # retains an expected `need` of them under a uniform-within-bin
            # estimate and drops the remaining ~h - need. Since f <= 256 the
            # fine cut can only ever RAISE the boundary, so it is a strict
            # subset of the coarse candidate set -- a skewed bin can therefore
            # only cause a MISS (certificate-flagged, exactly repaired), never
            # a wrong answer. h > need is guaranteed by the crossing condition
            # that selected tb, so the divide is safe; need can be 0 (the
            # prefix landed exactly on top_k), which floors f at 1 rather than
            # collecting nothing.
            #
            # The fine cut is disabled when margin > 0: `need` counts samples
            # in the ORIGINAL crossing bin, while margin then shifts tb to a
            # lower bin, so mHist[r, tb] would no longer be the histogram of
            # the bin the interpolation refers to. margin exists precisely to
            # widen the boundary deliberately, so the whole shifted bin is the
            # intended behaviour there. Both operands are Constexpr, so this
            # costs nothing at runtime and generates identical code at the
            # margin == 0 default.
            bnd = -cutlass.Float32.inf
            if tb < 255:
                tbb = tb
                sub = I32(256)
                if cutlass.const_expr(fine != 0 and margin == 0):
                    # Skip the +-0 straddle bin: there tbb is widened to 128
                    # below and mHist[r, tbb] no longer describes the bin the
                    # cut lands in.
                    if tbb != 127:
                        h = I32(mHist[r, tbb])
                        if h > need:
                            f = (I32(256) * need + h - I32(1)) // h
                            if f < I32(1):
                                f = I32(1)
                            if f < I32(256):
                                sub = f
                if tbb == 127:
                    tbb = I32(128)
                bnd = cutlass.Float32(_f16_key_boundary((tbb << 8) + sub))
            mBnd[r] = bnd


@cute.kernel
def _refine_threshold_kernel(
    mSub: cute.Tensor,
    mThr: cute.Tensor,
    mBnd: cute.Tensor,
    q_rows: I32,
):
    """Per row: replace the coarse cut with the measured fine cut. mSub is the
    refine pass's sub-code histogram of the crossing bin tb = mThr[r, 0], and
    need = mThr[r, 1] is how many of its samples the rank budget still needs.
    The cut keeps sub-codes [0, f) for the smallest f whose prefix holds
    `need`, so at least the budgeted number of samples stay above the
    boundary; f >= 1 as in the coarse rule. Both passes see the same samples,
    so the prefix always reaches `need` by f = 256 (the whole bin, i.e. the
    coarse cut). Collect-all rows (tb >= 255) and the +-0 straddle bin (127,
    which the coarse rule widens) keep the boundary _threshold_kernel wrote."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    r = bidx * 256 + tidx
    if r < q_rows:
        tb = I32(mThr[r, 0])
        if tb < 255 and tb != 127:
            need = I32(mThr[r, 1])
            cum = I32(0)
            cut = I32(256)
            run = I32(1)
            for s in cutlass.range(256):
                cum = cum + I32(mSub[r, s])
                if run == 1 and cum >= need:
                    cut = I32(s) + I32(1)
                    run = I32(0)
            mBnd[r] = cutlass.Float32(_f16_key_boundary((tb << 8) + cut))


@cute.kernel
def _compact_select_kernel(
    mDefIdx: cute.Tensor,
    mDefVal: cute.Tensor,
    mTieIdx: cute.Tensor,
    mTieVal: cute.Tensor,
    mCursors: cute.Tensor,
    mThr: cute.Tensor,
    mStarts: cute.Tensor,
    mEnds: cute.Tensor,
    mOutIdx: cute.Tensor,
    mOutProb: cute.Tensor,
    tie_cap: cutlass.Constexpr,
    top_k: cutlass.Constexpr,
    single_list: cutlass.Constexpr = False,
    mFlags: cute.Tensor = None,
    mFallbackPred: cute.Tensor = None,
    mFlagKinds: cute.Tensor = None,
):
    # mFlagKinds (optional, single_list): cumulative failure rows by cause,
    # [0] seed too tight, [1] candidate buffer overflow.
    # single_list (fusion="filtered"): the tie arrays hold ONE mixed candidate
    # list = every element with key <= the seeded threshold; the definite
    # arrays are unused. Exactness verification is the cursor count: if the
    # row wrote >= top_k candidates (or covers its whole length), the true
    # top-k is provably inside; otherwise the seed was too tight and the row
    # is appended to mFlags (count in slot 0, row ids after it) for fallback.
    # Winners = refine with
    # need = top_k over the whole list (or take-all when it fits).
    """Stage-2 pass-3: one CTA per row, SEL_THR (=512) threads, each owning
    top_k // SEL_THR winner slots (K in {512, 1024, 2048}).

    winners = all definite (count == above < top_k by construction) + the
    need largest tie candidates (exact 4-round byte radix refine on the
    cached fp32 values, ties at the exact boundary filled in arrival order --
    bitwise-equal values are interchangeable under the SSA gate). Winners
    are bitonic-sorted ascending by column index (pads = clamp(end-1, 0)
    sort to the tail; a pad ties with a real winner only at idx end-1, where
    the val-descending tiebreak puts the real one first), then the SSA
    predict softmax runs over the winner scores (pads contribute 0)."""
    tidx, _, _ = cute.arch.thread_idx()
    r, _, _ = cute.arch.block_idx()
    E = cutlass.const_expr(top_k // SEL_THR)

    smem = utils.SmemAllocator()
    sIdx = cute.make_tensor(smem.allocate_array(I32, top_k, byte_alignment=128), cute.make_layout((top_k,)))
    sVal = cute.make_tensor(smem.allocate_array(F32, top_k, byte_alignment=128), cute.make_layout((top_k,)))
    sH = cute.make_tensor(smem.allocate_array(I32, 256, byte_alignment=128), cute.make_layout((256,)))
    sRed = cute.make_tensor(smem.allocate_array(F32, 32, byte_alignment=128), cute.make_layout((32,)))
    sC = cute.make_tensor(smem.allocate_array(I32, 4, byte_alignment=128), cute.make_layout((4,)))
    # winner-emission bitmap + per-thread popcount-prefix partials (rows
    # longer than 32 * BMAP_WORDS fall back to the bitonic path)
    sBm = cute.make_tensor(smem.allocate_array(I32, BMAP_WORDS, byte_alignment=128), cute.make_layout((BMAP_WORDS,)))
    sScan = cute.make_tensor(smem.allocate_array(I32, SEL_THR, byte_alignment=128), cute.make_layout((SEL_THR,)))

    nd = I32(0)
    ntw = I32(0)
    if cutlass.const_expr(single_list):
        ntw = I32(mCursors[r, 0])
    else:
        nd = I32(mCursors[r, 0])
        if nd > I32(top_k):
            nd = I32(top_k)
        ntw = I32(mCursors[r, 1])
    ntie = ntw
    if ntie > I32(tie_cap):
        ntie = I32(tie_cap)
    need = I32(0)
    if cutlass.const_expr(single_list):
        # exactness check + refine budget from the cursor itself
        length = I32(mEnds[r]) - I32(mStarts[r])
        if tidx == 0:
            _bad = I32(0)
            if length > I32(top_k) and ntw < I32(top_k):
                _bad = I32(1)  # seed too tight: true top-k not covered
                if cutlass.const_expr(mFlagKinds is not None):
                    cute.arch.atomic_add(mFlagKinds.iterator, I32(1), scope="gpu")
            if ntw > I32(tie_cap):
                _bad = I32(1)  # candidate buffer overflowed
                if cutlass.const_expr(mFlagKinds is not None):
                    cute.arch.atomic_add(mFlagKinds.iterator + I32(1), I32(1), scope="gpu")
            if _bad == 1:
                pos = cute.arch.atomic_add(mFlags.iterator, I32(1), scope="gpu")
                mFlags[pos + I32(1)] = I32(r)
                mFallbackPred[0] = True
        need = I32(top_k)
        if ntie < need:
            need = ntie
    else:
        need = I32(mThr[r, 1])
    row_end = I32(mEnds[r])

    _iksl = _ik_b("sel_load")
    # ---- 1. definite winners straight into the winner buffer ----
    if cutlass.const_expr(not single_list):
        for _sl in cutlass.range_constexpr(E):
            _i = I32(_sl * SEL_THR) + tidx
            if _i < nd:
                sIdx[_i] = mDefIdx[r, _i]
                sVal[_i] = mDefVal[r, _i]
    if tidx == 0:
        sC[0] = nd  # winner-append cursor
    cute.arch.barrier()

    if cutlass.const_expr(single_list):
        # take-all shortcut: the whole list fits in the winner buffer
        if ntie <= I32(top_k):
            for _sl in cutlass.range_constexpr(E):
                _ti = I32(_sl * SEL_THR) + tidx
                if _ti < ntie:
                    sIdx[_ti] = mTieIdx[r, _ti]
                    sVal[_ti] = F32(mTieVal[r, _ti])
            if tidx == 0:
                sC[0] = ntie
            need = I32(0)  # skip the refine
    cute.arch.barrier()

    _ik_e(_iksl)
    # ---- 2. exact radix refine over the tie candidates (need > 0) ----
    _iksr = _ik_b("sel_refine")
    if need > 0:
        prefix = cutlass.Uint32(0)
        pmask = cutlass.Uint32(0)
        need_rem = need
        for rnd in cutlass.range_constexpr(4):
            shift = cutlass.const_expr((3 - rnd) * 8)
            for b in range(tidx, 256, SEL_THR):
                sH[b] = 0
            cute.arch.barrier()
            for i in range(tidx, ntie, SEL_THR):
                o = _ordered32(F32(mTieVal[r, i]))
                if (o & pmask) == prefix:
                    bb = I32((o >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF))
                    _red_smem_add_u32(sH.iterator + bb, I32(1))
            cute.arch.barrier()
            if tidx == 0:
                cum = I32(0)
                above_r = I32(0)
                tbin = I32(255)
                run = I32(1)
                for b in cutlass.range(256):
                    h = sH[b]
                    if run == 1 and cum + h > need_rem:
                        tbin = I32(b)
                        above_r = cum
                        run = I32(0)
                    cum = cum + h
                sC[1] = tbin
                sC[2] = above_r
            cute.arch.barrier()
            tbin_v = sC[1]
            prefix = prefix | (cutlass.Uint32(tbin_v) << cutlass.Uint32(shift))
            pmask = pmask | (cutlass.Uint32(0xFF) << cutlass.Uint32(shift))
            need_rem = need_rem - sC[2]
        # prefix is now the exact ordered key T of the boundary winner;
        # take everything strictly larger (ordered < T) plus the first
        # need_rem exactly-T candidates by arrival order.
        if tidx == 0:
            sC[3] = need_rem
        cute.arch.barrier()
        _ik_e(_iksr)
        _iksc = _ik_b("sel_collect")
        for i in range(tidx, ntie, SEL_THR):
            o = _ordered32(F32(mTieVal[r, i]))
            if o < prefix:
                pos = atomic_add(sC.iterator, I32(1))
                if pos < I32(top_k):
                    sIdx[pos] = mTieIdx[r, i]
                    sVal[pos] = F32(mTieVal[r, i])
            elif o == prefix:
                t = atomic_add(sC.iterator + 3, I32(-1))
                if t > 0:
                    pos = atomic_add(sC.iterator, I32(1))
                    if pos < I32(top_k):
                        sIdx[pos] = mTieIdx[r, i]
                        sVal[pos] = F32(mTieVal[r, i])
        _ik_e(_iksc)
    cute.arch.barrier()

    # ---- 3. pad, sort by index ascending, predict softmax ----
    _ikss = _ik_b("sel_sort")
    total = sC[0]
    _padv = row_end - I32(1)
    if _padv < 0:
        _padv = I32(0)
    row_start_v = I32(mStarts[r])
    length_v = row_end - row_start_v
    use_bmap = length_v <= I32(32 * BMAP_WORDS) and length_v > 0
    # register stash for the bitmap scatter (created outside the dynamic
    # branch -- rmem allocation inside an scf region is a DSL trap)
    wIdx = cute.make_rmem_tensor(cute.make_layout((E,)), I32)
    wVal = cute.make_rmem_tensor(cute.make_layout((E,)), F32)
    if use_bmap:
        # ---- bitmap rank-scatter: winner column ids are UNIQUE, so each
        # winner's sorted position is exactly the popcount prefix of its bit
        # in a per-row bitmap -- no sort at all (the bitonic path measured
        # 27us/row at K=2048: 66 barrier rounds). ~13 barriers total. ----
        # 1. stash my winners in registers, zero the bitmap
        for _sl in cutlass.range_constexpr(E):
            _bmi = I32(_sl * SEL_THR) + tidx
            wIdx[_sl] = I32(0)
            wVal[_sl] = -cutlass.Float32.inf
            if _bmi < total:
                wIdx[_sl] = sIdx[_bmi]
                wVal[_sl] = F32(sVal[_bmi])
        for _w in cutlass.range_constexpr(BMAP_WPT):
            sBm[I32(_w * SEL_THR) + tidx] = 0
        cute.arch.barrier()
        # 2. set winner bits (local coords bound the bitmap by row length)
        for _sl in cutlass.range_constexpr(E):
            _bmi = I32(_sl * SEL_THR) + tidx
            if _bmi < total:
                loc = wIdx[_sl] - row_start_v
                _red_smem_or_u32(sBm.iterator + (loc >> 5), I32(1) << (loc & I32(31)))
        cute.arch.barrier()
        # 3. per-thread popcount partials over BMAP_WPT consecutive words,
        # then a Hillis-Steele block scan (exclusive prefix per thread chunk)
        pc = I32(0)
        for _w in cutlass.range_constexpr(BMAP_WPT):
            pc = pc + I32(cute.arch.popc(sBm[I32(tidx * BMAP_WPT) + _w]))
        sScan[tidx] = pc
        cute.arch.barrier()
        _off = 1
        for _st in cutlass.range_constexpr(9):
            v_prev = I32(0)
            if tidx >= I32(_off):
                v_prev = sScan[tidx - I32(_off)]
            cute.arch.barrier()
            if tidx >= I32(_off):
                sScan[tidx] = sScan[tidx] + v_prev
            cute.arch.barrier()
            _off = _off * 2
        # 4. scatter my winners to their ranks (registers -> smem)
        for _sl in cutlass.range_constexpr(E):
            _bmi = I32(_sl * SEL_THR) + tidx
            if _bmi < total:
                loc = wIdx[_sl] - row_start_v
                wq = loc >> 5
                tq_ = wq // I32(BMAP_WPT)
                base = I32(0)
                if tq_ > 0:
                    base = sScan[tq_ - 1]
                wj = tq_ * I32(BMAP_WPT)
                while wj < wq:
                    base = base + I32(cute.arch.popc(sBm[wj]))
                    wj = wj + 1
                mask = (I32(1) << (loc & I32(31))) - I32(1)
                rank = base + I32(cute.arch.popc(sBm[wq] & mask))
                sIdx[rank] = wIdx[_sl]
                sVal[rank] = wVal[_sl]
        # 5. pads fill the tail slots
        for _sl in cutlass.range_constexpr(E):
            _bmi = I32(_sl * SEL_THR) + tidx
            if _bmi >= total:
                sIdx[_bmi] = _padv
                sVal[_bmi] = -cutlass.Float32.inf
        cute.arch.barrier()
    else:
        for _sl in cutlass.range_constexpr(E):
            _bmi = I32(_sl * SEL_THR) + tidx
            if _bmi >= total:
                sIdx[_bmi] = _padv
                sVal[_bmi] = -cutlass.Float32.inf
    # TODO(row length > 32 * BMAP_WORDS = 256K, i.e. 512K/1M contexts): the
    # fallback below costs ~66 barrier rounds vs the bitmap path's ~13. The
    # right upgrade is a SEGMENTED bitmap -- split the row into 256K-column
    # segments and run set-bits -> popcount-prefix -> scatter per segment,
    # accumulating the rank base across segments (segment i's winners rank at
    # [sum of winners in earlier segments] + in-segment prefix). SMEM stays
    # 32KB; barriers scale by segment count (2-4), still far under bitonic.
    # Do this when 512K+ tiers enter the workload; today every tested tier
    # (Tk <= 256K) takes the bitmap path and this branch is correctness-only.
    # bitonic fallback (rows longer than the bitmap; per-CTA-uniform branch,
    # so the barriers inside are legal): over top_k elements, E per thread
    # (pair (i, i^j) is handled by whichever slot enumerates the LOWER index
    # -- exactly once); key = (idx asc, val desc) so a real winner precedes
    # a same-index pad.
    if use_bmap == False:  # noqa: E712  (DSL runtime bool)
        for kk in cutlass.range_constexpr(int(top_k).bit_length() - 1):
            kbit = cutlass.const_expr(2 << kk)
            for jj in cutlass.range_constexpr(kk + 1):
                jbit = cutlass.const_expr((2 << kk) >> (jj + 1))
                cute.arch.barrier()
                for _sl in cutlass.range_constexpr(E):
                    i = I32(_sl * SEL_THR) + tidx
                    p = i ^ I32(jbit)
                    if p > i:
                        ia = sIdx[i]
                        ib = sIdx[p]
                        va = sVal[i]
                        vb = sVal[p]
                        gt = ia > ib or (ia == ib and va < vb)
                        lt = ia < ib or (ia == ib and va > vb)
                        do_swap = I32(0)
                        if (i & I32(kbit)) == 0:
                            if gt:
                                do_swap = I32(1)
                        else:
                            if lt:
                                do_swap = I32(1)
                        if do_swap == 1:
                            sIdx[i] = ib
                            sIdx[p] = ia
                            sVal[i] = vb
                            sVal[p] = va
    cute.arch.barrier()
    _ik_e(_ikss)
    _iksp = _ik_b("sel_predict")

    lm = -cutlass.Float32.inf
    for _sl in cutlass.range_constexpr(E):
        _x = F32(sVal[I32(_sl * SEL_THR) + tidx])
        if _x > lm:
            lm = _x
    # pads carry -inf already, so lm feeds the max reduce directly
    m = _ssa_block_reduce_max(lm, sRed, tidx, SEL_THR // 32)
    cute.arch.barrier()
    ls = cutlass.Float32(0.0)
    for _sl in cutlass.range_constexpr(E):
        _x = F32(sVal[I32(_sl * SEL_THR) + tidx])
        if _x > -cutlass.Float32.inf:
            ls = ls + cute.exp2((_x - m) * cutlass.Float32(_SSA_LOG2E), fastmath=True)
    s = _ssa_block_reduce_sum(ls, sRed, tidx, SEL_THR // 32)
    inv = cute.rcp(s)
    for _sl in cutlass.range_constexpr(E):
        _i = I32(_sl * SEL_THR) + tidx
        _x = F32(sVal[_i])
        mOutIdx[r, _i] = sIdx[_i]
        if _x > -cutlass.Float32.inf:
            mOutProb[r, _i] = cute.exp2((_x - m) * cutlass.Float32(_SSA_LOG2E), fastmath=True) * inv
        else:
            mOutProb[r, _i] = cutlass.Float32(0.0)
    _ik_e(_iksp)


@cute.kernel
def _filter_kmajor_kernel(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mThr: cute.Tensor,
    mBnd: cute.Tensor,
    mStarts: cute.Tensor,
    mEnds: cute.Tensor,
    mCursors: cute.Tensor,
    mTieIdx: cute.Tensor,
    mTieVal: cute.Tensor,
    S_mma: cute.TiledMma,
    lQs: cute.ComposedLayout,
    lKs: cute.ComposedLayout,
    lFlat: cute.ComposedLayout,
    lFlatK: cute.ComposedLayout,
    S_mma_h: cute.TiledMma,
    q_rows: cutlass.Constexpr,
    q_blocks: cutlass.Constexpr,
    tie_cap: cutlass.Constexpr,
):
    """K-major filtered collect: ONE CTA per 256-col K tile, looping over all
    Q row-blocks. The row-major layout re-streamed the 64MB K matrix once per
    Q block (NCU: 4.3GB DRAM read, 22% L2 hit); here each K tile is loaded
    from DRAM exactly once and the 2MB Q matrix is L2-resident for every CTA
    (all CTAs walk the same Q sequence). Filtered mode only: no S, no
    histogram -- the epilogue is the fp32-boundary classify + compact append.
    SMEM: K 64KB resident + Q 32KB streamed + bounds ~2KB -> 2 CTAs/SM."""
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    jt, _, _ = cute.arch.block_idx()

    smem = utils.SmemAllocator()
    # K tile resident for the whole kernel; Q block re-staged per iteration
    pQK = smem.allocate_array(BF16, BLK * D + NT * D, byte_alignment=1024)
    pSE = smem.allocate_array(I32, 2 * BLK, byte_alignment=128)
    pBnd = smem.allocate_array(F32, BLK, byte_alignment=128)
    mbar = smem.allocate_array(cutlass.Int64, 1, byte_alignment=16)

    sQ = cute.make_tensor(cute.recast_ptr(pQK, lQs.inner), lQs.outer)
    sK0 = cute.make_tensor(cute.recast_ptr(pQK + BLK * D, lKs.inner), lKs.outer)
    sQ2 = cute.make_tensor(cute.recast_ptr(pQK, lFlat.inner), lFlat.outer)
    sK2_0 = cute.make_tensor(cute.recast_ptr(pQK + BLK * D, lFlatK.inner), lFlatK.outer)
    sSE = cute.make_tensor(pSE, cute.make_layout((2 * BLK,)))
    sBnd = cute.make_tensor(pBnd, cute.make_layout((BLK,)))

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar + 0, 1)
    cute.arch.mbarrier_init_fence()

    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(barrier_for_retrieve=tmem_bar)
    tmem.allocate(NT)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(F32)
    tmem.relinquish_alloc_permit()

    tS_f = S_mma.make_fragment_C(S_mma.partition_shape_C((BLK, NT)))
    tS = cute.make_tensor(tmem_ptr, tS_f.layout)
    fQ = S_mma.make_fragment_A(sQ)
    fK0 = S_mma.make_fragment_B(sK0)

    g2s_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        BF16,
        num_bits_per_copy=128,
    )
    tiled_g2s = cute.make_tiled_copy_tv(g2s_atom, cute.make_layout((1, 16)), cute.make_layout((1, 8)))
    half = lane // 16
    thr_g2s = tiled_g2s.get_slice(lane % 16)
    qbase = mQ.iterator.toint()
    kbase = mK.iterator.toint()
    nwarp = cutlass.const_expr(THREADS // 32)

    # ---- stage the CTA's K tile once ----
    for i in cutlass.range_constexpr(NT // (nwarp * 2)):
        rr = i * (nwarp * 2) + warp_idx * 2 + half
        cute.copy(
            tiled_g2s,
            thr_g2s.partition_S(
                cute.make_tensor(cute.make_ptr(BF16, kbase + (jt * NT + rr) * (D * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, D)))
            ),
            thr_g2s.partition_D(cute.local_tile(sK2_0, (1, D), (rr, 0))),
        )
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier()

    ld32 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), F32)
    tSh_f = S_mma_h.make_fragment_C(S_mma_h.partition_shape_C((BLK, BLK)))
    cp_h = tcgen05.make_tmem_copy(ld32, cute.make_tensor(tmem_ptr, tSh_f.layout)[(None, None), 0, 0])
    ltid = tidx % 128
    grp = tidx // 128
    thr_h = cp_h.get_slice(ltid)
    idC = cute.make_identity_tensor((BLK, BLK))
    tr_cO = thr_h.partition_D(idC)
    nslh = cutlass.const_expr(cute.size(tr_cO, mode=[2]))
    rSh = cute.make_rmem_tensor(tr_cO[None, None, 0].shape, F32)

    nk = cutlass.const_expr(cute.size(fQ, mode=[2]))
    ph = I32(0)
    qb = I32(0)
    while qb < I32(q_blocks):
        # ---- stage Q block qb + this block's bounds/boundary ----
        for i in cutlass.range_constexpr(BLK // (nwarp * 2)):
            rr = i * (nwarp * 2) + warp_idx * 2 + half
            cute.copy(
                tiled_g2s,
                thr_g2s.partition_S(
                    cute.make_tensor(cute.make_ptr(BF16, qbase + (qb * BLK + rr) * (D * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, D)))
                ),
                thr_g2s.partition_D(cute.local_tile(sQ2, (1, D), (rr, 0))),
            )
        cute.arch.cp_async_commit_group()
        if tidx < BLK:
            rg0 = qb * BLK + tidx
            st0 = I32(0)
            en0 = I32(0)
            bn0 = cutlass.Float32.inf
            if rg0 < q_rows:
                st0 = I32(mStarts[rg0])
                en0 = I32(mEnds[rg0])
                bn0 = F32(mBnd[rg0])
            sSE[tidx] = st0
            sSE[BLK + tidx] = en0
            sBnd[tidx] = bn0
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()

        if warp_idx == 0:
            S_mma.set(tcgen05.Field.ACCUMULATE, False)
            for kb in cutlass.range_constexpr(nk):
                cute.gemm(S_mma, tS, fQ[None, None, kb, 0], fK0[None, None, kb, 0], tS)
                S_mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(mbar + 0)
        cute.arch.mbarrier_wait(mbar + 0, ph)
        ph = ph ^ 1

        for h in cutlass.range_constexpr(2):
            tS_h = cute.make_tensor(tmem_ptr + h * BLK, tSh_f.layout)[(None, None), 0, 0]
            tr_Sh = thr_h.partition_S(tS_h)
            col_base = jt * I32(NT) + I32(h * BLK)
            for sl in cutlass.range_constexpr(nslh):
                if sl == grp:
                    cute.copy(cp_h, tr_Sh[None, None, sl], rSh)
                    crd0 = tr_cO[None, None, sl][0]
                    crdL = tr_cO[None, None, sl][cutlass.const_expr(cute.size(rSh.shape) - 1)]
                    r0 = crd0[0]
                    my_st = sSE[r0]
                    my_en = sSE[BLK + r0]
                    rg1 = qb * BLK + r0
                    bnd = F32(sBnd[r0])
                    c_lo = col_base + crd0[1]
                    c_hi = col_base + crdL[1]
                    curp = mCursors.iterator + rg1 * 2
                    filtered_mask = I32(0)
                    filtered_count = I32(0)
                    filtered_hit = I32(0)
                    filtered_base = I32(0)
                    filtered_offset = I32(0)
                    filtered_position = I32(0)
                    filtered_col = I32(0)
                    filtered_value = F32(0.0)
                    if c_lo >= my_st and c_hi < my_en:
                        for j in cutlass.range_constexpr(cute.size(rSh.shape)):
                            filtered_hit = cutlass.select_(F32(rSh[j]) * F32(SCALE) > bnd, I32(1), I32(0))
                            filtered_mask = filtered_mask | (filtered_hit << j)
                            filtered_count = filtered_count + filtered_hit
                        if filtered_count > 0:
                            filtered_base = atomic_add(curp, filtered_count)
                            for j in cutlass.range_constexpr(cute.size(rSh.shape)):
                                if ((filtered_mask >> j) & 1) == 1:
                                    filtered_position = filtered_base + filtered_offset
                                    filtered_offset = filtered_offset + 1
                                    if filtered_position < I32(tie_cap):
                                        mTieIdx[rg1, filtered_position] = col_base + tr_cO[None, None, sl][j][1]
                                        mTieVal[rg1, filtered_position] = F32(rSh[j]) * F32(SCALE)
                    else:
                        for j in cutlass.range_constexpr(cute.size(rSh.shape)):
                            filtered_col = col_base + tr_cO[None, None, sl][j][1]
                            if filtered_col >= my_st and filtered_col < my_en:
                                filtered_value = F32(rSh[j]) * F32(SCALE)
                                if filtered_value > bnd:
                                    filtered_position = atomic_add(curp, I32(1))
                                    if filtered_position < I32(tie_cap):
                                        mTieIdx[rg1, filtered_position] = filtered_col
                                        mTieVal[rg1, filtered_position] = filtered_value
            cute.arch.fence_view_async_tmem_load()
            cute.arch.barrier()
        qb = qb + 1

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


class _SimplifiedSparseFilteredKMajor:
    """K-major filtered collect launcher: grid = one CTA per 256-col K tile."""

    def __init__(self, query_len: int, query_pad: int, key_pad: int, topk: int = TOPK):
        self.q_nblk = query_pad // BLK
        self.k_tiles = key_pad // NT
        self.q_rows = query_len
        self.topk = topk
        self.tcap = _tie_cap(topk, key_pad)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        thr: cute.Tensor,
        bnd: cute.Tensor,
        starts: cute.Tensor,
        ends: cute.Tensor,
        cursors: cute.Tensor,
        candidx: cute.Tensor,
        candval: cute.Tensor,
        stream,
    ):
        S_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, tcgen05.CtaGroup.ONE, (BLK, BLK))
        S_mma_w = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, tcgen05.CtaGroup.ONE, (BLK, NT))
        lQs = sm100_utils.make_smem_layout_a(S_mma, (BLK, BLK, D), BF16, 1)
        lKs_w = sm100_utils.make_smem_layout_b(S_mma_w, (BLK, NT, D), BF16, 1)
        a128 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, BF16)
        lFlat = cute.tile_to_shape(a128, (BLK, D), order=(0, 1))
        lFlatK = cute.tile_to_shape(a128, (NT, D), order=(0, 1))
        _filter_kmajor_kernel(
            q,
            k,
            thr,
            bnd,
            starts,
            ends,
            cursors,
            candidx,
            candval,
            S_mma_w,
            lQs,
            lKs_w,
            lFlat,
            lFlatK,
            S_mma,
            q_rows=self.q_rows,
            q_blocks=self.q_nblk,
            tie_cap=self.tcap,
        ).launch(
            grid=(self.k_tiles, 1, 1),
            block=(THREADS, 1, 1),
            min_blocks_per_mp=2,
            stream=stream,
        )


# ---------------------------------------------------------------------------
# k1 sampler: row and key extents are runtime arguments (one compile per top_k, as
# for the other filtered kernels); stratified positions, the lazy keep and the
# Poisson budget are the only modes; rows with at most the candidate capacity
# of keys take all, as the refine path does; branch-local scalars are seeded
# with their DSL types and the arithmetic avoids intrinsics newer than CuTe
# DSL 4.6.


def _k1_col(pos, bi, key_pad, stride):
    """Global key column sampled at position pos by q block bi: one
    pseudo-random key inside stratum [stride*pos, stride*(pos+1)), hashed from
    (pos, bi) so different q blocks see different phases. A stratum starting
    at or after key_pad returns its start (>= key_pad), which every row's
    bounds test rejects; callers clamp loads separately. Branch-free."""
    U32 = cutlass.Uint32
    U64 = cutlass.Uint64
    lo = pos * I32(stride)
    hi = lo + I32(stride)
    hi = cutlass.select_(hi > key_pad, key_pad, hi)
    width = hi - lo
    width = cutlass.select_(width > I32(1), width, I32(1))
    h = U32(pos) ^ (U32(bi) * U32(0x9E3779B9)) ^ U32(0xD1B54A35)
    h = h ^ (h >> U32(16))
    h = h * U32(0x7FEB352D)
    h = h ^ (h >> U32(15))
    h = h * U32(0x846CA68B)
    h = h ^ (h >> U32(16))
    off = (U64(h) * U64(U32(width))) >> U64(32)
    return lo + I32(U32(off))


def _k1_ceil_pos(x):
    """ceil(x) for x >= 0."""
    c = F32(I32(x))
    return cutlass.select_(c < x, c + F32(1.0), c)


@cute.jit
def _k1_poisson_budget(mu, ln_d, lane):
    """min b with P(Poisson(mu) >= b) <= exp(-ln_d), warp-parallel: lane l owns
    terms i = 4l..4l+3 of exp(i ln mu - mu - lgamma(i+1)); lgamma by a warp
    prefix of ln j, the upper tail by a warp suffix sum; the first i whose
    tail drops to the risk is the budget (ballot). 128 terms cover mu <= 48;
    above that (or if no lane qualifies) the Bernstein bound is returned."""
    i0 = lane * I32(4)
    l1 = cute.math.log(F32(i0 + I32(1)))
    l2 = cute.math.log(F32(i0 + I32(2)))
    l3 = cute.math.log(F32(i0 + I32(3)))
    l4 = cute.math.log(F32(i0 + I32(4)))
    p1 = l1
    p2 = p1 + l2
    p3 = p2 + l3
    p4 = p3 + l4
    # exclusive warp prefix of the lane totals p4 -> lgamma(4l + 1)
    acc = p4
    for d in cutlass.range_constexpr(5):
        o = cute.arch.shuffle_sync_up(acc, 1 << d, mask_and_clamp=0)
        acc = acc + cutlass.select_(lane >= I32(1 << d), o, F32(0.0))
    base = acc - p4
    mu_c = cutlass.select_(mu < F32(1e-6), F32(1e-6), mu)
    lnmu = cute.math.log(mu_c)
    t0 = cute.math.exp(F32(i0) * lnmu - mu_c - base)
    t1 = cute.math.exp(F32(i0 + I32(1)) * lnmu - mu_c - (base + p1))
    t2 = cute.math.exp(F32(i0 + I32(2)) * lnmu - mu_c - (base + p2))
    t3 = cute.math.exp(F32(i0 + I32(3)) * lnmu - mu_c - (base + p3))
    s3 = t3
    s2 = s3 + t2
    s1 = s2 + t1
    s0 = s1 + t0
    # inclusive warp suffix of the lane totals s0
    tot = s0
    for d in cutlass.range_constexpr(5):
        o = cute.arch.shuffle_sync_down(tot, 1 << d, mask_and_clamp=31)
        tot = tot + cutlass.select_(lane + I32(1 << d) < I32(32), o, F32(0.0))
    above = tot - s0  # tail strictly beyond this lane
    delta = F32(math.exp(-ln_d))
    sel_l = I32(-1)
    sel_l = cutlass.select_(above + s3 <= delta, i0 + I32(3), sel_l)
    sel_l = cutlass.select_(above + s2 <= delta, i0 + I32(2), sel_l)
    sel_l = cutlass.select_(above + s1 <= delta, i0 + I32(1), sel_l)
    sel_l = cutlass.select_(above + s0 <= delta, i0, sel_l)
    bal = I32(cute.arch.vote_ballot_sync(sel_l >= I32(0)))
    low = bal & (I32(0) - bal)
    src = I32(cute.arch.popc(low - I32(1)))
    sel = cute.arch.shuffle_sync(sel_l, src)
    # Bernstein bound; sqrt(x) = exp(log(x) / 2) for x > 0.
    bern = _k1_ceil_pos(mu_c + F32(ln_d / 3.0) + cute.math.exp(F32(0.5) * cute.math.log(F32(2.0 * ln_d) * mu_c)))
    bf = F32(sel)
    bf = cutlass.select_(bal == I32(0), bern, bf)
    bf = cutlass.select_(mu > F32(48.0), bern, bf)
    bf = cutlass.select_(bf < F32(1.0), F32(1.0), bf)
    return bf


@cute.jit
def _k1_row_budget(length, m, top_k, ln_d, keep_b, lane):
    """Per-row sampled rank budget (identical on every lane of the row's
    warp): the Poisson budget at mu = top_k * m / length, clamped to
    [1, keep_b] (budget <= keep_b is the keep's exactness invariant)."""
    len_f = F32(length)
    len_f = cutlass.select_(len_f < F32(1.0), F32(1.0), len_f)
    m_f = F32(m)
    m_f = cutlass.select_(m_f < F32(1.0), F32(1.0), m_f)
    bf = _k1_poisson_budget(F32(top_k) * m_f / len_f, ln_d, lane)
    budget = I32(bf)
    budget = cutlass.select_(budget < I32(1), I32(1), budget)
    budget = cutlass.select_(budget > I32(keep_b), I32(keep_b), budget)
    return budget


@cute.jit
def _k1_radix_select_desc(sKw, sBw, warp_idx, lane, n_vals, budget):
    """budget-th largest of the warp's n_vals bias-flipped keys in
    sKw[warp_idx*n_vals : +n_vals]: a 4-pass byte radix with a warp-parallel,
    branch-free crossing search (sBw: 256 bins per warp). Returns the
    selected key; the caller un-flips it."""
    n_it = cutlass.const_expr((n_vals + 31) // 32)
    prefix = I32(0)
    rem = budget
    radix_idx = I32(0)
    radix_key = I32(0)
    for p in cutlass.range_constexpr(4):
        shift = cutlass.const_expr(24 - 8 * p)
        hi_mask = cutlass.const_expr((0, -16777216, -65536, -256)[p])
        for b8 in cutlass.range_constexpr(8):
            sBw[warp_idx * I32(256) + I32(b8 * 32) + lane] = I32(0)
        cute.arch.sync_warp()
        for i in cutlass.range_constexpr(n_it):
            radix_idx = I32(i * 32) + lane
            radix_key = I32(sKw[warp_idx * I32(n_vals) + cutlass.select_(radix_idx < I32(n_vals), radix_idx, I32(n_vals - 1))])
            if radix_idx < I32(n_vals) and ((radix_key ^ prefix) & I32(hi_mask)) == 0:
                _red_smem_add_u32(sBw.iterator + warp_idx * I32(256) + ((radix_key >> shift) & I32(0xFF)), I32(1))
        cute.arch.sync_warp()
        # Lane l owns bins 255-8l .. 248-8l (lane order = descending bins);
        # an inclusive shuffle prefix counts everything above each lane's
        # group, and the lowest lane whose group reaches rem holds the
        # crossing bin.
        b0 = I32(255) - lane * I32(8)
        c0 = I32(sBw[warp_idx * I32(256) + b0])
        c1 = I32(sBw[warp_idx * I32(256) + b0 - I32(1)])
        c2 = I32(sBw[warp_idx * I32(256) + b0 - I32(2)])
        c3 = I32(sBw[warp_idx * I32(256) + b0 - I32(3)])
        c4 = I32(sBw[warp_idx * I32(256) + b0 - I32(4)])
        c5 = I32(sBw[warp_idx * I32(256) + b0 - I32(5)])
        c6 = I32(sBw[warp_idx * I32(256) + b0 - I32(6)])
        c7 = I32(sBw[warp_idx * I32(256) + b0 - I32(7)])
        s0 = c0
        s1 = s0 + c1
        s2 = s1 + c2
        s3 = s2 + c3
        s4 = s3 + c4
        s5 = s4 + c5
        s6 = s5 + c6
        s7 = s6 + c7
        pre = s7
        for d in cutlass.range_constexpr(5):
            o = cute.arch.shuffle_sync_up(pre, 1 << d, mask_and_clamp=0)
            pre = pre + cutlass.select_(lane >= I32(1 << d), o, I32(0))
        acc = pre - s7
        sel_l = I32(-1)
        abv_l = I32(0)
        sel_l = cutlass.select_(acc + s7 >= rem, b0 - I32(7), sel_l)
        abv_l = cutlass.select_(acc + s7 >= rem, acc + s6, abv_l)
        sel_l = cutlass.select_(acc + s6 >= rem, b0 - I32(6), sel_l)
        abv_l = cutlass.select_(acc + s6 >= rem, acc + s5, abv_l)
        sel_l = cutlass.select_(acc + s5 >= rem, b0 - I32(5), sel_l)
        abv_l = cutlass.select_(acc + s5 >= rem, acc + s4, abv_l)
        sel_l = cutlass.select_(acc + s4 >= rem, b0 - I32(4), sel_l)
        abv_l = cutlass.select_(acc + s4 >= rem, acc + s3, abv_l)
        sel_l = cutlass.select_(acc + s3 >= rem, b0 - I32(3), sel_l)
        abv_l = cutlass.select_(acc + s3 >= rem, acc + s2, abv_l)
        sel_l = cutlass.select_(acc + s2 >= rem, b0 - I32(2), sel_l)
        abv_l = cutlass.select_(acc + s2 >= rem, acc + s1, abv_l)
        sel_l = cutlass.select_(acc + s1 >= rem, b0 - I32(1), sel_l)
        abv_l = cutlass.select_(acc + s1 >= rem, acc + s0, abv_l)
        sel_l = cutlass.select_(acc + s0 >= rem, b0, sel_l)
        abv_l = cutlass.select_(acc + s0 >= rem, acc, abv_l)
        bal = I32(cute.arch.vote_ballot_sync(sel_l >= I32(0)))
        low = bal & (I32(0) - bal)
        src = I32(cute.arch.popc(low - I32(1)))
        sel = cute.arch.shuffle_sync(sel_l, src)
        above = cute.arch.shuffle_sync(abv_l, src)
        prefix = prefix | (sel << shift)
        rem = rem - above
        cute.arch.sync_warp()
    return prefix


@cute.jit
def _k1_emit(mThr, mBnd, rg, length, m, budget, prefix, trivial_k, active, lane):
    """Write the row's (tb, need, bnd): bnd one fp32 notch below the selected
    sample (skipping -0.0, which compares equal to +0.0). Rows with at most
    trivial_k keys, fewer samples than the budget, or no finite selected
    sample take all (tb 256, bnd -inf)."""
    vk = prefix ^ I32(-0x80000000)
    vkm = vk - I32(1)
    vkm = cutlass.select_(vkm == I32(-1), I32(-2), vkm)
    cut = (
        cutlass.select_(length > I32(trivial_k), I32(1), I32(0))
        * cutlass.select_(m >= budget, I32(1), I32(0))
        * cutlass.select_(vk > I32(-0x7F800001), I32(1), I32(0))
    )
    bnd = cutlass.select_(cut == I32(1), _skey_f32(vkm), F32(-math.inf))
    tb = cutlass.select_(cut == I32(1), I32(0), I32(256))
    need = cutlass.select_(cut == I32(1), budget, I32(0))
    if active and lane == 0:
        mThr[rg, 0] = tb
        mThr[rg, 1] = need
        mBnd[rg] = bnd
    cute.arch.sync_warp()


@cute.kernel
def _k1_sample_threshold_kernel(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mStarts: cute.Tensor,
    mEnds: cute.Tensor,
    mThr: cute.Tensor,
    mBnd: cute.Tensor,
    mParts: cute.Tensor,
    mCnt: cute.Tensor,
    S_mma: cute.TiledMma,
    lQs: cute.ComposedLayout,
    lKs: cute.ComposedLayout,
    lFlat: cute.ComposedLayout,
    lFlatK: cute.ComposedLayout,
    S_mma_h: cute.TiledMma,
    q_rows: I32,
    key_pad: I32,
    s_tiles: I32,
    stride: cutlass.Constexpr,
    top_k: cutlass.Constexpr,
    keep_b: cutlass.Constexpr,
    ln_d: cutlass.Constexpr,
    k_db: cutlass.Constexpr,
    span_skip: cutlass.Constexpr,
    trivial_k: cutlass.Constexpr,
    n_parts: cutlass.Constexpr = 1,
    n_tile: cutlass.Constexpr = NT,
):
    """k1 sampler: sample GEMM and per-row cut in one kernel, one CTA per
    128-row q block (n_parts CTAs when the block's sample tiles are split).

    The CTA streams every sample tile of its block (n_tile positions per
    tile, position p -> key column _k1_col(p)), runs the tcgen05 GEMM against
    the resident Q tile and inserts each in-span sampled score into the
    per-thread smem keep (four keeps per row, one per warp-group slice).
    Every sample of a row passes through this CTA, so the union of the row's
    keeps holds its top keep_b samples, or lower values if the lazy keep
    dropped some (over-collect only). One warp per row then extracts the
    budget-th largest of the union (budget <= keep_b) and emits the boundary.
    With n_parts > 1 each CTA writes its keeps and sample counts to mParts /
    mCnt and _k1_merge_kernel extracts. Exactness never depends on any of
    this (cursor certificate + exact repair)."""
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    bi, pi, _ = cute.arch.block_idx()
    nwarp = cutlass.const_expr(THREADS // 32)
    n_vals = cutlass.const_expr(4 * keep_b)

    smem = utils.SmemAllocator()
    pQK = smem.allocate_array(BF16, BLK * D + (2 * n_tile * D if k_db else n_tile * D), byte_alignment=1024)
    se_words = cutlass.const_expr(2 * BLK + 32)
    pSE = smem.allocate_array(I32, se_words, byte_alignment=128)
    pKC = smem.allocate_array(I32, THREADS, byte_alignment=128)
    pKG = smem.allocate_array(F32, THREADS, byte_alignment=128)
    pKI = smem.allocate_array(I32, THREADS, byte_alignment=128)
    pCnt = smem.allocate_array(I32, THREADS, byte_alignment=128)
    pKeep = smem.allocate_array(F32, THREADS * keep_b, byte_alignment=128)
    mbar = smem.allocate_array(cutlass.Int64, 1, byte_alignment=16)

    sQ = cute.make_tensor(cute.recast_ptr(pQK, lQs.inner), lQs.outer)
    sK0 = cute.make_tensor(cute.recast_ptr(pQK + BLK * D, lKs.inner), lKs.outer)
    sQ2 = cute.make_tensor(cute.recast_ptr(pQK, lFlat.inner), lFlat.outer)
    sK2_0 = cute.make_tensor(cute.recast_ptr(pQK + BLK * D, lFlatK.inner), lFlatK.outer)
    if cutlass.const_expr(k_db):
        sK1 = cute.make_tensor(cute.recast_ptr(pQK + BLK * D + n_tile * D, lKs.inner), lKs.outer)
        sK2_1 = cute.make_tensor(cute.recast_ptr(pQK + BLK * D + n_tile * D, lFlatK.inner), lFlatK.outer)
    sSE = cute.make_tensor(pSE, cute.make_layout((se_words,)))
    sKC = cute.make_tensor(pKC, cute.make_layout((THREADS,)))
    sKG = cute.make_tensor(pKG, cute.make_layout((THREADS,)))
    sKI = cute.make_tensor(pKI, cute.make_layout((THREADS,)))
    sCnt = cute.make_tensor(pCnt, cute.make_layout((THREADS,)))
    sKeep = cute.make_tensor(pKeep, cute.make_layout((keep_b, THREADS), stride=(THREADS, 1)))
    # After the tile loop the K buffer is dead: it becomes the per-warp merge
    # scratch (nwarp x n_vals keys, then nwarp x 256 radix bins).
    sKw = cute.make_tensor(cute.recast_ptr(pQK + BLK * D, dtype=I32), cute.make_layout((nwarp * n_vals,)))
    sBw = cute.make_tensor(cute.recast_ptr(pQK + BLK * D + 2 * nwarp * n_vals, dtype=I32), cute.make_layout((nwarp * 256,)))

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar + 0, 1)
    cute.arch.mbarrier_init_fence()

    # ---- bounds, keep state ----
    if tidx < BLK:
        rg0 = bi * BLK + tidx
        st0 = I32(0)
        en0 = I32(0)
        if rg0 < q_rows:
            st0 = I32(mStarts[rg0])
            en0 = I32(mEnds[rg0])
        sSE[tidx] = st0
        sSE[BLK + tidx] = en0
    sKC[tidx] = I32(0)
    sKG[tidx] = -cutlass.Float32.inf
    sKI[tidx] = I32(0)
    sCnt[tidx] = I32(0)
    for s in cutlass.range_constexpr(keep_b):
        sKeep[s, tidx] = -cutlass.Float32.inf

    # ---- the block's sample-tile interval [jb, je) (a tile is n_tile
    # positions = n_tile * stride key columns), clamped to its live span ----
    jb = I32(0)
    je = s_tiles
    if cutlass.const_expr(span_skip):
        sk = cutlass.const_expr(se_words - 32)
        if tidx == 0:
            sSE[sk] = key_pad
            sSE[sk + 1] = I32(0)
        cute.arch.barrier()
        if tidx < BLK:
            if (bi * BLK + tidx) < q_rows:
                cute.arch.atomic_min(sSE.iterator + sk, sSE[tidx], sem="relaxed", scope="cta")
                cute.arch.atomic_max(sSE.iterator + (sk + 1), sSE[BLK + tidx], sem="relaxed", scope="cta")
        cute.arch.barrier()
        lo_t = sSE[sk] // I32(n_tile * stride)
        hi_t = (sSE[sk + 1] + I32(n_tile * stride - 1)) // I32(n_tile * stride)
        if jb < lo_t:
            jb = lo_t
        if je > hi_t:
            je = hi_t
        if jb > je:
            jb = je
    else:
        cute.arch.barrier()
    if cutlass.const_expr(n_parts > 1):
        # this CTA's slice of the block's (clamped) tile interval
        plen = je - jb
        pje = jb + ((pi + I32(1)) * plen) // I32(n_parts)
        jb = jb + (pi * plen) // I32(n_parts)
        je = pje

    # ---- TMEM ----
    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(barrier_for_retrieve=tmem_bar)
    tmem.allocate(n_tile)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(F32)
    tmem.relinquish_alloc_permit()

    tS_f = S_mma.make_fragment_C(S_mma.partition_shape_C((BLK, n_tile)))
    tS = cute.make_tensor(tmem_ptr, tS_f.layout)
    fQ = S_mma.make_fragment_A(sQ)
    fK0 = S_mma.make_fragment_B(sK0)
    if cutlass.const_expr(k_db):
        fK1 = S_mma.make_fragment_B(sK1)

    g2s_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        BF16,
        num_bits_per_copy=128,
    )
    tiled_g2s = cute.make_tiled_copy_tv(g2s_atom, cute.make_layout((1, 16)), cute.make_layout((1, 8)))
    half = lane // 16
    thr_g2s = tiled_g2s.get_slice(lane % 16)
    qbase = mQ.iterator.toint()
    kbase = mK.iterator.toint()

    # ---- stage Q once ----
    if jb < je:
        for i in cutlass.range_constexpr(BLK // (nwarp * 2)):
            q_row = i * (nwarp * 2) + warp_idx * 2 + half
            cute.copy(
                tiled_g2s,
                thr_g2s.partition_S(
                    cute.make_tensor(
                        cute.make_ptr(BF16, qbase + (bi * BLK + q_row) * (D * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, D))
                    )
                ),
                thr_g2s.partition_D(cute.local_tile(sQ2, (1, D), (q_row, 0))),
            )
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier()

    # ---- epilogue partitions ----
    ld32 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), F32)
    tSh_f = S_mma_h.make_fragment_C(S_mma_h.partition_shape_C((BLK, BLK)))
    cp_h = tcgen05.make_tmem_copy(ld32, cute.make_tensor(tmem_ptr, tSh_f.layout)[(None, None), 0, 0])
    ltid = tidx % 128
    grp = tidx // 128
    thr_h = cp_h.get_slice(ltid)
    idC = cute.make_identity_tensor((BLK, BLK))
    tr_cO = thr_h.partition_D(idC)
    nslh = cutlass.const_expr(cute.size(tr_cO, mode=[2]))
    rSh = cute.make_rmem_tensor(tr_cO[None, None, 0].shape, F32)
    nfrag = cutlass.const_expr(cute.size(rSh.shape))
    nk = cutlass.const_expr(cute.size(fQ, mode=[2]))

    # reverse traversal (high columns first) so the keep gates tighten early
    jrev = jb + je - I32(1)
    # K-load scalars, assigned in sibling runtime branches below: seeded.
    k_slot = I32(0)
    k_row = I32(0)
    jnx = I32(0)

    # ---- prologue: the first processed tile (jrev - jb) ----
    if jb < je:
        for i in cutlass.range_constexpr(n_tile // (nwarp * 2)):
            k_slot = i * (nwarp * 2) + warp_idx * 2 + half
            k_row = _k1_col((jrev - jb) * n_tile + k_slot, bi, key_pad, stride)
            k_row = cutlass.select_(k_row > key_pad - I32(1), key_pad - I32(1), k_row)
            cute.copy(
                tiled_g2s,
                thr_g2s.partition_S(
                    cute.make_tensor(cute.make_ptr(BF16, kbase + k_row * (D * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, D)))
                ),
                thr_g2s.partition_D(cute.local_tile(sK2_0, (1, D), (k_slot, 0))),
            )
    cute.arch.cp_async_commit_group()

    ph = I32(0)
    st = I32(0)
    jt = jb
    while jt < je:
        jte = jrev - jt
        if cutlass.const_expr(not k_db):
            if jt > jb:
                for i in cutlass.range_constexpr(n_tile // (nwarp * 2)):
                    k_slot = i * (nwarp * 2) + warp_idx * 2 + half
                    k_row = _k1_col(jte * n_tile + k_slot, bi, key_pad, stride)
                    k_row = cutlass.select_(k_row > key_pad - I32(1), key_pad - I32(1), k_row)
                    cute.copy(
                        tiled_g2s,
                        thr_g2s.partition_S(
                            cute.make_tensor(cute.make_ptr(BF16, kbase + k_row * (D * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, D)))
                        ),
                        thr_g2s.partition_D(cute.local_tile(sK2_0, (1, D), (k_slot, 0))),
                    )
                cute.arch.cp_async_commit_group()
        elif jt + I32(1) < je:
            jnx = jrev - (jt + I32(1))
            if st == 0:
                for i in cutlass.range_constexpr(n_tile // (nwarp * 2)):
                    k_slot = i * (nwarp * 2) + warp_idx * 2 + half
                    k_row = _k1_col(jnx * n_tile + k_slot, bi, key_pad, stride)
                    k_row = cutlass.select_(k_row > key_pad - I32(1), key_pad - I32(1), k_row)
                    cute.copy(
                        tiled_g2s,
                        thr_g2s.partition_S(
                            cute.make_tensor(cute.make_ptr(BF16, kbase + k_row * (D * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, D)))
                        ),
                        thr_g2s.partition_D(cute.local_tile(sK2_1, (1, D), (k_slot, 0))),
                    )
            else:
                for i in cutlass.range_constexpr(n_tile // (nwarp * 2)):
                    k_slot = i * (nwarp * 2) + warp_idx * 2 + half
                    k_row = _k1_col(jnx * n_tile + k_slot, bi, key_pad, stride)
                    k_row = cutlass.select_(k_row > key_pad - I32(1), key_pad - I32(1), k_row)
                    cute.copy(
                        tiled_g2s,
                        thr_g2s.partition_S(
                            cute.make_tensor(cute.make_ptr(BF16, kbase + k_row * (D * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, D)))
                        ),
                        thr_g2s.partition_D(cute.local_tile(sK2_0, (1, D), (k_slot, 0))),
                    )
            cute.arch.cp_async_commit_group()
        if cutlass.const_expr(not k_db):
            cute.arch.cp_async_wait_group(0)
        elif jt + I32(1) < je:
            cute.arch.cp_async_wait_group(1)
        else:
            cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()

        # ---- UMMA ----
        if warp_idx == 0:
            S_mma.set(tcgen05.Field.ACCUMULATE, False)
            if cutlass.const_expr(not k_db):
                for kb in cutlass.range_constexpr(nk):
                    cute.gemm(S_mma, tS, fQ[None, None, kb, 0], fK0[None, None, kb, 0], tS)
                    S_mma.set(tcgen05.Field.ACCUMULATE, True)
            elif st == 0:
                for kb in cutlass.range_constexpr(nk):
                    cute.gemm(S_mma, tS, fQ[None, None, kb, 0], fK0[None, None, kb, 0], tS)
                    S_mma.set(tcgen05.Field.ACCUMULATE, True)
            else:
                for kb in cutlass.range_constexpr(nk):
                    cute.gemm(S_mma, tS, fQ[None, None, kb, 0], fK1[None, None, kb, 0], tS)
                    S_mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(mbar + 0)
        cute.arch.mbarrier_wait(mbar + 0, ph)
        ph = ph ^ 1

        # ---- epilogue: TMEM -> regs -> per-thread keep ----
        for h in cutlass.range_constexpr(n_tile // BLK):
            tS_h = cute.make_tensor(tmem_ptr + h * BLK, tSh_f.layout)[(None, None), 0, 0]
            tr_Sh = thr_h.partition_S(tS_h)
            pos_base = jte * I32(n_tile) + I32(h * BLK)
            for sl in cutlass.range_constexpr(nslh):
                if sl == grp:
                    cute.copy(cp_h, tr_Sh[None, None, sl], rSh)
                    crd0 = tr_cO[None, None, sl][0]
                    crdL = tr_cO[None, None, sl][cutlass.const_expr(nfrag - 1)]
                    r0 = crd0[0]
                    my_st = sSE[r0]
                    my_en = sSE[BLK + r0]
                    # Columns are monotone in position (strata are disjoint
                    # and ordered), so the fragment is wholly in span when
                    # its first stratum's start and last stratum's end are.
                    c_lo = (pos_base + crd0[1]) * I32(stride)
                    c_hi = (pos_base + crdL[1]) * I32(stride) + I32(stride - 1)
                    kg = F32(sKG[tidx])
                    # CuTe joins this runtime branch through SSA form: seed
                    # every branch-local scalar with its DSL type.
                    k1_col = I32(0)
                    k1_val = F32(0.0)
                    if c_lo >= my_st and c_hi < my_en:
                        sCnt[tidx] = I32(sCnt[tidx]) + I32(nfrag)
                        for j in cutlass.range_constexpr(nfrag):
                            k1_val = F32(rSh[j]) * F32(SCALE)
                            if k1_val > kg:
                                _k1_keep_insert(sKeep, sKC, sKG, sKI, tidx, k1_val, keep_b)
                    else:
                        for j in cutlass.range_constexpr(nfrag):
                            k1_col = _k1_col(pos_base + tr_cO[None, None, sl][j][1], bi, key_pad, stride)
                            if k1_col >= my_st and k1_col < my_en:
                                sCnt[tidx] = I32(sCnt[tidx]) + I32(1)
                                k1_val = F32(rSh[j]) * F32(SCALE)
                                if k1_val > kg:
                                    _k1_keep_insert(sKeep, sKC, sKG, sKI, tidx, k1_val, keep_b)
            cute.arch.fence_view_async_tmem_load()
            cute.arch.barrier()
        st = st ^ 1
        jt = jt + I32(1)

    # ---- per-row extraction: one warp per row, BLK / nwarp rows per warp
    # (the K buffer is dead and holds the merge scratch) ----
    cute.arch.barrier()
    if cutlass.const_expr(n_parts == 1):
        n_it = cutlass.const_expr((n_vals + 31) // 32)
        keep_idx = I32(0)
        keep_grp = I32(0)
        keep_key = I32(0)
        for ri in cutlass.range_constexpr(BLK // nwarp):
            row = I32(ri * nwarp) + warp_idx
            row_g = bi * BLK + row
            length = I32(sSE[BLK + row]) - I32(sSE[row])
            m = I32(sCnt[row]) + I32(sCnt[BLK + row]) + I32(sCnt[2 * BLK + row]) + I32(sCnt[3 * BLK + row])
            budget = _k1_row_budget(length, m, top_k, ln_d, keep_b, lane)
            # the row's four keeps as bias-flipped (unsigned-monotone) keys
            for i in cutlass.range_constexpr(n_it):
                keep_idx = cutlass.select_(I32(i * 32) + lane < I32(n_vals), I32(i * 32) + lane, I32(n_vals - 1))
                keep_grp = keep_idx // I32(keep_b)
                keep_key = _f32_skey(F32(sKeep[keep_idx - keep_grp * I32(keep_b), keep_grp * I32(BLK) + row]))
                if I32(i * 32) + lane < I32(n_vals):
                    sKw[warp_idx * I32(n_vals) + keep_idx] = keep_key ^ I32(-0x80000000)
            cute.arch.sync_warp()
            prefix = _k1_radix_select_desc(sKw, sBw, warp_idx, lane, n_vals, budget)
            _k1_emit(mThr, mBnd, row_g, length, m, budget, prefix, trivial_k, row_g < q_rows, lane)
            cute.arch.sync_warp()
    else:
        # ---- partial: stage the block's keeps row-major as flipped keys (two
        # 64-row halves through the dead K buffer) and write them coalesced:
        # mParts[row, part * n_vals + j], j = group * keep_b + slot ----
        n_stage = cutlass.const_expr(64 * n_vals)
        sStage = cute.make_tensor(cute.recast_ptr(pQK + BLK * D, dtype=I32), cute.make_layout((n_stage,)))
        stage_idx = I32(0)
        stage_row = I32(0)
        stage_j = I32(0)
        stage_grp = I32(0)
        stage_key = I32(0)
        for c in cutlass.range_constexpr(2):
            for i in cutlass.range_constexpr((n_stage + THREADS - 1) // THREADS):
                stage_idx = cutlass.select_(I32(i * THREADS) + tidx < I32(n_stage), I32(i * THREADS) + tidx, I32(n_stage - 1))
                stage_row = stage_idx // I32(n_vals)
                stage_j = stage_idx - stage_row * I32(n_vals)
                stage_grp = stage_j // I32(keep_b)
                stage_key = _f32_skey(F32(sKeep[stage_j - stage_grp * I32(keep_b), stage_grp * I32(BLK) + I32(c * 64) + stage_row]))
                if I32(i * THREADS) + tidx < I32(n_stage):
                    sStage[stage_idx] = stage_key ^ I32(-0x80000000)
            cute.arch.barrier()
            for i in cutlass.range_constexpr((n_stage + THREADS - 1) // THREADS):
                stage_idx = cutlass.select_(I32(i * THREADS) + tidx < I32(n_stage), I32(i * THREADS) + tidx, I32(n_stage - 1))
                stage_row = stage_idx // I32(n_vals)
                stage_j = stage_idx - stage_row * I32(n_vals)
                if I32(i * THREADS) + tidx < I32(n_stage):
                    mParts[bi * BLK + I32(c * 64) + stage_row, pi * I32(n_vals) + stage_j] = I32(sStage[stage_idx])
            cute.arch.barrier()
        if tidx < BLK:
            mCnt[bi * BLK + tidx, pi] = I32(sCnt[tidx]) + I32(sCnt[BLK + tidx]) + I32(sCnt[2 * BLK + tidx]) + I32(sCnt[3 * BLK + tidx])

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.kernel
def _k1_merge_kernel(
    mParts: cute.Tensor,
    mCnt: cute.Tensor,
    mStarts: cute.Tensor,
    mEnds: cute.Tensor,
    mThr: cute.Tensor,
    mBnd: cute.Tensor,
    q_rows: I32,
    n_parts: cutlass.Constexpr,
    keep_b: cutlass.Constexpr,
    top_k: cutlass.Constexpr,
    ln_d: cutlass.Constexpr,
    trivial_k: cutlass.Constexpr,
):
    """k1 sampler, n_parts > 1: one warp per row merges the n_parts x 4
    partial keeps (already bias-flipped keys) and emits (tb, need, bnd) as
    the single-CTA path does. Each part's keep is exact-or-lower over its
    slice, so the budget-th largest of the union is at most the true one:
    over-collect only, never a miss."""
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    bx, _, _ = cute.arch.block_idx()
    nwarp = cutlass.const_expr(K1M_THREADS // 32)
    n_vals = cutlass.const_expr(n_parts * 4 * keep_b)
    n_it = cutlass.const_expr((n_vals + 31) // 32)
    smem = utils.SmemAllocator()
    pKw = smem.allocate_array(I32, nwarp * n_vals, byte_alignment=128)
    pBw = smem.allocate_array(I32, nwarp * 256, byte_alignment=128)
    sKw = cute.make_tensor(pKw, cute.make_layout((nwarp * n_vals,)))
    sBw = cute.make_tensor(pBw, cute.make_layout((nwarp * 256,)))
    rg = bx * I32(nwarp) + warp_idx
    length = I32(0)
    m = I32(0)
    if rg < q_rows:
        length = I32(mEnds[rg]) - I32(mStarts[rg])
        for pp in cutlass.range_constexpr(n_parts):
            m = m + I32(mCnt[rg, pp])
    budget = _k1_row_budget(length, m, top_k, ln_d, keep_b, lane)
    for i in cutlass.range_constexpr(n_it):
        if I32(i * 32) + lane < I32(n_vals):
            sKw[warp_idx * I32(n_vals) + I32(i * 32) + lane] = I32(mParts[rg, I32(i * 32) + lane])
    cute.arch.sync_warp()
    prefix = _k1_radix_select_desc(sKw, sBw, warp_idx, lane, n_vals, budget)
    _k1_emit(mThr, mBnd, rg, length, m, budget, prefix, trivial_k, rg < q_rows, lane)


class _SimplifiedSparseSampleThresholdK1:
    """fusion="filtered" k1 sampler (SSA_FUSED_SAMPLER=k1): writes thr / bnd
    for the collect pass. Grid: one CTA per 128-row q block, times n_parts
    (see _k1_compiled); with n_parts > 1 a merge kernel follows."""

    def __init__(self, topk: int, tcap: int, n_parts: int, span_skip: bool = SAMPLE_SPAN_SKIP):
        self.topk = topk
        self.stride, self.keep_b, self.k_db, _ = _k1_geometry(topk)
        self.trivial_k = tcap
        self.span_skip = span_skip
        self.n_parts = n_parts
        self.ln_d = math.log(1.0 / K1_RISK)

    def scratch_shapes(self, rows):
        """(parts, counts) shapes; one part binds small placeholders (each
        with a dimension > 1, so the dynamic layout has a leading dim)."""
        if self.n_parts > 1:
            return (rows, self.n_parts * 4 * self.keep_b), (rows, self.n_parts)
        return (1, 4), (1, 2)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        starts: cute.Tensor,
        ends: cute.Tensor,
        thr: cute.Tensor,
        bnd: cute.Tensor,
        parts: cute.Tensor,
        cnt: cute.Tensor,
        query_rows: I32,
        key_tiles: I32,
        stream,
    ):
        S_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, tcgen05.CtaGroup.ONE, (BLK, BLK))
        S_mma_w = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, tcgen05.CtaGroup.ONE, (BLK, NT))
        lQs = sm100_utils.make_smem_layout_a(S_mma, (BLK, BLK, D), BF16, 1)
        lKs_w = sm100_utils.make_smem_layout_b(S_mma_w, (BLK, NT, D), BF16, 1)
        a128 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, BF16)
        lFlat = cute.tile_to_shape(a128, (BLK, D), order=(0, 1))
        lFlatK = cute.tile_to_shape(a128, (NT, D), order=(0, 1))
        key_pad = key_tiles * I32(NT)
        n_pos = (key_pad + I32(self.stride - 1)) // I32(self.stride)
        s_tiles = (n_pos + I32(NT - 1)) // I32(NT)
        _k1_sample_threshold_kernel(
            q,
            k,
            starts,
            ends,
            thr,
            bnd,
            parts,
            cnt,
            S_mma_w,
            lQs,
            lKs_w,
            lFlat,
            lFlatK,
            S_mma,
            q_rows=query_rows,
            key_pad=key_pad,
            s_tiles=s_tiles,
            stride=self.stride,
            top_k=self.topk,
            keep_b=self.keep_b,
            ln_d=self.ln_d,
            k_db=self.k_db,
            span_skip=self.span_skip,
            trivial_k=self.trivial_k,
            n_parts=self.n_parts,
        ).launch(
            grid=(cute.ceil_div(query_rows, BLK), self.n_parts, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )
        if cutlass.const_expr(self.n_parts > 1):
            _k1_merge_kernel(
                parts,
                cnt,
                starts,
                ends,
                thr,
                bnd,
                q_rows=query_rows,
                n_parts=self.n_parts,
                keep_b=self.keep_b,
                top_k=self.topk,
                ln_d=self.ln_d,
                trivial_k=self.trivial_k,
            ).launch(
                grid=(cute.ceil_div(query_rows, K1M_THREADS // 32), 1, 1),
                block=(K1M_THREADS, 1, 1),
                stream=stream,
            )


class _SimplifiedSparseHistOnlyFused:
    """Stage-2 pass-1 launcher: GEMM + per-row coarse histogram, no S."""

    def __init__(self, query_len: int, query_pad: int, key_pad: int):
        self.q_nblk = query_pad // BLK
        self.k_tiles = key_pad // NT
        self.n_chunks = (self.k_tiles + C_TILES - 1) // C_TILES
        self.q_rows = query_len

    @cute.jit
    def __call__(self, q: cute.Tensor, k: cute.Tensor, hist: cute.Tensor, starts: cute.Tensor, ends: cute.Tensor):
        S_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, tcgen05.CtaGroup.ONE, (BLK, BLK))
        S_mma_w = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, tcgen05.CtaGroup.ONE, (BLK, NT))
        lQs = sm100_utils.make_smem_layout_a(S_mma, (BLK, BLK, D), BF16, 1)
        lKs_w = sm100_utils.make_smem_layout_b(S_mma_w, (BLK, NT, D), BF16, 1)
        a128 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, BF16)
        lFlat = cute.tile_to_shape(a128, (BLK, D), order=(0, 1))
        lFlatK = cute.tile_to_shape(a128, (NT, D), order=(0, 1))
        _score_hist_kernel_tcgen05(
            q,
            k,
            hist,
            starts,
            ends,
            S_mma_w,
            lQs,
            lKs_w,
            lFlat,
            lFlatK,
            S_mma,
            q_rows=self.q_rows,
            k_tiles_total=self.k_tiles,
            write_s=False,
        ).launch(
            grid=(self.n_chunks, self.q_nblk, 1),
            block=(THREADS, 1, 1),
        )


class _SimplifiedSparseCollectFused:
    """Stage-2 pass-2 launcher: GEMM recompute + classify/collect, no S."""

    def __init__(self, query_len: int, query_pad: int, key_pad: int, topk: int = TOPK):
        self.q_nblk = query_pad // BLK
        self.k_tiles = key_pad // NT
        self.n_chunks = (self.k_tiles + C_TILES - 1) // C_TILES
        self.q_rows = query_len
        self.topk = topk
        self.tcap = _tie_cap(topk, key_pad)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        hist: cute.Tensor,
        thr: cute.Tensor,
        starts: cute.Tensor,
        ends: cute.Tensor,
        cursors: cute.Tensor,
        didx: cute.Tensor,
        dval: cute.Tensor,
        tieidx: cute.Tensor,
        tieval: cute.Tensor,
    ):
        S_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, tcgen05.CtaGroup.ONE, (BLK, BLK))
        S_mma_w = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, tcgen05.CtaGroup.ONE, (BLK, NT))
        lQs = sm100_utils.make_smem_layout_a(S_mma, (BLK, BLK, D), BF16, 1)
        lKs_w = sm100_utils.make_smem_layout_b(S_mma_w, (BLK, NT, D), BF16, 1)
        a128 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, BF16)
        lFlat = cute.tile_to_shape(a128, (BLK, D), order=(0, 1))
        lFlatK = cute.tile_to_shape(a128, (NT, D), order=(0, 1))
        _score_hist_kernel_tcgen05(
            q,
            k,
            hist,
            starts,
            ends,
            S_mma_w,
            lQs,
            lKs_w,
            lFlat,
            lFlatK,
            S_mma,
            q_rows=self.q_rows,
            k_tiles_total=self.k_tiles,
            write_s=False,
            collect=True,
            mThr=thr,
            mCursors=cursors,
            mDefIdx=didx,
            mDefVal=dval,
            mTieIdx=tieidx,
            mTieVal=tieval,
            tie_cap=self.tcap,
            top_k=self.topk,
        ).launch(
            grid=(self.n_chunks, self.q_nblk, 1),
            block=(THREADS, 1, 1),
        )


class _SimplifiedSparseSampleHistFused:
    """fusion="filtered" sample pass: GEMM + histogram over 1/SAMPLE_STRIDE
    of the keys -- every SAMPLE_STRIDE'th key (token) or 256-col tile (tile).

    Token mode iterates sample-space tiles of NT positions (tile_stride 1)
    and loads K row SAMPLE_STRIDE * position; the grid is the same size."""

    def __init__(self, query_len: int, query_pad: int, key_pad: int, span_skip: bool = SAMPLE_SPAN_SKIP):
        self.q_nblk = query_pad // BLK
        self.k_tiles = key_pad // NT
        s_tiles = (self.k_tiles + SAMPLE_STRIDE - 1) // SAMPLE_STRIDE
        self.n_chunks = (s_tiles + SAMPLE_CTILES - 1) // SAMPLE_CTILES
        self.q_rows = query_len
        self.span_skip = span_skip

    def _launch(self, q, k, hist, starts, ends, query_rows, key_tiles, stream, thr=None, refine=False, sample_scores=None):
        # Traced from the cute.jit __call__ methods below; SAMPLE_MODE is a
        # module constant, so the plain Python branch picks one launch.
        S_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, tcgen05.CtaGroup.ONE, (BLK, BLK))
        S_mma_w = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, tcgen05.CtaGroup.ONE, (BLK, NT))
        lQs = sm100_utils.make_smem_layout_a(S_mma, (BLK, BLK, D), BF16, 1)
        lKs_w = sm100_utils.make_smem_layout_b(S_mma_w, (BLK, NT, D), BF16, 1)
        a128 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, BF16)
        lFlat = cute.tile_to_shape(a128, (BLK, D), order=(0, 1))
        lFlatK = cute.tile_to_shape(a128, (NT, D), order=(0, 1))
        if SAMPLE_MODE == "token":
            s_tiles = (key_tiles + I32(SAMPLE_STRIDE - 1)) // I32(SAMPLE_STRIDE)
            _score_hist_kernel_tcgen05(
                q,
                k,
                hist,
                starts,
                ends,
                S_mma_w,
                lQs,
                lKs_w,
                lFlat,
                lFlatK,
                S_mma,
                q_rows=query_rows,
                k_tiles_total=s_tiles,
                write_s=False,
                tile_stride=1,
                c_tiles=SAMPLE_CTILES,
                span_skip=self.span_skip,
                mThr=thr,
                refine=refine,
                col_scale=SAMPLE_STRIDE,
                mScrSampleScores=sample_scores,
            ).launch(
                grid=(cute.ceil_div(s_tiles, SAMPLE_CTILES), cute.ceil_div(query_rows, BLK), 1),
                block=(THREADS, 1, 1),
                stream=stream,
            )
        else:
            _score_hist_kernel_tcgen05(
                q,
                k,
                hist,
                starts,
                ends,
                S_mma_w,
                lQs,
                lKs_w,
                lFlat,
                lFlatK,
                S_mma,
                q_rows=query_rows,
                k_tiles_total=key_tiles,
                write_s=False,
                tile_stride=SAMPLE_STRIDE,
                c_tiles=SAMPLE_CTILES,
                span_skip=self.span_skip,
                mThr=thr,
                refine=refine,
                mScrSampleScores=sample_scores,
            ).launch(
                grid=(cute.ceil_div(key_tiles, SAMPLE_STRIDE * SAMPLE_CTILES), cute.ceil_div(query_rows, BLK), 1),
                block=(THREADS, 1, 1),
                stream=stream,
            )

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        hist: cute.Tensor,
        starts: cute.Tensor,
        ends: cute.Tensor,
        query_rows: I32,
        key_tiles: I32,
        stream,
        sample_scores: cute.Tensor = None,
    ):
        self._launch(q, k, hist, starts, ends, query_rows, key_tiles, stream, sample_scores=sample_scores)


class _SimplifiedSparseSampleRefineFused(_SimplifiedSparseSampleHistFused):
    """fusion="filtered" refine pass: the sample pass again (same samples,
    same scores), histogramming the sub-codes of each row's crossing bin."""

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        sub_hist: cute.Tensor,
        thr: cute.Tensor,
        starts: cute.Tensor,
        ends: cute.Tensor,
        query_rows: I32,
        key_tiles: I32,
        stream,
    ):
        self._launch(q, k, sub_hist, starts, ends, query_rows, key_tiles, stream, thr=thr, refine=True)


class _SimplifiedSparseFilteredCollect:
    """fusion="filtered" main pass: the ONLY full GEMM -- filter-collect
    against the seeded per-row threshold, no histogram, no S."""

    def __init__(
        self,
        query_len: int,
        query_pad: int,
        key_pad: int,
        topk: int = TOPK,
        mask_walk: bool = MASK_WALK,
        fast_cnt: bool = FAST_CNT,
        raw_cmp: bool = RAW_CMP,
        c_tiles: int = None,
        span_skip: bool = SPAN_SKIP,
        scr_counts: bool = False,
    ):
        self.scr_counts = scr_counts
        self.q_nblk = query_pad // BLK
        self.k_tiles = key_pad // NT
        # Column tiles per CTA. Finer filtered chunks let the scheduler smooth
        # the ragged per-row [start, end) workload; total GEMM work is
        # unchanged. None selects the measured filtered-pass default.
        self.c_tiles = FILTER_C_TILES if c_tiles is None else int(c_tiles)
        self.n_chunks = (self.k_tiles + self.c_tiles - 1) // self.c_tiles
        self.q_rows = query_len
        self.topk = topk
        self.tcap = _tie_cap(topk, key_pad)
        self.mask_walk = mask_walk
        self.fast_cnt = fast_cnt
        # RAW_CMP stages raw TMEM values, so it needs the walk's staging buf.
        self.raw_cmp = raw_cmp and mask_walk
        self.span_skip = span_skip

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        hist: cute.Tensor,
        thr: cute.Tensor,
        bnd: cute.Tensor,
        starts: cute.Tensor,
        ends: cute.Tensor,
        cursors: cute.Tensor,
        candidx: cute.Tensor,
        candval: cute.Tensor,
        query_rows: I32,
        key_tiles: I32,
        stream,
    ):
        S_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, tcgen05.CtaGroup.ONE, (BLK, BLK))
        S_mma_w = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, tcgen05.CtaGroup.ONE, (BLK, NT))
        lQs = sm100_utils.make_smem_layout_a(S_mma, (BLK, BLK, D), BF16, 1)
        lKs_w = sm100_utils.make_smem_layout_b(S_mma_w, (BLK, NT, D), BF16, 1)
        a128 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, BF16)
        lFlat = cute.tile_to_shape(a128, (BLK, D), order=(0, 1))
        lFlatK = cute.tile_to_shape(a128, (NT, D), order=(0, 1))
        _score_hist_kernel_tcgen05(
            q,
            k,
            hist,
            starts,
            ends,
            S_mma_w,
            lQs,
            lKs_w,
            lFlat,
            lFlatK,
            S_mma,
            q_rows=query_rows,
            k_tiles_total=key_tiles,
            write_s=False,
            filtered=True,
            mThr=thr,
            mBnd=bnd,
            mCursors=cursors,
            mTieIdx=candidx,
            mTieVal=candval,
            tie_cap=self.tcap,
            k_db=False,
            c_tiles=self.c_tiles,
            span_skip=self.span_skip,
            mask_walk=self.mask_walk,
            fast_cnt=self.fast_cnt,
            raw_cmp=self.raw_cmp,
            scr_counts=self.scr_counts,
        ).launch(
            grid=(cute.ceil_div(key_tiles, self.c_tiles), cute.ceil_div(query_rows, BLK), 1),
            block=(THREADS, 1, 1),
            # 2 CTAs/SM: NCU measured the filter latency-bound at 25% warp
            # occupancy (128 regs/thread + the K double buffer both capped
            # it at 1 CTA); single K buffer fits 2, launch bounds cap regs.
            min_blocks_per_mp=2,
            stream=stream,
        )


class _SimplifiedSparseThresholdSelect:
    """Launchers for the small Stage-2 / filtered kernels."""

    def __init__(self, query_len: int, topk: int = TOPK, key_pad: int = 262144, sample_rank: int = 0, fine_bnd: bool = None):
        self.q_rows = query_len
        self.topk = topk
        self.tcap = _tie_cap(topk, key_pad)
        # Seeded-threshold budget. sample_rank=0 -> the module default
        # ceil(RANK_NUM * (topk / SAMPLE_STRIDE) / RANK_DEN); an explicit
        # value lets a sweep drive several budgets from one process.
        unit = topk // SAMPLE_STRIDE
        self.sample_rank = (
            int(sample_rank)
            if sample_rank
            else max(
                1,
                (SAMPLE_RANK_NUM * unit + SAMPLE_RANK_DEN - 1) // SAMPLE_RANK_DEN,
            )
        )
        self.fine_bnd = FINE_BOUNDARY if fine_bnd is None else bool(fine_bnd)

    @cute.jit
    def threshold(self, hist: cute.Tensor, starts: cute.Tensor, ends: cute.Tensor, thr: cute.Tensor):
        _threshold_kernel(
            hist,
            starts,
            ends,
            thr,
            q_rows=self.q_rows,
            top_k=self.topk,
        ).launch(
            grid=((self.q_rows + 255) // 256, 1, 1),
            block=(256, 1, 1),
        )

    @cute.jit
    def threshold_sample(self, hist: cute.Tensor, starts: cute.Tensor, ends: cute.Tensor, thr: cute.Tensor, bnd: cute.Tensor, query_rows: I32, stream):
        # Sampled histogram, tuned for a near-zero seed-miss rate (measured:
        # unbiased scan + margin 1 missed short rows):
        #   - rows that fit in the candidate buffer skip sampling entirely:
        #     tb = 256 collects the whole row, exact by construction and
        #     comfortably under TIE_CAP;
        #   - longer rows (>= 512 samples) scan with an oversampled budget
        #     (cum > sample_rank targets a multiple of the true TOPK above
        #     the bin, a Chernoff-safe cover of the true top-TOPK) + margin.
        #     The coarse rule uses 2x (rank 64 at K=512); the fine within-bin
        #     boundary pairs with 9/4 (rank 72), see FINE_BOUNDARY.
        _threshold_kernel(
            hist,
            starts,
            ends,
            thr,
            q_rows=query_rows,
            top_k=self.sample_rank,
            trivial_k=self.tcap,
            margin=SAMPLE_MARGIN,
            mBnd=bnd,
            fine=1 if self.fine_bnd else 0,
        ).launch(
            grid=(cute.ceil_div(query_rows, 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.jit
    def threshold_refine(self, sub_hist: cute.Tensor, thr: cute.Tensor, bnd: cute.Tensor, query_rows: I32, stream):
        _refine_threshold_kernel(
            sub_hist,
            thr,
            bnd,
            query_rows,
        ).launch(
            grid=(cute.ceil_div(query_rows, 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.jit
    def select_single(
        self,
        candidx: cute.Tensor,
        candval: cute.Tensor,
        cursors: cute.Tensor,
        thr: cute.Tensor,
        starts: cute.Tensor,
        ends: cute.Tensor,
        oidx: cute.Tensor,
        oprob: cute.Tensor,
        flags: cute.Tensor,
        fallback_pred: cute.Tensor,
        query_rows: I32,
        stream,
        flag_kinds: cute.Tensor = None,
    ):
        _compact_select_kernel(
            candidx,
            candval,
            candidx,
            candval,
            cursors,
            thr,
            starts,
            ends,
            oidx,
            oprob,
            tie_cap=self.tcap,
            top_k=self.topk,
            single_list=True,
            mFlags=flags,
            mFallbackPred=fallback_pred,
            mFlagKinds=flag_kinds,
        ).launch(
            grid=(query_rows, 1, 1),
            block=(SEL_THR, 1, 1),
            stream=stream,
        )

    @cute.jit
    def select(
        self,
        didx: cute.Tensor,
        dval: cute.Tensor,
        tieidx: cute.Tensor,
        tieval: cute.Tensor,
        cursors: cute.Tensor,
        thr: cute.Tensor,
        starts: cute.Tensor,
        ends: cute.Tensor,
        oidx: cute.Tensor,
        oprob: cute.Tensor,
    ):
        _compact_select_kernel(
            didx,
            dval,
            tieidx,
            tieval,
            cursors,
            thr,
            starts,
            ends,
            oidx,
            oprob,
            tie_cap=self.tcap,
            top_k=self.topk,
        ).launch(
            grid=(self.q_rows, 1, 1),
            block=(SEL_THR, 1, 1),
        )


_state = {}


@torch.no_grad()
def run_fused(
    q_idx,
    k_idx,
    topk_indices,
    predict,
    key_starts,
    key_ends,
):
    """Fused rect SSA indexer: score + p1 histogram in one kernel, then the
    top-k with ``external_hist`` (p1 scan skipped). Same contract as the
    unfused ``run()``: GLOBAL ascending selected ids, ``clamp(end-1, 0)``
    padding, predict = softmax over the selected scores."""
    raise ValueError("run_fused/fusion='hist' is an unqualified experimental route; " "use the production auto, off, or filtered route")

    from .cute_dsl_filter_topk import cute_dsl_topk_prefill_wrapper

    query_len = q_idx.shape[0]
    key_len = k_idx.shape[0]
    if key_starts is None or key_ends is None:
        raise ValueError("fused SSA indexer requires explicit key bounds")
    if key_starts.shape != (query_len,) or key_ends.shape != (query_len,):
        raise ValueError("SSA indexer expects one contiguous [start,end) pair per query row")
    if key_starts.dtype != torch.int32 or key_ends.dtype != torch.int32:
        raise TypeError("SSA indexer key_starts/key_ends must be int32")

    query_pad = ((query_len + BLK - 1) // BLK) * BLK
    key_pad = ((key_len + NT - 1) // NT) * NT
    q = q_idx.view(query_len, D)
    k = k_idx.view(key_len, D)
    if query_pad != query_len:
        qp = torch.zeros((query_pad, D), dtype=q.dtype, device=q.device)
        qp[:query_len].copy_(q)
        q = qp
    if key_pad != key_len:
        kp = torch.zeros((key_pad, D), dtype=k.dtype, device=k.device)
        kp[:key_len].copy_(k)
        k = kp

    topk = _derive_topk(topk_indices, query_len)
    out_idx = topk_indices.view(query_len, topk)
    out_prob = predict.view(query_len, topk)
    starts = key_starts.contiguous()
    ends = key_ends.contiguous()

    cache_key = (query_len, key_len, query_pad, key_pad, topk)
    st = _state.get(cache_key)
    if st is None:
        dev = q.device
        ws = torch.empty((query_pad, key_pad), dtype=torch.float32, device=dev)
        hist = torch.zeros((query_len, 256), dtype=torch.int32, device=dev)
        op = _SimplifiedSparseScoreHistFused(query_len, query_pad, key_pad)
        score_compiled = cute.compile(
            op,
            from_dlpack(q, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(k, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(ws, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(hist, assumed_align=16, enable_tvm_ffi=True),
            from_dlpack(starts, assumed_align=4, enable_tvm_ffi=True),
            from_dlpack(ends, assumed_align=4, enable_tvm_ffi=True),
            options="--enable-tvm-ffi",
        )
        st = dict(ws=ws, hist=hist, score=score_compiled)
        _state[cache_key] = st

    hist = st["hist"]
    hist.zero_()
    st["score"](q, k, st["ws"], hist, starts, ends)
    cute_dsl_topk_prefill_wrapper(
        st["ws"][:query_len],
        starts,
        ends,
        topk,
        return_val=False,
        in_kernel_sort=True,
        ssa_epilogue=True,
        ssa_tri_input=False,
        logical_num_cols=key_pad,
        subtract_row_start=False,
        out_indices=out_idx,
        out_probs=out_prob,
        external_hist=hist,
    )


_state_full = {}


@torch.no_grad()
def run_fused_full(
    q_idx,
    k_idx,
    topk_indices,
    predict,
    key_starts,
    key_ends,
):
    """Stage-2 fully fused rect SSA indexer: S is never materialized.

    pass-1 GEMM + per-row coarse histogram -> threshold kernel -> pass-2 GEMM
    recompute + compact (idx, val) candidate collection -> compact select
    (exact radix refine + index-ascending sort + predict softmax). Both GEMM
    passes use the identical MMA tiling, so the recomputed scores are bitwise
    identical and the threshold semantics are exact. Same output contract as
    the unfused ``run()``. Workspace: hist 256/row + definite 512/row + tie
    candidates TIE_CAP/row (vs the unfused q_pad x k_pad fp32 S).
    """
    raise ValueError("run_fused_full/fusion='full' is an unqualified experimental route; " "use the production auto, off, or filtered route")

    query_len = q_idx.shape[0]
    key_len = k_idx.shape[0]
    if key_starts is None or key_ends is None:
        raise ValueError("fused SSA indexer requires explicit key bounds")
    if key_starts.shape != (query_len,) or key_ends.shape != (query_len,):
        raise ValueError("SSA indexer expects one contiguous [start,end) pair per query row")
    if key_starts.dtype != torch.int32 or key_ends.dtype != torch.int32:
        raise TypeError("SSA indexer key_starts/key_ends must be int32")

    query_pad = ((query_len + BLK - 1) // BLK) * BLK
    key_pad = ((key_len + NT - 1) // NT) * NT
    q = q_idx.view(query_len, D)
    k = k_idx.view(key_len, D)
    if query_pad != query_len:
        qp = torch.zeros((query_pad, D), dtype=q.dtype, device=q.device)
        qp[:query_len].copy_(q)
        q = qp
    if key_pad != key_len:
        kp = torch.zeros((key_pad, D), dtype=k.dtype, device=k.device)
        kp[:key_len].copy_(k)
        k = kp

    topk = _derive_topk(topk_indices, query_len)
    tcap = _tie_cap(topk, key_pad)
    out_idx = topk_indices.view(query_len, topk)
    out_prob = predict.view(query_len, topk)
    starts = key_starts.contiguous()
    ends = key_ends.contiguous()

    cache_key = (query_len, key_len, query_pad, key_pad, topk)
    st = _state_full.get(cache_key)
    if st is None:
        dev = q.device
        hist = torch.zeros((query_len, 256), dtype=torch.int32, device=dev)
        thr = torch.zeros((query_len, 2), dtype=torch.int32, device=dev)
        cursors = torch.zeros((query_len, 2), dtype=torch.int32, device=dev)
        didx = torch.empty((query_len, topk), dtype=torch.int32, device=dev)
        dval = torch.empty((query_len, topk), dtype=torch.float32, device=dev)
        tieidx = torch.empty((query_len, tcap), dtype=torch.int32, device=dev)
        tieval = torch.empty((query_len, tcap), dtype=torch.float32, device=dev)

        def _c(t, align=16):
            return from_dlpack(t, assumed_align=align, enable_tvm_ffi=True)

        p1 = cute.compile(
            _SimplifiedSparseHistOnlyFused(query_len, query_pad, key_pad),
            _c(q),
            _c(k),
            _c(hist),
            _c(starts, 4),
            _c(ends, 4),
            options="--enable-tvm-ffi",
        )
        ts = _SimplifiedSparseThresholdSelect(query_len, topk, key_pad)
        thrk = cute.compile(
            ts.threshold,
            _c(hist),
            _c(starts, 4),
            _c(ends, 4),
            _c(thr, 8),
            options="--enable-tvm-ffi",
        )
        p2 = cute.compile(
            _SimplifiedSparseCollectFused(query_len, query_pad, key_pad, topk),
            _c(q),
            _c(k),
            _c(hist),
            _c(thr, 8),
            _c(starts, 4),
            _c(ends, 4),
            _c(cursors, 8),
            _c(didx),
            _c(dval),
            _c(tieidx),
            _c(tieval),
            options="--enable-tvm-ffi",
        )
        p3 = cute.compile(
            ts.select,
            _c(didx),
            _c(dval),
            _c(tieidx),
            _c(tieval),
            _c(cursors, 8),
            _c(thr, 8),
            _c(starts, 4),
            _c(ends, 4),
            _c(out_idx),
            _c(out_prob),
            options="--enable-tvm-ffi",
        )
        st = dict(hist=hist, thr=thr, cursors=cursors, didx=didx, dval=dval, tieidx=tieidx, tieval=tieval, p1=p1, thrk=thrk, p2=p2, p3=p3)
        _state_full[cache_key] = st

    st["hist"].zero_()
    st["cursors"].zero_()
    st["p1"](q, k, st["hist"], starts, ends)
    st["thrk"](st["hist"], starts, ends, st["thr"])
    st["p2"](q, k, st["hist"], st["thr"], starts, ends, st["cursors"], st["didx"], st["dval"], st["tieidx"], st["tieval"])
    st["p3"](st["didx"], st["dval"], st["tieidx"], st["tieval"], st["cursors"], st["thr"], starts, ends, out_idx, out_prob)


_state_filtered = {}
_compiled_filtered = {}
_compiled_k1 = {}
_retired_filtered = []
_retired_filtered_lock = Lock()


def _k1_compiled(compile_key, dev, topk, tcap, query_capacity, query_len, key_pad, q, k, starts, ends):
    """(compiled k1 sampler, launcher) for a state of this query capacity.

    One CTA per 128-row q block fills one wave of SMs only for long queries,
    so a state splits each block's sample tiles over n_parts CTAs: as many as
    one wave holds, at most 8. Chosen per state (its partial-keep scratch is
    rows x n_parts x 4 x keep_b words) and compiled once per (compile_key,
    n_parts)."""
    device_index = dev.index if dev.index is not None else torch.cuda.current_device()
    nsm = torch.cuda.get_device_properties(device_index).multi_processor_count
    n_parts = min(8, max(1, nsm // max(1, query_capacity // BLK)))
    key = (*compile_key, n_parts)
    entry = _compiled_k1.get(key)
    if entry is None:
        launcher = _SimplifiedSparseSampleThresholdK1(topk, tcap, n_parts)
        parts_shape, cnt_shape = launcher.scratch_shapes(query_capacity)
        # Signature examples; the kernels are runtime-shaped.
        thr = torch.empty((query_capacity, 2), dtype=torch.int32, device=dev)
        bnd = torch.empty((query_capacity,), dtype=torch.float32, device=dev)
        parts = torch.empty(parts_shape, dtype=torch.int32, device=dev)
        cnt = torch.empty(cnt_shape, dtype=torch.int32, device=dev)

        def _c(t, align=16):
            return from_dlpack(t, assumed_align=align, enable_tvm_ffi=True).mark_layout_dynamic()

        compiled = cute.compile(
            launcher,
            _c(q),
            _c(k),
            _c(starts, 4),
            _c(ends, 4),
            _c(thr, 8),
            _c(bnd, 4),
            _c(parts, 4),
            _c(cnt, 4),
            I32(query_len),
            I32(key_pad // NT),
            stream=cute.runtime.make_fake_stream(),
            options="--enable-tvm-ffi",
        )
        entry = _compiled_k1[key] = (compiled, launcher)
    return entry


def _filtered_scratch(st, dev):
    """Main-path buffers for one call; the repair graph never reads them.

    Kernels run on the caller stream, which is also the allocation stream, so
    the caching allocator reuses these blocks only for later caller work.  The
    zero-initialized ones match a fresh state; the candidates are rewritten.
    """
    rows, width = st["query_capacity"], st["tcap_buf"]
    scratch = dict(
        # The k1 sampler has no histogram, and the filtered collect never
        # reads its histogram argument, so one row binds it.
        hist=torch.zeros((1 if SAMPLER == "k1" else rows, 256), dtype=torch.int32, device=dev),
        thr=torch.zeros((rows, 2), dtype=torch.int32, device=dev),
        bnd=torch.zeros((rows,), dtype=torch.float32, device=dev),
        cursors=torch.zeros((rows, 2), dtype=torch.int32, device=dev),
        candidx=torch.empty((rows, width), dtype=torch.int32, device=dev),
        candval=torch.empty((rows, width), dtype=torch.float32, device=dev),
    )
    if SAMPLER == "k1":
        # Split q blocks' partial keeps and sample counts (every slot is
        # written before the merge reads it; placeholders with one part).
        parts_shape, cnt_shape = st["k1_obj"].scratch_shapes(rows)
        scratch["k1_parts"] = torch.empty(parts_shape, dtype=torch.int32, device=dev)
        scratch["k1_cnt"] = torch.empty(cnt_shape, dtype=torch.int32, device=dev)
    return scratch


def _gvr_state(st, topk, dev):
    """Buffers of the optional GVR selector, created on its first use."""
    gvr = st.get("gvr")
    if gvr is None:
        from .gvr_topk import gvr_host

        rows = st["query_capacity"]
        gvr = st["gvr"] = dict(
            counts=torch.empty((rows,), dtype=torch.int32, device=dev),
            gvr_counts=torch.empty((rows,), dtype=torch.int32, device=dev),
            # GVR's streaming family publishes through this self-restoring zero
            # slab. Keep it local to the caller-stream state so concurrent
            # streams never share it.
            gvr_workspace=torch.zeros(gvr_host.workspace_bytes(), dtype=torch.uint8, device=dev),
            positions=torch.empty((rows, topk), dtype=torch.int32, device=dev),
            sel_vals=torch.empty((rows, topk), dtype=torch.float32, device=dev),
        )
    return gvr


def _filtered_capacity(length, alignment, headroom=True):
    # Headroom absorbs physical packed-length jitter without another repair
    # graph capture. Main launches still use actual Tq/Tk. Storage remains
    # O(Tq*K + Tk), with at most 12.5% plus one alignment tile of slack.
    # A caller-declared maximum (reserve_fused_filtered) needs no headroom.
    if headroom:
        length += (length + 7) // 8
    return ((length + alignment - 1) // alignment) * alignment


def _filtered_cache_event(event, started, **details):
    if os.environ.get("SSA_FILTERED_CACHE_DIAGNOSTICS", "0") != "1":
        return
    # Miss-only host timing: no tensor read or CUDA synchronization. Optional
    # diagnostics must never turn a successful operation into a failed run.
    try:
        _logger.warning(
            "SSA_FILTERED_CACHE %s",
            json.dumps(
                dict(
                    event=event,
                    rank=os.environ.get("RANK"),
                    pid=os.getpid(),
                    wall_time=time.time(),
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    **details,
                )
            ),
        )
    except Exception:
        pass


@contextmanager
def _graph_capture(graph, stream):
    # torch.cuda.graph also synchronizes the device and empties the CUDA and
    # pinned-host caches on entry. In a training step that frees every cached
    # block (seconds), which later work then allocates again. The repair
    # graph's buffers exist before capture. Capture is otherwise the same:
    # a private pool, global capture mode, on ``stream``.
    with torch.cuda.stream(stream):
        graph.capture_begin()
        try:
            yield
        finally:
            graph.capture_end()


def _release_retired_filtered():
    # Growing storage replaces graph-owned addresses. Keep the old graph and
    # tensors alive until all preceding caller-stream work has completed.
    # query() is nonblocking; this list is empty on normal steady-state hits.
    with _retired_filtered_lock:
        _retired_filtered[:] = [(done, state) for done, state in _retired_filtered if not done.query()]


def _conditional_graph_body():
    try:
        from torch._higher_order_ops.cudagraph_conditional_nodes import _if_body
    except ImportError as error:
        raise NotImplementedError(
            "simplified_sparse_attention filtered indexing requires PyTorch CUDA conditional graph support "
            f"(tested with PyTorch 2.13); found {torch.__version__}."
        ) from error
    return _if_body


def _filtered_state(dev, dtype, topk, query_len, key_len, exact=False):
    """Persistent filtered state of the current stream, grown to these extents.

    New storage gets 12.5% headroom unless ``exact`` (a caller-declared
    maximum). Storage never shrinks; outgrown storage is retired.
    """
    _conditional_graph_body()
    query_pad = ((query_len + BLK - 1) // BLK) * BLK
    key_pad = ((key_len + NT - 1) // NT) * NT
    tcap = _tie_cap(topk, key_pad)
    caller_stream = torch.cuda.current_stream(dev)
    compile_key = (dev.index, topk, MASK_WALK, FAST_CNT, RAW_CMP)
    cache_key = (*compile_key, caller_stream.cuda_stream)
    if _retired_filtered:
        _release_retired_filtered()
    st = _state_filtered.get(cache_key)
    if st is None or query_pad > st["query_capacity"] or key_pad > st["key_capacity"]:
        state_started = time.perf_counter()
        previous_state = st
        query_capacity = max(st["query_capacity"] if st else 0, _filtered_capacity(query_len, BLK, headroom=not exact))
        key_capacity = max(st["key_capacity"] if st else 0, _filtered_capacity(key_len, C_TILES * NT, headroom=not exact))
        # The conditional repair graph embeds tensor addresses.  Keep its
        # inputs and outputs in stream-local persistent storage, then copy at
        # the public boundary.  This avoids per-step graph recapture when an
        # odd shape needs padding or the adapter returns fresh output tensors.
        # These buffers are linear in Tq/Tk (or Tq*K), never Tq*Tk.  The main
        # path's scratch (histogram, thresholds, cursors, candidates) is not
        # in the graph: every call allocates its own (see _filtered_scratch).
        q = torch.zeros((query_capacity, D), dtype=dtype, device=dev)
        k = torch.zeros((key_capacity, D), dtype=dtype, device=dev)
        starts = torch.zeros((query_capacity,), dtype=torch.int32, device=dev)
        ends = torch.zeros((query_capacity,), dtype=torch.int32, device=dev)
        out_idx = torch.empty((query_capacity, topk), dtype=torch.int32, device=dev)
        out_prob = torch.empty((query_capacity, topk), dtype=torch.float32, device=dev)
        # candidx/candval physical width padded to a float4 multiple so the
        # gvr_compact_select path (SSA_FILTERED_SELECT=gvr) can hand candval to
        # GVR (which requires a %4 row stride). The collect still bounds writes
        # by the logical tie_cap; the extra columns are unused. Harmless for the
        # native compact_select path (indexed by stride).
        tcap_buf = (tcap + 15) // 16 * 16
        # flags[0] is the number of certificate failures; flags[1:] is the
        # compact, device-produced row-id list used by the row-only fallback.
        flags = torch.zeros((query_capacity + 1,), dtype=torch.int32, device=dev)
        fallback_stats = torch.zeros((2,), dtype=torch.int32, device=dev)
        # Optional diagnostics (see SSA_FUSED_FALLBACK_KINDS/SCR_COUNTS).
        flag_kinds = torch.zeros((2,), dtype=torch.int32, device=dev) if _FALLBACK_KINDS else None
        scr_sample_scores = None
        if _SCR_COUNTS:
            sample_tiles = (key_capacity // NT + SAMPLE_STRIDE - 1) // SAMPLE_STRIDE
            scr_sample_scores = torch.empty((4, sample_tiles * NT), dtype=torch.float32, device=dev)
        # CUDA conditional graphs reject the 64 nodes required by an 8K/128
        # row-by-row repair graph.  Seven nodes are known-good on the target
        # runtime (the 769-row odd gate captures seven), so group consecutive
        # repair chunks under at most seven predicates.  Gather/scatter guard
        # every slot against flags[0], preserving exact row-only repair; only
        # harmless dummy work in the last active group is overcomputed.
        # Group sizes grow geometrically (1, 1, 2, 4, 8, 16 chunks, then the
        # rest), so a call with a few failing rows -- the k1 sampler's rare
        # capacity overflows -- repairs one 128-row chunk instead of a
        # uniform seventh of the capacity (11 chunks at Tq 8K, about 3.3 ms at
        # 1M keys); a call failing every row still runs every chunk once.
        max_repair_chunks = (query_capacity + BLK - 1) // BLK
        repair_group_chunks = []  # (first chunk, chunk count) per group
        first = 0
        while first < max_repair_chunks:
            g = len(repair_group_chunks)
            size = max_repair_chunks - first if g == 6 else min(max_repair_chunks - first, 1 << max(0, g - 1))
            repair_group_chunks.append((first, size))
            first += size
        repair_group_count = len(repair_group_chunks)
        fallback_group_pred = torch.zeros((repair_group_count,), dtype=torch.bool, device=dev)
        fallback_group_offsets = torch.tensor(
            [first * BLK for first, _ in repair_group_chunks],
            dtype=torch.int32,
            device=dev,
        )
        fb_q = torch.zeros((BLK, D), dtype=q.dtype, device=dev)
        fb_starts = torch.zeros((BLK,), dtype=torch.int32, device=dev)
        fb_ends = torch.ones((BLK,), dtype=torch.int32, device=dev)
        fb_idx = torch.empty((BLK, topk), dtype=torch.int32, device=dev)
        fb_prob = torch.empty((BLK, topk), dtype=torch.float32, device=dev)

        fb_score = torch.empty((BLK, key_capacity), dtype=torch.float32, device=dev)
        compiled = _compiled_filtered.get(compile_key)
        if compiled is None:
            compile_started = time.perf_counter()
            # Signature examples of the per-call main-path scratch; the kernels
            # are runtime-shaped and reused when the state grows.
            hist = torch.empty((query_capacity, 256), dtype=torch.int32, device=dev)
            thr = torch.empty((query_capacity, 2), dtype=torch.int32, device=dev)
            cursors = torch.empty((query_capacity, 2), dtype=torch.int32, device=dev)
            bnd = torch.empty((query_capacity,), dtype=torch.float32, device=dev)
            candidx = torch.empty((query_capacity, tcap_buf), dtype=torch.int32, device=dev)
            candval = torch.empty((query_capacity, tcap_buf), dtype=torch.float32, device=dev)

            def _c(t, align=16):
                return from_dlpack(t, assumed_align=align, enable_tvm_ffi=True).mark_layout_dynamic()

            main_stream = cute.runtime.make_fake_stream()
            # Diagnostic variants take one extra trailing argument, passed
            # at compile and launch only when enabled. The k1 sampler replaces
            # the sample, threshold and refine kernels (see _k1_compiled).
            ps = (
                None
                if SAMPLER == "k1"
                else cute.compile(
                    _SimplifiedSparseSampleHistFused(query_capacity, query_capacity, key_capacity),
                    _c(q),
                    _c(k),
                    _c(hist),
                    _c(starts, 4),
                    _c(ends, 4),
                    I32(query_len),
                    I32(key_pad // NT),
                    stream=main_stream,
                    options="--enable-tvm-ffi",
                    **({"sample_scores": _c(scr_sample_scores)} if _SCR_COUNTS else {}),
                )
            )
            # With the measured fine cut, the threshold kernel writes the
            # coarse boundary and the refine kernels replace it.
            ts = _SimplifiedSparseThresholdSelect(query_capacity, topk, key_capacity, fine_bnd=False if FINE_REFINE else None)
            thrk = (
                None
                if SAMPLER == "k1"
                else cute.compile(
                    ts.threshold_sample,
                    _c(hist),
                    _c(starts, 4),
                    _c(ends, 4),
                    _c(thr, 8),
                    _c(bnd, 4),
                    I32(query_len),
                    stream=main_stream,
                    options="--enable-tvm-ffi",
                )
            )
            pr = thrr = None
            if FINE_REFINE and SAMPLER != "k1":
                pr = cute.compile(
                    _SimplifiedSparseSampleRefineFused(query_capacity, query_capacity, key_capacity),
                    _c(q),
                    _c(k),
                    _c(hist),
                    _c(thr, 8),
                    _c(starts, 4),
                    _c(ends, 4),
                    I32(query_len),
                    I32(key_pad // NT),
                    stream=main_stream,
                    options="--enable-tvm-ffi",
                )
                thrr = cute.compile(
                    ts.threshold_refine,
                    _c(hist),
                    _c(thr, 8),
                    _c(bnd, 4),
                    I32(query_len),
                    stream=main_stream,
                    options="--enable-tvm-ffi",
                )
            if KMAJOR:
                pf = cute.compile(
                    _SimplifiedSparseFilteredKMajor(query_capacity, query_pad, key_pad, topk),
                    _c(q),
                    _c(k),
                    _c(thr, 8),
                    _c(bnd, 4),
                    _c(starts, 4),
                    _c(ends, 4),
                    _c(cursors, 8),
                    _c(candidx),
                    _c(candval),
                    stream=main_stream,
                    options="--enable-tvm-ffi",
                )
            else:
                pf = cute.compile(
                    _SimplifiedSparseFilteredCollect(query_capacity, query_capacity, key_capacity, topk, scr_counts=_SCR_COUNTS),
                    _c(q),
                    _c(k),
                    _c(hist),
                    _c(thr, 8),
                    _c(bnd, 4),
                    _c(starts, 4),
                    _c(ends, 4),
                    _c(cursors, 8),
                    _c(candidx),
                    _c(candval),
                    I32(query_len),
                    I32(key_pad // NT),
                    stream=main_stream,
                    options="--enable-tvm-ffi",
                )
            p3 = cute.compile(
                ts.select_single,
                _c(candidx),
                _c(candval),
                _c(cursors, 8),
                _c(thr, 8),
                _c(starts, 4),
                _c(ends, 4),
                _c(out_idx),
                _c(out_prob),
                _c(flags, 4),
                _c(fallback_group_pred, 1),
                I32(query_len),
                stream=main_stream,
                options="--enable-tvm-ffi",
                **({"flag_kinds": _c(flag_kinds, 4)} if _FALLBACK_KINDS else {}),
            )
            del hist, thr, cursors, bnd, candidx, candval
            fallback_stream = cute.runtime.make_fake_stream()
            fb_gather = cute.compile(
                _SimplifiedSparseFallbackGatherRows(),
                _c(flags, 4),
                _c(q),
                _c(starts, 4),
                _c(ends, 4),
                _c(fb_q),
                _c(fb_starts, 4),
                _c(fb_ends, 4),
                I32(0),
                stream=fallback_stream,
                options="--enable-tvm-ffi",
            )
            fb_scatter = cute.compile(
                _SimplifiedSparseFallbackScatterRows(topk),
                _c(flags, 4),
                _c(fb_idx),
                _c(fb_prob),
                _c(out_idx),
                _c(out_prob),
                _c(fallback_stats, 4),
                I32(0),
                stream=fallback_stream,
                options="--enable-tvm-ffi",
            )
            # Prepare the exact 128-row repair pipeline without launching it. This
            # lets a conditional graph be captured for a new pointer set without
            # the old empty repair launch or host synchronization on the common
            # zero-certificate-failure path.
            from .dense_gemm_sm100 import DenseGemmKernel

            fb_gemm = DenseGemmKernel(F32, False, (128, 256), (1, 1), True)
            fb_mnkl = (BLK, key_capacity, D, 1)
            if not fb_gemm.can_implement(fb_mnkl, BF16, F32, "k", "k", "n"):
                raise ValueError(f"fallback dense_gemm cannot implement {fb_mnkl}")

            def _gemm_c(t):
                return from_dlpack(t, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1)

            fb_a_c = _gemm_c(fb_q.unsqueeze(-1))
            fb_b_c = _gemm_c(k.unsqueeze(-1))
            fb_c_c = _gemm_c(fb_score.unsqueeze(-1))
            fb_score_compiled = cute.compile(
                fb_gemm,
                fb_a_c,
                fb_b_c,
                fb_c_c,
                stream=fallback_stream,
                epilogue_op=lambda x: x * SCALE,
                options="--enable-tvm-ffi",
            )
            compiled = dict(ps=ps, thrk=thrk, pr=pr, thrr=thrr, pf=pf, p3=p3, fb_score_compiled=fb_score_compiled, fb_gather=fb_gather, fb_scatter=fb_scatter)
            _compiled_filtered[compile_key] = compiled
            _filtered_cache_event("compile", compile_started, key=compile_key, query_len=query_len, key_len=key_len)
        k1 = k1_obj = None
        if SAMPLER == "k1":
            k1, k1_obj = _k1_compiled(compile_key, dev, topk, tcap, query_capacity, query_len, key_pad, q, k, starts, ends)
        st = dict(
            q=q,
            k=k,
            starts=starts,
            ends=ends,
            k1=k1,
            k1_obj=k1_obj,
            out_idx=out_idx,
            out_prob=out_prob,
            tcap_buf=tcap_buf,
            flags=flags,
            fallback_stats=fallback_stats,
            flag_kinds=flag_kinds,
            scr_sample_scores=scr_sample_scores,
            fallback_group_pred=fallback_group_pred,
            fallback_group_offsets=fallback_group_offsets,
            repair_group_count=repair_group_count,
            repair_group_chunks=repair_group_chunks,
            fb_q=fb_q,
            fb_starts=fb_starts,
            fb_ends=fb_ends,
            fb_idx=fb_idx,
            fb_prob=fb_prob,
            fb_score=fb_score,
            query_capacity=query_capacity,
            key_capacity=key_capacity,
            fallback_graphs={},
            **compiled,
        )
        if previous_state is not None:
            # The stream lock orders repair waits and output copies. Retain
            # old graph addresses until that work completes, without a sync.
            st["fallback_stats"].copy_(previous_state["fallback_stats"])
            if flag_kinds is not None:
                st["flag_kinds"].copy_(previous_state["flag_kinds"])
            retired_done = torch.cuda.Event()
            retired_done.record(caller_stream)
            with _retired_filtered_lock:
                _retired_filtered.append((retired_done, previous_state))
        _state_filtered[cache_key] = st
        _filtered_cache_event(
            "state",
            state_started,
            key=cache_key,
            query_len=query_len,
            key_len=key_len,
            query_capacity=query_capacity,
            key_capacity=key_capacity,
            grew=previous_state is not None,
        )
    return st, cache_key


def _filtered_repair_graph(st, graph_key, topk, cache_key, query_len, key_len):
    """Conditional exact-repair graph over ``st``'s storage, captured once."""
    import cuda.bindings.driver as _cuda

    q, k, starts, ends = st["q"], st["k"], st["starts"], st["ends"]
    out_idx, out_prob = st["out_idx"], st["out_prob"]
    graph_entry = st["fallback_graphs"].get(graph_key)
    if graph_entry is None:
        capture_started = time.perf_counter()
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SSA filtered sparse fallback graph must be warmed before " "entering an outer CUDA graph capture")
        import tvm_ffi

        _if_body = _conditional_graph_body()

        from .cute_dsl_filter_topk.filtered_top_k_prefill_varlen import _prepare_topk_prefill

        capture_stream = torch.cuda.Stream(device=q.device)
        capture_stream.wait_stream(torch.cuda.current_stream(q.device))
        fb_score_args = (
            st["fb_q"].unsqueeze(-1),
            k.unsqueeze(-1),
            st["fb_score"].unsqueeze(-1),
        )
        fb_topk_launch = _prepare_topk_prefill(
            st["fb_score"],
            st["fb_starts"],
            st["fb_ends"],
            topk,
            # This kernel only runs for rows whose bounded candidate list
            # overflowed.  Enter the exact GMEM reread path directly: the
            # optimistic REREAD policy's runtime-overflow branch corrupts
            # all-equal 65K rows, while REREAD_ALWAYS is exact at both 65K
            # and 131K.  The repair score tile remains fixed at 128 x Tk, so
            # this changes neither the O(Tk) repair workspace nor the common
            # path.
            overflow_policy="REREAD_ALWAYS",
            return_val=False,
            in_kernel_sort=True,
            ssa_epilogue=True,
            ssa_tri_input=False,
            logical_num_cols=st["key_capacity"],
            out_indices=st["fb_idx"],
            out_probs=st["fb_prob"],
            subtract_row_start=False,
            # Exact repair must not use the float-compare classifier: at the
            # signed-zero bin straddle it drops an exact +0.0 K-th score and
            # leaves one output slot unwritten.
            boundary_classify=False,
        )

        # Resolve every TVM-FFI CUDA function through the loader ABI. This is
        # library setup only: no empty repair kernel and no device synchronize.
        # The first real launch can therefore be captured inside the false-by-
        # default conditional node.
        device_index = q.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        _preload_tvm_ffi_cuda(st["fb_gather"], device_index)
        _preload_tvm_ffi_cuda(st["fb_score_compiled"], device_index)
        _preload_tvm_ffi_cuda(fb_topk_launch.launcher, device_index)
        _preload_tvm_ffi_cuda(st["fb_scatter"], device_index)

        def _launch_sparse_chunk(offset):
            # ``_if_body`` switches PyTorch's current stream to the CUDA
            # conditional node's child-capture stream.  Resolve that stream
            # here, inside the body: using the outer graph-capture stream
            # launches these kernels outside the child graph, so replay sees
            # stale/zero fallback buffers and scatters invalid routing IDs.
            child_stream_arg = _cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
            st["fb_gather"](st["flags"], q, starts, ends, st["fb_q"], st["fb_starts"], st["fb_ends"], offset, child_stream_arg)
            st["fb_score_compiled"](*fb_score_args, child_stream_arg)
            # This launcher obtains its stream from TVM-FFI rather than an
            # explicit argument.  Rebind the environment after entering the
            # conditional body so it observes the same child stream.
            with tvm_ffi.use_torch_stream():
                fb_topk_launch.run()
            st["fb_scatter"](st["flags"], st["fb_idx"], st["fb_prob"], out_idx, out_prob, st["fallback_stats"], offset, child_stream_arg)

        graph = torch.cuda.CUDAGraph()
        group_count = st["repair_group_count"]

        def _capture_group(group):
            pred = st["fallback_group_pred"][group]
            first_chunk, chunk_count = st["repair_group_chunks"][group]
            with _if_body(pred):
                for chunk in range(first_chunk, first_chunk + chunk_count):
                    _launch_sparse_chunk(chunk * BLK)

        with torch.cuda.stream(capture_stream):
            with tvm_ffi.use_torch_stream(_graph_capture(graph, capture_stream)):
                for group in range(group_count):
                    _capture_group(group)
        torch.cuda.current_stream(q.device).wait_stream(capture_stream)

        graph_entry = {
            "graph": graph,
            "capture_stream": capture_stream,
            # Keep the synchronization events alive. Stream.wait_stream()
            # creates temporary events whose destruction can block the host
            # until the caller's queued work completes.
            "main_done": torch.cuda.Event(),
            "repair_done": torch.cuda.Event(),
            "fb_topk_launch": fb_topk_launch,
            # Pin every address embedded in the graph against allocator reuse.
            "refs": (q, k, starts, ends, out_idx, out_prob),
        }
        st["fallback_graphs"][graph_key] = graph_entry
        _filtered_cache_event(
            "capture",
            capture_started,
            key=cache_key,
            selector=graph_key,
            query_len=query_len,
            key_len=key_len,
            query_capacity=st["query_capacity"],
            key_capacity=st["key_capacity"],
        )
        # Match the existing launch-wrapper pointer-cache policy.
        if len(st["fallback_graphs"]) > 8:
            oldest = next(iter(st["fallback_graphs"]))
            if oldest != graph_key:
                st["fallback_graphs"].pop(oldest)
    return graph_entry


def _run_fused_filtered_impl(
    q_idx,
    k_idx,
    topk_indices,
    predict,
    key_starts,
    key_ends,
):
    """Seeded filtered single-pass bounded SSA indexer (fusion="filtered").

    The full GEMM runs exactly ONCE: a 1/SAMPLE_STRIDE column sample pass
    builds a per-row seeded coarse-threshold estimate (+SAMPLE_MARGIN bins),
    and supplies the full pass's FP32 collect boundary. The full pass appends
    a compact (idx, fp32 value) list without materializing S. The candidate
    cursor count is the exact correctness certificate (>= TOPK and <= cap
    proves the true top-k is inside). Compact select refines the exact top-k,
    sorts ascending, and computes the predict softmax. Certificate failures
    are compacted on device and repaired exactly by the conditional row graph;
    SSA_FUSED_CHECK=1 instead reads the certificate and fails loudly for
    validation.
    """
    if os.environ.get("SSA_FUSED_NOSYNC", "0") != "0":
        raise ValueError("SSA_FUSED_NOSYNC is an unsupported approximate benchmark path; " "production filtered top-k always runs certificate repair")
    if os.environ.get("SSA_FUSED_KMAJOR", "0") != "0":
        raise ValueError("SSA_FUSED_KMAJOR is an unqualified experimental route; " "production filtered top-k uses the qualified row-major kernel")
    query_len = q_idx.shape[0]
    key_len = k_idx.shape[0]
    if key_starts is None or key_ends is None:
        raise ValueError("fused SSA indexer requires explicit key bounds")
    if key_starts.shape != (query_len,) or key_ends.shape != (query_len,):
        raise ValueError("SSA indexer expects one contiguous [start,end) pair per query row")
    if key_starts.dtype != torch.int32 or key_ends.dtype != torch.int32:
        raise TypeError("SSA indexer key_starts/key_ends must be int32")

    query_pad = ((query_len + BLK - 1) // BLK) * BLK
    key_pad = ((key_len + NT - 1) // NT) * NT
    q_src = q_idx.view(query_len, D)
    k_src = k_idx.view(key_len, D)

    topk = _derive_topk(topk_indices, query_len)
    tcap = _tie_cap(topk, key_pad)
    caller_out_idx = topk_indices.view(query_len, topk)
    caller_out_prob = predict.view(query_len, topk)

    dev = q_src.device
    st, cache_key = _filtered_state(dev, q_src.dtype, topk, query_len, key_len)
    caller_stream_id = torch.cuda.current_stream(dev).cuda_stream

    q = st["q"]
    k = st["k"]
    starts = st["starts"]
    ends = st["ends"]
    out_idx = st["out_idx"]
    out_prob = st["out_prob"]
    # A smaller shape may reuse storage containing an earlier larger input.
    # Only the final loaded MMA tiles need zero padding; graph repair masks
    # every key by the current per-row bounds, including the capacity tail.
    q[query_len:query_pad].zero_()
    k[key_len:key_pad].zero_()
    q[:query_len].copy_(q_src)
    k[:key_len].copy_(k_src)
    starts[query_len:].zero_()
    ends[query_len:].zero_()
    starts[:query_len].copy_(key_starts)
    ends[:query_len].copy_(key_ends)

    def _copy_outputs():
        caller_out_idx.copy_(out_idx[:query_len])
        caller_out_prob.copy_(out_prob[:query_len])

    scratch = _filtered_scratch(st, dev)

    # Keep the native compact selector as the qualified default. The updated
    # GVR selector remains available for explicit qualification and comparison.
    select_impl = os.environ.get("SSA_FILTERED_SELECT", "compact")
    if select_impl not in ("gvr", "compact"):
        raise ValueError("SSA_FILTERED_SELECT must be 'gvr' or 'compact', " f"got {select_impl!r}")

    # The compiled TVM-FFI functions take the caller's CUDA stream as an
    # explicit runtime argument.  Using TVM-FFI's environment-stream context
    # here changes a process-global stream slot and can synchronize the old
    # stream on context entry; it also allowed launchers compiled without a
    # stream parameter to fall through to stream 0.
    import cuda.bindings.driver as _cuda

    caller_stream_arg = _cuda.CUstream(caller_stream_id)

    def _launch_main():
        st["flags"].zero_()
        st["fallback_group_pred"].zero_()
        if st["k1"] is not None:
            # k1: sample GEMM + per-row cut in one kernel (+ merge) -> thr/bnd.
            st["k1"](q, k, starts, ends, scratch["thr"], scratch["bnd"], scratch["k1_parts"], scratch["k1_cnt"], query_len, key_pad // NT, caller_stream_arg)
        elif _SCR_COUNTS:
            st["scr_sample_scores"].fill_(float("nan"))
            st["ps"](q, k, scratch["hist"], starts, ends, query_len, key_pad // NT, caller_stream_arg, st["scr_sample_scores"])
        else:
            st["ps"](q, k, scratch["hist"], starts, ends, query_len, key_pad // NT, caller_stream_arg)
        if st["k1"] is None:
            st["thrk"](scratch["hist"], starts, ends, scratch["thr"], scratch["bnd"], query_len, caller_stream_arg)
        if st["pr"] is not None:
            # The coarse histogram is consumed (the collect pass ignores it);
            # reuse it for the crossing bins' sub-codes.
            scratch["hist"].zero_()
            st["pr"](q, k, scratch["hist"], scratch["thr"], starts, ends, query_len, key_pad // NT, caller_stream_arg)
            st["thrr"](scratch["hist"], scratch["thr"], scratch["bnd"], query_len, caller_stream_arg)
        if KMAJOR:
            st["pf"](q, k, scratch["thr"], scratch["bnd"], starts, ends, scratch["cursors"], scratch["candidx"], scratch["candval"], caller_stream_arg)
        else:
            st["pf"](
                q,
                k,
                scratch["hist"],
                scratch["thr"],
                scratch["bnd"],
                starts,
                ends,
                scratch["cursors"],
                scratch["candidx"],
                scratch["candval"],
                query_len,
                key_pad // NT,
                caller_stream_arg,
            )
        if select_impl == "gvr":
            from .gvr_topk.wrapper import gvr_compact_select

            gvr = _gvr_state(st, topk, dev)
            gvr["counts"].copy_(scratch["cursors"][:, 0])
            gvr_compact_select(
                scratch["candidx"],
                scratch["candval"],
                gvr["counts"],
                starts,
                ends,
                topk,
                out_indices=out_idx,
                out_probs=out_prob,
                flags=st["flags"],
                tie_cap=tcap,
                fallback_pred=st["fallback_group_pred"],
                workspace={
                    "gvr_counts": gvr["gvr_counts"],
                    "positions": gvr["positions"],
                    "sel_vals": gvr["sel_vals"],
                    "gvr_workspace": gvr["gvr_workspace"],
                },
            )
        else:
            extra = (st["flag_kinds"],) if _FALLBACK_KINDS else ()
            st["p3"](
                scratch["candidx"],
                scratch["candval"],
                scratch["cursors"],
                scratch["thr"],
                starts,
                ends,
                out_idx,
                out_prob,
                st["flags"],
                st["fallback_group_pred"],
                query_len,
                caller_stream_arg,
                *extra,
            )
        # Materialize at most seven group predicates from the device
        # certificate.  The conditional graph then runs only enough groups to
        # cover the compact failure list, without reading its size on host.
        torch.gt(
            st["flags"][0],
            st["fallback_group_offsets"],
            out=st["fallback_group_pred"],
        )
        if _SCR_COUNTS and _SCR_OBSERVER is not None:
            unit = topk // SAMPLE_STRIDE
            _observe_fused_scr(
                dict(
                    query_len=query_len,
                    key_len=key_len,
                    topk=topk,
                    tcap=tcap,
                    sample_mode=SAMPLE_MODE,
                    sample_stride=SAMPLE_STRIDE,
                    sample_rank=max(1, (SAMPLE_RANK_NUM * unit + SAMPLE_RANK_DEN - 1) // SAMPLE_RANK_DEN),
                    cursors=scratch["cursors"][:query_len],
                    thr=scratch["thr"][:query_len],
                    bnd=scratch["bnd"][:query_len],
                    flags=st["flags"],
                    starts=starts[:query_len],
                    ends=ends[:query_len],
                    q=q[:query_len],
                    k=k[:key_len],
                    sample_scores=st["scr_sample_scores"],
                    sample_rows=(0, query_len // 3, 2 * query_len // 3, query_len - 1),
                )
            )

    global _TOTAL_CALLS, _FALLBACK_CALLS, _FALLBACK_ROWS
    _TOTAL_CALLS += 1
    fallback_mode = os.environ.get("SSA_FUSED_FALLBACK_MODE", "rows")
    check_only = os.environ.get("SSA_FUSED_CHECK", "0") == "1"
    if fallback_mode not in ("rows", "batch"):
        raise ValueError("SSA_FUSED_FALLBACK_MODE must be 'rows' or 'batch', " f"got {fallback_mode!r}")

    # Validation's fail-loud mode and the retained whole-batch bisect use the
    # eager certificate read deliberately.  Production row repair below never
    # reads the device count on the host.
    if check_only or fallback_mode == "batch":
        _launch_main()
        nbad = int(st["flags"][0].item())
        if nbad == 0:
            _copy_outputs()
            return
        if check_only:
            raise RuntimeError(f"fusion=filtered: {nbad} rows missed by the seeded threshold " "(raise SSA_FUSED_MARGIN / SSA_FUSED_TIE_CAP)")
        _FALLBACK_CALLS += 1
        _FALLBACK_ROWS += nbad
        if _FALLBACK_CALLS % _LOG_EVERY == 0:
            _logger.warning(
                "SSA filtered fusion: seeded threshold too tight for "
                "%d/%d rows (Tk=%d, K=%d) -- recomputing the batch exactly. "
                "Cumulative: %d/%d calls fell back, %d rows.",
                nbad,
                query_len,
                key_len,
                topk,
                _FALLBACK_CALLS,
                _TOTAL_CALLS,
                _FALLBACK_ROWS,
            )
        from .indexer_forward_topk_tcgen05 import run as _run_exact

        _run_exact(
            q_idx,
            k_idx,
            topk_indices,
            predict,
            key_starts,
            key_ends,
            triangular=False,
            fusion="off",
            topk_impl=_fallback_topk_impl(key_len),
            # Whole-batch fallback is retained only as a validation/bisect
            # oracle.  Match the production row-repair backend exactly so an
            # inherited REREAD-overflow bug cannot invalidate the comparator.
            topk_overflow_policy="REREAD_ALWAYS",
        )
        return

    # Production sparse repair: capture at most seven conditional groups of
    # exact 128-row repair chunks.  All embedded pointers refer to the
    # stream-local state above, so one graph serves changing caller tensors.
    # After warmup there is no host .item(), graph recapture, or host sync.
    graph_entry = _filtered_repair_graph(st, select_impl, topk, cache_key, query_len, key_len)

    # A CUDAGraph replays on its capture stream, not necessarily the caller's
    # current stream.  Add fresh event dependencies on every invocation:
    # main/certificate on caller -> conditional repair on capture stream ->
    # subsequent caller work.  The old one-time waits around graph capture did
    # not order later replays and could expose stale outputs on nondefault
    # streams.
    _launch_main()
    caller_stream = torch.cuda.current_stream(q.device)
    repair_stream = graph_entry["capture_stream"]
    graph_entry["main_done"].record(caller_stream)
    repair_stream.wait_event(graph_entry["main_done"])
    with torch.cuda.stream(repair_stream):
        graph_entry["graph"].replay()
        graph_entry["repair_done"].record(repair_stream)
    caller_stream.wait_event(graph_entry["repair_done"])
    _copy_outputs()


@torch.no_grad()
@serialize_workspace
def run_fused_filtered(
    q_idx,
    k_idx,
    topk_indices,
    predict,
    key_starts,
    key_ends,
):
    """Run the bounded filtered path on PyTorch's current CUDA stream."""
    return _run_fused_filtered_impl(
        q_idx,
        k_idx,
        topk_indices,
        predict,
        key_starts,
        key_ends,
    )


@torch.no_grad()
def reserve_fused_filtered(max_query_len, max_key_len, topk):
    """Size the current stream's filtered state for these maximum extents.

    Allocates the persistent storage without headroom, compiles the kernels
    and captures the repair graph, so later calls within the maxima neither
    reallocate nor recapture. A larger call still grows the state.
    """
    dev = torch.device("cuda", torch.cuda.current_device())
    with stream_lock(dev):
        st, cache_key = _filtered_state(dev, torch.bfloat16, topk, max_query_len, max_key_len, exact=True)
        _filtered_repair_graph(st, os.environ.get("SSA_FILTERED_SELECT", "compact"), topk, cache_key, max_query_len, max_key_len)
