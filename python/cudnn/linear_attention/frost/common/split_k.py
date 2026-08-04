# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split-K sequence partitioning for ALL chunked linear-attention kernels:
GDN (scalar ``gate (T, HO)``, b_t=64) and KDA / GDN-2 (per-key-channel
``gate (T, HO, DK)``, b_t=16) share this one three-kernel pipeline.

The chunked kernels are persistent one-CTA-per-(batch, head) kernels; when
``B * HO`` does not fill the SMs, long sequences serialize on a few CTAs.
This module cuts each (batch, head) sequence into independent work items at
"forgetting horizon" boundaries: a cut is placed only where the running
log2 of the forget gates saturates below a threshold on both sides, so the
recurrent state (forward) and the state gradient (backward) entering an
item can be reconstructed from a short warmup window recomputed by the item
itself — no state is exchanged between items.

Work-item table row (``WORK_ITEM_FIELDS`` x int32, chunk units)::

    [batch_idx, head_idx, wstart, wend, cstart, cend]

The item OWNS (writes outputs for) chunks ``[wstart, wend)``.  The forward
kernel COMPUTES chunks ``[cstart, wend)`` — ``[cstart, wstart)`` is the
left warmup that rebuilds the incoming state from zero (accurate to
``2^log2_threshold`` because the gate decay over the window saturates).
The backward kernel computes ``[wstart, cend)`` — ``[wend, cend)`` is the
right warmup for the reverse dH recurrence (the forward states come exactly
from the per-chunk H checkpoints).  ``cstart == 0`` items seed the true
initial state; ``cend == num_chunks`` items seed the true ``d_final_state``
— so the un-cut degenerate item ``(0, nc, 0, nc)`` reproduces the serial
kernel exactly.

Piece choice per (batch, head): spans never exceed ``ideal_chunks`` (total
work / SM count), so outlier-long sequences are always cut down to the
batch-wide grain.  When the grid is small (``n_tiles < 2 * num_sms``) a
wave-quantized search widens the cut further to fill the machine; in the
many-tile regime cutting only adds per-item overhead, so the cap alone
decides.

The pipeline (one fixed shape regardless of gate kind):

1. scan: a flat data-parallel grid — CTA ``(x, h)`` covers chunk-scratch
   rows ``[x * WARPS, (x + 1) * WARPS)`` of head ``h``, one warp per chunk
   — reduces the gate to per-chunk values in the caller's GMEM scratch
   (channel gates: max over the per-channel clamped-log2 sums, a cut is
   valid only when EVERY channel saturated; scalar gates: the plain sum).
   Only chunks within ``WARMUP_CAP_CHUNKS`` of a candidate boundary are
   read: skipping chunks can only RAISE the (negative) horizon sums, so the
   windowed scan never accepts an unsaturated cut, it can only skip one.
   Uncut tiles never touch the gate.
2. walk: one CTA per (batch, head) probes every candidate boundary in
   parallel (one warp per boundary, one lane per window chunk), then
   thread 0 walks the probe results and emits work items into the caller's
   ``item_scratch``.
3. order: one CTA bitonic-sorts the emitted items into ``work_items``,
   longest ``[cstart, cend)`` first, so the main kernels' ticket scheduler
   consumes them in LPT order — the makespan tail is set by whatever starts
   last, so the big items must go first.  This is what keeps ragged varlen
   batches balanced without cutting them.
"""

import math

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.cute.runtime import from_dlpack

WORK_ITEM_FIELDS = 6
WARMUP_CAP_CHUNKS = 32  # hard warmup cap: a cut must saturate within one warp of chunks per side
MAX_BLOCKS = 2048  # piece-count ceiling; host clamps ideal_chunks so the per-tile block count fits
WARP_SIZE = 32
WARPS = 8
THREADS_PER_BLOCK = WARPS * WARP_SIZE
SCAN_WARPS = 4
SCAN_THREADS = SCAN_WARPS * WARP_SIZE
SCAN_ROWS_PER_WARP = 4  # consecutive chunk rows per scan warp (amortizes the batch lookup + piece choice)
SCAN_TOKEN_STRIDE = 4  # sample every Nth token of a chunk: skipped tokens only RAISE the negative horizon sums (sound), and the safe-gate sigmoid is SFU-bound
ORDER_THREADS = 1024
ORDER_ELEMS = 4
ORDER_CAPACITY = (
    ORDER_THREADS * ORDER_ELEMS
)  # sort capacity (32 KB SMEM); the kernel always launches — past this the device-side branch copies through unsorted (>25 items/SM, LPT is noise there)
OVERHEAD_TOKENS = 256  # per-item fixed cost for the piece model: state reseed + pipeline refill + typical warmup
P_WINDOW = 16  # fill-regime piece-count search width
P_BELOW = 8  # how far below the ideal-cap floor the fill-regime search may go

_DEFAULT_LOG2_THRESHOLD = -10.0 / math.log(2.0)  # e^-10, in log2 units
RCP_LN2 = 1.4426950408889634  # 1/ln(2): natural-log gates -> the scan's log2 domain


def compute_ideal_chunks(total_tokens: int, n_heads_out: int, num_sms: int, b_t: int) -> int:
    """Ideal per-work-item span in chunks: total work across all (b,h)
    tiles divided by the SM count, floored at one chunk and clamped so the
    per-tile block count fits ``MAX_BLOCKS``."""
    total_chunks = -(-total_tokens // b_t)
    ideal_tokens = -(-(total_tokens * n_heads_out) // num_sms)
    ideal_chunks = max(1, -(-ideal_tokens // b_t))
    return max(ideal_chunks, -(-total_chunks // MAX_BLOCKS))


def chunk_scratch_rows(total_tokens: int, batch_size: int, b_t: int) -> int:
    """Rows of the ``(rows, HO)`` fp32 chunk-value scratch: per-batch chunk
    ranges are based at ``cu[b] // b_t + b``, so one extra row per sequence
    covers the ceil rounding."""
    return total_tokens // b_t + batch_size


def max_work_items(total_tokens: int, batch_size: int, n_heads_out: int, ideal_chunks: int, b_t: int, num_sms: int) -> int:
    """Provable upper bound on the emitted item count: spans are capped at
    ``ideal_chunks`` (at most ``ceil(nc / ideal) + 1`` pieces per (b, h)),
    zero-length sequences emit one item each, and the fill-regime search
    can widen any tile by up to ``P_WINDOW - 1`` extra pieces."""
    total_chunks = -(-total_tokens // b_t)
    window = P_WINDOW - 1 if batch_size * n_heads_out < 2 * num_sms else 0
    return n_heads_out * (-(-total_chunks // ideal_chunks) + (2 + window) * batch_size)


@cute.jit
def decode_work_item(cfg, tile_idx, cu_seqlens, mWorkItems):
    """Tile decode shared by every warp body of the main kernels.

    Legacy mode maps ``tile_idx`` to a full (batch, head) sequence; split-K
    mode reads the work-item row.  Returns
    ``(batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b,
    wstart, wend, cstart, cend)`` with chunk-unit bounds."""
    if cutlass.const_expr(cfg.split_k):
        batch_idx = mWorkItems[tile_idx, 0]
        head_idx = mWorkItems[tile_idx, 1]
        wstart = mWorkItems[tile_idx, 2]
        wend = mWorkItems[tile_idx, 3]
        cstart = mWorkItems[tile_idx, 4]
        cend = mWorkItems[tile_idx, 5]
        batch_start = cu_seqlens[batch_idx]
        batch_end = cu_seqlens[batch_idx + 1]
        seqlen_b = batch_end - batch_start
        num_chunks_b = cute.ceil_div(seqlen_b, cfg.b_t)
    else:
        batch_idx = tile_idx // cfg.n_heads_out
        head_idx = tile_idx % cfg.n_heads_out
        batch_start = cu_seqlens[batch_idx]
        batch_end = cu_seqlens[batch_idx + 1]
        seqlen_b = batch_end - batch_start
        num_chunks_b = cute.ceil_div(seqlen_b, cfg.b_t)
        wstart = cutlass.Int32(0)
        wend = num_chunks_b
        cstart = cutlass.Int32(0)
        cend = num_chunks_b
    return batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend


@cute.jit
def _emit_item(mWorkItems, mCount, batch_idx, head_idx, wstart, wend, cstart, cend):
    count_addr = mCount.iterator.toint()
    wi = inline_ptx(
        "atom.global.add.s32 {$w0}, [{$r0}], 1;",
        write_only_types=[cutlass.Int32],
        read_only_args=[count_addr],
    )
    mWorkItems[wi, 0] = batch_idx
    mWorkItems[wi, 1] = head_idx
    mWorkItems[wi, 2] = wstart
    mWorkItems[wi, 3] = wend
    mWorkItems[wi, 4] = cstart
    mWorkItems[wi, 5] = cend


@cute.jit
def _clamped_log2(log_gate: cutlass.Constexpr[bool], gate_val: cutlass.Float32) -> cutlass.Float32:
    """Log2-domain decay increment, clamped to <= 0 (a gate > 1 must not
    relax the horizon)."""
    if cutlass.const_expr(log_gate):
        lg = gate_val * cutlass.Float32(RCP_LN2)
    else:
        lg = cute.math.log2(gate_val + cutlass.Float32(1e-10), fastmath=True)
    return lg if lg < cutlass.Float32(0.0) else cutlass.Float32(0.0)


@cute.jit
def _piece_choice(overhead_chunks: cutlass.Constexpr[int], num_chunks_b, n_tiles, num_sms, ideal_chunks):
    """Per-tile piece choice.  Returns ``(span, num_blocks)``."""
    # even spread: spans never exceed ideal_chunks (total work / SM count)
    p_hi = num_chunks_b if num_chunks_b < cutlass.Int32(MAX_BLOCKS) else cutlass.Int32(MAX_BLOCKS)
    p_hi = p_hi if p_hi > 0 else cutlass.Int32(1)
    p = (num_chunks_b + ideal_chunks - cutlass.Int32(1)) // ideal_chunks
    p = p if p > 0 else cutlass.Int32(1)
    p = p if p < p_hi else p_hi
    if n_tiles < cutlass.Int32(2) * num_sms:
        # fill regime: SMs to spare — search piece counts around the cap on
        # the wave-quantized makespan estimate (the cap only binds in the
        # many-tile regime, where its even spread kills varlen tails)
        p_start = p - cutlass.Int32(P_BELOW)
        p_start = p_start if p_start > 0 else cutlass.Int32(1)
        best = cutlass.Int32(2147483647)
        for dp in cutlass.range_constexpr(P_WINDOW):
            cand = p_start + cutlass.Int32(dp)
            cand = cand if cand < p_hi else p_hi
            span_c = (num_chunks_b + cand - cutlass.Int32(1)) // cand
            waves = (n_tiles * cand + num_sms - cutlass.Int32(1)) // num_sms
            est = waves * (span_c + cutlass.Int32(overhead_chunks))
            hit = est < best
            best = est if hit else best
            p = cand if hit else p
        # the estimate flatters marginal cuts (measured: <25% predicted gain
        # loses to serial); cut only on a clear margin over uncut
        est1 = ((n_tiles + num_sms - cutlass.Int32(1)) // num_sms) * (num_chunks_b + cutlass.Int32(overhead_chunks))
        if cutlass.Int32(4) * best > cutlass.Int32(3) * est1:
            p = cutlass.Int32(1)
    span = cutlass.Int32(0)
    num_blocks = cutlass.Int32(0)
    if num_chunks_b > 0:
        span = (num_chunks_b + p - cutlass.Int32(1)) // p
        num_blocks = (num_chunks_b + span - cutlass.Int32(1)) // span
    return span, num_blocks


@cute.jit
def _tile_spans(b_t: cutlass.Constexpr[int], overhead_chunks: cutlass.Constexpr[int], n_heads_out, n_tiles, num_sms, ideal_chunks, mCuSeqlens, tile):
    """Per-tile decode + piece choice.  Returns ``(batch_idx, head_idx,
    batch_start, batch_end, num_chunks_b, cv_base, span, num_blocks)``;
    ``cv_base`` is the tile's row base in the GMEM chunk scratch."""
    batch_idx = tile // n_heads_out
    head_idx = tile % n_heads_out
    batch_start = mCuSeqlens[batch_idx]
    batch_end = mCuSeqlens[batch_idx + 1]
    seqlen_b = batch_end - batch_start
    num_chunks_b = cute.ceil_div(seqlen_b, b_t)
    cv_base = batch_start // cutlass.Int32(b_t) + batch_idx
    span, num_blocks = _piece_choice(overhead_chunks, num_chunks_b, n_tiles, num_sms, ideal_chunks)
    return batch_idx, head_idx, batch_start, batch_end, num_chunks_b, cv_base, span, num_blocks


@cute.jit
def _near_boundary(c, span, num_blocks):
    """True iff chunk ``c`` lies in the walk's read window of a candidate
    boundary: suffix ``[j*span - W, j*span)`` or prefix ``[j*span, j*span
    + W)`` for some ``j`` in ``[1, num_blocks)``."""
    j0 = c // span
    cm = c - j0 * span
    w = cutlass.Int32(WARMUP_CAP_CHUNKS)
    pre = (cm < w) and (j0 >= cutlass.Int32(1))
    suf = (span - cm <= w) and (j0 < num_blocks - cutlass.Int32(1))
    return pre or suf


@cute.kernel
def _scan_kernel(
    b_t: cutlass.Constexpr[int],
    log_gate: cutlass.Constexpr[bool],
    safe_gate: cutlass.Constexpr[bool],
    gate_channels: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    has_sched: cutlass.Constexpr[bool],
    n_heads_out: cutlass.Int32,
    n_tiles: cutlass.Int32,
    num_sms: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    gate_scale_log2: cutlass.Float32,
    mGate: cute.Tensor,
    mALog: cute.Tensor | None,
    mDtBias: cute.Tensor | None,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor,
    mCount: cute.Tensor,
    mSched: cute.Tensor | None,
):
    """Flat windowed chunk scan, per-channel gate (KDA / GDN-2): CTA
    ``(x, h)`` covers 16 chunk-scratch rows of head ``h``, one warp per
    chunk, lane ``l`` owning channels ``[l*cpl, (l+1)*cpl)``.  Chunks
    outside every cut window — and whole tiles the piece choice leaves
    uncut — never touch the gate.  CTA (0, 0) also zeroes the item count
    for the walk and the main kernels' scheduler ticket ring (the kernels
    leave the ring dirty on exit, so every launch needs a fresh build)."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()
    tidx = cutlass.Int32(tidx)
    head_idx = cutlass.Int32(bidx[1])
    if cutlass.Int32(bidx[0]) == 0 and head_idx == 0 and tidx == 0:
        mCount[0] = cutlass.Int32(0)
        if cutlass.const_expr(has_sched):
            i = cutlass.Int32(0)
            while i < mSched.shape[0]:
                mSched[i] = cutlass.Int32(0)
                i = i + cutlass.Int32(1)
    lidx = tidx % cutlass.Int32(WARP_SIZE)
    widx = tidx // cutlass.Int32(WARP_SIZE)
    row0 = (cutlass.Int32(bidx[0]) * cutlass.Int32(SCAN_WARPS) + widx) * cutlass.Int32(SCAN_ROWS_PER_WARP)

    # batch of the warp's first row: largest b with cu[b] // b_t + b <= row0
    lo = cutlass.Int32(0)
    hi = batch_size - cutlass.Int32(1)
    while lo < hi:
        mid = (lo + hi + cutlass.Int32(1)) // cutlass.Int32(2)
        take = mCuSeqlens[mid] // cutlass.Int32(b_t) + mid <= row0
        lo = mid if take else lo
        hi = hi if take else mid - cutlass.Int32(1)
    batch_idx = lo
    batch_start = mCuSeqlens[batch_idx]
    batch_end = mCuSeqlens[batch_idx + 1]
    num_chunks_b = cute.ceil_div(batch_end - batch_start, b_t)
    cv_base = batch_start // cutlass.Int32(b_t) + batch_idx
    # the piece choice is per batch: computed once here and again only when
    # a row crosses into the next batch — NOT per row (on uncut tiles this
    # is the scan's entire cost)
    span, num_blocks = _piece_choice(overhead_chunks, num_chunks_b, n_tiles, num_sms, ideal_chunks)
    for rr in cutlass.range_constexpr(SCAN_ROWS_PER_WARP):
        row = row0 + cutlass.Int32(rr)
        while (batch_idx + cutlass.Int32(1) < batch_size) and (mCuSeqlens[batch_idx + 1] // cutlass.Int32(b_t) + batch_idx + cutlass.Int32(1) <= row):
            batch_idx = batch_idx + cutlass.Int32(1)
            batch_start = mCuSeqlens[batch_idx]
            batch_end = mCuSeqlens[batch_idx + 1]
            num_chunks_b = cute.ceil_div(batch_end - batch_start, b_t)
            cv_base = batch_start // cutlass.Int32(b_t) + batch_idx
            span, num_blocks = _piece_choice(overhead_chunks, num_chunks_b, n_tiles, num_sms, ideal_chunks)
        c = row - cv_base
        if (c >= 0) and (c < num_chunks_b) and (num_blocks > 1):
            if _near_boundary(c, span if span > 0 else cutlass.Int32(1), num_blocks):
                # chunk value = max over channels of the per-channel
                # clamped-log2 sums (each lane owns a contiguous channel run)
                cpl = gate_channels // WARP_SIZE
                row_elems = n_heads_out * cutlass.Int32(gate_channels)
                lane_base = cutlass.Int64(head_idx * cutlass.Int32(gate_channels) + lidx * cutlass.Int32(cpl))
                gate_addr = mGate.iterator.toint() + lane_base * cutlass.Int64(4)
                gate_ptr = mGate.iterator + lane_base
                a_exp = cutlass.Float32(1.0)
                dt_vals = cutlass.Array(cutlass.Float32, cpl)
                if cutlass.const_expr(safe_gate):
                    # per-head rate + per-lane channel biases, fixed for the whole chunk
                    a_exp = cute.math.exp2(mALog[head_idx].to(cutlass.Float32) * cutlass.Float32(RCP_LN2), fastmath=True)
                    for q in cutlass.range_constexpr(cpl):
                        dt_vals[q] = mDtBias[head_idx, lidx * cutlass.Int32(cpl) + cutlass.Int32(q)].to(cutlass.Float32)
                ch_acc = cutlass.Array(cutlass.Float32, cpl, alignment=16)
                for q in cutlass.range_constexpr(cpl):
                    ch_acc[q] = cutlass.Float32(0.0)
                oob = cutlass.Float32(0.0) if cutlass.const_expr(log_gate) else cutlass.Float32(1.0)
                for tt in cutlass.range(0, b_t, SCAN_TOKEN_STRIDE, unroll_full=True):
                    pos = batch_start + c * cutlass.Int32(b_t) + cutlass.Int32(tt)
                    inb = pos < batch_end
                    pos_r = pos if inb else batch_start
                    grow = cutlass.Int64(pos_r) * cutlass.Int64(row_elems)
                    if cutlass.const_expr(cpl % 4 == 0):
                        for q4 in cutlass.range_constexpr(cpl // 4):
                            addr = gate_addr + (grow + cutlass.Int64(4 * q4)) * cutlass.Int64(4)
                            g0, g1, g2, g3 = inline_ptx(
                                "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
                                write_only_types=[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32],
                                read_only_args=[addr],
                            )
                            for qq, gvq in enumerate((g0, g1, g2, g3)):
                                q = 4 * q4 + qq
                                if cutlass.const_expr(safe_gate):
                                    # the main kernel's transform, in log2 domain: scale * sigmoid(exp(a_log) * (g + dt_bias))
                                    half = cutlass.Float32(0.5)
                                    sig = cute.math.tanh(a_exp * (gvq + dt_vals[q]) * half, approx=True) * half + half
                                    contrib = gate_scale_log2 * sig
                                    contrib = contrib if inb else cutlass.Float32(0.0)
                                else:
                                    gvq = gvq if inb else oob
                                    contrib = _clamped_log2(log_gate, gvq)
                                ch_acc[q] = ch_acc[q] + contrib
                    else:
                        for q in cutlass.range_constexpr(cpl):
                            gv = (gate_ptr + grow + cutlass.Int32(q)).load()
                            if cutlass.const_expr(safe_gate):
                                half = cutlass.Float32(0.5)
                                sig = cute.math.tanh(a_exp * (gv + dt_vals[q]) * half, approx=True) * half + half
                                contrib = gate_scale_log2 * sig
                                contrib = contrib if inb else cutlass.Float32(0.0)
                            else:
                                gv = gv if inb else oob
                                contrib = _clamped_log2(log_gate, gv)
                            ch_acc[q] = ch_acc[q] + contrib
                m = ch_acc[0]
                for q in cutlass.range_constexpr(1, cpl):
                    m = m if m > ch_acc[q] else ch_acc[q]
                for off in [1, 2, 4, 8, 16]:
                    other = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, m, off, 31, kind=nvvm.Shfl.BFLY))
                    m = m if m > other else other
                if lidx == 0:
                    mChunkVals[cv_base + c, head_idx] = m


@cute.kernel
def _scan_scalar_kernel(
    b_t: cutlass.Constexpr[int],
    log_gate: cutlass.Constexpr[bool],
    overhead_chunks: cutlass.Constexpr[int],
    has_sched: cutlass.Constexpr[bool],
    n_heads_out: cutlass.Int32,
    n_tiles: cutlass.Int32,
    num_sms: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    mGate: cute.Tensor,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor,
    mCount: cute.Tensor,
    mSched: cute.Tensor | None,
):
    """Scalar-gate scan (GDN): CTA ``(x, hg)`` covers 16 chunk-scratch rows
    for heads ``[hg*32, (hg+1)*32)``; lane ``l`` owns head ``hg*32 + l``, so
    gate reads and chunk-value writes are coalesced across lanes and every
    lane accumulates its own head — no reduction.  CTA (0, 0) zeroes the
    item count and the scheduler ring, as in the per-channel scan."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()
    tidx = cutlass.Int32(tidx)
    if cutlass.Int32(bidx[0]) == 0 and cutlass.Int32(bidx[1]) == 0 and tidx == 0:
        mCount[0] = cutlass.Int32(0)
        if cutlass.const_expr(has_sched):
            i = cutlass.Int32(0)
            while i < mSched.shape[0]:
                mSched[i] = cutlass.Int32(0)
                i = i + cutlass.Int32(1)
    lidx = tidx % cutlass.Int32(WARP_SIZE)
    widx = tidx // cutlass.Int32(WARP_SIZE)
    h = cutlass.Int32(bidx[1]) * cutlass.Int32(WARP_SIZE) + lidx
    h_ok = h < n_heads_out
    h_r = h if h_ok else n_heads_out - cutlass.Int32(1)
    row0 = (cutlass.Int32(bidx[0]) * cutlass.Int32(SCAN_WARPS) + widx) * cutlass.Int32(SCAN_ROWS_PER_WARP)

    # batch of the warp's first row: largest b with cu[b] // b_t + b <= row0
    lo = cutlass.Int32(0)
    hi = batch_size - cutlass.Int32(1)
    while lo < hi:
        mid = (lo + hi + cutlass.Int32(1)) // cutlass.Int32(2)
        take = mCuSeqlens[mid] // cutlass.Int32(b_t) + mid <= row0
        lo = mid if take else lo
        hi = hi if take else mid - cutlass.Int32(1)
    batch_idx = lo
    batch_start = mCuSeqlens[batch_idx]
    batch_end = mCuSeqlens[batch_idx + 1]
    num_chunks_b = cute.ceil_div(batch_end - batch_start, b_t)
    cv_base = batch_start // cutlass.Int32(b_t) + batch_idx
    span, num_blocks = _piece_choice(overhead_chunks, num_chunks_b, n_tiles, num_sms, ideal_chunks)
    for rr in cutlass.range_constexpr(SCAN_ROWS_PER_WARP):
        row = row0 + cutlass.Int32(rr)
        while (batch_idx + cutlass.Int32(1) < batch_size) and (mCuSeqlens[batch_idx + 1] // cutlass.Int32(b_t) + batch_idx + cutlass.Int32(1) <= row):
            batch_idx = batch_idx + cutlass.Int32(1)
            batch_start = mCuSeqlens[batch_idx]
            batch_end = mCuSeqlens[batch_idx + 1]
            num_chunks_b = cute.ceil_div(batch_end - batch_start, b_t)
            cv_base = batch_start // cutlass.Int32(b_t) + batch_idx
            span, num_blocks = _piece_choice(overhead_chunks, num_chunks_b, n_tiles, num_sms, ideal_chunks)
        c = row - cv_base
        if (c >= 0) and (c < num_chunks_b) and (num_blocks > 1):
            if _near_boundary(c, span if span > 0 else cutlass.Int32(1), num_blocks):
                oob = cutlass.Float32(0.0) if cutlass.const_expr(log_gate) else cutlass.Float32(1.0)
                acc = cutlass.Float32(0.0)
                for tt in cutlass.range(0, b_t, SCAN_TOKEN_STRIDE, unroll_full=True):
                    pos = batch_start + c * cutlass.Int32(b_t) + cutlass.Int32(tt)
                    inb = pos < batch_end
                    pos_r = pos if inb else batch_start
                    gv = (mGate.iterator + cutlass.Int64(pos_r) * cutlass.Int64(n_heads_out) + h_r).load()
                    gv = gv if inb else oob
                    acc = acc + _clamped_log2(log_gate, gv)
                if h_ok:
                    mChunkVals[cv_base + c, h] = acc


@cute.kernel
def _walk_kernel(
    b_t: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    n_heads_out: cutlass.Int32,
    n_tiles: cutlass.Int32,
    num_sms: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor,
    mStaging: cute.Tensor,
    mCount: cute.Tensor,
):
    """One CTA per (batch, head): every candidate boundary is probed by its
    own warp (one lane per window chunk, warp-scan for the smallest
    saturating warmup), then thread 0 walks the probe results and emits the
    items."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    tidx = cutlass.Int32(tidx)

    batch_idx, head_idx, batch_start, batch_end, num_chunks_b, cv_base, span, num_blocks = _tile_spans(
        b_t, overhead_chunks, n_heads_out, n_tiles, num_sms, ideal_chunks, mCuSeqlens, cutlass.Int32(bidx)
    )
    if num_blocks <= 1:
        # single piece: no cuts, nothing scanned
        if tidx == 0:
            _emit_item(mStaging, mCount, batch_idx, head_idx, cutlass.Int32(0), num_chunks_b, cutlass.Int32(0), num_chunks_b)

    else:
        # packed per-boundary probe results: warm_b | warm_f << 8, 0 = no cut
        sWarm = cutlass.Array(cutlass.Int32, MAX_BLOCKS, space=cutlass.AddressSpace.smem, alignment=16)
        lidx = tidx % cutlass.Int32(WARP_SIZE)
        widx = tidx // cutlass.Int32(WARP_SIZE)
        big = cutlass.Int32(2 * WARMUP_CAP_CHUNKS)
        j = widx + cutlass.Int32(1)
        while j < num_blocks:
            wend = j * span
            # fwd warmup: smallest chunk suffix of [0, wend) that saturates
            idx = wend - cutlass.Int32(1) - lidx
            ok_b = idx >= 0
            v = mChunkVals[cv_base + (idx if ok_b else cutlass.Int32(0)), head_idx]
            acc = v if ok_b else cutlass.Float32(0.0)
            for off in [1, 2, 4, 8, 16]:
                o = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, acc, off, 0, kind=nvvm.Shfl.UP))
                acc = acc + (o if lidx >= cutlass.Int32(off) else cutlass.Float32(0.0))
            cand = lidx + cutlass.Int32(1) if (ok_b and acc <= log2_thresh) else big
            for off in [1, 2, 4, 8, 16]:
                other = cutlass.Int32(nvvm.shfl_sync(0xFFFFFFFF, cand, off, 31, kind=nvvm.Shfl.BFLY))
                cand = cand if cand < other else other
            warm_b = cand if cand < big else cutlass.Int32(0)
            # bwd warmup: smallest chunk prefix of [wend, nc) that saturates
            idx = wend + lidx
            ok_f = idx < num_chunks_b
            v = mChunkVals[cv_base + (idx if ok_f else cutlass.Int32(0)), head_idx]
            acc = v if ok_f else cutlass.Float32(0.0)
            for off in [1, 2, 4, 8, 16]:
                o = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, acc, off, 0, kind=nvvm.Shfl.UP))
                acc = acc + (o if lidx >= cutlass.Int32(off) else cutlass.Float32(0.0))
            cand = lidx + cutlass.Int32(1) if (ok_f and acc <= log2_thresh) else big
            for off in [1, 2, 4, 8, 16]:
                other = cutlass.Int32(nvvm.shfl_sync(0xFFFFFFFF, cand, off, 31, kind=nvvm.Shfl.BFLY))
                cand = cand if cand < other else other
            warm_f = cand if cand < big else cutlass.Int32(0)
            if lidx == 0:
                packed = warm_b + warm_f * cutlass.Int32(256)
                packed = packed if (warm_b > 0 and warm_f > 0) else cutlass.Int32(0)
                sWarm[j] = packed
            j = j + cutlass.Int32(WARPS)
        nvvm.barrier_cta_sync()

        if tidx == 0:
            prev_cut = cutlass.Int32(0)  # wstart of the open item, chunk units
            cur_cstart = cutlass.Int32(0)  # cstart of the open item, chunk units
            jj = cutlass.Int32(1)
            while jj < num_blocks:
                r = sWarm[jj]
                if r != 0:
                    wend = jj * span
                    warm_b = r % cutlass.Int32(256)
                    warm_f = r // cutlass.Int32(256)
                    cend = wend + warm_f
                    cend = cend if cend < num_chunks_b else num_chunks_b
                    _emit_item(mStaging, mCount, batch_idx, head_idx, prev_cut, wend, cur_cstart, cend)
                    cur_cstart = wend - warm_b
                    prev_cut = wend
                jj = jj + cutlass.Int32(1)
            _emit_item(mStaging, mCount, batch_idx, head_idx, prev_cut, num_chunks_b, cur_cstart, num_chunks_b)


@cute.kernel
def _order_kernel(
    mStaging: cute.Tensor,
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
):
    """LPT ordering (single CTA): bitonic-sort the staged items by span
    ``cend - cstart``, longest first, and gather into the final table, so
    the ticket scheduler starts the big items before the filler."""
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    n = mCount[0]
    if n > cutlass.Int32(ORDER_CAPACITY):
        # dozens of items per SM: LPT stops mattering, copy through
        i = tidx
        while i < n:
            for f in cutlass.range_constexpr(WORK_ITEM_FIELDS):
                mWorkItems[i, cutlass.Int32(f)] = mStaging[i, cutlass.Int32(f)]
            i = i + cutlass.Int32(ORDER_THREADS)
    else:
        sKey = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
        sIdx = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
        sSpread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
        if tidx == 0:
            sSpread[0] = cutlass.Int32(2147483647)
            sSpread[1] = cutlass.Int32(-2147483648)
        b_pad = cutlass.Int32(1)
        while b_pad < n:
            b_pad = b_pad * cutlass.Int32(2)
        nvvm.barrier_cta_sync()
        kmin = cutlass.Int32(2147483647)
        kmax = cutlass.Int32(-2147483648)
        for e in cutlass.range_constexpr(ORDER_ELEMS):
            i = tidx + cutlass.Int32(e * ORDER_THREADS)
            if i < n:
                key = mStaging[i, 5] - mStaging[i, 4]
                sKey[i] = key
                sIdx[i] = i
                kmin = kmin if kmin < key else key
                kmax = kmax if kmax > key else key
            elif i < b_pad:
                # pads sort to the end of the descending order
                sKey[i] = cutlass.Int32(-2147483648)
                sIdx[i] = i
        nvvm.atomicrmw("min", sSpread.data_ptr(0), kmin, space=nvvm.SharedSpace.shared_cta)
        nvvm.atomicrmw("max", sSpread.data_ptr(1), kmax, space=nvvm.SharedSpace.shared_cta)
        nvvm.barrier_cta_sync()
        if sSpread[0] == sSpread[1]:
            # every key equal (uniform batches): any order is LPT, copy through
            i2 = tidx
            while i2 < n:
                for f in cutlass.range_constexpr(WORK_ITEM_FIELDS):
                    mWorkItems[i2, cutlass.Int32(f)] = mStaging[i2, cutlass.Int32(f)]
                i2 = i2 + cutlass.Int32(ORDER_THREADS)
        else:
            k = cutlass.Int32(2)
            while k <= b_pad:
                j = k // cutlass.Int32(2)
                while j > 0:
                    for e in cutlass.range_constexpr(ORDER_ELEMS):
                        i = tidx + cutlass.Int32(e * ORDER_THREADS)
                        if i < b_pad:
                            l = i ^ j
                            if l > i:
                                ki = sKey[i]
                                kl = sKey[l]
                                up = (i // k) % cutlass.Int32(2) == 0
                                dn = (i // k) % cutlass.Int32(2) == 1
                                swap = (up and ki < kl) or (dn and ki > kl)
                                if swap:
                                    sKey[i] = kl
                                    sKey[l] = ki
                                    si = sIdx[i]
                                    sIdx[i] = sIdx[l]
                                    sIdx[l] = si
                    nvvm.barrier_cta_sync()
                    j = j // cutlass.Int32(2)
                k = k * cutlass.Int32(2)
            for e in cutlass.range_constexpr(ORDER_ELEMS):
                i = tidx + cutlass.Int32(e * ORDER_THREADS)
                if i < n:
                    src = sIdx[i]
                    for f in cutlass.range_constexpr(WORK_ITEM_FIELDS):
                        mWorkItems[i, cutlass.Int32(f)] = mStaging[src, cutlass.Int32(f)]


@cute.jit
def _launch(
    b_t: cutlass.Constexpr[int],
    log_gate: cutlass.Constexpr[bool],
    safe_gate: cutlass.Constexpr[bool],
    gate_channels: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    has_sched: cutlass.Constexpr[bool],
    n_heads_out: cutlass.Int32,
    n_tiles: cutlass.Int32,
    num_sms: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    gate_scale_log2: cutlass.Float32,
    mGate: cute.Tensor,
    mALog: cute.Tensor | None,
    mDtBias: cute.Tensor | None,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor,
    mStaging: cute.Tensor,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mSched: cute.Tensor | None,
    n_scan_ctas: cutlass.Int32,
    n_walk_ctas: cutlass.Int32,
    stream: cuda.CUstream,
) -> None:
    if cutlass.const_expr(gate_channels > 0):
        _scan_kernel(
            b_t,
            log_gate,
            safe_gate,
            gate_channels,
            overhead_chunks,
            has_sched,
            n_heads_out,
            n_tiles,
            num_sms,
            ideal_chunks,
            batch_size,
            gate_scale_log2,
            mGate,
            mALog,
            mDtBias,
            mCuSeqlens,
            mChunkVals,
            mCount,
            mSched,
        ).launch(
            grid=(n_scan_ctas, n_heads_out, 1),
            block=(SCAN_THREADS, 1, 1),
            stream=stream,
        )
    else:
        _scan_scalar_kernel(
            b_t,
            log_gate,
            overhead_chunks,
            has_sched,
            n_heads_out,
            n_tiles,
            num_sms,
            ideal_chunks,
            batch_size,
            mGate,
            mCuSeqlens,
            mChunkVals,
            mCount,
            mSched,
        ).launch(
            grid=(n_scan_ctas, (n_heads_out + cutlass.Int32(WARP_SIZE - 1)) // cutlass.Int32(WARP_SIZE), 1),
            block=(SCAN_THREADS, 1, 1),
            stream=stream,
        )
    _walk_kernel(
        b_t,
        overhead_chunks,
        n_heads_out,
        n_tiles,
        num_sms,
        ideal_chunks,
        log2_thresh,
        mCuSeqlens,
        mChunkVals,
        mStaging,
        mCount,
    ).launch(
        grid=(n_walk_ctas, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
        stream=stream,
    )
    _order_kernel(
        mStaging,
        mCount,
        mWorkItems,
    ).launch(
        grid=(1, 1, 1),
        block=(ORDER_THREADS, 1, 1),
        stream=stream,
    )


_compiled_cache = {}


def build_split_table(
    gate,
    cu_seqlens,
    work_items,
    work_count,
    *,
    ideal_chunks,
    n_tiles,
    num_sms,
    b_t,
    chunk_scratch,
    item_scratch,
    log2_threshold=None,
    log_gate=False,
    safe_gate=False,
    a_log=None,
    dt_bias=None,
    gate_lower_bound=None,
    sched_ctr=None,
    stream,
) -> None:
    """Fill ``work_items``/``work_count`` with the split-K partition of
    ``gate`` / ``cu_seqlens (B+1,) int32``, LPT-ordered (longest item
    first).  A 2-D ``(total_tokens, HO)`` gate is the scalar GDN kind; a
    3-D ``(total_tokens, HO, DK)`` gate is the per-key-channel KDA / GDN-2
    kind.  With ``log_gate`` the gate values are natural-log decay instead
    of raw linear alpha.  With ``safe_gate`` (per-channel only) the gate
    holds RAW logits and the scan applies the KDA safe-gate transform
    ``gate_lower_bound * sigmoid(exp(a_log) * (g + dt_bias))`` per element,
    so cuts land on true decay values.

    ``work_items`` and ``item_scratch`` are ``(max_items,
    WORK_ITEM_FIELDS)`` int32 with ``max_items >= max_work_items(...)``;
    ``work_count`` is ``(1,)`` int32 (zeroed here by the scan kernel, as
    is every cell of ``sched_ctr`` when passed — the main kernels' int32
    ticket rings, ``(2,)`` per kernel launch that consumes this table,
    since the kernels leave their ring dirty on exit);
    ``chunk_scratch`` is ``(>= chunk_scratch_rows(total_tokens,
    B, b_t), HO)`` fp32 (contents managed here).  Runs entirely on device
    — no host synchronization."""
    if log2_threshold is None:
        log2_threshold = _DEFAULT_LOG2_THRESHOLD
    if len(gate.shape) not in (2, 3):
        raise ValueError(f"gate must be (total_tokens, HO) or (total_tokens, HO, DK), got {tuple(gate.shape)}")
    gate_channels = gate.shape[2] if len(gate.shape) == 3 else 0
    if safe_gate and gate_channels == 0:
        raise ValueError("safe_gate applies to per-channel gates only")
    if safe_gate and (a_log is None or dt_bias is None or gate_lower_bound is None):
        raise ValueError("safe_gate requires a_log, dt_bias, and gate_lower_bound")
    if not safe_gate:
        a_log = None
        dt_bias = None
    if gate_channels and gate_channels % WARP_SIZE != 0:
        raise ValueError(f"per-channel gate dim must be a multiple of {WARP_SIZE}, got {gate_channels}")
    if gate_channels and gate_channels % 128 == 0 and gate.data_ptr() % 16 != 0:
        raise ValueError("per-channel gate base must be 16-byte aligned (vectorized scan loads)")
    gate_scale_log2 = float(gate_lower_bound) * RCP_LN2 if safe_gate else 0.0
    n_heads_out = gate.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    n_walk_ctas = batch_size * n_heads_out
    need_rows = chunk_scratch_rows(gate.shape[0], batch_size, b_t)
    n_scan_ctas = -(-need_rows // (SCAN_WARPS * SCAN_ROWS_PER_WARP))
    if len(chunk_scratch.shape) != 2 or chunk_scratch.shape[0] < need_rows or chunk_scratch.shape[1] != n_heads_out:
        raise ValueError(f"chunk_scratch must be (>= {need_rows}, {n_heads_out}) fp32, got {tuple(chunk_scratch.shape)}")
    if tuple(item_scratch.shape) != tuple(work_items.shape) or work_items.shape[1] != WORK_ITEM_FIELDS:
        raise ValueError(
            f"item_scratch must match work_items (max_items, {WORK_ITEM_FIELDS}) int32, got {tuple(item_scratch.shape)} vs {tuple(work_items.shape)}"
        )
    overhead_chunks = max(1, OVERHEAD_TOKENS // b_t)
    cu_stream = cuda.CUstream(int(stream))

    key = (b_t, n_heads_out, bool(log_gate), bool(safe_gate), gate_channels, sched_ctr is not None)
    if key not in _compiled_cache:

        def _dyn(t):
            c = from_dlpack(t, assumed_align=4)
            c.mark_compact_shape_dynamic(mode=0, stride_order=tuple(range(len(t.shape))), divisibility=1)
            return c

        _compiled_cache[key] = cute.compile(
            _launch,
            b_t,
            bool(log_gate),
            bool(safe_gate),
            gate_channels,
            overhead_chunks,
            sched_ctr is not None,
            cutlass.Int32(n_heads_out),
            cutlass.Int32(n_tiles),
            cutlass.Int32(num_sms),
            cutlass.Int32(ideal_chunks),
            cutlass.Int32(batch_size),
            cutlass.Float32(log2_threshold),
            cutlass.Float32(gate_scale_log2),
            _dyn(gate),
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if safe_gate else None,
            _dyn(dt_bias) if safe_gate else None,
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(),
            _dyn(chunk_scratch),
            _dyn(item_scratch),
            _dyn(work_items),
            _dyn(work_count),
            from_dlpack(sched_ctr, assumed_align=4).mark_layout_dynamic() if sched_ctr is not None else None,
            cutlass.Int32(n_scan_ctas),
            cutlass.Int32(n_walk_ctas),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    _compiled_cache[key](
        n_heads_out,
        n_tiles,
        num_sms,
        ideal_chunks,
        batch_size,
        float(log2_threshold),
        float(gate_scale_log2),
        gate,
        a_log,
        dt_bias,
        cu_seqlens,
        chunk_scratch,
        item_scratch,
        work_items,
        work_count,
        sched_ctr,
        n_scan_ctas,
        n_walk_ctas,
        cu_stream,
    )
