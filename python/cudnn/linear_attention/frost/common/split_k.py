# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split-K sequence partitioning for the chunked linear-attention kernels:
GDN (scalar ``gate (T, HO)``, b_t=64) and KDA / GDN-2 (per-key-channel
``gate (T, HO, DK)``, b_t=16).

The main kernels are persistent, one CTA per (batch, head); when ``B * HO``
does not fill the SMs, long sequences serialize on a few CTAs.  This module
cuts a sequence into independent work items where the forget gates saturate
on BOTH sides of the cut, so the recurrent state (forward) and its gradient
(backward) can be rebuilt from a short warmup the item recomputes itself.
No state is exchanged between items.

Work-item row (``WORK_ITEM_FIELDS`` x int32, chunk units)::

    [batch_idx, head_idx, write_start, write_end, compute_start, compute_end, batch_start, batch_end]

The item writes outputs for ``[write_start, write_end)``.  Forward computes
``[compute_start, write_end)``, backward ``[write_start, compute_end)``; the
extra chunks on each side are the warmups, accurate to ``2^log2_threshold``.
``compute_start == 0`` seeds the true initial state and ``compute_end ==
num_chunks`` the true ``d_final_state``, so the uncut item ``(0, nc, 0, nc)``
reproduces the serial kernel exactly.

The pipeline:

0. plan: one thread settles the batch-wide span the piece choice may be
   overridden with and leaves it in the chunk scratch's last row.  It is
   batch-global, and deriving it inside the scan and the walk instead puts its
   code in kernels that run tens of thousands of warps.
1. scan: reduce the gate to one value per chunk into the caller's GMEM
   scratch (channel gates: max over the per-channel clamped-log2 sums, so a
   cut needs EVERY channel saturated; scalar gates: the plain sum).  Only
   chunks near a candidate boundary are read; skipping chunks only raises the
   negative sums, so the window can miss a cut but never accept a bad one.
2. walk: one CTA per (batch, head), one warp per candidate boundary.  A cut is
   not pinned to ``j*span``: the warp scores positions within ``SNAP_CHUNKS``
   on ``warmup_before + warmup_after + |offset|`` and keeps the best, clamped
   to ``span // 2`` so adjacent cuts cannot cross.  Thread 0 emits the items.
3. order: bitonic-sort into ``work_items``, longest ``[compute_start,
   compute_end)`` first, so the ticket scheduler runs LPT, and zero the
   scheduler rings (dirty on exit).

With ``split=False`` scan and walk never launch and the prologue kernel
synthesizes the uncut item per (batch, head) from ``cu_seqlens`` alone: that
no-cuts table serves batch-invariant mode and checkpoint cadences a cut may
not cross.
"""

import math
from typing import NamedTuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.cute.runtime import from_dlpack

from cudnn.frost.buffers import data_ptr

from cudnn.frost.tile_dsl.pointwise import sigmoid, softplus

WORK_ITEM_FIELDS = 8
WARMUP_CAP_CHUNKS = 32  # hard warmup cap: a cut must saturate within one warp of chunks per side
SNAP_CHUNKS = 32  # cut-position search half-width
MAX_BLOCKS = 2048  # piece-count ceiling; host clamps ideal_chunks so the per-tile block count fits
WARP_SIZE = 32
WARPS = 8
THREADS_PER_BLOCK = WARPS * WARP_SIZE
SCAN_WARPS = 4
SCAN_THREADS = SCAN_WARPS * WARP_SIZE
SCAN_ROWS_PER_WARP_MAX = 4  # consecutive chunk rows per scan warp; scan_rows_per_warp() shrinks it to fill the SMs
SCAN_TOKEN_STRIDE = 1  # tokens sampled per chunk; a stride > 1 scales the effective threshold by it
OVERHEAD_TOKENS = 256  # per-item fixed cost for the piece model: state reseed + pipeline refill + typical warmup
P_WINDOW = 16  # fill-regime piece-count search width
P_BELOW = 8  # how far below the ideal-cap floor the fill-regime search may go

ORDER_THREADS = 1024
ORDER_ELEMENTS = 4
ORDER_CAPACITY = ORDER_THREADS * ORDER_ELEMENTS  # sort capacity (32 KB SMEM); past this the device-side branch copies through unsorted

DEFAULT_LOG2_THRESHOLD = -10.0 / math.log(2.0)  # e^-10, in log2 units
RCP_LN2 = 1.4426950408889634  # 1/ln(2): natural-log gates -> the scan's log2 domain


def compute_ideal_chunks(total_tokens: int, n_heads_out: int, num_sms: int, b_t: int) -> int:
    """Ideal per-work-item span in chunks: total work across all (b,h)
    tiles divided by the SM count, floored at one chunk and clamped so the
    per-tile block count fits ``MAX_BLOCKS``."""
    total_chunks = -(-total_tokens // b_t)
    ideal_tokens = -(-(total_tokens * n_heads_out) // num_sms)
    ideal_chunks = max(1, -(-ideal_tokens // b_t))
    return max(ideal_chunks, -(-total_chunks // MAX_BLOCKS))


def scan_rows_per_warp(need_rows: int, grid_y: int, num_sms: int) -> int:
    """Chunk rows each scan warp walks.

    Rows are the scan's only parallelism axis besides heads, and heads give a
    scalar-gate launch just ``ceil(HO / 32)`` blocks, so a long sequence at low
    batch can leave the grid at a fraction of the SMs.  Walking fewer rows per
    warp widens the grid (and shrinks the kernel, since the row loop is
    unrolled), but the per-warp prologue -- batch search plus piece choice --
    is then paid more often, which costs a channel-gate launch that already
    fills the machine.  So take the largest stride that still fills it."""
    for rows in (SCAN_ROWS_PER_WARP_MAX, 2, 1):
        if rows == 1 or -(-need_rows // (SCAN_WARPS * rows)) * grid_y >= num_sms:
            return rows


def chunk_scratch_rows(total_tokens: int, batch_size: int, b_t: int) -> int:
    """Rows of the ``(rows, HO)`` fp32 chunk-value scratch: per-batch chunk
    ranges are based at ``cu[b] // b_t + b``, so one extra row per sequence
    covers the ceil rounding, plus the last row, which carries the plan
    (:func:`frost_split_k_plan`) rather than a chunk value."""
    return total_tokens // b_t + batch_size + 1


def max_work_items(total_tokens: int, batch_size: int, n_heads_out: int, ideal_chunks: int, b_t: int, num_sms: int) -> int:
    """Provable upper bound on the emitted item count: spans are capped at
    ``ideal_chunks`` (at most ``ceil(nc / ideal) + 1`` pieces per (b, h)),
    zero-length sequences emit one item each, and the fill-regime search
    can widen any tile by up to ``P_WINDOW - 1`` extra pieces."""
    total_chunks = -(-total_tokens // b_t)
    window = P_WINDOW - 1 if batch_size * n_heads_out < 2 * num_sms else 0
    return n_heads_out * (-(-total_chunks // ideal_chunks) + (2 + window) * batch_size)


@cute.jit
def decode_work_item(cfg, tile_idx, mWorkItems):
    """Tile decode shared by every warp body of the main kernels: read the
    work-item row (an uncut table row IS the whole sequence).  Returns
    ``(batch_idx, head_idx, batch_start, batch_end, batch_seqlen,
    batch_num_chunks, write_start, write_end, compute_start,
    compute_end)`` with chunk-unit bounds."""
    batch_idx = mWorkItems[tile_idx, 0]
    head_idx = mWorkItems[tile_idx, 1]
    write_start = mWorkItems[tile_idx, 2]
    write_end = mWorkItems[tile_idx, 3]
    compute_start = mWorkItems[tile_idx, 4]
    compute_end = mWorkItems[tile_idx, 5]
    batch_start = mWorkItems[tile_idx, 6]
    batch_end = mWorkItems[tile_idx, 7]
    batch_seqlen = batch_end - batch_start
    batch_num_chunks = cute.ceil_div(batch_seqlen, cfg.b_t)
    return batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end


@cute.jit
def emit_item(mWorkItems, mCount, batch_idx, head_idx, write_start, write_end, compute_start, compute_end, batch_start, batch_end):
    count_addr = mCount.iterator.toint()
    slot = inline_ptx(
        "atom.global.add.s32 {$w0}, [{$r0}], 1;",
        write_only_types=[cutlass.Int32],
        read_only_args=[count_addr],
    )
    mWorkItems[slot, 0] = batch_idx
    mWorkItems[slot, 1] = head_idx
    mWorkItems[slot, 2] = write_start
    mWorkItems[slot, 3] = write_end
    mWorkItems[slot, 4] = compute_start
    mWorkItems[slot, 5] = compute_end
    mWorkItems[slot, 6] = batch_start
    mWorkItems[slot, 7] = batch_end


@cute.jit
def clamped_log2(log_gate: cutlass.Constexpr[bool], gate_val: cutlass.Float32) -> cutlass.Float32:
    """Log2-domain decay increment, clamped to <= 0 (a gate > 1 must not
    relax the horizon)."""
    if cutlass.const_expr(log_gate):
        lg = gate_val * cutlass.Float32(RCP_LN2)
    else:
        lg = cute.math.log2(gate_val + cutlass.Float32(1e-10), fastmath=True)
    return lg if lg < cutlass.Float32(0.0) else cutlass.Float32(0.0)


@cute.jit
def piece_choice(overhead_chunks: cutlass.Constexpr[int], num_sms: cutlass.Constexpr[int], batch_num_chunks, n_tiles, ideal_chunks):
    """Per-tile piece choice.  Returns ``(span, num_blocks)``."""
    # ---- even-spread cap: no span past ideal_chunks ----------------------------------
    p_hi = batch_num_chunks if batch_num_chunks < cutlass.Int32(MAX_BLOCKS) else cutlass.Int32(MAX_BLOCKS)
    p_hi = p_hi if p_hi > 0 else cutlass.Int32(1)
    p = (batch_num_chunks + ideal_chunks - cutlass.Int32(1)) // ideal_chunks
    p = p if p > 0 else cutlass.Int32(1)
    p = p if p < p_hi else p_hi
    if n_tiles < cutlass.Int32(2) * num_sms:

        # ---- fill regime: wave-quantized search around the cap -----------------------
        p_start = p - cutlass.Int32(P_BELOW)
        p_start = p_start if p_start > 0 else cutlass.Int32(1)
        best = cutlass.Int32(2147483647)
        for dp in cutlass.range(P_WINDOW):
            candidate = p_start + cutlass.Int32(dp)
            candidate = candidate if candidate < p_hi else p_hi
            span_c = (batch_num_chunks + candidate - cutlass.Int32(1)) // candidate
            waves = (n_tiles * candidate + num_sms - cutlass.Int32(1)) // num_sms
            estimate = waves * (span_c + cutlass.Int32(overhead_chunks))
            hit = estimate < best
            best = estimate if hit else best
            p = candidate if hit else p

        # ---- margin over uncut -------------------------------------------------------
        uncut_estimate = ((n_tiles + num_sms - cutlass.Int32(1)) // num_sms) * (batch_num_chunks + cutlass.Int32(overhead_chunks))
        if cutlass.Int32(4) * best > cutlass.Int32(3) * uncut_estimate:
            p = cutlass.Int32(1)
    span = cutlass.Int32(0)
    num_blocks = cutlass.Int32(0)
    if batch_num_chunks > 0:
        span = (batch_num_chunks + p - cutlass.Int32(1)) // p
        num_blocks = (batch_num_chunks + span - cutlass.Int32(1)) // span
    return span, num_blocks


@cute.jit
def common_span(
    b_t: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    n_heads_out: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    n_tiles,
    ideal_chunks,
    batch_size,
    mCuSeqlens,
):
    """Span for :func:`span_choice` to force on every tile the per-sequence
    choice left uncut, or 0 to keep that choice.

    :func:`piece_choice` scores a piece count with ``waves = n_tiles *
    candidate``, which assumes EVERY tile splits into ``candidate`` pieces.  A
    ragged batch does not, so the sequence that sets the makespan
    under-splits.  A span shared by the batch has an exact CTA count ``sum_b
    ceil(nc_b / span)``, which is what the search below scores.

    The answer depends on the batch alone, so each kernel evaluates it ONCE
    and combines it per tile.  Folded into the per-tile choice instead it is
    re-derived by every thread of the scan and walk grids and costs an order
    of magnitude more than the scan it gates."""
    chosen = cutlass.Int32(0)
    if n_tiles < num_sms:

        # ---- ragged? a uniform batch makes piece_choice's wave count exact -----------
        max_chunks = cutlass.Int32(0)
        min_chunks = cutlass.Int32(2147483647)
        b = cutlass.Int32(0)
        while b < batch_size:
            nc = cute.ceil_div(cutlass.Int32(mCuSeqlens[b + 1]) - cutlass.Int32(mCuSeqlens[b]), b_t)
            max_chunks = max_chunks if max_chunks > nc else nc
            min_chunks = min_chunks if min_chunks < nc else nc
            b = b + cutlass.Int32(1)

        if min_chunks < max_chunks:

            # ---- grid the per-sequence choice actually builds ------------------------
            blocks = cutlass.Int32(0)
            b = cutlass.Int32(0)
            while b < batch_size:
                nc = cute.ceil_div(cutlass.Int32(mCuSeqlens[b + 1]) - cutlass.Int32(mCuSeqlens[b]), b_t)
                _, nb = piece_choice(overhead_chunks, num_sms, nc, n_tiles, ideal_chunks)
                blocks = blocks + nb
                b = b + cutlass.Int32(1)

            # ---- underfilled: search a span shared by the batch ----------------------
            if blocks * n_heads_out < num_sms:
                best = cutlass.Int32(2147483647)
                common = max_chunks
                for dp in cutlass.range(P_WINDOW):
                    s = (max_chunks + cutlass.Int32(dp)) // (cutlass.Int32(dp) + cutlass.Int32(1))
                    ctas = cutlass.Int32(0)
                    b = cutlass.Int32(0)
                    while b < batch_size:
                        nc = cute.ceil_div(cutlass.Int32(mCuSeqlens[b + 1]) - cutlass.Int32(mCuSeqlens[b]), b_t)
                        ctas = ctas + cute.ceil_div(nc, s)
                        b = b + cutlass.Int32(1)
                    ctas = ctas * n_heads_out
                    waves = (ctas + num_sms - cutlass.Int32(1)) // num_sms
                    estimate = waves * (s + cutlass.Int32(overhead_chunks))
                    hit = estimate < best
                    best = estimate if hit else best
                    common = s if hit else common
                uncut = ((n_tiles + num_sms - cutlass.Int32(1)) // num_sms) * (max_chunks + cutlass.Int32(overhead_chunks))
                chosen = common if cutlass.Int32(4) * best <= cutlass.Int32(3) * uncut else cutlass.Int32(0)
    return chosen


@cute.jit
def span_choice(overhead_chunks: cutlass.Constexpr[int], num_sms: cutlass.Constexpr[int], batch_num_chunks, n_tiles, ideal_chunks, common):
    """Per-tile piece choice, overridden by the batch-wide ``common`` span
    from :func:`common_span` (0 when it does not apply) whenever the
    per-sequence choice left this tile in one piece.  Returns ``(span,
    num_blocks)``."""
    span, num_blocks = piece_choice(overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks)
    if (n_tiles < num_sms) and (num_blocks == cutlass.Int32(1)) and (common > cutlass.Int32(0)):
        span = common if common < batch_num_chunks else batch_num_chunks
        num_blocks = cute.ceil_div(batch_num_chunks, span) if span > 0 else cutlass.Int32(0)
    return span, num_blocks


@cute.jit
def near_boundary(c, span, num_blocks):
    """True iff chunk ``c`` lies in the walk's read window of a candidate
    boundary: suffix ``[j*span - W, j*span)`` or prefix ``[j*span, j*span
    + W)`` for some ``j`` in ``[1, num_blocks)``.  ``W`` spans the warmup
    cap plus the snap search."""
    j0 = c // span
    cm = c - j0 * span
    w = cutlass.Int32(WARMUP_CAP_CHUNKS + SNAP_CHUNKS)
    pre = (cm < w) and (j0 >= cutlass.Int32(1))
    suf = (span - cm <= w) and (j0 < num_blocks - cutlass.Int32(1))
    return pre or suf


@cute.jit
def saturating_length(lane_idx, idx, ok, chunk_value_base, head_idx, mChunkVals, log2_thresh, big):
    """Smallest window length whose chunk values sum past log2_thresh, lane
    l holding the window's l-th chunk (idx, valid iff ok): a warp
    prefix-sum, then a warp min over the lanes that crossed.  Returns 0 when no
    length within the warp saturates."""
    v = mChunkVals[chunk_value_base + (idx if ok else cutlass.Int32(0)), head_idx]
    acc = v if ok else cutlass.Float32(0.0)
    for off in [1, 2, 4, 8, 16]:
        o = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, acc, off, 0, kind=nvvm.Shfl.UP))
        acc = acc + (o if lane_idx >= cutlass.Int32(off) else cutlass.Float32(0.0))
    candidate = lane_idx + cutlass.Int32(1) if (ok and acc <= log2_thresh) else big
    for off in [1, 2, 4, 8, 16]:
        other = cutlass.Int32(nvvm.shfl_sync(0xFFFFFFFF, candidate, off, 31, kind=nvvm.Shfl.BFLY))
        candidate = candidate if candidate < other else other
    return candidate if candidate < big else cutlass.Int32(0)


@cute.jit
def probe_warmup_fwd(lane_idx, x, chunk_value_base, head_idx, mChunkVals, log2_thresh, big):
    """Forward state warmup at cut position x: the smallest chunk suffix
    [x - s, x) that saturates."""
    idx = x - cutlass.Int32(1) - lane_idx
    return saturating_length(lane_idx, idx, idx >= 0, chunk_value_base, head_idx, mChunkVals, log2_thresh, big)


@cute.jit
def probe_warmup_bwd(lane_idx, x, limit, chunk_value_base, head_idx, mChunkVals, log2_thresh, big):
    """Reverse dstate warmup at cut position x: the smallest chunk prefix
    [x, x + s) that saturates."""
    idx = x + lane_idx
    return saturating_length(lane_idx, idx, idx < limit, chunk_value_base, head_idx, mChunkVals, log2_thresh, big)


@cute.jit
def read_plan(mChunkVals):
    """The batch-wide span :func:`frost_split_k_plan` left in the scratch's
    last row.  Spans are small integers, exact in fp32."""
    return mChunkVals[mChunkVals.shape[0] - cutlass.Int32(1), 0].to(cutlass.Int32)


@cute.kernel
def frost_split_k_plan(
    b_t: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    n_heads_out: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor,
    mCount: cute.Tensor,
):
    """One thread: settle the batch-wide span once and leave it in the chunk
    scratch's last row for the scan and the walk, and zero the item count.

    :func:`common_span` is batch-global, so ONE evaluation is all that is
    needed -- but inlining it into the scan and the walk also puts its code
    (two ``piece_choice`` expansions plus a ``P_WINDOW`` search) in kernels
    that run tens of thousands of warps, and that costs those kernels ~7 us
    and ~5 us respectively even on shapes whose grid is too full for the
    probe to run at all.  Here it costs one small launch."""
    tidx, _, _ = cute.arch.thread_idx()
    if cutlass.Int32(tidx) == 0:
        mCount[0] = cutlass.Int32(0)
        common = common_span(b_t, overhead_chunks, n_heads_out, num_sms, n_tiles, ideal_chunks, batch_size, mCuSeqlens)
        mChunkVals[mChunkVals.shape[0] - cutlass.Int32(1), 0] = cutlass.Float32(common)


@cute.kernel
def frost_split_k_scan_channel(
    b_t: cutlass.Constexpr[int],
    scan_rows: cutlass.Constexpr[int],
    log_gate: cutlass.Constexpr[bool],
    safe_gate: cutlass.Constexpr[bool],
    gate_channels: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    n_heads_out: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    gate_scale_log2: cutlass.Float32,
    mGate: cute.Tensor,
    mALog: cute.Tensor | None,
    mDtBias: cute.Tensor | None,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor,
    mCount: cute.Tensor,
):
    """Windowed chunk scan, per-channel gate (KDA / GDN-2): CTA ``(x, h)``
    covers 16 chunk-scratch rows of head ``h``, one warp per chunk, lane
    ``l`` owning channels ``[l*cpl, (l+1)*cpl)``.  Chunks outside a cut
    window never touch the gate.  The item count is the plan kernel's."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()
    tidx = cutlass.Int32(tidx)
    head_idx = cutlass.Int32(bidx[1])
    lane_idx = tidx % cutlass.Int32(WARP_SIZE)
    widx = tidx // cutlass.Int32(WARP_SIZE)
    row0 = (cutlass.Int32(bidx[0]) * cutlass.Int32(SCAN_WARPS) + widx) * cutlass.Int32(scan_rows)
    common = read_plan(mChunkVals)

    lo = cutlass.Int32(0)
    hi = batch_size - cutlass.Int32(1)
    while lo < hi:
        mid = (lo + hi + cutlass.Int32(1)) // cutlass.Int32(2)
        take = cutlass.Int32(mCuSeqlens[mid]) // cutlass.Int32(b_t) + mid <= row0
        lo = mid if take else lo
        hi = hi if take else mid - cutlass.Int32(1)
    batch_idx = lo
    batch_start = cutlass.Int32(mCuSeqlens[batch_idx])
    batch_end = cutlass.Int32(mCuSeqlens[batch_idx + 1])
    batch_num_chunks = cute.ceil_div(batch_end - batch_start, b_t)
    chunk_value_base = batch_start // cutlass.Int32(b_t) + batch_idx
    span, num_blocks = span_choice(overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks, common)

    for rr in cutlass.range_constexpr(scan_rows):
        row = row0 + cutlass.Int32(rr)
        while (batch_idx + cutlass.Int32(1) < batch_size) and (
            cutlass.Int32(mCuSeqlens[batch_idx + 1]) // cutlass.Int32(b_t) + batch_idx + cutlass.Int32(1) <= row
        ):
            batch_idx = batch_idx + cutlass.Int32(1)
            batch_start = cutlass.Int32(mCuSeqlens[batch_idx])
            batch_end = cutlass.Int32(mCuSeqlens[batch_idx + 1])
            batch_num_chunks = cute.ceil_div(batch_end - batch_start, b_t)
            chunk_value_base = batch_start // cutlass.Int32(b_t) + batch_idx
            span, num_blocks = span_choice(overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks, common)
        c = row - chunk_value_base
        if (c >= 0) and (c < batch_num_chunks) and (num_blocks > 1):
            if near_boundary(c, span if span > 0 else cutlass.Int32(1), num_blocks):
                # ---- per-channel clamped-log2 sums, one channel run per lane ---------
                cpl = gate_channels // WARP_SIZE
                row_elements = cutlass.Int32(mGate.stride[0])
                lane_base = cutlass.Int64(head_idx * cutlass.Int32(mGate.stride[1]) + lane_idx * cutlass.Int32(cpl))
                gate_addr = mGate.iterator.toint() + lane_base * cutlass.Int64(4)
                gate_ptr = mGate.iterator + lane_base
                a_exp = cutlass.Float32(1.0)
                dt_vals = cutlass.Array(cutlass.Float32, cpl)
                if cutlass.const_expr(safe_gate):
                    a_exp = cute.math.exp2(mALog[head_idx].to(cutlass.Float32) * cutlass.Float32(RCP_LN2), fastmath=True)
                    for q in cutlass.range_constexpr(cpl):
                        dt_vals[q] = mDtBias[head_idx, lane_idx * cutlass.Int32(cpl) + cutlass.Int32(q)].to(cutlass.Float32)
                ch_acc = cutlass.Array(cutlass.Float32, cpl, alignment=16)
                for q in cutlass.range_constexpr(cpl):
                    ch_acc[q] = cutlass.Float32(0.0)
                oob = cutlass.Float32(0.0) if cutlass.const_expr(log_gate) else cutlass.Float32(1.0)
                for tt in cutlass.range(0, b_t, SCAN_TOKEN_STRIDE, unroll_full=True):
                    pos = batch_start + c * cutlass.Int32(b_t) + cutlass.Int32(tt)
                    inb = pos < batch_end
                    pos_r = pos if inb else batch_start
                    grow = cutlass.Int64(pos_r) * cutlass.Int64(row_elements)
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
                                    sig = sigmoid(a_exp * (gvq + dt_vals[q]))
                                    contrib = gate_scale_log2 * sig
                                    contrib = contrib if inb else cutlass.Float32(0.0)
                                else:
                                    gvq = gvq if inb else oob
                                    contrib = clamped_log2(log_gate, gvq)
                                ch_acc[q] = ch_acc[q] + contrib
                    else:
                        for q in cutlass.range_constexpr(cpl):
                            gv = (gate_ptr + grow + cutlass.Int32(q)).load()
                            if cutlass.const_expr(safe_gate):
                                sig = sigmoid(a_exp * (gv + dt_vals[q]))
                                contrib = gate_scale_log2 * sig
                                contrib = contrib if inb else cutlass.Float32(0.0)
                            else:
                                gv = gv if inb else oob
                                contrib = clamped_log2(log_gate, gv)
                            ch_acc[q] = ch_acc[q] + contrib

                # ---- chunk value = max over channels, then over lanes ----------------
                m = ch_acc[0]
                for q in cutlass.range_constexpr(1, cpl):
                    m = m if m > ch_acc[q] else ch_acc[q]
                for off in [1, 2, 4, 8, 16]:
                    other = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, m, off, 31, kind=nvvm.Shfl.BFLY))
                    m = m if m > other else other
                if lane_idx == 0:
                    mChunkVals[chunk_value_base + c, head_idx] = m


@cute.kernel
def frost_split_k_scan_scalar(
    b_t: cutlass.Constexpr[int],
    scan_rows: cutlass.Constexpr[int],
    log_gate: cutlass.Constexpr[bool],
    safe_gate: cutlass.Constexpr[bool],
    overhead_chunks: cutlass.Constexpr[int],
    n_heads_out: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    mGate: cute.Tensor,
    mALog: cute.Tensor | None,
    mDtBias: cute.Tensor | None,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor,
    mCount: cute.Tensor,
):
    """Windowed chunk scan, scalar gate (GDN): CTA ``(x, hg)`` covers 16
    chunk-scratch rows for heads ``[hg*32, (hg+1)*32)``; lane ``l`` owns head
    ``hg*32 + l``, so reads and writes coalesce and no reduction is needed.
    With ``safe_gate`` each token contributes ``-exp(A_log[h]) * softplus(g +
    dt_bias[h]) * RCP_LN2``.  The item count is the plan kernel's."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()
    tidx = cutlass.Int32(tidx)
    lane_idx = tidx % cutlass.Int32(WARP_SIZE)
    widx = tidx // cutlass.Int32(WARP_SIZE)
    h = cutlass.Int32(bidx[1]) * cutlass.Int32(WARP_SIZE) + lane_idx
    h_ok = h < n_heads_out
    h_r = h if h_ok else cutlass.Int32(n_heads_out - 1)
    a = cutlass.Float32(0.0)
    bias = cutlass.Float32(0.0)
    if cutlass.const_expr(safe_gate):
        a = -cute.math.exp2(mALog[h_r].to(cutlass.Float32) * cutlass.Float32(RCP_LN2), fastmath=True) * cutlass.Float32(RCP_LN2)
        bias = mDtBias[h_r].to(cutlass.Float32)
    row0 = (cutlass.Int32(bidx[0]) * cutlass.Int32(SCAN_WARPS) + widx) * cutlass.Int32(scan_rows)
    common = read_plan(mChunkVals)

    lo = cutlass.Int32(0)
    hi = batch_size - cutlass.Int32(1)
    while lo < hi:
        mid = (lo + hi + cutlass.Int32(1)) // cutlass.Int32(2)
        take = cutlass.Int32(mCuSeqlens[mid]) // cutlass.Int32(b_t) + mid <= row0
        lo = mid if take else lo
        hi = hi if take else mid - cutlass.Int32(1)
    batch_idx = lo
    batch_start = cutlass.Int32(mCuSeqlens[batch_idx])
    batch_end = cutlass.Int32(mCuSeqlens[batch_idx + 1])
    batch_num_chunks = cute.ceil_div(batch_end - batch_start, b_t)
    chunk_value_base = batch_start // cutlass.Int32(b_t) + batch_idx
    span, num_blocks = span_choice(overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks, common)

    for rr in cutlass.range_constexpr(scan_rows):
        row = row0 + cutlass.Int32(rr)
        while (batch_idx + cutlass.Int32(1) < batch_size) and (
            cutlass.Int32(mCuSeqlens[batch_idx + 1]) // cutlass.Int32(b_t) + batch_idx + cutlass.Int32(1) <= row
        ):
            batch_idx = batch_idx + cutlass.Int32(1)
            batch_start = cutlass.Int32(mCuSeqlens[batch_idx])
            batch_end = cutlass.Int32(mCuSeqlens[batch_idx + 1])
            batch_num_chunks = cute.ceil_div(batch_end - batch_start, b_t)
            chunk_value_base = batch_start // cutlass.Int32(b_t) + batch_idx
            span, num_blocks = span_choice(overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks, common)
        c = row - chunk_value_base
        if (c >= 0) and (c < batch_num_chunks) and (num_blocks > 1):
            if near_boundary(c, span if span > 0 else cutlass.Int32(1), num_blocks):
                # ---- clamped-log2 sum over the chunk, one head per lane --------------
                oob = cutlass.Float32(0.0) if cutlass.const_expr(log_gate) else cutlass.Float32(1.0)
                acc = cutlass.Float32(0.0)
                for tt in cutlass.range(0, b_t, SCAN_TOKEN_STRIDE, unroll_full=True):
                    pos = batch_start + c * cutlass.Int32(b_t) + cutlass.Int32(tt)
                    inb = pos < batch_end
                    pos_r = pos if inb else batch_start
                    gv = (mGate.iterator + cutlass.Int64(pos_r) * cutlass.Int64(mGate.stride[0]) + h_r).load()
                    if cutlass.const_expr(safe_gate):
                        contrib = a * softplus(gv + bias)
                        acc = acc + (contrib if inb else cutlass.Float32(0.0))
                    else:
                        gv = gv if inb else oob
                        acc = acc + clamped_log2(log_gate, gv)
                if h_ok:
                    mChunkVals[chunk_value_base + c, h] = acc


@cute.kernel
def frost_split_k_walk(
    b_t: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    n_heads_out: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    n_tiles: cutlass.Int32,
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
    tidx = cutlass.Int32(tidx)
    tile = cutlass.Int32(cute.arch.block_idx()[0])

    common = read_plan(mChunkVals)

    # ---- tile decode: chunk_value_base is the row base in the chunk scratch ----------
    batch_idx = tile // n_heads_out
    head_idx = tile % n_heads_out
    batch_start = cutlass.Int32(mCuSeqlens[batch_idx])
    batch_end = cutlass.Int32(mCuSeqlens[batch_idx + 1])
    batch_num_chunks = cute.ceil_div(batch_end - batch_start, b_t)
    chunk_value_base = batch_start // cutlass.Int32(b_t) + batch_idx
    span, num_blocks = span_choice(overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks, common)
    if num_blocks <= 1:
        # ---- nothing to scan: emit the whole sequence --------------------------------
        if tidx == 0:
            emit_item(mStaging, mCount, batch_idx, head_idx, cutlass.Int32(0), batch_num_chunks, cutlass.Int32(0), batch_num_chunks, batch_start, batch_end)

    else:
        # ---- scan the chunk sums to accept cuts --------------------------------------
        sWarmup = cutlass.Array(cutlass.Int32, MAX_BLOCKS, space=cutlass.AddressSpace.smem, alignment=16)
        lane_idx = tidx % cutlass.Int32(WARP_SIZE)
        widx = tidx // cutlass.Int32(WARP_SIZE)
        big = cutlass.Int32(2 * WARMUP_CAP_CHUNKS)
        half = span // cutlass.Int32(2)
        half = half if half < cutlass.Int32(SNAP_CHUNKS) else cutlass.Int32(SNAP_CHUNKS)
        j = widx + cutlass.Int32(1)
        while j < num_blocks:
            target = j * span
            best_key = cutlass.Int32(2147483647)
            best_pack = cutlass.Int32(0)
            # ---- offsets in |d| order, pruned once no remainder can win --------------
            i = cutlass.Int32(0)
            i_last = cutlass.Int32(2) * half
            searching = True
            while searching:
                ad = (i + cutlass.Int32(1)) // cutlass.Int32(2)
                d = ad if i % cutlass.Int32(2) == cutlass.Int32(1) else -ad
                x = target + d
                in_range = (x > cutlass.Int32(0)) and (x < batch_num_chunks)
                x_r = x if in_range else cutlass.Int32(1)
                warmup_before = probe_warmup_fwd(lane_idx, x_r, chunk_value_base, head_idx, mChunkVals, log2_thresh, big)
                warmup_after = probe_warmup_bwd(lane_idx, x_r, batch_num_chunks, chunk_value_base, head_idx, mChunkVals, log2_thresh, big)
                key = (warmup_before + warmup_after + ad) * cutlass.Int32(64) + ad
                live = in_range and (warmup_before > 0) and (warmup_after > 0) and (key < best_key)
                best_key = key if live else best_key
                best_pack = (warmup_before + warmup_after * cutlass.Int32(256) + (d + cutlass.Int32(SNAP_CHUNKS)) * cutlass.Int32(65536)) if live else best_pack
                i = i + cutlass.Int32(1)
                ad_next = (i + cutlass.Int32(1)) // cutlass.Int32(2)
                searching = (i <= i_last) and ((cutlass.Int32(2) + ad_next) * cutlass.Int32(64) < best_key)
            if lane_idx == 0:
                sWarmup[j] = best_pack
            j = j + cutlass.Int32(WARPS)
        nvvm.barrier_cta_sync()

        # ---- emit: thread 0 walks the accepted cuts in order -------------------------
        if tidx == 0:
            prev_cut = cutlass.Int32(0)  # write_start of the open item, chunk units
            current_compute_start = cutlass.Int32(0)  # compute_start of the open item, chunk units
            jj = cutlass.Int32(1)
            while jj < num_blocks:
                r = sWarmup[jj]
                if r != 0:
                    warmup_before = r % cutlass.Int32(256)
                    warmup_after = (r // cutlass.Int32(256)) % cutlass.Int32(256)
                    write_end = jj * span + r // cutlass.Int32(65536) - cutlass.Int32(SNAP_CHUNKS)
                    if write_end - prev_cut >= warmup_before + warmup_after:
                        compute_end = write_end + warmup_after
                        compute_end = compute_end if compute_end < batch_num_chunks else batch_num_chunks
                        emit_item(mStaging, mCount, batch_idx, head_idx, prev_cut, write_end, current_compute_start, compute_end, batch_start, batch_end)
                        current_compute_start = write_end - warmup_before
                        prev_cut = write_end
                jj = jj + cutlass.Int32(1)
            emit_item(mStaging, mCount, batch_idx, head_idx, prev_cut, batch_num_chunks, current_compute_start, batch_num_chunks, batch_start, batch_end)


@cute.jit
def gen_item_bounds(b_t: cutlass.Constexpr[int], n_heads_out, mCuSeqlens, item):
    """(batch, head, batch_start, batch_end, num_chunks) of the uncut
    whole-sequence item ``item`` — a no-cuts table row is pure geometry."""
    batch_idx = item // n_heads_out
    head_idx = item % n_heads_out
    batch_start = cutlass.Int32(mCuSeqlens[batch_idx])
    batch_end = cutlass.Int32(mCuSeqlens[batch_idx + 1])
    batch_num_chunks = cute.ceil_div(batch_end - batch_start, b_t)
    return batch_idx, head_idx, batch_start, batch_end, batch_num_chunks


@cute.jit
def write_item(
    gen: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    n_heads_out,
    mCuSeqlens,
    mStaging,
    mWorkItems,
    dst,
    src,
):
    """Final-table row ``dst`` from source item ``src``: the walk's staged
    row, or (``gen``) the synthesized uncut item ``(0, nc, 0, nc)``."""
    if cutlass.const_expr(gen):
        batch_idx, head_idx, batch_start, batch_end, batch_num_chunks = gen_item_bounds(b_t, n_heads_out, mCuSeqlens, src)
        mWorkItems[dst, 0] = batch_idx
        mWorkItems[dst, 1] = head_idx
        mWorkItems[dst, 2] = cutlass.Int32(0)
        mWorkItems[dst, 3] = batch_num_chunks
        mWorkItems[dst, 4] = cutlass.Int32(0)
        mWorkItems[dst, 5] = batch_num_chunks
        mWorkItems[dst, 6] = batch_start
        mWorkItems[dst, 7] = batch_end
    else:
        for f in cutlass.range_constexpr(WORK_ITEM_FIELDS):
            mWorkItems[dst, cutlass.Int32(f)] = mStaging[src, cutlass.Int32(f)]


@cute.jit
def order_body(
    gen: cutlass.Constexpr[bool],
    has_scheduler: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    n_threads: cutlass.Constexpr[int],
    order_elements: cutlass.Constexpr[int],
    tidx,
    n_heads_out: cutlass.Int32,
    n_tiles: cutlass.Int32,
    mCuSeqlens: cute.Tensor,
    mStaging: cute.Tensor | None,
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mScheduler: cute.Tensor | None,
    sKey,
    sIdx,
    sSpread,
):
    """Bitonic-sort the staged items by ``compute_end - compute_start``,
    longest first, into ``work_items``; with ``gen`` synthesize the uncut item
    per (batch, head) from ``cu_seqlens`` instead.  Thread 0 also zeroes the
    scheduler ticket rings.  Caller owns ``sKey``/``sIdx`` (``n_threads *
    order_elements`` Int32 each) and a 2-cell ``sSpread``.  CTA-wide barriers
    inside: every thread of the calling CTA must reach it."""
    capacity = cutlass.const_expr(n_threads * order_elements)

    # ---- zero the scheduler ticket rings ---------------------------------------------
    if cutlass.const_expr(has_scheduler):
        if tidx == 0:
            si = cutlass.Int32(0)
            while si < mScheduler.shape[0]:
                mScheduler[si] = cutlass.Int32(0)
                si = si + cutlass.Int32(1)

    if cutlass.const_expr(gen):
        n = n_tiles
        if tidx == 0:
            mCount[0] = n_tiles
    else:
        n = mCount[0]

    # ---- copy through unsorted when past SMEM sort capacity --------------------------
    if n > cutlass.Int32(capacity):
        i = tidx
        while i < n:
            write_item(gen, b_t, n_heads_out, mCuSeqlens, mStaging, mWorkItems, i, i)
            i = i + cutlass.Int32(n_threads)
    else:
        if tidx == 0:
            sSpread[0] = cutlass.Int32(2147483647)
            sSpread[1] = cutlass.Int32(-2147483648)

        b_pad = cutlass.Int32(1)
        while b_pad < n:
            b_pad = b_pad * cutlass.Int32(2)
        nvvm.barrier_cta_sync()
        kmin = cutlass.Int32(2147483647)
        kmax = cutlass.Int32(-2147483648)
        for e in cutlass.range_constexpr(order_elements):
            i = tidx + cutlass.Int32(e * n_threads)
            if i < n:
                if cutlass.const_expr(gen):
                    batch_idx, head_idx, batch_start, batch_end, batch_num_chunks = gen_item_bounds(b_t, n_heads_out, mCuSeqlens, i)
                    key = batch_num_chunks
                else:
                    key = mStaging[i, 5] - mStaging[i, 4]
                sKey[i] = key
                sIdx[i] = i
                kmin = kmin if kmin < key else key
                kmax = kmax if kmax > key else key
            elif i < b_pad:
                sKey[i] = cutlass.Int32(-2147483648)
                sIdx[i] = i

        nvvm.atomicrmw("min", sSpread.data_ptr(0), kmin, space=nvvm.SharedSpace.shared_cta)
        nvvm.atomicrmw("max", sSpread.data_ptr(1), kmax, space=nvvm.SharedSpace.shared_cta)
        nvvm.barrier_cta_sync()
        if sSpread[0] == sSpread[1]:
            # ---- every key equal and nothing to sort ---------------------------------
            i2 = tidx
            while i2 < n:
                write_item(gen, b_t, n_heads_out, mCuSeqlens, mStaging, mWorkItems, i2, i2)
                i2 = i2 + cutlass.Int32(n_threads)
        else:
            # ---- bitonic sort, descending: k = subsequence width, j = distance -------
            k = cutlass.Int32(2)
            while k <= b_pad:
                j = k // cutlass.Int32(2)
                while j > 0:
                    for e in cutlass.range_constexpr(order_elements):
                        i = tidx + cutlass.Int32(e * n_threads)
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

            # ---- write the table in sorted order -------------------------------------
            for e in cutlass.range_constexpr(order_elements):
                i = tidx + cutlass.Int32(e * n_threads)
                if i < n:
                    src = sIdx[i]
                    write_item(gen, b_t, n_heads_out, mCuSeqlens, mStaging, mWorkItems, i, src)


@cute.jit
def launch(
    split: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    scan_rows: cutlass.Constexpr[int],
    log_gate: cutlass.Constexpr[bool],
    safe_gate: cutlass.Constexpr[bool],
    gate_channels: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    has_scheduler: cutlass.Constexpr[bool],
    n_heads_out: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    gate_scale_log2: cutlass.Float32,
    mGate: cute.Tensor | None,
    mALog: cute.Tensor | None,
    mDtBias: cute.Tensor | None,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor | None,
    mStaging: cute.Tensor | None,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mScheduler: cute.Tensor | None,
    n_scan_ctas: cutlass.Int32,
    n_walk_ctas: cutlass.Int32,
    stream: cuda.CUstream,
) -> None:
    if cutlass.const_expr(split):
        frost_split_k_plan(
            b_t,
            overhead_chunks,
            n_heads_out,
            num_sms,
            n_tiles,
            ideal_chunks,
            batch_size,
            mCuSeqlens,
            mChunkVals,
            mCount,
        ).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)
        if cutlass.const_expr(gate_channels > 0):
            frost_split_k_scan_channel(
                b_t,
                scan_rows,
                log_gate,
                safe_gate,
                gate_channels,
                overhead_chunks,
                n_heads_out,
                num_sms,
                n_tiles,
                ideal_chunks,
                batch_size,
                gate_scale_log2,
                mGate,
                mALog,
                mDtBias,
                mCuSeqlens,
                mChunkVals,
                mCount,
            ).launch(
                grid=(n_scan_ctas, n_heads_out, 1),
                block=(SCAN_THREADS, 1, 1),
                stream=stream,
            )
        else:
            frost_split_k_scan_scalar(
                b_t,
                scan_rows,
                log_gate,
                safe_gate,
                overhead_chunks,
                n_heads_out,
                num_sms,
                n_tiles,
                ideal_chunks,
                batch_size,
                mGate,
                mALog,
                mDtBias,
                mCuSeqlens,
                mChunkVals,
                mCount,
            ).launch(
                grid=(n_scan_ctas, -(-n_heads_out // WARP_SIZE), 1),
                block=(SCAN_THREADS, 1, 1),
                stream=stream,
            )
        frost_split_k_walk(
            b_t,
            overhead_chunks,
            n_heads_out,
            num_sms,
            n_tiles,
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


compiled_cache = {}


class TableRecipe(NamedTuple):
    """Build-time facts of one split-table launch: everything static is
    settled once so :func:`run_table` can replay the call as a straight
    line.  Produced by :func:`build_split_table`."""

    compiled: object
    split: bool
    safe_gate: bool
    n_heads_out: int
    n_tiles: int
    num_sms: int
    ideal_chunks: int
    batch_size: int
    log2_threshold: float
    gate_scale_log2: float
    n_scan_ctas: int
    n_walk_ctas: int


def run_table(r, gate, a_log, dt_bias, cu_seqlens, chunk_scratch, item_scratch, work_items, work_count, scheduler_counter, stream) -> None:
    """The lowered split-table launch: no validation, no key build.  Only
    buffers move between calls; every scalar comes from the recipe."""
    r.compiled(
        r.n_tiles,
        r.ideal_chunks,
        r.batch_size,
        r.log2_threshold,
        r.gate_scale_log2,
        gate if r.split else None,
        a_log if r.safe_gate else None,
        dt_bias if r.safe_gate else None,
        cu_seqlens,
        chunk_scratch if r.split else None,
        item_scratch if r.split else None,
        work_items,
        work_count,
        scheduler_counter,
        r.n_scan_ctas,
        r.n_walk_ctas,
        cuda.CUstream(int(stream)),
    )


def build_split_table(
    gate,
    cu_seqlens,
    work_items,
    work_count,
    *,
    ideal_chunks=None,
    n_tiles,
    num_sms,
    b_t,
    chunk_scratch=None,
    item_scratch=None,
    log2_threshold=None,
    log_gate=False,
    safe_gate=False,
    a_log=None,
    dt_bias=None,
    gate_lower_bound=None,
    scheduler_counter=None,
    split=True,
    stream,
) -> "TableRecipe":
    """Fill ``work_items``/``work_count`` with the split-K partition of
    ``gate`` / ``cu_seqlens (B+1,) int32``.  A 2-D ``(total_tokens, HO)`` gate
    is the scalar GDN kind, 3-D ``(total_tokens, HO, DK)`` the per-key-channel
    KDA / GDN-2 kind; ``log_gate`` means natural-log decay rather than linear
    alpha.  With ``safe_gate`` the gate holds RAW logits and the scan applies
    the matching transform so cuts land on true decay: per-channel
    ``gate_lower_bound * sigmoid(exp(a_log) * (g + dt_bias))``
    (``gate_lower_bound`` required), scalar ``-exp(a_log[h]) * softplus(g +
    dt_bias[h])``.

    ``split=False`` is rejected: each main kernel's prologue synthesizes the
    no-cuts table itself (``order_gen``).

    ``work_items`` and ``item_scratch`` are ``(max_items, WORK_ITEM_FIELDS)``
    int32 with ``max_items >= max_work_items(...)``; ``work_count`` is
    ``(1,)`` int32; ``chunk_scratch`` is ``(>= chunk_scratch_rows(...), HO)``
    fp32, whose LAST row carries the plan rather than a chunk value.
    ``scheduler_counter`` cells are zeroed by the order phase, the count by the
    plan.  Runs entirely on device — no host synchronization."""
    if log2_threshold is None:
        log2_threshold = DEFAULT_LOG2_THRESHOLD
    if len(gate.shape) not in (2, 3):
        raise ValueError(f"gate must be (total_tokens, HO) or (total_tokens, HO, DK), got {tuple(gate.shape)}")
    gate_channels = gate.shape[2] if len(gate.shape) == 3 else 0
    if safe_gate and (a_log is None or dt_bias is None):
        raise ValueError("safe_gate requires a_log and dt_bias")
    if safe_gate and gate_channels > 0 and gate_lower_bound is None:
        raise ValueError("per-channel safe_gate requires gate_lower_bound")
    if not safe_gate:
        a_log = None
        dt_bias = None
    gate_scale_log2 = float(gate_lower_bound) * RCP_LN2 if safe_gate and gate_channels > 0 else 0.0
    n_heads_out = gate.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    if not split:
        raise ValueError("split=False has no host-side stage: each main kernel's prologue synthesizes the no-cuts table (order_gen)")
    if ideal_chunks is None or chunk_scratch is None or item_scratch is None:
        raise ValueError("split=True requires ideal_chunks, chunk_scratch, and item_scratch")
    if gate_channels and gate_channels % WARP_SIZE != 0:
        raise ValueError(f"per-channel gate dim must be a multiple of {WARP_SIZE}, got {gate_channels}")
    if gate_channels and gate_channels % 128 == 0 and data_ptr(gate) % 16 != 0:
        raise ValueError("per-channel gate base must be 16-byte aligned (vectorized scan loads)")
    n_walk_ctas = batch_size * n_heads_out
    need_rows = chunk_scratch_rows(gate.shape[0], batch_size, b_t)
    grid_y = n_heads_out if gate_channels > 0 else -(-n_heads_out // WARP_SIZE)
    scan_rows = scan_rows_per_warp(need_rows, grid_y, num_sms)
    n_scan_ctas = -(-need_rows // (SCAN_WARPS * scan_rows))
    if len(chunk_scratch.shape) != 2 or chunk_scratch.shape[0] < need_rows or chunk_scratch.shape[1] != n_heads_out:
        raise ValueError(f"chunk_scratch must be (>= {need_rows}, {n_heads_out}) fp32, got {tuple(chunk_scratch.shape)}")
    if tuple(item_scratch.shape) != tuple(work_items.shape) or work_items.shape[1] != WORK_ITEM_FIELDS:
        raise ValueError(
            f"item_scratch must match work_items (max_items, {WORK_ITEM_FIELDS}) int32, got {tuple(item_scratch.shape)} vs {tuple(work_items.shape)}"
        )
    overhead_chunks = max(1, OVERHEAD_TOKENS // b_t)
    cu_stream = cuda.CUstream(int(stream))

    key = (
        bool(split),
        b_t,
        scan_rows,
        n_heads_out,
        num_sms,
        bool(log_gate),
        bool(safe_gate),
        gate_channels,
        scheduler_counter is not None,
        str(cu_seqlens.dtype),
    )
    if key not in compiled_cache:

        dt_bias_c = None
        if safe_gate:
            dt_bias_c = from_dlpack(dt_bias, assumed_align=4)
            dt_bias_c.mark_compact_shape_dynamic(mode=0, stride_order=tuple(range(len(dt_bias.shape))), divisibility=1)
        chunk_scratch_c = None
        item_scratch_c = None
        if split:
            chunk_scratch_c = from_dlpack(chunk_scratch, assumed_align=4)
            chunk_scratch_c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            item_scratch_c = from_dlpack(item_scratch, assumed_align=4)
            item_scratch_c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_items_c = from_dlpack(work_items, assumed_align=4)
        work_items_c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_count_c = from_dlpack(work_count, assumed_align=4)
        work_count_c.mark_compact_shape_dynamic(mode=0, stride_order=(0,), divisibility=1)
        compiled_cache[key] = cute.compile(
            launch,
            bool(split),
            b_t,
            int(scan_rows),
            bool(log_gate),
            bool(safe_gate),
            gate_channels,
            overhead_chunks,
            scheduler_counter is not None,
            int(n_heads_out),
            int(num_sms),
            cutlass.Int32(n_tiles),
            cutlass.Int32(ideal_chunks),
            cutlass.Int32(batch_size),
            cutlass.Float32(log2_threshold),
            cutlass.Float32(gate_scale_log2),
            from_dlpack(gate, assumed_align=4).mark_layout_dynamic(leading_dim=len(gate.shape) - 1) if split else None,
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if safe_gate else None,
            dt_bias_c,
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(),
            chunk_scratch_c,
            item_scratch_c,
            work_items_c,
            work_count_c,
            from_dlpack(scheduler_counter, assumed_align=4).mark_layout_dynamic() if scheduler_counter is not None else None,
            cutlass.Int32(n_scan_ctas),
            cutlass.Int32(n_walk_ctas),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    compiled_cache[key](
        n_tiles,
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
        scheduler_counter,
        n_scan_ctas,
        n_walk_ctas,
        cu_stream,
    )
    return TableRecipe(
        compiled_cache[key],
        bool(split),
        bool(safe_gate),
        n_heads_out,
        n_tiles,
        num_sms,
        int(ideal_chunks),
        batch_size,
        float(log2_threshold),
        float(gate_scale_log2),
        n_scan_ctas,
        n_walk_ctas,
    )


frost_split_k_plan.set_name_prefix("cudnn", remove_cutlass_symbol=True)
frost_split_k_scan_channel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
frost_split_k_scan_scalar.set_name_prefix("cudnn", remove_cutlass_symbol=True)
frost_split_k_walk.set_name_prefix("cudnn", remove_cutlass_symbol=True)
