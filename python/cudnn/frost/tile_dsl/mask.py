# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.cute.arch.nvvm_wrappers import inline_ptx

from .constants import MASK_CAUSAL, MASK_NONE, MASK_PADDED, MASK_SWA  # noqa: F401

_NEG_INF_BITS = -3.4028235e38


def apply_mask_chunk(
    reg_S,
    q_abs,
    kv_col_base,
    seq_kv_len,
    window_left: int,
    mask_flags: int,
    N: int = 64,
    bottom_right: int = 0,
    causal_diag=None,
    mask_value: float = _NEG_INF_BITS,
    window_right: int = 0,
):
    # mask_value: what a masked score becomes.  Default is the legacy finite
    # sentinel; the f16 prefill kernels pass float("-inf") so a fully-masked
    # row's max stays -inf under any scale and the canonical
    # `max == -inf -> substitute 0` guard (row_max_for_exp2) applies.
    if cutlass.const_expr(mask_flags == MASK_NONE):
        return reg_S

    neg_inf = cutlass.Float32(mask_value)
    # The whole band shifts with the diagonal: under BOTTOM_RIGHT the SWA
    # lower limit is q + (S_kv - S_q) - W — the same causal_diag offset the
    # upper (causal) limit uses below. Top-left keeps the plain q - W.
    q_minus_w = None
    if mask_flags & MASK_SWA:
        swa_base = (q_abs + causal_diag) if bottom_right else q_abs
        q_minus_w = swa_base - cutlass.Int32(window_left)
    # window_right is the compile-time diagonal-band right bound (cuDNN
    # diagonal_band_right_bound): kv columns up to q + window_right (plus the
    # bottom-right diagonal offset) stay unmasked. 0 = plain causal.
    if cutlass.const_expr((mask_flags & MASK_CAUSAL) and bottom_right):
        q_caus_lim = q_abs + causal_diag
    else:
        q_caus_lim = q_abs
    if cutlass.const_expr((mask_flags & MASK_CAUSAL) and window_right != 0):
        q_caus_lim = q_caus_lim + cutlass.Int32(window_right)

    elems = []
    for i in range(N):
        kv_abs = kv_col_base + cutlass.Int32(i)
        masked = None
        if cutlass.const_expr(mask_flags & MASK_PADDED):
            term = kv_abs >= seq_kv_len
            masked = term if masked is None else (masked | term)
        if cutlass.const_expr(mask_flags & MASK_CAUSAL):
            term = kv_abs > q_caus_lim
            masked = term if masked is None else (masked | term)
        if cutlass.const_expr(mask_flags & MASK_SWA):
            term = kv_abs < q_minus_w
            masked = term if masked is None else (masked | term)
        val = cutlass.Float32(
            arith.select(
                masked.ir_value(),
                neg_inf.ir_value(),
                reg_S[i].ir_value(),
            )
        )
        elems.append(val)
    return cutlass.Vector.from_elements(tuple(elems), cutlass.Float32)


# ---------------------------------------------------------------------------
# Per-cell mask, "bits" form: one keep-word per 32 columns, register-to-predicate
# ---------------------------------------------------------------------------
#
# `apply_mask_chunk` above compares EVERY cell against every active bound and
# ORs the terms: per cell one IADD (kv_col_base + i, never folded into the
# compare), one ISETP per term and one FSEL per term (the i1 OR lowers to nested
# selects), so a causal+SWA tile costs 5 instructions per cell and a padded
# causal+SWA one 7 -- 51-72 % of a masked softmax tile's instructions, serialized
# in front of the exp burst (sm_107a listings, 2026-09-22: a masked KV tile ran
# 805-1602 instructions per lane against 358-451 for a dense one).  The information
# content of any of these masks is one or two band EDGES per row, so the form
# below spends its instructions there instead:
#
#   1. per 32-column word, build the KEEP mask from the band edges with two
#      saturating shifts (`shr.u32` for the upper edge, `shl.b32` for the lower;
#      PTX clamps a shift amount >= 32 to a zero result, which is exactly the
#      "edge is outside this word" case) -- ~7 integer ops per word, per lane;
#   2. per cell, test one bit of the word and feed the SAME `arith.select`
#      `apply_mask_chunk` uses.  ptxas turns 32 consecutive bit tests into
#      `R2P` (register -> 7 predicates) + one `FSEL` per cell, so the per-cell
#      cost is 1.4-1.6 instructions regardless of how many mask terms are on
#      (sm_107a / sm_100a micro-kernel listings, 2026-09-22, against 3.0-7.6 for
#      the per-cell form; the same idiom the block-sparse-attention SM90 backward
#      uses, `predicate_bitmask_below`).
#
# Semantics contract (shared with `apply_mask_chunk`, bitwise):
#   - a masked cell becomes `mask_value` (default: the finite fp32-min sentinel;
#     pass float("-inf") for the true -inf form); an unmasked cell is passed
#     through untouched (NaN payloads included);
#   - bounds are PER LANE (one lane = one q row): `lo`/`hi` are Int32 values
#     derived from that lane's q row, `kv_col_base` is the chunk's first
#     absolute kv column; a fully-masked row yields `mask_value` in every cell,
#     so the caller's `row_max_for_exp2` guard applies exactly as today;
#   - a `tcgen05.ld 32x32b` chunk is contiguous along columns per lane (register
#     k = column k), so NO column remap is needed here -- unlike the WGMMA
#     accumulator layout the BSA kernel remaps with `sm90_col_to_predicate_idx`.

MASK_FORM_CELLS = "cells"  # apply_mask_chunk: per-cell compare + select
MASK_FORM_BITS = "bits"  # apply_mask_chunk_bits: keep-word + register-to-predicate select
MASK_FORMS = (MASK_FORM_CELLS, MASK_FORM_BITS)

MASK_WORD_COLS = 32  # columns per keep-word = bits per register

# The band arithmetic (`lo - kv_col_base`, `hi - kv_col_base`, the per-word shift count) is Int32.
# With every absolute row / column index below 2**28 (a TMA coordinate keeps S far under that) a
# window bound below this limit cannot wrap it; a bound at or past it can, and a wrapped `lo` makes
# the bits form mask EVERYTHING where `apply_mask_chunk` masks nothing.  Enforced at trace time on
# the Python-int bounds (no instruction); pinned by `test_bits_form_domain_guard`.
MASK_BOUND_LIMIT = 1 << 30


def keep_below_word(hi_rel, s: int):
    """Keep-word of 32-column word ``s``: bit ``i`` is set iff column ``32 s + i < hi_rel``.

    ``hi_rel`` is the exclusive upper bound RELATIVE to the chunk base (an Int32, per lane;
    any value, including negative or past the chunk).  Spelled in PTX because the
    saturating shift is the whole point: LLVM treats a shift by >= 32 as poison, PTX
    ``shr.u32`` clamps it and yields 0 (= every column of this word is at or past ``hi``).
    The ``max.s32`` clamps the other side (``hi`` past the word -> shift 0 -> all kept)."""
    n_masked = cutlass.Int32(MASK_WORD_COLS * (s + 1)) - hi_rel
    return inline_ptx(
        "{ .reg .b32 n, ones; mov.b32 ones, 0xFFFFFFFF; max.s32 n, {$r0}, 0; shr.u32 {$w0}, ones, n; }",
        write_only_types=[cutlass.Uint32],
        read_only_args=[n_masked],
    )


def keep_from_word(lo_rel, s: int):
    """Keep-word of 32-column word ``s``: bit ``i`` is set iff column ``32 s + i >= lo_rel``.

    The lower-edge twin of :func:`keep_below_word`: ``shl.b32`` of all-ones by the number
    of columns below ``lo`` in this word, clamped at 0 (``lo`` left of the word -> all kept)
    and saturating at >= 32 (``lo`` right of the word -> 0, all masked)."""
    n_masked = lo_rel - cutlass.Int32(MASK_WORD_COLS * s)
    return inline_ptx(
        "{ .reg .b32 n, ones; mov.b32 ones, 0xFFFFFFFF; max.s32 n, {$r0}, 0; shl.b32 {$w0}, ones, n; }",
        write_only_types=[cutlass.Uint32],
        read_only_args=[n_masked],
    )


def band_mask_words(lo, hi, kv_col_base, n_cols: int):
    """Per-lane keep-words for the ``n_cols`` columns starting at absolute kv column
    ``kv_col_base``: bit ``i`` of word ``s`` is set iff column ``c = kv_col_base + 32 s + i``
    satisfies ``keep(c) = (lo is None or c >= lo) & (hi is None or c < hi)``.

    ``lo`` / ``hi`` are per-lane Int32 absolute column bounds (``hi`` exclusive), or ``None``
    for a side no mask term defines -- that side folds out at trace time (one shift per word
    for a one-sided mask, two for a band), never an INT_MIN/INT_MAX sentinel, so ``lo - base``
    cannot wrap as long as the caller keeps every bound within ``MASK_BOUND_LIMIT`` of the chunk
    (:func:`apply_mask_chunk_bits` refuses a wider window).  Returns a tuple of
    ``ceil(n_cols / 32)`` ``Uint32`` words.  Trace-time helper (plain Python over traced
    values), like :func:`apply_mask_chunk`."""
    if lo is None and hi is None:
        raise ValueError("band_mask_words: at least one of lo / hi must be given (a mask with no edge is no mask)")
    n_words = (n_cols + MASK_WORD_COLS - 1) // MASK_WORD_COLS
    hi_rel = None if hi is None else hi - kv_col_base
    lo_rel = None if lo is None else lo - kv_col_base
    words = []
    for s in range(n_words):
        word = None
        if hi_rel is not None:
            word = keep_below_word(hi_rel, s)
        if lo_rel is not None:
            above = keep_from_word(lo_rel, s)
            word = above if word is None else (word & above)
        words.append(word)
    return tuple(words)


def apply_mask_words(reg_S, words, mask_value: float = _NEG_INF_BITS, n_cols: int = None):
    """``reg_S[c] if keep-bit(c) else mask_value`` over an ``n_cols``-wide register chunk.

    ``words`` is the tuple :func:`band_mask_words` returns (bit ``i`` of word ``s`` = column
    ``32 s + i`` is KEPT).  The inner loop is a Python ``range`` over a compile-time bit
    index, which is what lets ptxas see 32 consecutive single-bit tests of one register and
    emit ``R2P`` + one ``FSEL`` per cell.  Same ``arith.select`` as :func:`apply_mask_chunk`
    (only the predicate derivation differs), so the two forms are bitwise identical for the
    same mask set, for either sentinel, NaN payloads included."""
    if n_cols is None:
        n_cols = MASK_WORD_COLS * len(words)
    if len(words) * MASK_WORD_COLS < n_cols:
        raise ValueError(f"apply_mask_words: {len(words)} word(s) cover {len(words) * MASK_WORD_COLS} columns, chunk has {n_cols}")
    neg_inf = cutlass.Float32(mask_value)
    elems = []
    for s, word in enumerate(words):
        for i in range(MASK_WORD_COLS):
            c = MASK_WORD_COLS * s + i
            if c >= n_cols:
                break
            keep = cutlass.Boolean(word & cutlass.Uint32(1 << i))
            elems.append(cutlass.Float32(arith.select(keep.ir_value(), reg_S[c].ir_value(), neg_inf.ir_value())))
    return cutlass.Vector.from_elements(tuple(elems), cutlass.Float32)


def apply_mask_chunk_bits(
    reg_S,
    q_abs,
    kv_col_base,
    seq_kv_len,
    window_left: int,
    mask_flags: int,
    N: int = 64,
    bottom_right: int = 0,
    causal_diag=None,
    mask_value: float = _NEG_INF_BITS,
    window_right: int = 0,
):
    """:func:`apply_mask_chunk` with the same signature and the same masked set, in the
    "bits" form.  The three terms map onto one band ``[lo, hi)`` per lane:

    - CAUSAL masks ``kv > q_caus_lim``  ->  ``hi = q_caus_lim + 1`` with the same
      ``q_caus_lim = q_abs (+ causal_diag under bottom_right) (+ window_right)``;
    - PADDED masks ``kv >= seq_kv_len``  ->  ``hi = min(hi, seq_kv_len)``;
    - SWA masks ``kv < q_minus_w``  ->  ``lo = q_minus_w`` (the same bottom-right anchor).

    A side no term defines is ``None`` and folds out.  Fully-masked rows, the sentinel and
    the caller's tile-level trimming are exactly as for :func:`apply_mask_chunk`.

    Domain: a compile-time ``window_left`` / ``window_right`` at or past ``MASK_BOUND_LIMIT``
    raises at trace time (see the constant); the per-cell form has no such limit because it
    never subtracts the chunk base."""
    if cutlass.const_expr(mask_flags == MASK_NONE):
        return reg_S

    if (mask_flags & MASK_SWA) and isinstance(window_left, int) and window_left >= MASK_BOUND_LIMIT:
        raise ValueError(f"apply_mask_chunk_bits: window_left must be < {MASK_BOUND_LIMIT} (got {window_left}); the Int32 band arithmetic would wrap")
    if (mask_flags & MASK_CAUSAL) and isinstance(window_right, int) and window_right >= MASK_BOUND_LIMIT:
        raise ValueError(f"apply_mask_chunk_bits: window_right must be < {MASK_BOUND_LIMIT} (got {window_right}); the Int32 band arithmetic would wrap")

    lo = None
    hi = None
    if mask_flags & MASK_SWA:
        swa_base = (q_abs + causal_diag) if bottom_right else q_abs
        lo = swa_base - cutlass.Int32(window_left)
    if mask_flags & MASK_CAUSAL:
        q_caus_lim = (q_abs + causal_diag) if bottom_right else q_abs
        if window_right != 0:
            q_caus_lim = q_caus_lim + cutlass.Int32(window_right)
        hi = q_caus_lim + cutlass.Int32(1)
    if mask_flags & MASK_PADDED:
        hi = seq_kv_len if hi is None else cute.math.min(hi, seq_kv_len)

    words = band_mask_words(lo, hi, kv_col_base, N)
    return apply_mask_words(reg_S, words, mask_value=mask_value, n_cols=N)


def apply_mask_chunk_form(
    reg_S,
    q_abs,
    kv_col_base,
    seq_kv_len,
    window_left: int,
    mask_flags: int,
    N: int = 64,
    bottom_right: int = 0,
    causal_diag=None,
    mask_value: float = _NEG_INF_BITS,
    window_right: int = 0,
    *,
    form: str,
):
    """:func:`apply_mask_chunk` (``form=MASK_FORM_CELLS``) or :func:`apply_mask_chunk_bits`
    (``form=MASK_FORM_BITS``) behind one signature.  A kernel picks the form with ONE
    module-level constant (``MASK_FORM``) that every masked call site passes -- the
    ``DESC_VERSION`` discipline -- so an A/B of the two lowerings is a constant flip and a
    test can count call sites against the constant.  ``form`` is keyword-only and has no
    default: a call site that forgets it fails at trace time instead of silently picking one."""
    kw = dict(N=N, bottom_right=bottom_right, causal_diag=causal_diag, mask_value=mask_value, window_right=window_right)
    if form == MASK_FORM_BITS:
        return apply_mask_chunk_bits(reg_S, q_abs, kv_col_base, seq_kv_len, window_left, mask_flags, **kw)
    if form == MASK_FORM_CELLS:
        return apply_mask_chunk(reg_S, q_abs, kv_col_base, seq_kv_len, window_left, mask_flags, **kw)
    raise ValueError(f"apply_mask_chunk_form: form must be one of {MASK_FORMS}, got {form!r}")


# ---------------------------------------------------------------------------
# Tile-level mask bounds: which kv TILES a q tile has to visit at all.
# ---------------------------------------------------------------------------
#
# `apply_mask_chunk` above is the per-CELL mask. This is the per-TILE one, and
# it is where the work actually gets saved: under a causal band a q tile only
# intersects the kv tiles up to its own diagonal, so the whole upper triangle of
# tiles is never issued -- roughly half the MMAs at causal. The returned range
# also splits into a middle sub-range `[unmasked_lo, unmasked_hi)` where no cell
# can be masked, so only the diagonal (and padding-tail) tiles pay for
# `apply_mask_chunk` at all.
#
# Lives here rather than beside one pass's kernels because both the forward and
# the backward need exactly this arithmetic, and it is pure: it reads mask
# parameters and tile geometry, nothing pass-specific.


class KvLoopBounds(NamedTuple):
    left: object
    unmasked_lo: object
    unmasked_hi: object
    right: object


def _div_up(a, b):
    return (a + cutlass.Int32(b - 1)) // cutlass.Int32(b)


def swa_kv_lo_tile(anchor_row, window_left: int, tile_n: int):
    """First KV tile a left window of ``window_left`` keeps for ``anchor_row``
    (the q row plus the bottom-right diagonal); 0 while the window still
    reaches kv 0.  The forward's KV-loop lower bound, and the backward's
    deterministic dQ relay -- a kv-tile's turn on a q-tile is its rank among
    that q-tile's visitors, which start here.
    """
    cond = anchor_row > cutlass.Int32(window_left)
    delta = anchor_row - cutlass.Int32(window_left)
    return cutlass.Int32(
        arith.select(
            cond.ir_value(),
            (delta // cutlass.Int32(tile_n)).ir_value(),
            cutlass.Int32(0).ir_value(),
        )
    )


def compute_kv_loop_bounds(
    q_row_coord,
    seqlen_q,
    seq_kv_len,
    window_left: int,
    mask_flags: int,
    tile_n: int,
    cga_tile_m: int,
    bottom_right: bool = False,
    window_right: int = 0,
) -> KvLoopBounds:
    # window_right: compile-time diagonal-band right bound (cuDNN
    # diagonal_band_right_bound) — the causal upper limit is widened by
    # window_right columns. 0 = plain causal; folds out entirely.
    left = cutlass.Int32(0)
    right = _div_up(seq_kv_len, tile_n)

    if cutlass.const_expr(bottom_right):
        causal_diag = seq_kv_len - seqlen_q
    else:
        causal_diag = cutlass.Int32(0)

    if cutlass.const_expr(mask_flags & MASK_CAUSAL):
        kv_hi_caus = _div_up(q_row_coord + cutlass.Int32(cga_tile_m + window_right) + causal_diag, tile_n)
        right = cute.math.min(right, kv_hi_caus)

    if cutlass.const_expr(mask_flags & MASK_SWA):
        # The whole band shifts with the diagonal: under BOTTOM_RIGHT the SWA
        # lower bound is q + (S_kv - S_q) - W, same anchor the causal upper
        # bound uses (causal_diag folds to 0 for top-left).
        left = cute.math.max(left, swa_kv_lo_tile(q_row_coord + causal_diag, window_left, tile_n))

    unmasked_hi = right
    if cutlass.const_expr(mask_flags & MASK_PADDED):
        unaligned = (seq_kv_len % cutlass.Int32(tile_n)) != cutlass.Int32(0)
        lo_pad = cutlass.Int32(
            arith.select(
                unaligned.ir_value(),
                (right - cutlass.Int32(1)).ir_value(),
                right.ir_value(),
            )
        )
        unmasked_hi = cute.math.min(unmasked_hi, lo_pad)
    if cutlass.const_expr(mask_flags & MASK_CAUSAL):
        lo_caus = (q_row_coord + cutlass.Int32(window_right) + causal_diag) // cutlass.Int32(tile_n)
        unmasked_hi = cute.math.min(unmasked_hi, lo_caus)
    unmasked_hi = cute.math.max(unmasked_hi, left)

    unmasked_lo = left
    if cutlass.const_expr(mask_flags & MASK_SWA):
        anchor = q_row_coord + causal_diag + cutlass.Int32(cga_tile_m - 1 - window_left)
        swa_unmasked_lo = _div_up(anchor, tile_n)
        cond = anchor > cutlass.Int32(0)
        swa_unmasked_lo = cutlass.Int32(
            arith.select(
                cond.ir_value(),
                swa_unmasked_lo.ir_value(),
                cutlass.Int32(0).ir_value(),
            )
        )
        unmasked_lo = cute.math.max(unmasked_lo, swa_unmasked_lo)

    unmasked_lo = cute.math.min(unmasked_lo, unmasked_hi)

    return KvLoopBounds(
        left=left,
        unmasked_lo=unmasked_lo,
        unmasked_hi=unmasked_hi,
        right=right,
    )


# The KV-major dual: a backward CTA owns one KV tile and loops over Q tiles, so
# it needs the range of q tiles that attend its kv rows.  Causal trims from
# BELOW (q rows above the diagonal never see this kv tile), the sliding window
# trims from ABOVE (rows more than window_left past the tile never see it).
# Same inputs and conventions as compute_kv_loop_bounds; the causal diagonal is
# derived the same way, and the right-band widening may be a runtime value
# (the backward keeps it dynamic).


class QLoopBounds(NamedTuple):
    lo: object  # first q tile attending the kv tile (inclusive)
    hi: object  # one past the last; hi <= lo means the kv tile has no work


def compute_q_loop_bounds(
    kv_row_coord,
    seqlen_q,
    seq_kv_len,
    n_q_tiles,
    window_left: int,
    mask_flags: int,
    tile_q: int,
    tile_kv: int,
    bottom_right: bool = False,
    window_right=0,
) -> QLoopBounds:
    if cutlass.const_expr(bottom_right):
        causal_diag = seq_kv_len - seqlen_q
    else:
        causal_diag = cutlass.Int32(0)

    lo = cutlass.Int32(0)
    if cutlass.const_expr(mask_flags & MASK_CAUSAL):
        # kv rows [kv_row_coord, +tile_kv) are attended by q whenever
        # kv <= q + causal_diag + window_right: the first such q is the tile's
        # first row minus the diagonal minus the band widening (the widening
        # is subtracted so the straddling rows of the previous tile keep their
        # dQ and this tile keeps their dK/dV contribution).
        q_lo_abs = kv_row_coord - causal_diag - window_right
        lo = cute.math.max(q_lo_abs // cutlass.Int32(tile_q), cutlass.Int32(0))

    hi = n_q_tiles
    if cutlass.const_expr(mask_flags & MASK_SWA):
        # The window keeps kv >= q + causal_diag - window_left, so no q at or
        # past kv_row_coord + tile_kv + window_left - causal_diag attends the tile.
        q_hi_abs = kv_row_coord + cutlass.Int32(tile_kv + window_left) - causal_diag
        q_hi_abs = cute.math.max(q_hi_abs, cutlass.Int32(0))
        hi = cute.math.min(_div_up(q_hi_abs, tile_q), n_q_tiles)

    return QLoopBounds(lo=lo, hi=hi)
