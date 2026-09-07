# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith

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
        swa_base = q_row_coord + causal_diag
        cond = swa_base > cutlass.Int32(window_left)
        delta = swa_base - cutlass.Int32(window_left)
        kv_lo_swa = cutlass.Int32(
            arith.select(
                cond.ir_value(),
                (delta // cutlass.Int32(tile_n)).ir_value(),
                cutlass.Int32(0).ir_value(),
            )
        )
        left = cute.math.max(left, kv_lo_swa)

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
