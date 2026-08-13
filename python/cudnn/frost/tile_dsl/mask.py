# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import cutlass
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
