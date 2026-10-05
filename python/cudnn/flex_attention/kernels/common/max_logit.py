# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device helpers for the per-query-head maximum attention logit."""

import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32


@cute.jit
def init_max_logit(
    sMaxLogit: cute.Tensor,
    tidx: Int32,
) -> None:
    """Initialize one CTA-local maximum for each query head in the tile."""

    if tidx < cute.size(sMaxLogit):
        sMaxLogit[tidx] = -Float32.inf


@cute.jit
def update_max_logit(
    sMaxLogit: cute.Tensor,
    row_max: Float32,
    row_is_valid: Boolean,
    head_offset: Int32,
    softmax_scale: Float32,
) -> None:
    """Accumulate one valid row maximum into CTA-local shared memory."""

    row_has_visible_score = row_max != -Float32.inf and row_max == row_max
    if row_is_valid and row_has_visible_score:
        value = Float32(0.0) if softmax_scale == Float32(0.0) else row_max * softmax_scale
        cute.arch.atomic_fmax(
            sMaxLogit.iterator + head_offset,
            value,
            sem="relaxed",
            scope="cta",
        )


@cute.jit
def store_max_logit(
    sMaxLogit: cute.Tensor,
    mMaxLogit: cute.Tensor,
    tidx: Int32,
    head_begin: Int32,
) -> None:
    """Commit one CTA-local maximum per query head to the device result."""

    if tidx < cute.size(sMaxLogit):
        value = sMaxLogit[tidx]
        if value != -Float32.inf:
            cute.arch.atomic_fmax(
                mMaxLogit.iterator + head_begin + tidx,
                value,
                sem="relaxed",
                scope="gpu",
            )


__all__ = [
    "init_max_logit",
    "store_max_logit",
    "update_max_logit",
]
