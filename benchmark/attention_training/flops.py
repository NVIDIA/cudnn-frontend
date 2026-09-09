# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for calculating attention operation counts."""

from typing import Optional


def count_causal_nonmasked_elems(
    q_seqlen: int,
    kv_seqlen: int,
    attn_mask: str,
    sliding_window_size: Optional[int] = None,
) -> int:
    """Count unmasked (query, key/value) pairs for a causal SDPA mask.

    Args:
        q_seqlen: Query sequence length.
        kv_seqlen: Key/value sequence length.
        attn_mask: ``top_left`` or ``bottom_right`` causal alignment.
        sliding_window_size: Optional sliding window size.
    """
    if attn_mask not in ("top_left", "bottom_right"):
        raise ValueError(f"Unsupported causal attn mask for counting: {attn_mask}")

    diagonal_offset = kv_seqlen - q_seqlen if attn_mask == "bottom_right" else 0
    total = 0

    for q_idx in range(q_seqlen):
        row_end = min(kv_seqlen - 1, q_idx + diagonal_offset)
        if row_end < 0:
            continue

        if sliding_window_size is None:
            row_start = 0
        elif attn_mask == "top_left":
            row_start = max(0, q_idx - sliding_window_size + 1)
        else:
            # For bottom-right alignment, anchor the window to the causal diagonal.
            row_start = max(0, row_end - sliding_window_size + 1)

        if row_start <= row_end:
            total += row_end - row_start + 1

    return total
