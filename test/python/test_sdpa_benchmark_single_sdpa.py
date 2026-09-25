# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from typing import Optional

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from benchmark.attention_training.flops import count_causal_nonmasked_elems  # noqa: E402

pytestmark = pytest.mark.L0


def _reference_causal_nonmasked_elems(q_seqlen: int, kv_seqlen: int, attn_mask: str, sliding_window_size: Optional[int]):
    diagonal_offset = kv_seqlen - q_seqlen if attn_mask == "bottom_right" else 0
    total = 0

    for q_idx in range(q_seqlen):
        for kv_idx in range(kv_seqlen):
            distance_from_diagonal = kv_idx - (q_idx + diagonal_offset)
            if distance_from_diagonal > 0:
                continue
            if sliding_window_size is not None and distance_from_diagonal <= -sliding_window_size:
                continue
            total += 1

    return total


def test_count_causal_nonmasked_elems_square_top_left():
    assert count_causal_nonmasked_elems(4, 4, "top_left") == 10


def test_count_causal_nonmasked_elems_rectangular_top_left():
    assert count_causal_nonmasked_elems(5, 3, "top_left") == 12
    assert count_causal_nonmasked_elems(3, 6, "top_left") == 6


def test_count_causal_nonmasked_elems_rectangular_bottom_right():
    assert count_causal_nonmasked_elems(5, 3, "bottom_right") == 6
    assert count_causal_nonmasked_elems(3, 6, "bottom_right") == 15


def test_count_causal_nonmasked_elems_sliding_window():
    assert count_causal_nonmasked_elems(6, 6, "top_left", sliding_window_size=3) == 15
    assert count_causal_nonmasked_elems(5, 3, "bottom_right", sliding_window_size=2) == 5


def test_count_causal_nonmasked_elems_matches_reference():
    cases = [
        (4, 4, "top_left", None),
        (5, 3, "top_left", None),
        (3, 6, "top_left", 2),
        (4, 4, "bottom_right", None),
        (5, 3, "bottom_right", 2),
        (3, 6, "bottom_right", 3),
    ]

    for q_seqlen, kv_seqlen, attn_mask, sliding_window_size in cases:
        assert count_causal_nonmasked_elems(
            q_seqlen=q_seqlen,
            kv_seqlen=kv_seqlen,
            attn_mask=attn_mask,
            sliding_window_size=sliding_window_size,
        ) == _reference_causal_nonmasked_elems(
            q_seqlen=q_seqlen,
            kv_seqlen=kv_seqlen,
            attn_mask=attn_mask,
            sliding_window_size=sliding_window_size,
        )
