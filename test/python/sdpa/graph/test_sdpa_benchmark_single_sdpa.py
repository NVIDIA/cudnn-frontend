# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from typing import Optional

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
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


def _backward_graph_call_kwargs(src: str):
    """``{method: {keyword, ...}}`` for every ``<graph>.sdpa*backward(...)`` call the harness builds."""
    import ast

    out = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.startswith("sdpa") and node.func.attr.endswith("backward"):
            out.setdefault(node.func.attr, set()).update(k.arg for k in node.keywords if k.arg)
    return out


def test_backward_graphs_carry_the_sliding_window_the_flop_model_counts():
    """A ``--sliding_window_size`` run must be rated on the graph that RAN.  The half and fp8 backward calls carry the band
    (the fp8 one as ``left_bound`` alone: the binding folds ``right_bound = 0`` out of the causal flags and raises when both
    are given), and the backward FLOP count goes through ``bwd_sliding_window`` -- None for the mxfp8 backward, whose call
    carries no band.  Until 2026-09-28 the fp8 backward built a PLAIN-causal graph under a window while the FLOP model counted
    window pairs: the sm107 d256 fp8 SWA640 cell read 148 TFLOPS on 11.14 ms, its main kernel 3.1x the windowed one's time."""
    src = (_REPO_ROOT / "benchmark" / "attention_training" / "benchmark_single_sdpa.py").read_text()
    kw = _backward_graph_call_kwargs(src)
    assert "diagonal_band_left_bound" in kw["sdpa_backward"], "the half backward call dropped the band"
    assert "left_bound" in kw["sdpa_fp8_backward"], "the fp8 backward call dropped the band (a window run measures a plain-causal graph)"
    assert "right_bound" not in kw["sdpa_fp8_backward"], "sdpa_fp8_backward raises when use_causal_mask and right_bound are both set"
    assert "left_bound" not in kw["sdpa_mxfp8_backward"], "the mxfp8 backward carries the band now: drop the mxfp8 arm of bwd_sliding_window"
    assert 'bwd_sliding_window = None if args.data_type == "mxfp8" else args.sliding_window_size' in src
    assert src.count("bwd_sliding_window,") == 1, "the backward tflops_per_sec call must take bwd_sliding_window, not args.sliding_window_size"
