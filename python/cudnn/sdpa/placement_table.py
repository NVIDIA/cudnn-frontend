# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measured placement table: FROST row vs the backend, per shape class, generated from frozen sweeps.

A class is read off the graph facts (row, device, dtype, head pair, phase, layout, mask, sink, MHA vs GQA). Each entry
is a monotone rule on ONE work scalar, ``log2(h_q * b * s_q * s_kv)``: ``always``, ``never``, ``ge T`` (FROST leads
from T up) or ``le T`` (FROST leads up to T). Entries are produced by ``benchmark/placement_gen`` from a training grid
and qualified on a frozen holdout; every entry names its study. Do not edit entries by hand -- regenerate.

:func:`lookup` answers ``True`` (lead), ``False`` (trail) or ``None`` (no entry, or a feature no study measured); the
hand-written rules in ``sdpa/fwd/placement.py`` / ``sdpa/bwd/heuristics.py`` decide every ``None``.
"""

from __future__ import annotations

import math
from typing import Dict, NamedTuple, Optional, Tuple

import cudnn


class Entry(NamedTuple):
    direction: str  # "always" | "never" | "ge" | "le"
    log2_work: Optional[float]  # the threshold for "ge" / "le"
    study: str  # the frozen study that produced it (grid + holdout hashes live in its record)


def _dtype_name(dtype) -> Optional[str]:
    if dtype == cudnn.data_type.BFLOAT16:
        return "bf16"
    if dtype == cudnn.data_type.HALF:
        return "f16"
    return None


def _unmeasured(facts) -> bool:
    """Features no study sweeps: such a graph keeps the hand-written rules."""
    return bool(
        getattr(facts, "is_fp8", False)
        or getattr(facts, "is_mxfp8", False)
        or facts.has_bias
        or facts.has_dropout
        or getattr(facts, "has_score_mod", False)
        or getattr(facts, "has_alibi", False)
        or getattr(facts, "has_block_mask", False)
        or facts.right_band_widening
        or getattr(facts, "deterministic", False)
        or getattr(facts, "has_dbias", False)
        or getattr(facts, "o_block_scale", False)
        or getattr(facts, "shape_overrides", False)
    )


def class_key(row: str, facts) -> Optional[Tuple]:
    dtype = _dtype_name(facts.dtype)
    if dtype is None or _unmeasured(facts) or not facts.h_kv or facts.h_q % facts.h_kv:
        return None
    if facts.s_q == 1:
        phase = "q1"
    elif facts.s_q <= 4:
        phase = "mtp"
    else:
        phase = "prefill"
    if facts.has_paged_kv:
        layout = "paged"
    elif facts.thd:
        total = facts.max_total_seq_len_q
        layout = "thd_ragged" if total is not None and total < facts.b * facts.s_q else "thd"
    else:
        layout = "dense"
    if facts.window_left is not None:
        mask = "window"
    elif facts.causal and facts.s_q > 1:  # one query row: causal masks nothing
        mask = "causal_br" if facts.bottom_right else "causal_tl"
    else:
        mask = "none"
    group = "mha" if facts.h_q == facts.h_kv else "gqa"
    return (row, tuple(facts.device_cc or ()), dtype, (facts.d_qk, facts.d_v), phase, layout, mask, bool(facts.has_sink), group)


def log2_work(facts) -> float:
    return math.log2(max(1, facts.h_q * facts.b * facts.s_q * facts.s_kv))


def lookup(row: str, facts) -> Optional[bool]:
    key = class_key(row, facts)
    entry = TABLE.get(key) if key is not None else None
    if entry is None:
        return None
    if entry.direction in ("always", "never"):
        return entry.direction == "always"
    w = log2_work(facts)
    return w >= entry.log2_work if entry.direction == "ge" else w <= entry.log2_work


# ---------------------------------------------------------------------------------------------------------------- generated
# benchmark/placement_gen/gen.py --emit; one block per study, keys sorted. Do not edit by hand.
TABLE: Dict[Tuple, Entry] = {
    # study sm100_d576_decode_20261027 (full-power B200, cuDNN 9.28.0.38; frozen 128-case grid, 40-case holdout)
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (576, 512), "mtp", "paged", "causal_br", False, "gqa"): Entry("always", None, "sm100_d576_decode_20261027"),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (576, 512), "q1", "paged", "none", False, "gqa"): Entry("ge", 19.5, "sm100_d576_decode_20261027"),
}
