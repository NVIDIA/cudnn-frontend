# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measured placement table: FROST row vs the backend, per shape class, generated from frozen sweeps.

A class is read off the graph facts (row, device, dtype, head pair, phase, layout, mask, sink, MHA vs GQA, and whether
S_kv is off the 128-key tile). Each entry
is a monotone rule on ONE scalar from :data:`FEATURES` -- the roofline-shaped cost proxy :func:`log2_work` or the
parallel-tile count :func:`log2_tiles`, whichever the class's data is monotone in: ``always``, ``never``, ``ge T`` (FROST leads
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
    log2_work: Optional[float]  # the threshold for "ge" / "le", in the entry's feature
    study: str  # the frozen study that produced it (grid + holdout hashes live in its record)
    feature: str = "cost"  # which scalar the threshold reads: a key of FEATURES


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
    elif facts.causal and not (facts.s_q == 1 and facts.bottom_right):  # one bottom-right row sees every key: no mask
        mask = "causal_br" if facts.bottom_right else "causal_tl"
    else:
        mask = "none"
    group = "mha" if facts.h_q == facts.h_kv else "gqa"
    # Off the 128-key tile dense KV cannot split and paged KV measures differently near the boundary: its own class, which
    # an on-tile study never answers.
    kv_off_tile = facts.s_kv % 128 != 0
    return (row, tuple(facts.device_cc or ()), dtype, (facts.d_qk, facts.d_v), phase, layout, mask, bool(facts.has_sink), group, kv_off_tile)


COST_BALANCE = 16.0
"""FLOPs per byte at which the cost proxy switches from bytes to FLOPs. Chosen from the data, not the part's peak: with
16, one threshold per class fits the three B200 studies at once (prefill-shaped backward is FLOP-bound, d256 decode is
KV-byte-bound and independent of h_q, D576 MLA decode with up to 128 query heads is FLOP-bound); 8, 32 and the
part's ~280 each lose one of them."""


def log2_work(facts) -> float:
    """log2 of a roofline-shaped cost proxy, max(FLOPs / COST_BALANCE, bytes), for bf16 / f16 Q, K, V, O."""
    width = facts.d_qk + facts.d_v
    flops = 2.0 * facts.b * facts.h_q * facts.s_q * facts.s_kv * width
    nbytes = 2.0 * facts.b * (facts.h_kv * facts.s_kv + facts.h_q * facts.s_q) * width
    return math.log2(max(1.0, flops / COST_BALANCE, nbytes))


def log2_tiles(facts) -> float:
    """log2 of the grid's parallel tiles before any KV split: b * h_kv * ceil(s_q * G / 128) (packed Q rows per KV head)."""
    group = max(1, facts.h_q // max(1, facts.h_kv))
    return math.log2(max(1, facts.b * facts.h_kv * -(-facts.s_q * group // 128)))


FEATURES = {"cost": log2_work, "tiles": log2_tiles}
"""The scalars a class may be thresholded on; the generator picks, per class, the one its data is monotone in."""


def lookup(row: str, facts) -> Optional[bool]:
    key = class_key(row, facts)
    entry = TABLE.get(key) if key is not None else None
    if entry is None:
        return None
    if entry.direction in ("always", "never"):
        return entry.direction == "always"
    w = FEATURES[entry.feature](facts)
    return w >= entry.log2_work if entry.direction == "ge" else w <= entry.log2_work


# ---------------------------------------------------------------------------------------------------------------- generated
# benchmark/placement_gen/gen.py --emit; one block per study, keys sorted. Do not edit by hand.
TABLE: Dict[Tuple, Entry] = {
    # study sm100_d256_decode_v3_20261028 (full-power B200, cuDNN 9.28.0.38; frozen 200 on-tile + 200 off-tile grid + 48 off-tile dense at 34-60 tiles)
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (256, 256), "q1", "dense", "none", False, "gqa", False): Entry("ge", 26.001, "sm100_d256_decode_v3_20261028"),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (256, 256), "q1", "dense", "none", False, "gqa", True): Entry(
        "ge", 5.044, "sm100_d256_decode_v3_20261028", "tiles"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (256, 256), "q1", "paged", "none", False, "gqa", False): Entry("ge", 26.002, "sm100_d256_decode_v3_20261028"),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (256, 256), "q1", "paged", "none", False, "gqa", True): Entry("ge", 24.532, "sm100_d256_decode_v3_20261028"),
    # study sm100_d576_decode_v2_20261028 (full-power B200, cuDNN 9.28.0.38; frozen 128-case grid on the tile + the same grid off it)
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (576, 512), "mtp", "paged", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm100_d576_decode_v2_20261028"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (576, 512), "mtp", "paged", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm100_d576_decode_v2_20261028"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (576, 512), "q1", "paged", "none", False, "gqa", False): Entry("ge", 26.588, "sm100_d576_decode_v2_20261028"),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (576, 512), "q1", "paged", "none", False, "gqa", True): Entry("ge", 26.131, "sm100_d576_decode_v2_20261028"),
}
