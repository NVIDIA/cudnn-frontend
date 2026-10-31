# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measured placement table: FROST row vs the backend, per shape class, generated from frozen sweeps.

A class is read off the graph facts (row, device, dtype, head pair, phase, layout, mask, sink, MHA vs GQA, and whether
S_kv is off the 128-key tile; a decode-shaped GQA graph the row cannot pack is its own group). Each entry
is a monotone rule on ONE scalar from :data:`FEATURES` -- the roofline-shaped cost proxy :func:`log2_work`, the
parallel-tile count :func:`log2_tiles`, the query rows :func:`log2_q_rows` or the KV per request :func:`log2_kv`, whichever the class's data is monotone in: ``always``, ``never``, ``ge T`` (FROST leads
from T up) or ``le T`` (FROST leads up to T). Entries are produced by ``benchmark/placement_gen`` from a training grid
and qualified on a frozen holdout; every entry names its study. Do not edit entries by hand -- regenerate.

:func:`lookup` answers ``True`` (lead), ``False`` (trail) or ``None`` (no entry, or a feature no study measured); the
hand-written rules in ``sdpa/fwd/placement.py`` / ``sdpa/bwd/heuristics.py`` decide every ``None``.
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, NamedTuple, Optional, Tuple

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


def _unpacked_decode(row: str, facts) -> bool:
    """A decode-shaped GQA graph the FROST forward row cannot pack (every query head re-reads K/V): a different kernel regime
    from the packed graphs the studies measured (paged d512 at G = 12-96 ran 5-21x slower than its packed neighbours). The
    row's own heuristics decide it, so the class follows the kernel rather than a shape proxy."""
    if not row.startswith("sdpa_fwd") or facts.s_q > 4 or facts.h_q == facts.h_kv:
        return False
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next((s for s in ENGINE_SPECS if s.name == row), None)
    if spec is None:
        return False
    caps = spec.capabilities
    return heuristics._pack_gqa_group(caps, facts, 128, heuristics._pack_gqa_eligible(caps, facts, 128)) == 1


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
    group = "mha" if facts.h_q == facts.h_kv else ("gqa_unpacked" if _unpacked_decode(row, facts) else "gqa")
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


def log2_q_rows(facts) -> float:
    """log2 of the query rows in flight, b * h_q * s_q: what fills the SMs when KV cannot split and one tile runs all heads."""
    return math.log2(max(1, facts.b * facts.h_q * facts.s_q))


def log2_kv(facts) -> float:
    """log2 of the KV tokens per request: what amortizes a kernel's fixed per-launch cost over its KV loop."""
    return math.log2(max(1, facts.s_kv))


FEATURES = {"cost": log2_work, "tiles": log2_tiles, "rows": log2_q_rows, "kv": log2_kv}
"""The scalars a class may be thresholded on; the generator picks, per class, the one its data is monotone in."""


PART_TRANSFERS: Dict[Tuple[int, int], Tuple[Tuple[int, int], FrozenSet[str]]] = {
    # study sm103_transfer_20261031 (B300 SXM6 1100 W, cuDNN 9.29.0.17): a cost-stratified sample of each listed study's frozen
    # cc 10.0 grid (201 cases) plus 56 off-lattice holdout cases, timed on cc 10.3 and scored through the full placement:
    # regret 1.000-1.025 per family (holdout 1.000) against develop's 1.035-2.98; of 115 changed routes one is slower than
    # develop (1.11x, the 32-tile d256 boundary the cc 10.0 grid also measured as mixed). Only these studies answer for
    # cc 10.3 -- a new cc 10.0 study does not, until a transfer check names it here.
    (10, 3): (
        (10, 0),
        frozenset(
            {
                "sm100_d256_decode_v3_20261028",
                "sm100_d576_decode_v2_20261028",
                "sm100_d512_paged_decode_b100_20261030",
                "sm100_d512_paged_context_20261030",
                "sm100_d256_paged_context_20261030",
                "sm100_d576_dense_decode_20261030",
            }
        ),
    ),
}
"""A part with no study of its own may borrow a measured part's entries, study by study, once a transfer check timed that
study's classes on it."""


def _answer(entry: Entry, facts) -> bool:
    if entry.direction in ("always", "never"):
        return entry.direction == "always"
    w = FEATURES[entry.feature](facts)
    return w >= entry.log2_work if entry.direction == "ge" else w <= entry.log2_work


def lookup(row: str, facts) -> Optional[bool]:
    key = class_key(row, facts)
    if key is None:
        return None
    entry = TABLE.get(key)
    if entry is not None:
        return _answer(entry, facts)
    if key[1] not in PART_TRANSFERS:
        return None
    base, studies = PART_TRANSFERS[key[1]]
    entry = TABLE.get(key[:1] + (base,) + key[2:])
    if entry is None or entry.study not in studies:
        return None
    return _answer(entry, facts)


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
    # study sm100_d512_paged_decode_b100_20261030 (cc 10.0 B100 850 W, cuDNN 9.29.0.17; 120 on-tile + 27 off-tile cells, one KV head,
    # h_q 8-128, b 1-128, S_kv 4k-128k, page 16/64, fill 0.25/1; the same threshold on the B200 grid: regret 1.002)
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (512, 512), "q1", "paged", "none", False, "gqa", False): Entry(
        "ge", 26.5, "sm100_d512_paged_decode_b100_20261030"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (512, 512), "q1", "paged", "none", False, "gqa", True): Entry(
        "ge", 26.486, "sm100_d512_paged_decode_b100_20261030"
    ),
    # study sm100_d512_paged_context_20261030 (cuDNN 9.29.0.17; one KV head, h_q 8-128, THD Q 16-8192 over S_kv 16-128k, b 1-4; B200 for
    # the 512+ grid, B100 for off-tile and short-KV cells; short KV runs the kernel's fixed cost, so the on-tile class reads S_kv)
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (512, 512), "prefill", "paged", "causal_br", False, "gqa", False): Entry(
        "ge", 9.0, "sm100_d512_paged_context_20261030", "kv"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (512, 512), "prefill", "paged", "causal_br", False, "gqa", True): Entry(
        "ge", 30.5, "sm100_d512_paged_context_20261030"
    ),
    # study sm100_d256_paged_context_20261030 (cuDNN 9.29.0.17; GQA 4-32x, THD Q 16-8192 over S_kv 16-128k, page 16/64, b 1-4; same split
    # of parts as the d512 context study)
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (256, 256), "prefill", "paged", "causal_br", False, "gqa", False): Entry(
        "ge", 9.0, "sm100_d256_paged_context_20261030", "kv"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (256, 256), "prefill", "paged", "causal_br", False, "gqa", True): Entry(
        "ge", 28.999, "sm100_d256_paged_context_20261030"
    ),
    # study sm100_d576_dense_decode_20261030 (cuDNN 9.29.0.17; dense K/V, h_q 8-128, b 1-128, S_kv 4k-128k: on the tile on B200,
    # off it on B100 plus 18 cells at 384-768 query rows; off the tile KV cannot split, so the unsplit tile follows b * h_q)
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (576, 512), "q1", "dense", "none", False, "gqa", False): Entry("ge", 26.09, "sm100_d576_dense_decode_20261030"),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (576, 512), "q1", "dense", "none", False, "gqa", True): Entry(
        "ge", 8.293, "sm100_d576_dense_decode_20261030", "rows"
    ),
}
