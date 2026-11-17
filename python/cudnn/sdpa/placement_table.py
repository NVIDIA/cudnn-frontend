# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measured placement table: FROST row vs the backend, per shape class, generated from frozen sweeps.

A class is read off the graph facts (row, device, dtype, head pair, phase, layout, mask, sink, MHA vs GQA, and whether
S_kv is off the 128-key tile; a decode-shaped GQA graph the row cannot pack is its own group). A bounded declaration
(``shape_overrides``: one plan serves every live shape up to the declared one, e.g. FlashInfer's ragged path) is a layout of
its own (``*_bounded``), keyed and thresholded on what it declares; an exact graph's entry never answers it, and a bounded
prefill declaring at most one 128-row Q tile per request is ``prefill_short_q``. SM120 backward on
cc 12.0 separates square prefills below that first tile into ``prefill_short_square``; their studies cannot be
borrowed from longer off-tile squares. Each entry
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


DETERMINISTIC_KEYED_ROWS = frozenset({"sdpa_bwd_sm120", "sdpa_bwd_sm80"})
SM80_DET_NARROW_WINDOW_MAX_LEFT = 383  # window_left at or below keys `window_narrow` on sdpa_bwd_sm80 prefill_det (256 measured, 512 not)
SM120_PAGED_BODY_MAX_S_Q = 16  # sdpa_fwd_prefill_sm120 paged d128 up to this s_q, off the decode tile, keys `gqa_body`
"""Rows whose deterministic graphs are classes of their own (phase ``<phase>_det``): the row's deterministic body is a different kernel
(SM120: the two-kernel dS / dQ split), so those graphs neither share the default classes' entries nor count as unmeasured."""


def _unmeasured(facts, row: Optional[str] = None) -> bool:
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
        or (getattr(facts, "deterministic", False) and row not in DETERMINISTIC_KEYED_ROWS)
        or getattr(facts, "has_dbias", False)
        or getattr(facts, "o_block_scale", False)
    )


def _unpacked_decode(row: str, facts) -> bool:
    """A decode-shaped GQA graph the FROST forward row cannot pack (every query head re-reads K/V): a different kernel regime
    from the packed graphs the studies measured (paged d512 at G = 12-96 ran 5-21x slower than its packed neighbours). The
    row's own heuristics decide it, so the class follows the kernel rather than a shape proxy."""
    if not row.startswith("sdpa_fwd") or facts.s_q > 4 or facts.h_q == facts.h_kv:
        return False
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS, sm100_legacy_policy_spec

    spec = (
        sm100_legacy_policy_spec(row) if row in ("sdpa_fwd_prefill_sm90", "sdpa_fwd_prefill_sm107") else next((s for s in ENGINE_SPECS if s.name == row), None)
    )
    if spec is None:
        return False
    from cudnn.sdpa.fwd.kernels.sm100.d256_f16_contract import in_pilot_domain
    from cudnn.sdpa.fwd.kernels.sm100.d512_f16_contract import in_domain as in_wide_half_domain
    from cudnn.sdpa.fwd.kernels.sm100.d192_half_contract import in_domain as in_d192_half_domain
    from cudnn.sdpa.fwd.kernels.sm100.small_half_contract import in_domain as in_small_half_domain
    from cudnn.sdpa.fwd.kernels.sm100.d512_fp8_role_split.contract import LEGACY_ROW as FP8_LEGACY_ROW, in_domain as in_d512_fp8_domain
    from cudnn.sdpa.fwd.kernels.sm100.d256_mxfp8_single_q.contract import LEGACY_ROW as MXFP8_LEGACY_ROW, in_domain as in_d256_mxfp8_domain
    from cudnn.sdpa.fwd.kernels.sm100.d256_fp8_single_q.contract import in_domain as in_d256_fp8_domain
    from cudnn.sdpa.fwd.kernels.sm100.d192_fp8_multi_q.contract import in_domain as in_d192_fp8_domain
    from cudnn.sdpa.fwd.kernels.sm100.small_fp8_contract import in_domain as in_small_fp8_domain
    from cudnn.sdpa.fwd.kernels.sm107.d512_fp8_role_split import contract as sm107_d512_fp8
    from cudnn.sdpa.fwd.kernels.sm107.d192_fp8_multi_q import contract as sm107_d192_fp8
    from cudnn.sdpa.fwd.kernels.sm107.d256_fp8_single_q import contract as sm107_d256_fp8
    from cudnn.sdpa.fwd.kernels.sm107.d128_fp8_multi_q import contract as sm107_small_fp8

    from cudnn.sdpa.fwd.kernels.sm120.d512_f16_paired.contract import LEGACY_ROW as SM120_LEGACY_ROW, in_domain as in_sm120_paired_domain

    caps = (
        heuristics.sm100_legacy_policy_spec().capabilities
        if row == "sdpa_fwd_prefill_sm100"
        and (in_pilot_domain(facts) or in_wide_half_domain(facts) or in_d192_half_domain(facts) or in_small_half_domain(facts))
        else spec.capabilities
    )
    from cudnn.sdpa.fwd.kernels.sm120.d512_fp8_paired import contract as fp8_wide
    from cudnn.sdpa.fwd.kernels.sm120.d512_mxfp8_paired import contract as mxfp8_wide

    from cudnn.sdpa.fwd.kernels.sm100.d128_mxfp8_multi_q.contract import in_domain as in_small_mxfp8_domain
    from cudnn.sdpa.fwd.kernels.sm100.d192_mxfp8_multi_q.contract import in_domain as in_mid_mxfp8_domain

    if row == MXFP8_LEGACY_ROW and (in_d256_mxfp8_domain(facts) or in_small_mxfp8_domain(facts) or in_mid_mxfp8_domain(facts)):
        caps = heuristics.sm100_legacy_policy_spec(row).capabilities
    from cudnn.sdpa.fwd.kernels.sm120 import half_exact_contract as sm120_half
    from cudnn.sdpa.fwd.kernels.sm120 import quantized_general_contract as sm120_quant_general

    sm120_wide = ((SM120_LEGACY_ROW, in_sm120_paired_domain), (sm120_half.LEGACY_ROW, sm120_half.in_domain)) + tuple(
        (m.LEGACY_ROW, m.in_domain) for m in (fp8_wide, mxfp8_wide)
    )
    sm120_wide += tuple((row, sm120_quant_general.in_domain) for row in sm120_quant_general.LEGACY_ROWS)
    from cudnn.sdpa.fwd.kernels.sm80 import half_flavor_contract as sm80_half

    sm120_wide += ((sm80_half.LEGACY_ROW, sm80_half.in_domain), (sm107_d512_fp8.LEGACY_ROW, sm107_d512_fp8.in_domain))
    from cudnn.sdpa.fwd.kernels.sm100 import d512_mxfp8_contract as d512_mxfp8

    sm120_wide += ((d512_mxfp8.LEGACY_ROW, d512_mxfp8.in_domain),)
    sm120_wide += tuple((owner.LEGACY_ROW, owner.in_domain) for owner in (sm107_d192_fp8, sm107_d256_fp8, sm107_small_fp8))
    from cudnn.sdpa.fwd.kernels.sm107 import mxfp8_contract as sm107_mxfp8

    sm120_wide += ((sm107_mxfp8.LEGACY_ROW, sm107_mxfp8.in_domain),)
    if (row == FP8_LEGACY_ROW and (in_d512_fp8_domain(facts) or in_d256_fp8_domain(facts) or in_d192_fp8_domain(facts) or in_small_fp8_domain(facts))) or any(
        row == legacy and in_domain(facts) for legacy, in_domain in sm120_wide
    ):
        caps = heuristics.sm100_legacy_policy_spec(row).capabilities
    return heuristics._pack_gqa_group(caps, facts, 128, heuristics._pack_gqa_eligible(caps, facts, 128)) == 1


def _sm120_paged_body(row: str, facts) -> bool:
    """SM120 paged d128 MTP / short prefill that the decode tile declines (packed rows past 32) runs the prefill body, which leads
    only once enough query rows are in flight; the decode tile's group rule does not transfer (sm120_paged_d128_body_20261117).
    The row's own tile choice decides, so the class follows the kernel. Padded Q only: no study timed THD Q here."""
    if row != "sdpa_fwd_prefill_sm120" or not facts.has_paged_kv or facts.thd or not 1 < facts.s_q <= SM120_PAGED_BODY_MAX_S_Q:
        return False
    if (facts.d_qk, facts.d_v) != (128, 128):
        return False
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next((s for s in ENGINE_SPECS if s.name == row), None)
    return spec is not None and heuristics._sm120_decode_tile_m(spec.capabilities, facts) == 0


def _packed_split(row: str, facts) -> bool:
    """A paged GQA prefix the SM100 row packs and splits (``heuristics.sm100_paged_packed_split_choice``): a different kernel
    regime from the unsplit graphs beside it (B200 D64: 0.02-0.61 of the backend split, 0.24-1.08 unsplit), so it is a class
    of its own, read from the row's own decision as ``_unpacked_decode`` is."""
    if row != "sdpa_fwd_prefill_sm100" or not facts.has_paged_kv or facts.h_q == facts.h_kv:
        return False
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next((s for s in ENGINE_SPECS if s.name == row), None)
    if spec is None:
        return False
    from cudnn.sdpa.fwd.kernels.sm100.d256_f16_contract import in_pilot_domain
    from cudnn.sdpa.fwd.kernels.sm100.d512_f16_contract import in_domain as in_wide_half_domain
    from cudnn.sdpa.fwd.kernels.sm100.small_half_contract import in_domain as in_small_half_domain

    caps = (
        heuristics.sm100_legacy_policy_spec().capabilities
        if in_pilot_domain(facts) or in_wide_half_domain(facts) or in_small_half_domain(facts)
        else spec.capabilities
    )
    return heuristics.sm100_paged_packed_split_choice(caps, facts) > 1


_VERIFY_ROWS = frozenset({"sdpa_fwd_prefill_sm80", "sdpa_fwd_prefill_sm90"})
_VERIFY_MAX_S_Q = 16  # placement.DECODE_SHAPED_MAX_S_Q: the spec-decode verify depth


def class_key(row: str, facts, verify: bool = True) -> Optional[Tuple]:
    dtype = _dtype_name(facts.dtype)
    if dtype is None or _unmeasured(facts, row) or not facts.h_kv or facts.h_q % facts.h_kv:
        return None
    if facts.s_q == 1:
        phase = "q1"
    elif facts.s_q <= 4:
        phase = "mtp"
    elif row == "sdpa_bwd_sm120" and facts.device_cc == (12, 0) and facts.s_q == facts.s_kv and facts.s_q < 128:
        # Separate the first partial 128-key placement tile from later partial tiles. On RTX 5090, short wide
        # squares lose while long squares win; one threshold fitted to both sacrifices the long-S leads (#2349).
        # No study has qualified this new class yet, so it falls back to the existing backend-first order.
        # This is row/part-local: GB10 and every other row keep their existing classification and studies.
        phase = "prefill_short_square"
    elif verify and row in _VERIFY_ROWS and facts.s_q <= _VERIFY_MAX_S_Q and facts.h_q != facts.h_kv:
        # SM80 / SM90 GQA spec-decode verify rows run the packed split-KV decode tile, a different regime from the prefill bodies the
        # older prefill studies of these rows measured (sm80_dense_prefill_*: s_q 8-16 GQA there predates SM80 split-KV, #2399).
        phase = "verify"
    else:
        phase = "prefill"
    bounded = bool(getattr(facts, "shape_overrides", False))
    sm100_paged_d64 = row == "sdpa_fwd_prefill_sm100" and facts.has_paged_kv and (facts.d_qk, facts.d_v) == (64, 64)
    if phase == "prefill" and facts.s_q <= 128 and (bounded or sm100_paged_d64):
        # One 128-row Q tile per request at most, where the backend switches to its short-row engine and FROST runs one Q tile
        # per head: FlashInfer's short-row declaration, and SM100 paged D64 (FROST 0.24-0.46 of the backend past a 2k cache,
        # 0.67-1.08 above one Q tile; study sm100_paged_d64_d256_prefill_20261111).
        phase = "prefill_short_q"
    if getattr(facts, "deterministic", False):
        phase += "_det"  # only DETERMINISTIC_KEYED_ROWS get here
    if facts.has_paged_kv:
        layout = "paged"
    elif facts.thd:
        total = facts.max_total_seq_len_q
        layout = "thd_ragged" if total is not None and total < facts.b * facts.s_q else "thd"
    else:
        layout = "dense"
    if bounded:
        layout += "_bounded"
    bottom_right = facts.bottom_right or facts.s_q == facts.s_kv  # a square reads the same from either corner
    if facts.window_left is not None:
        # A100 deterministic backward: a window of up to ~384 keys wins from S ~900, a 1024-key one only from S ~3k; neither S nor
        # S / window orders both (sm80_det_bwd_d64mha_v4_20261117), so narrow windows key apart on this row only.
        narrow = row == "sdpa_bwd_sm80" and phase == "prefill_det" and facts.window_left <= SM80_DET_NARROW_WINDOW_MAX_LEFT
        mask = "window_narrow" if narrow else "window"
    elif facts.causal and not (facts.s_q == 1 and bottom_right):  # one bottom-right row sees every key: no mask
        mask = "causal_br" if bottom_right else "causal_tl"
    else:
        mask = "none"
    if facts.h_q == facts.h_kv:
        group = "mha"
    elif _unpacked_decode(row, facts):
        group = "gqa_unpacked"
    else:
        group = "gqa_split" if _packed_split(row, facts) else "gqa"
    if group == "gqa" and _sm120_paged_body(row, facts):
        group = "gqa_body"
    # Off the 128-key tile dense KV cannot split and paged KV measures differently near the boundary: its own class, which
    # an on-tile study never answers.
    kv_off_tile = facts.s_kv % 128 != 0
    if row == "sdpa_fwd_prefill_sm90" and (facts.d_qk, facts.d_v) == (128, 128) and layout == "dense" and phase in ("prefill", "verify"):
        # The row's own decision (Rule S12: a key follows the kernel): the D128 template's entries answer the graphs it serves;
        # a graph it declines (per-batch lengths, unpadded bottom-right S_q > S_kv, ...) runs the 64 x 64 tile, which no
        # study of these classes timed, so it keeps the hand-written rules.
        from cudnn.sdpa.fwd.config_sm90 import d128_graph_decline

        if d128_graph_decline(facts) is not None:
            return None
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


def log2_q(facts) -> float:
    """log2 of the query tokens per request: separates decode-shaped graphs (one Q tile) from prefill in a class that holds both."""
    return math.log2(max(1, facts.s_q))


FEATURES = {"cost": log2_work, "tiles": log2_tiles, "rows": log2_q_rows, "kv": log2_kv, "q": log2_q}
"""The scalars a class may be thresholded on; the generator picks, per class, the one its data is monotone in."""


PART_TRANSFERS: Dict[Tuple[int, int], Tuple[Tuple[int, int], FrozenSet[str]]] = {
    # study sm103_transfer_20261031 (B300 SXM6 1100 W, cuDNN 9.29.0.17): a cost-stratified sample of each listed study's frozen
    # cc 10.0 grid (201 cases) plus 56 off-lattice holdout cases, timed on cc 10.3 and scored through the full placement:
    # regret 1.000-1.025 per family (holdout 1.000) against develop's 1.035-2.98; of 115 changed routes one is slower than
    # develop (1.11x, the 32-tile d256 boundary the cc 10.0 grid also measured as mixed). Only these studies answer for
    # cc 10.3 -- a new cc 10.0 study does not, until a transfer check names it here. Entries keyed (10, 3) themselves (a
    # study run on cc 10.3, e.g. the d256 backward below) answer first and need no transfer.
    # sm100_d256_decode_v4_924_20261116 replaces v3 after its own check (B300 SXM6 AC 1100 W, cuDNN 9.24.0.43; 80 cost-stratified
    # train cells): regret 1.002, worst 1.038; the 15 verdicts that differ from v3 run 0.861 of v3's choice, worst 1.032.
    # sm100_d576_dense_decode_rt924_20261030 replaces its parent study after the same check (20 train cells): regret 1.000.
    # sm100_d576_decode_v3_924_20261116 (D576 paged s_q 1 on the tile): a 20-cell check missed the regret bound (1.040, worst 1.339;
    # every changed verdict still beat v2, 0.751), so the class was studied on cc 10.3 itself (B300, 103 train cells): its own fit is
    # the same threshold, cost 25.503, and a fresh 20-cell cc 10.3 holdout scores 1.000 vs v2's 1.058, no cell slower.
    # sm100_paged_d64_d256_prefill_20261111 after its own check (study b300_d64_transfer_20261117: B300 SXM6 AC 1100 W, cuDNN
    # 9.24.0.43, FE 9bdf9a8ab; 76 of its d64 train cells + its 45 off-lattice d64 holdout cells through FlashInfer's paged path):
    # regret 1.020 vs today's cc 10.3 default 1.155; the 31 changed verdicts run 0.614 (worst 0.928). Unsplit short-Q d64 (the
    # GB300 nightly's 3-4x serving-prefill loss) goes 3.39 -> 1.00 on the tile. Its d256 entries are the split classes, which
    # cc 10.3 already led through the split rule.
    # sm100_d576_prefill_20261117 after its own check (B300 SXM6 AC 1100 W, cuDNN 9.24.0.43, FE 6f17f2973; all 384 of its B200 cells):
    # every twin passes, 278 changed verdicts at 0.130-0.153 of cc 10.3's default, worst 0.456.
    (10, 3): (
        (10, 0),
        frozenset(
            {
                "sm100_d256_decode_v4_924_20261116",
                "sm100_d576_decode_v2_20261028",
                "sm100_d512_paged_decode_b100_20261030",
                "sm100_d512_paged_context_20261030",
                "sm100_d256_paged_context_20261030",
                "sm100_d576_dense_decode_rt924_20261030",
                "sm100_d576_decode_v3_924_20261116",
                "sm100_paged_d64_d256_prefill_20261111",
                "sm100_d576_prefill_20261117",
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


def matched_entry(row: str, facts) -> Optional[Entry]:
    """The entry :func:`lookup` answers from: the graph's own class, else a part transfer's borrowed class. A ``verify`` class
    no study answers keeps the ``prefill`` class it had before that phase existed."""
    key = class_key(row, facts)
    entry = _class_entry(key)
    if entry is None and key is not None and key[4] == "verify":
        entry = _class_entry(class_key(row, facts, verify=False))
    return entry


DTYPE_TRANSFERS: Dict[Tuple[str, str], Tuple[str, FrozenSet[str]]] = {
    # study sm90_d128_h100_refit_20261114: 24 f16 cells of its classes timed against the bf16 entries, none slower than the measured
    # default beyond the noise band.
    ("sdpa_fwd_prefill_sm90", "f16"): ("bf16", frozenset({"sm90_d128_h100_refit_20261114"})),
    # study sm100_paged_f16_transfer_20261112: FlashInfer's exact paged f16 prefill timed against these studies' bf16 verdicts.
    # sm100_d256_decode_v4_924_20261116: 32 of its train cells timed in f16 (B200, cuDNN 9.24.0.43): regret 1.000, worst 1.015.
    ("sdpa_fwd_prefill_sm100", "f16"): (
        "bf16",
        frozenset({"sm100_paged_d64_d256_prefill_20261111", "sm100_d256_paged_context_20261030", "sm100_d256_decode_v4_924_20261116"}),
    ),
}
"""A dtype with no study of its own may borrow another dtype's entries on a row, study by study, once a transfer check timed
those studies' classes in it."""


def _class_entry(key: Optional[Tuple]) -> Optional[Entry]:
    entry = _measured_entry(key)
    if entry is None and key is not None and (key[0], key[2]) in DTYPE_TRANSFERS:
        base, studies = DTYPE_TRANSFERS[(key[0], key[2])]
        borrowed = _measured_entry(key[:2] + (base,) + key[3:])
        entry = borrowed if borrowed is not None and borrowed.study in studies else None
    return entry


def _measured_entry(key: Optional[Tuple]) -> Optional[Entry]:
    if key is None:
        return None
    entry = TABLE.get(key)
    if entry is not None:
        return entry
    if key[1] not in PART_TRANSFERS:
        return None
    base, studies = PART_TRANSFERS[key[1]]
    entry = TABLE.get(key[:1] + (base,) + key[2:])
    return entry if entry is not None and entry.study in studies else None


def lookup(row: str, facts) -> Optional[bool]:
    entry = matched_entry(row, facts)
    return None if entry is None else _answer(entry, facts)


# ---------------------------------------------------------------------------------------------------------------- generated
# benchmark/placement_gen/gen.py --emit; one block per study, keys sorted. Do not edit by hand.
TABLE: Dict[Tuple, Entry] = {
    # study sm100_d256_decode_v4_924_20261116 (full-power B200, cuDNN 9.24.0.43; v3's frozen 448-cell grid re-timed with the backend's
    # split-KV reduction kernel counted, see benchmark/placement_gen/README.md)
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (256, 256), "q1", "dense", "none", False, "gqa", False): Entry(
        "ge", 23.507, "sm100_d256_decode_v4_924_20261116"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (256, 256), "q1", "dense", "none", False, "gqa", True): Entry(
        "ge", 4.5, "sm100_d256_decode_v4_924_20261116", "tiles"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (256, 256), "q1", "paged", "none", False, "gqa", False): Entry(
        "always", None, "sm100_d256_decode_v4_924_20261116"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (256, 256), "q1", "paged", "none", False, "gqa", True): Entry(
        "always", None, "sm100_d256_decode_v4_924_20261116"
    ),
    # study sm100_d576_prefill_20261117 (B200 1000 W, cuDNN 9.24.0.43, FE 8014a2c75; benchmark/attention_inference's context graph:
    # absorbed-MLA (576, 512) MQA prefill chunks, h_q 8-128, b 1-16, S_q 8-2048, KV 4k-128k on and off the tile; dense BSHD and paged
    # THD with an exact total-Q; 320 train cells). FROST's prefill_d512_f16_2x2 leads everywhere: changed train cells 0.137-0.149
    # (worst 0.452); holdout at fresh h_q, scored once, 0.110-0.196 (worst 0.431). Only h_kv 1 was measured. cc 10.3 borrows these
    # entries (PART_TRANSFERS).
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (576, 512), "prefill", "dense", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm100_d576_prefill_20261117"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (576, 512), "prefill", "dense", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm100_d576_prefill_20261117"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (576, 512), "prefill", "paged", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm100_d576_prefill_20261117"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (576, 512), "prefill", "paged", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm100_d576_prefill_20261117"
    ),
    # study sm100_d576_decode_v2_20261028 (full-power B200, cuDNN 9.28.0.38; frozen 128-case grid on the tile + the same grid off it)
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (576, 512), "mtp", "paged", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm100_d576_decode_v2_20261028"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (576, 512), "mtp", "paged", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm100_d576_decode_v2_20261028"
    ),
    # study sm100_d576_decode_v3_924_20261116 (on the tile only): v2's grid with the backend re-timed on a full-power B200 at cuDNN
    # 9.24.0.43 (its split-KV reduction counted) plus a 40-cell supplement at cost 24.5-27 timed at develop 5304d19df; fresh 20-cell
    # boundary holdout: regret 1.059 vs v2's 1.293, no cell slower, 9 faster. Off the tile and MTP keep v2 (no holdout cell moved).
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (576, 512), "q1", "paged", "none", False, "gqa", False): Entry(
        "ge", 25.503, "sm100_d576_decode_v3_924_20261116"
    ),
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
    # study sm100_paged_d64_d256_prefill_20261111 (full-power B200, cuDNN 9.24.0.43; FlashInfer's paged cudnn path, recorded facts;
    # bf16 HND page 16/64, GQA 2-16, causal: D64 / D256 b 1-16, Q 8-1024, KV 2k-64k on and off the 128-key tile, plus D64 short
    # KV 128-1536, b 32-128 and Q 1024-8192; 752 train cells). ``gqa_split`` is the row's packed split (#2471); D256 unsplit
    # keeps sm100_d256_paged_context_20261030's entries above.
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (256, 256), "prefill", "paged", "causal_br", False, "gqa_split", False): Entry(
        "always", None, "sm100_paged_d64_d256_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (256, 256), "prefill", "paged", "causal_br", False, "gqa_split", True): Entry(
        "always", None, "sm100_paged_d64_d256_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (64, 64), "prefill", "paged", "causal_br", False, "gqa", False): Entry(
        "le", 11.5, "sm100_paged_d64_d256_prefill_20261111", "kv"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (64, 64), "prefill", "paged", "causal_br", False, "gqa", True): Entry(
        "le", 12.026, "sm100_paged_d64_d256_prefill_20261111", "kv"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (64, 64), "prefill", "paged", "causal_br", False, "gqa_split", False): Entry(
        "always", None, "sm100_paged_d64_d256_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (64, 64), "prefill", "paged", "causal_br", False, "gqa_split", True): Entry(
        "always", None, "sm100_paged_d64_d256_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (64, 64), "prefill_short_q", "paged", "causal_br", False, "gqa", False): Entry(
        "ge", 7.5, "sm100_paged_d64_d256_prefill_20261111", "kv"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (64, 64), "prefill_short_q", "paged", "causal_br", False, "gqa", True): Entry(
        "ge", 8.219, "sm100_paged_d64_d256_prefill_20261111", "kv"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (64, 64), "prefill_short_q", "paged", "causal_br", False, "gqa_split", False): Entry(
        "always", None, "sm100_paged_d64_d256_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (64, 64), "prefill_short_q", "paged", "causal_br", False, "gqa_split", True): Entry(
        "always", None, "sm100_paged_d64_d256_prefill_20261111"
    ),
    # study sm100_d576_dense_decode_rt924_20261030: study sm100_d576_dense_decode_20261030's grid (dense K/V, h_q 8-128, b 1-128,
    # S_kv 4k-128k; off the tile KV cannot split, so the unsplit tile follows b * h_q) with the backend re-timed on a full-power B200
    # at cuDNN 9.24.0.43, its split-KV reduction counted; FROST times from the original study; fresh 24-cell holdout at develop
    # 5304d19df: regret 1.000 vs the old entries' 1.031, no cell slower
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (576, 512), "q1", "dense", "none", False, "gqa", False): Entry(
        "ge", 25.588, "sm100_d576_dense_decode_rt924_20261030"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (576, 512), "q1", "dense", "none", False, "gqa", True): Entry(
        "ge", 8.293, "sm100_d576_dense_decode_rt924_20261030", "rows"
    ),
    # study sm107_paged_prefill_20261109 (208-SM Rubin, cuDNN 9.30.0.2; FlashInfer's paged cudnn path, recorded graph facts; bf16
    # D128 / D192x128 / D256 causal chunks and squares, MHA / GQA 4-8 / MQA, b 1-16, s_q 5-8192, page 16 / 64, HND / NHD; 953
    # train cells; D128 keeps the hand-written paged rules (no feature splits its GQA class); 96 off-tile + 48 on-tile holdout cells:
    # the entries never >5% behind the measured default, geo regret 1.00-1.02 vs 1.97-6.01; FROST also 0.03-0.61 of the backend on
    # the dense-Q decode graph at s_q 5-16)
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (192, 128), "prefill", "paged", "causal_br", False, "gqa", False): Entry(
        "ge", 9.0, "sm107_paged_prefill_20261109", "kv"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (192, 128), "prefill", "paged", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm107_paged_prefill_20261109"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (192, 128), "prefill", "paged", "causal_br", False, "mha", False): Entry(
        "ge", 9.0, "sm107_paged_prefill_20261109", "kv"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (192, 128), "prefill", "paged", "causal_br", False, "mha", True): Entry(
        "always", None, "sm107_paged_prefill_20261109"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "prefill", "paged", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm107_paged_prefill_20261109"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "prefill", "paged", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm107_paged_prefill_20261109"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "prefill", "paged", "causal_br", False, "mha", False): Entry(
        "always", None, "sm107_paged_prefill_20261109"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "prefill", "paged", "causal_br", False, "mha", True): Entry(
        "always", None, "sm107_paged_prefill_20261109"
    ),
    # study sm107_paged_mtp_20261110 (208-SM Rubin, cuDNN 9.30.0.3; benchmark/attention_inference's paged generation graph: dense Q,
    # shuffled HND page pool, per-request KV, recorded graph facts; bf16 MTP rows s_q 2-4 under the bottom-right diagonal at
    # d64 (sink, sink + window 128, neither) / d128 / d256 GQA and d512 / D576xV512 absorbed MLA and GQA; GQA 4-128, b 1-256,
    # page 16 / 32 / 64, declared KV 16 to 1M on and off the tile; calls timed N per captured graph; 1421 train cells (incl. a
    # consumed first holdout) + a fresh 144-cell holdout: never >5% behind the measured default. The absorbed-MLA s_q == 1 classes
    # are not included: d512 lost 1.32x on the holdout at h_kv 2, and D576 was not measured at h_kv > 1)
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (128, 128), "mtp", "paged", "causal_br", False, "gqa", False): Entry("ge", 18.793, "sm107_paged_mtp_20261110"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (128, 128), "mtp", "paged", "causal_br", False, "gqa", True): Entry("ge", 22.325, "sm107_paged_mtp_20261110"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "mtp", "paged", "causal_br", False, "gqa", False): Entry(
        "ge", 7.5, "sm107_paged_mtp_20261110", "kv"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "mtp", "paged", "causal_br", False, "gqa", True): Entry(
        "ge", 7.011, "sm107_paged_mtp_20261110", "kv"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (512, 512), "mtp", "paged", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm107_paged_mtp_20261110"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (512, 512), "mtp", "paged", "causal_br", False, "gqa", True): Entry("always", None, "sm107_paged_mtp_20261110"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (576, 512), "mtp", "paged", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm107_paged_mtp_20261110"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (576, 512), "mtp", "paged", "causal_br", False, "gqa", True): Entry("always", None, "sm107_paged_mtp_20261110"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "mtp", "paged", "causal_br", False, "gqa", False): Entry("ge", 7.5, "sm107_paged_mtp_20261110", "kv"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "mtp", "paged", "causal_br", False, "gqa", True): Entry(
        "ge", 5.749, "sm107_paged_mtp_20261110", "kv"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "mtp", "paged", "causal_br", True, "gqa", False): Entry("ge", 7.5, "sm107_paged_mtp_20261110", "kv"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "mtp", "paged", "causal_br", True, "gqa", True): Entry("ge", 6.915, "sm107_paged_mtp_20261110", "kv"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "mtp", "paged", "window", True, "gqa", False): Entry("always", None, "sm107_paged_mtp_20261110"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "mtp", "paged", "window", True, "gqa", True): Entry("always", None, "sm107_paged_mtp_20261110"),
    # study sm107_thd_prefill_20261111 (208-SM Rubin, cuDNN 9.30.0.3; FlashInfer's cudnn path, recorded graph facts; bf16 HND
    # paged page 16/64 and ragged THD, d64/d128 all masks, D192xV128 / d256 ragged all masks and paged mask-free (paged causal
    # D192/D256 is sm107_paged_prefill_20261109's); short-Q, chunks, squares; GQA groups 3-12 incl. 8/1 and 12/1, MHA 16-128.
    # Train 664 frozen cells, holdout 200, scored once; 14 entries dropped by the holdout. Ragged reaches FE as bounded
    # declarations (shape_overrides): those classes are the declaration's, scored on the realized shapes.)
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (128, 128), "prefill", "paged", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (128, 128), "prefill", "paged", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (128, 128), "prefill", "paged", "causal_br", False, "mha", False): Entry(
        "ge", 32.17, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (128, 128), "prefill", "paged", "causal_br", False, "mha", True): Entry(
        "ge", 13.304, "sm107_thd_prefill_20261111", "kv"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (128, 128), "prefill", "paged", "none", False, "gqa", False): Entry("le", 27.584, "sm107_thd_prefill_20261111"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (128, 128), "prefill", "paged", "none", False, "gqa", True): Entry("le", 26.976, "sm107_thd_prefill_20261111"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (128, 128), "prefill", "thd_bounded", "none", False, "mha", False): Entry(
        "le", 33.5, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (128, 128), "prefill_short_q", "thd_bounded", "causal_br", False, "gqa", False): Entry(
        "le", 45.5, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (128, 128), "prefill_short_q", "thd_bounded", "causal_br", False, "mha", False): Entry(
        "le", 38.0, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (192, 128), "prefill", "thd_bounded", "none", False, "mha", False): Entry(
        "ge", 55.322, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (192, 128), "prefill_short_q", "thd_bounded", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (192, 128), "prefill_short_q", "thd_bounded", "causal_br", False, "mha", False): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (192, 128), "prefill_short_q", "thd_bounded", "none", False, "gqa", False): Entry(
        "ge", 45.822, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "prefill", "paged", "none", False, "gqa", False): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "prefill", "paged", "none", False, "gqa", True): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "prefill", "paged", "none", False, "mha", False): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "prefill", "paged", "none", False, "mha", True): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "prefill", "thd_bounded", "causal_br", False, "gqa", False): Entry(
        "le", 26.5, "sm107_thd_prefill_20261111", "tiles"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "prefill_short_q", "thd_bounded", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "prefill_short_q", "thd_bounded", "causal_br", False, "mha", False): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "prefill_short_q", "thd_bounded", "none", False, "gqa", False): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "prefill_short_q", "thd_bounded", "none", False, "mha", False): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill", "paged", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill", "paged", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill", "paged", "causal_br", False, "mha", False): Entry(
        "le", 38.764, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill", "paged", "causal_br", False, "mha", True): Entry(
        "le", 39.287, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill", "paged", "none", False, "gqa", False): Entry("le", 34.475, "sm107_thd_prefill_20261111"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill", "paged", "none", False, "gqa", True): Entry("always", None, "sm107_thd_prefill_20261111"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill", "thd_bounded", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill", "thd_bounded", "causal_br", False, "mha", False): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill", "thd_bounded", "none", False, "gqa", False): Entry(
        "ge", 16.5, "sm107_thd_prefill_20261111", "kv"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill", "thd_bounded", "none", False, "mha", False): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill_short_q", "thd_bounded", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill_short_q", "thd_bounded", "causal_br", False, "mha", False): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill_short_q", "thd_bounded", "none", False, "gqa", False): Entry(
        "ge", 42.5, "sm107_thd_prefill_20261111"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill_short_q", "thd_bounded", "none", False, "mha", False): Entry(
        "always", None, "sm107_thd_prefill_20261111"
    ),
    # study sm80_thd_prefill_20261117 (A100 THD d256 GQA causal, trained at S_q >= 500): its four `always` entries are retired. The SM80
    # GQA prefill class starts at S_q 17, and an edge supplement at S_q 17-499 lost on every side (worst 1.13-1.53, geomean 0.68-0.88).
    # Re-studied over S_q 17-8192 as sm80_thd_d256_v2_20261117 (below).
    # study sm80_thd_d256_v2_20261117 (A100-SXM4 80GB, cuDNN 9.24.0.43, FE 835bbdb9b; the training harness's THD forward, recorded facts:
    # d256 GQA causal, squares and bottom-right chunks over S_q 17-8192, groups 2-8, b 8-96; 200 train cells). Ragged lengths lead from
    # S_q 2^7.5 on the tile and from 2^9.09 tiles off it (changed train 0.537 / 0.531); holdout at fresh heads, scored once, 0.516 /
    # 0.566 (worst 0.705). Equal lengths (`thd`) have no entry: their off-tile holdout side reached 1.101.
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "prefill", "thd_ragged", "causal_br", False, "gqa", False): Entry(
        "ge", 7.5, "sm80_thd_d256_v2_20261117", "q"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "prefill", "thd_ragged", "causal_br", False, "gqa", True): Entry(
        "ge", 9.085, "sm80_thd_d256_v2_20261117", "tiles"
    ),
    # study sm80_dense_prefill_20261031 (cuDNN 9.29.0.18; A100 80GB PCIe; dense bf16 context d64 / d128 / d192x128 / d256, causal,
    # mask-free and d64 sink / window + sink, s_q 8-8192 over S_kv on and off the tile; 2,448 train cells incl. two consumed
    # holdouts, 96-cell holdout 3; d128 GQA causal is non-monotone and keeps the hand-written order; d256 mask-free MHA,
    # dropped after holdout 3, is refit in round 4 below)
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (128, 128), "prefill", "dense", "causal_br", False, "mha", False): Entry(
        "le", 10.0, "sm80_dense_prefill_20261031", "kv"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (128, 128), "prefill", "dense", "causal_br", False, "mha", True): Entry(
        "le", 10.758, "sm80_dense_prefill_20261031", "kv"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (128, 128), "prefill", "dense", "none", False, "gqa", False): Entry("never", None, "sm80_dense_prefill_20261031"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (128, 128), "prefill", "dense", "none", False, "gqa", True): Entry("never", None, "sm80_dense_prefill_20261031"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (128, 128), "prefill", "dense", "none", False, "mha", False): Entry("never", None, "sm80_dense_prefill_20261031"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (128, 128), "prefill", "dense", "none", False, "mha", True): Entry("never", None, "sm80_dense_prefill_20261031"),
    # study sm80_d192_gqa_r5_20261112 replaces this study's four D192/V128 GQA entries (ge 8.085 on q): after native split-KV,
    # smaller tiles and packed GQA (#2399 / #2400 / #2403) FROST leads every short-q cell the threshold sent to the backend.
    # A100 80GB PCIe, cuDNN 9.29.0.18 (this study's), 512 train cells (GQA 8/1 / 16/2 / 32/8 / 64/8, b 1-16, s_q 16-4096,
    # S_kv 2k-16.5k on and off the tile, causal and mask-free) all FROST-first (backend-only regret 2.62 geomean, max 26x);
    # a 96-cell seeded holdout scored once: 64 changed cells at 0.28-0.46 geomean per class, worst 0.75.
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "prefill", "dense", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm80_d192_gqa_r5_20261112"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "prefill", "dense", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm80_d192_gqa_r5_20261112"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "prefill", "dense", "none", False, "gqa", False): Entry("always", None, "sm80_d192_gqa_r5_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "prefill", "dense", "none", False, "gqa", True): Entry("always", None, "sm80_d192_gqa_r5_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "prefill", "dense", "causal_br", False, "mha", False): Entry(
        "ge", 6.293, "sm80_dense_prefill_20261031", "tiles"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "prefill", "dense", "causal_br", False, "mha", True): Entry(
        "ge", 6.293, "sm80_dense_prefill_20261031", "tiles"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "causal_br", False, "gqa", False): Entry(
        "ge", 8.085, "sm80_dense_prefill_20261031", "q"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "causal_br", False, "gqa", True): Entry(
        "ge", 5.782, "sm80_dense_prefill_20261031", "tiles"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "causal_br", False, "mha", False): Entry(
        "ge", 6.293, "sm80_dense_prefill_20261031", "tiles"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "causal_br", False, "mha", True): Entry(
        "ge", 6.293, "sm80_dense_prefill_20261031", "tiles"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "causal_br", True, "gqa", False): Entry(
        "ge", 6.793, "sm80_dense_prefill_20261031", "q"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "causal_br", True, "gqa", True): Entry(
        "ge", 8.085, "sm80_dense_prefill_20261031", "q"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "causal_br", True, "mha", False): Entry(
        "ge", 5.793, "sm80_dense_prefill_20261031", "tiles"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "causal_br", True, "mha", True): Entry(
        "always", None, "sm80_dense_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "none", False, "gqa", False): Entry(
        "ge", 8.085, "sm80_dense_prefill_20261031", "q"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "none", False, "gqa", True): Entry("ge", 8.085, "sm80_dense_prefill_20261031", "q"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "none", False, "mha", False): Entry(
        "ge", 6.293, "sm80_dense_prefill_20261031", "tiles"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "none", False, "mha", True): Entry("ge", 28.534, "sm80_dense_prefill_20261031"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "window", True, "gqa", False): Entry("always", None, "sm80_dense_prefill_20261031"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "window", True, "gqa", True): Entry("always", None, "sm80_dense_prefill_20261031"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "window", True, "mha", False): Entry("always", None, "sm80_dense_prefill_20261031"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "window", True, "mha", True): Entry("always", None, "sm80_dense_prefill_20261031"),
    # study sm80_dense_prefill_r4_20261101 (round 4 after the SM80 short-query tiles: D128 / D256 S_q <= 128 and D192 S_q <= 64
    # re-timed, 575 cells, plus consumed holdout 3 outside that domain; an entry replaces its round-3 parent only where no
    # training cell runs > 5 % slower than under the parent -- the D192 GQA classes keep theirs; 108-cell holdout 4)
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "prefill", "dense", "causal_br", False, "mha", False): Entry(
        "always", None, "sm80_dense_prefill_r4_20261101"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "prefill", "dense", "causal_br", False, "mha", True): Entry(
        "always", None, "sm80_dense_prefill_r4_20261101"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "prefill", "dense", "none", False, "mha", False): Entry(
        "always", None, "sm80_dense_prefill_r4_20261101"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "prefill", "dense", "none", False, "mha", True): Entry(
        "always", None, "sm80_dense_prefill_r4_20261101"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "prefill", "dense", "causal_br", False, "gqa", False): Entry(
        "ge", 6.793, "sm80_dense_prefill_r4_20261101", "tiles"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "prefill", "dense", "causal_br", False, "gqa", True): Entry(
        "ge", 6.793, "sm80_dense_prefill_r4_20261101", "tiles"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "prefill", "dense", "none", False, "gqa", False): Entry(
        "ge", 6.793, "sm80_dense_prefill_r4_20261101", "tiles"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "prefill", "dense", "none", False, "gqa", True): Entry(
        "ge", 7.155, "sm80_dense_prefill_r4_20261101", "tiles"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "prefill", "dense", "none", False, "mha", False): Entry(
        "ge", 6.293, "sm80_dense_prefill_r4_20261101", "tiles"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "prefill", "dense", "none", False, "mha", True): Entry(
        "always", None, "sm80_dense_prefill_r4_20261101"
    ),
    # study sm80_f16_dense_prefill_20261106 (cuDNN 9.28.1.4; A100 80GB PCIe; dense f16 context d64 / d128, MHA and GQA 2-8x,
    # mask-free and bottom-right causal, s_q 5-8.8k and S_kv 128-35k drawn independently, on and off the 128-key tile, h_q 4-64,
    # b 1-16; 768 train cells, each distinct plan timed once with the plans interleaved; a 192-cell holdout off the training
    # lattice (h_q 10-80, GQA 2-7x, b 1-12). Three classes, each on and off the tile: d64 bottom-right causal MHA and GQA, and d128
    # mask-free MHA. The other ten f16 classes are non-monotone, regressed on the holdout, or lost their tile twin, and stay with
    # the hand rules. SM80 had no f16 entry.)
    ("sdpa_fwd_prefill_sm80", (8, 0), "f16", (128, 128), "prefill", "dense", "none", False, "mha", False): Entry(
        "le", 23.02, "sm80_f16_dense_prefill_20261106"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "f16", (128, 128), "prefill", "dense", "none", False, "mha", True): Entry(
        "le", 23.438, "sm80_f16_dense_prefill_20261106"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "f16", (64, 64), "prefill", "dense", "causal_br", False, "gqa", False): Entry(
        "ge", 9.603, "sm80_f16_dense_prefill_20261106", "q"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "f16", (64, 64), "prefill", "dense", "causal_br", False, "gqa", True): Entry(
        "ge", 6.937, "sm80_f16_dense_prefill_20261106", "q"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "f16", (64, 64), "prefill", "dense", "causal_br", False, "mha", False): Entry(
        "ge", 7.863, "sm80_f16_dense_prefill_20261106", "q"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "f16", (64, 64), "prefill", "dense", "causal_br", False, "mha", True): Entry(
        "ge", 8.217, "sm80_f16_dense_prefill_20261106", "q"
    ),
    # study sm80_dense_prefill_{bf16,f16}_20261107: train-only gen.py --margin 0.05 --emit, keeping always tile twins.
    # BF16 d64 windowed GQA remains excluded after the original qualification. FP16 d256 mask-free MHA/GQA also
    # retain their prior rules after native A100 qualification found repeatable losses; keep only MHA d64 windows.
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "window", False, "mha", False): Entry(
        "always", None, "sm80_dense_prefill_bf16_20261107"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "window", False, "mha", True): Entry(
        "always", None, "sm80_dense_prefill_bf16_20261107"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "f16", (64, 64), "prefill", "dense", "window", False, "mha", False): Entry(
        "always", None, "sm80_dense_prefill_f16_20261107"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "f16", (64, 64), "prefill", "dense", "window", False, "mha", True): Entry(
        "always", None, "sm80_dense_prefill_f16_20261107"
    ),
    # study sm80_decode_20261112 (A100 80GB PCIe, cuDNN 9.30.0.4; benchmark/attention_inference's generation graph, GQA only: d128 /
    # d256 dense and paged page 16, D576xV512 absorbed MLA dense and paged page 32, d64 with a sink (with and without a 128-key
    # window); s_q 1 / 2-4 (mtp) / 8-16 (verify), b 1-128, S_kv 4k-131k on and off the tile; 778 train cells + an 87-cell seeded
    # holdout scored once per class: mtp / verify 0.02-0.49 geomean of the measured default, worst 0.87. d256 q1 failed the holdout
    # (groups 12, 1.26-1.32x) and its paged q1 move had no holdout cell: no entry; d128 q1 dense is non-monotone: no entry)
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (128, 128), "mtp", "dense", "causal_br", False, "gqa", False): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (128, 128), "mtp", "dense", "causal_br", False, "gqa", True): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (128, 128), "mtp", "paged", "causal_br", False, "gqa", False): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (128, 128), "mtp", "paged", "causal_br", False, "gqa", True): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (128, 128), "q1", "paged", "none", False, "gqa", False): Entry("never", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (128, 128), "q1", "paged", "none", False, "gqa", True): Entry("never", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (128, 128), "verify", "dense", "causal_br", False, "gqa", False): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (128, 128), "verify", "dense", "causal_br", False, "gqa", True): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (128, 128), "verify", "paged", "causal_br", False, "gqa", False): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (128, 128), "verify", "paged", "causal_br", False, "gqa", True): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "mtp", "dense", "causal_br", False, "gqa", False): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "mtp", "dense", "causal_br", False, "gqa", True): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "mtp", "paged", "causal_br", False, "gqa", False): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "mtp", "paged", "causal_br", False, "gqa", True): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "verify", "dense", "causal_br", False, "gqa", False): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "verify", "dense", "causal_br", False, "gqa", True): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "verify", "paged", "causal_br", False, "gqa", False): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "verify", "paged", "causal_br", False, "gqa", True): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (576, 512), "mtp", "dense", "causal_br", False, "gqa", False): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (576, 512), "mtp", "dense", "causal_br", False, "gqa", True): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (576, 512), "mtp", "paged", "causal_br", False, "gqa", False): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (576, 512), "mtp", "paged", "causal_br", False, "gqa", True): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (576, 512), "q1", "dense", "none", False, "gqa", False): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (576, 512), "q1", "dense", "none", False, "gqa", True): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (576, 512), "q1", "paged", "none", False, "gqa", False): Entry("ge", 25.381, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (576, 512), "q1", "paged", "none", False, "gqa", True): Entry("ge", 25.376, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "mtp", "dense", "causal_br", True, "gqa", False): Entry("ge", 3.5, "sm80_decode_20261112", "tiles"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "mtp", "dense", "causal_br", True, "gqa", True): Entry("ge", 3.5, "sm80_decode_20261112", "tiles"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "mtp", "dense", "window", True, "gqa", False): Entry("always", None, "sm80_decode_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "mtp", "dense", "window", True, "gqa", True): Entry("always", None, "sm80_decode_20261112"),
    # study sm80_decode_gaps_20261117 (A100-SXM4 80GB, cuDNN 9.24.0.43, FE dbae6eec5; SM80 decode / MTP / verify classes no study
    # answered, found by re-timing the a100 nightly on the branch: ar_dit d128 MHA MTP, MiMo-V2 SWA verify (d192 x V128, window 128
    # + sink), gemma4 SWA paged d256 (window 1024), d192 x V128 and paged d256 single-row decode; 1580 train cells fitted under the
    # criterion, a 96-cell holdout scored once, plus a 24-cell supplement for the windowed single-row twins). FROST runs 0.28-0.55
    # of the backend on the windowed classes (the nightly ran them on the backend at 2.2-6x); dense d256 single-row decode failed
    # its holdout (1.53) and keeps the hand rules.
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (128, 128), "mtp", "dense", "causal_br", False, "mha", False): Entry(
        "le", 5.792, "sm80_decode_gaps_20261117", "tiles"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (128, 128), "mtp", "dense", "causal_br", False, "mha", True): Entry(
        "le", 5.792, "sm80_decode_gaps_20261117", "tiles"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "q1", "dense", "none", False, "gqa", False): Entry("ge", 23.824, "sm80_decode_gaps_20261117"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "q1", "dense", "none", False, "gqa", True): Entry("ge", 23.325, "sm80_decode_gaps_20261117"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "q1", "paged", "none", False, "gqa", False): Entry("ge", 8.5, "sm80_decode_gaps_20261117", "tiles"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "q1", "paged", "none", False, "gqa", True): Entry("ge", 30.823, "sm80_decode_gaps_20261117"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "verify", "dense", "window", True, "gqa", False): Entry("always", None, "sm80_decode_gaps_20261117"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "verify", "dense", "window", True, "gqa", True): Entry("always", None, "sm80_decode_gaps_20261117"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "verify", "paged", "window", True, "gqa", False): Entry("always", None, "sm80_decode_gaps_20261117"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "verify", "paged", "window", True, "gqa", True): Entry("always", None, "sm80_decode_gaps_20261117"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "mtp", "paged", "window", False, "gqa", False): Entry("always", None, "sm80_decode_gaps_20261117"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "mtp", "paged", "window", False, "gqa", True): Entry("always", None, "sm80_decode_gaps_20261117"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "q1", "paged", "none", False, "gqa", False): Entry("ge", 31.5, "sm80_decode_gaps_20261117"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "q1", "paged", "none", False, "gqa", True): Entry("ge", 31.001, "sm80_decode_gaps_20261117"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "q1", "paged", "window", False, "gqa", False): Entry("le", 6.5, "sm80_decode_gaps_20261117", "tiles"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "q1", "paged", "window", False, "gqa", True): Entry("le", 32.001, "sm80_decode_gaps_20261117"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "verify", "paged", "window", False, "gqa", False): Entry("always", None, "sm80_decode_gaps_20261117"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (256, 256), "verify", "paged", "window", False, "gqa", True): Entry("always", None, "sm80_decode_gaps_20261117"),
    # study sm80_paged_wide_verify_20261117 (A100-SXM4 80GB, cuDNN 9.24.0.43, FE 688f00136; paged speculative decode with wide heads: D576 x V512 and
    # d512 at s_q 5-16, d192 x V128 at s_q 2-16; groups 1-128, b 1-64, KV 8k / 128k on and off the tile; 960 train cells, a 100-cell
    # holdout over other head pairs scored once). FROST runs 0.13-0.20 of the backend; the a100 nightly (run-557) ran these on the
    # backend at 10-53x (kimi_k2 / deepseek_v3 verify, gemma4 global, MiMo-V2).
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "mtp", "paged", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm80_paged_wide_verify_20261117"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "mtp", "paged", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm80_paged_wide_verify_20261117"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "verify", "paged", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm80_paged_wide_verify_20261117"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "verify", "paged", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm80_paged_wide_verify_20261117"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (512, 512), "verify", "paged", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm80_paged_wide_verify_20261117"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (512, 512), "verify", "paged", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm80_paged_wide_verify_20261117"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (576, 512), "verify", "paged", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm80_paged_wide_verify_20261117"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (576, 512), "verify", "paged", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm80_paged_wide_verify_20261117"
    ),
    # study sm80_paged_mtp_sink_20261117 (A100-SXM4 80GB, cuDNN 9.24.0.43, FE 9bdf9a8ab; the same generation graph, paged page 16,
    # d64 with a sink, full and 128-key window, s_q 2-4, b 1-128, KV 4k-128k; 280 train cells, a 60-cell seeded holdout scored once:
    # the four classes moved 47 cells at 0.29-0.41 geomean, worst 1.07). The a100 nightly (run-547) had gpt-oss paged MTP at 3.9-8.4x.
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "mtp", "paged", "causal_br", True, "gqa", False): Entry(
        "ge", 3.5, "sm80_paged_mtp_sink_20261117", "tiles"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "mtp", "paged", "causal_br", True, "gqa", True): Entry(
        "ge", 4.0, "sm80_paged_mtp_sink_20261117", "tiles"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "mtp", "paged", "window", True, "gqa", False): Entry(
        "ge", 2.5, "sm80_paged_mtp_sink_20261117", "tiles"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (64, 64), "mtp", "paged", "window", True, "gqa", True): Entry("ge", 4.0, "sm80_paged_mtp_sink_20261117", "tiles"),
    # study sm80_d192_gqa_r5_20261112's D192/V128 GQA cells at s_q 8-16 now key the verify phase: 96 train cells (s_q 16) and the
    # holdout's s_q <= 16 cells all FROST-first (A100 80GB PCIe, cuDNN 9.29.0.18; the prefill-phase twins of these entries are #2499's)
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "verify", "dense", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm80_d192_gqa_r5_20261112"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "verify", "dense", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm80_d192_gqa_r5_20261112"
    ),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "verify", "dense", "none", False, "gqa", False): Entry("always", None, "sm80_d192_gqa_r5_20261112"),
    ("sdpa_fwd_prefill_sm80", (8, 0), "bf16", (192, 128), "verify", "dense", "none", False, "gqa", True): Entry("always", None, "sm80_d192_gqa_r5_20261112"),
    # study sm90_decode_gaps_20261117 (H100 80GB HBM3, cuDNN 9.24.0.43, FE dbae6eec5; SM90 decode classes no study answered, found by
    # re-timing the h200 nightly on the branch: paged d512 verify (gemma4 global) and paged d256 sliding-window decode / MTP / verify
    # (gemma4 SWA, window 1024); 763 train cells fitted under the criterion, a 48-cell holdout scored once). FROST runs 0.05 of the
    # backend on paged d512 verify (the nightly ran the backend at 32-84x) and 0.31-0.44 on the windowed d256 classes (2.8-3.8x).
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "mtp", "paged", "window", False, "gqa_unpacked", False): Entry(
        "always", None, "sm90_decode_gaps_20261117"
    ),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "mtp", "paged", "window", False, "gqa_unpacked", True): Entry(
        "always", None, "sm90_decode_gaps_20261117"
    ),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "q1", "paged", "window", False, "gqa_unpacked", False): Entry(
        "le", 6.0, "sm90_decode_gaps_20261117", "tiles"
    ),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "q1", "paged", "window", False, "gqa_unpacked", True): Entry(
        "le", 6.0, "sm90_decode_gaps_20261117", "tiles"
    ),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "verify", "paged", "window", False, "gqa", False): Entry("always", None, "sm90_decode_gaps_20261117"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "verify", "paged", "window", False, "gqa", True): Entry("always", None, "sm90_decode_gaps_20261117"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (512, 512), "verify", "paged", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm90_decode_gaps_20261117"
    ),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (512, 512), "verify", "paged", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm90_decode_gaps_20261117"
    ),
    # study sm90_decode_gaps_20261117, supplement (36 fresh cells, scored once): gemma4 SWA's own windowed paged d256 single-row /
    # MTP twins (groups 2 / 4) lead too: 0.32-0.41 of the backend, worst 0.68.
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "mtp", "paged", "window", False, "gqa", False): Entry("always", None, "sm90_decode_gaps_20261117"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "mtp", "paged", "window", False, "gqa", True): Entry("always", None, "sm90_decode_gaps_20261117"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "q1", "paged", "window", False, "gqa", False): Entry("always", None, "sm90_decode_gaps_20261117"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "q1", "paged", "window", False, "gqa", True): Entry("le", 32.001, "sm90_decode_gaps_20261117"),
    # study sm90_decode_20261112 (H200 700 W, cuDNN 9.30.0.4; the same graphs and lattice as sm80_decode_20261112 without d64 /
    # D576, which the SM90 row does not serve, plus d512 absorbed MLA dense and paged page 64; 820 train cells + an 88-cell holdout:
    # mtp / verify 0.01-0.20 geomean, d512 q1 0.25-0.33, d256 paged q1 0.94. Dense verify d128 lost 1.08-1.14x at 40/8 b24-96
    # (group 5) and d256 q1 1.74-1.83x (group 12): no entry for those classes)
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "paged", "causal_br", False, "gqa", False): Entry("always", None, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "paged", "causal_br", False, "gqa", True): Entry("always", None, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "q1", "paged", "none", False, "gqa", False): Entry("never", None, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "q1", "paged", "none", False, "gqa", True): Entry("never", None, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "verify", "paged", "causal_br", False, "gqa", False): Entry("always", None, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "verify", "paged", "causal_br", False, "gqa", True): Entry("always", None, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "mtp", "dense", "causal_br", False, "gqa", False): Entry("always", None, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "mtp", "dense", "causal_br", False, "gqa", True): Entry("always", None, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "mtp", "paged", "causal_br", False, "gqa", False): Entry("always", None, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "mtp", "paged", "causal_br", False, "gqa", True): Entry("always", None, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "q1", "paged", "none", False, "gqa", False): Entry("ge", 27.001, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "q1", "paged", "none", False, "gqa", True): Entry("ge", 2.0, "sm90_decode_20261112", "tiles"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "verify", "dense", "causal_br", False, "gqa", False): Entry("always", None, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "verify", "dense", "causal_br", False, "gqa", True): Entry("always", None, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "verify", "paged", "causal_br", False, "gqa", False): Entry("always", None, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "verify", "paged", "causal_br", False, "gqa", True): Entry("always", None, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (512, 512), "mtp", "dense", "causal_br", False, "gqa", False): Entry("always", None, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (512, 512), "mtp", "dense", "causal_br", False, "gqa", True): Entry("always", None, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (512, 512), "mtp", "paged", "causal_br", False, "gqa", False): Entry("always", None, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (512, 512), "mtp", "paged", "causal_br", False, "gqa", True): Entry("always", None, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (512, 512), "q1", "dense", "none", False, "gqa", False): Entry("ge", 24.5, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (512, 512), "q1", "dense", "none", False, "gqa", True): Entry("ge", 23.505, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (512, 512), "q1", "paged", "none", False, "gqa", False): Entry("ge", 24.5, "sm90_decode_20261112"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (512, 512), "q1", "paged", "none", False, "gqa", True): Entry("always", None, "sm90_decode_20261112"),
    # study sm90_d128_h100_refit_20261114 (1x H100 SXM (Lambda), cuDNN 9.27.0): exact dense d128 bf16 graphs the SM90 D128
    # template serves; 356 training cells from an earlier frozen sweep on the same part, the FROST time the plan this tree
    # ships, qualified on a fresh 136-cell holdout off the training lattice (odd heads and batches, off-tile lengths); legs
    # backend-only, FROST-first and default dispatch, profiler kernel time after an L2 flush. No entry (the fresh holdout failed
    # the generated entry or its twin's); _place_sm90_d128 answers them: mtp/gqa_unpacked/causal_br/off,
    # mtp/gqa_unpacked/causal_br/on, mtp/mha/none/off, mtp/mha/none/on, prefill/gqa/causal_br/off, prefill/gqa/causal_br/on,
    # prefill/gqa/none/off, prefill/gqa/none/on, prefill/mha/causal_br/off, prefill/mha/causal_br/on, prefill/mha/causal_tl/off,
    # prefill/mha/causal_tl/on, prefill/mha/none/off, prefill/mha/none/on, verify/gqa/causal_br/off, verify/gqa/causal_br/on,
    # verify/gqa/none/off, verify/gqa/none/on. Replaces an older study's entries for the same keys, refit here under this
    # tree's plan order: mtp/gqa/causal_br/on, mtp/gqa/causal_br/off.
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "dense", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm90_d128_h100_refit_20261114"
    ),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "dense", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm90_d128_h100_refit_20261114"
    ),
    # study sm90_dense_d256_chunk_v2_20261117 (H100 80GB, cuDNN 9.24.0.43, FE 7c63bf4ff; dense bottom-right d256 chunks, groups 2-16,
    # b 1-4, q 128-2048, KV 4k-131k with an off-tile lattice of its own; 726 train cells fitted under the criterion, a 72-cell holdout
    # of new head pairs scored once: 7 changed at 0.70 / 0.57, worst 1.06). FROST leads while few query tiles are in flight (the h200
    # nightly's qwen3.5 TP4 / TP8 chunks ran the backend at 1.8-4.1x); v1 failed at group 2, outside its train grid.
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "prefill", "dense", "causal_br", False, "gqa", False): Entry(
        "le", 6.085, "sm90_dense_d256_chunk_v2_20261117", "tiles"
    ),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (256, 256), "prefill", "dense", "causal_br", False, "gqa", True): Entry(
        "le", 5.407, "sm90_dense_d256_chunk_v2_20261117", "tiles"
    ),
    # study sm90_mha_mtp_v2_20261117 re-fits the MHA twins (H100 80GB, cuDNN 9.24.0.43, FE ba0b35370; heads 4-64 MHA, b 1-64, s_q 2/4,
    # KV 8k-128k with a full off-tile lattice; 240 train cells, a 60-cell holdout scored once: 34 changed at 0.54-0.59, worst 0.98).
    # FROST splits the cache where the backend does not, so few head-batch units over a long KV lead (h200 ar_dit 9/9 ran 6.4x).
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "dense", "causal_br", False, "mha", False): Entry(
        "le", 6.08, "sm90_mha_mtp_v2_20261117", "tiles"
    ),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "dense", "causal_br", False, "mha", True): Entry(
        "le", 6.29, "sm90_mha_mtp_v2_20261117", "tiles"
    ),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "dense", "none", False, "gqa", False): Entry("always", None, "sm90_d128_h100_refit_20261114"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "dense", "none", False, "gqa", True): Entry("always", None, "sm90_d128_h100_refit_20261114"),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "dense", "none", False, "gqa_unpacked", False): Entry(
        "never", None, "sm90_d128_h100_refit_20261114"
    ),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "mtp", "dense", "none", False, "gqa_unpacked", True): Entry(
        "le", 23.977, "sm90_d128_h100_refit_20261114"
    ),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "prefill", "dense", "causal_tl", False, "gqa", False): Entry(
        "ge", 8.293, "sm90_d128_h100_refit_20261114", "kv"
    ),
    ("sdpa_fwd_prefill_sm90", (9, 0), "bf16", (128, 128), "prefill", "dense", "causal_tl", False, "gqa", True): Entry(
        "always", None, "sm90_d128_h100_refit_20261114"
    ),
    # study sm107_dense_decode_20261031 (cuDNN 9.29.0.18; cc 10.7, 216 SMs; dense bf16 decode and MTP, s_q 1-4: d64 with the gpt-oss
    # masks (sink, window + sink, neither), GQA 4 / 8; d256 without a sink, GQA 4-16; b 1-128, S_kv 3k-128k on and off the tile;
    # 396 train + 52 scored random holdout cells incl. groups 5, 6, 12; d256 single-row decode keeps the hand order: its
    # frontier moves with the group, which one cost threshold per class cannot follow)
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "mtp", "dense", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm107_dense_decode_20261031"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "mtp", "dense", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm107_dense_decode_20261031"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "mtp", "dense", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm107_dense_decode_20261031"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "mtp", "dense", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm107_dense_decode_20261031"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "mtp", "dense", "causal_br", True, "gqa", False): Entry(
        "always", None, "sm107_dense_decode_20261031"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "mtp", "dense", "causal_br", True, "gqa", True): Entry("always", None, "sm107_dense_decode_20261031"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "mtp", "dense", "window", True, "gqa", False): Entry("always", None, "sm107_dense_decode_20261031"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "mtp", "dense", "window", True, "gqa", True): Entry("always", None, "sm107_dense_decode_20261031"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "q1", "dense", "none", False, "gqa", False): Entry("never", None, "sm107_dense_decode_20261031"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "q1", "dense", "none", False, "gqa", True): Entry("never", None, "sm107_dense_decode_20261031"),
    # study sm107_d64_sink_prefill_20261112 (208-SM Rubin, cuDNN 9.30.0.4; benchmark/attention_inference's context graph: dense d64
    # with a sink, bottom-right causal with and without a 128-key window, GQA 8, s_q 128-8192, S_kv 2k-131k on and off the tile,
    # b 1 / 4; 264 train cells (measured default regret 1.52, up to 37.8x) + a 48-cell seeded holdout scored once: 0.61 geomean
    # of the measured default per class, worst 1.08)
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill", "dense", "causal_br", True, "gqa", False): Entry(
        "le", 8.5, "sm107_d64_sink_prefill_20261112", "tiles"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill", "dense", "causal_br", True, "gqa", True): Entry(
        "always", None, "sm107_d64_sink_prefill_20261112"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill", "dense", "window", True, "gqa", False): Entry(
        "le", 8.5, "sm107_d64_sink_prefill_20261112", "tiles"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill", "dense", "window", True, "gqa", True): Entry(
        "always", None, "sm107_d64_sink_prefill_20261112"
    ),
    # study sm107_paged_d64_sink_prefill_20261117 (208-SM Rubin, cuDNN 9.30.0.4, FE 235172717; benchmark/attention_inference's paged
    # context graph (THD, exact total-Q, page 16): gpt-oss d64 with a sink, bottom-right causal, groups 3-8, b 1-8, S_q 128-4096, KV
    # 4k-128k on and off the tile; 100 train cells, measured default regret 1.28 / 1.45). Changed train cells 0.707 / 0.690; a 32-cell
    # holdout at fresh heads, scored once, 0.660 / 0.618 (worst 0.975); a supplement at S_q 5-96, 0.388 / 0.400 (worst 0.542). The
    # 128-key window twin has no entry: its on-tile rule led S_q 5 verify graphs at 1.7-2.0x in the supplement.
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill", "paged", "causal_br", True, "gqa", False): Entry(
        "le", 9.792, "sm107_paged_d64_sink_prefill_20261117", "tiles"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (64, 64), "prefill", "paged", "causal_br", True, "gqa", True): Entry(
        "always", None, "sm107_paged_d64_sink_prefill_20261117"
    ),
    # study sm120_decode_20261112 (RTX PRO 6000 Server, cuDNN 9.30.0.4; benchmark/attention_inference's generation graph, s_q 1-4:
    # paged d128 / d256 GQA page 16, d512 / D576xV512 absorbed MLA dense and paged, d64 sink (and window 128) paged MTP; b 1-64, S_kv
    # 8k-131k on and off the tile; 333 train cells + a seeded holdout scored once per class: 0.34-0.67 geomean of the measured default,
    # worst 0.65. The sink-only d64 MTP class lost 1.21x on the holdout and dense d512 s_q 1 stays with the #2497 rule: no entry.
    # Its paged d512 and D576 s_q 1 entries are retired: the group, which the key cannot carry, decides there
    # (sm120_paged_d512_q1_v2_20261117 and the D576 transfer check sm120_paged_d576_q1_xfer_20261117; fwd/placement._place_sm120_f16))
    # Its d128 paged MTP entries (sm120_paged_d128_mtp_20261117, `le 5.5 tiles`) are retired the same way: FROST's time is flat in
    # the GQA group and the backend's grows with it, so the group decides (sm120_paged_d128_mtp_v6_20261117; fwd/placement._place_sm120_f16).
    # study sm120_paged_d128_body_20261117 (RTX PRO 6000 Server, cuDNN 9.24.0.43, FE a09a291e7; paged d128 at s_q 2-16 past 32
    # packed rows, which the decode tile declines (`gqa_body`), causal and windowed; groups 2-16, b 1-64, KV 4k-131k on and off the
    # tile; 480 train cells fitted under the criterion, a 70-cell holdout scored once: 37 changed at 0.39-0.81, worst 0.985). The
    # prefill body runs 0.32-0.67 of the backend from about 2^9-2^11 query rows in flight and 1.5x behind it at b 1, so ripple /
    # laguna / step37 MTP and verify leave the backend (rtxpro6000 run-553, 1.9-2.7x) while one-request MTP keeps it.
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (128, 128), "mtp", "paged", "causal_br", False, "gqa_body", False): Entry(
        "ge", 8.792, "sm120_paged_d128_body_20261117", "rows"
    ),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (128, 128), "mtp", "paged", "causal_br", False, "gqa_body", True): Entry(
        "ge", 10.161, "sm120_paged_d128_body_20261117", "rows"
    ),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (128, 128), "mtp", "paged", "window", False, "gqa_body", False): Entry(
        "ge", 8.792, "sm120_paged_d128_body_20261117", "rows"
    ),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (128, 128), "mtp", "paged", "window", False, "gqa_body", True): Entry(
        "ge", 10.161, "sm120_paged_d128_body_20261117", "rows"
    ),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (128, 128), "prefill", "paged", "causal_br", False, "gqa_body", False): Entry(
        "ge", 10.792, "sm120_paged_d128_body_20261117", "rows"
    ),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (128, 128), "prefill", "paged", "causal_br", False, "gqa_body", True): Entry(
        "ge", 4.5, "sm120_paged_d128_body_20261117", "tiles"
    ),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (128, 128), "prefill", "paged", "window", False, "gqa_body", False): Entry(
        "ge", 4.5, "sm120_paged_d128_body_20261117", "tiles"
    ),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (128, 128), "prefill", "paged", "window", False, "gqa_body", True): Entry(
        "ge", 4.5, "sm120_paged_d128_body_20261117", "tiles"
    ),
    # study sm120_paged_window_decode_20261117 (RTX PRO 6000 Server, cuDNN 9.24.0.43, FE b9eb6fead; paged sliding-window decode at s_q 1,
    # d64 / d128 / d256, groups 2-16, windows 256-4096, KV 4k-131k on and off the tile; 256 train cells fitted under the criterion, a
    # 60-cell holdout scored once: 25 changed at 0.20-0.63, worst 1.00). The decode split clips KV to the window (c59994a55), so FROST
    # leads windowed decode the hand rule trails at d64 / d128 (gemma4 31B SWA at TP4 ran the backend at 3.7x; muse_spark SWA at 2.3-4.4x).
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (128, 128), "q1", "paged", "window", False, "gqa", False): Entry(
        "le", 5.5, "sm120_paged_window_decode_20261117", "tiles"
    ),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (128, 128), "q1", "paged", "window", False, "gqa", True): Entry(
        "le", 4.5, "sm120_paged_window_decode_20261117", "tiles"
    ),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (256, 256), "q1", "paged", "window", False, "gqa", False): Entry(
        "always", None, "sm120_paged_window_decode_20261117"
    ),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (256, 256), "q1", "paged", "window", False, "gqa", True): Entry(
        "always", None, "sm120_paged_window_decode_20261117"
    ),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (64, 64), "q1", "paged", "window", False, "gqa", False): Entry(
        "le", 5.5, "sm120_paged_window_decode_20261117", "tiles"
    ),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (64, 64), "q1", "paged", "window", False, "gqa", True): Entry(
        "le", 6.5, "sm120_paged_window_decode_20261117", "tiles"
    ),
    # study sm120_paged_window_decode_d192_v2_20261117 (RTX PRO 6000 Server, cuDNN 9.24.0.43, FE 9d464839d; paged sliding-window decode at s_q 1,
    # d192 x V128, groups 2-16 including ones that do not divide the decode tile, windows 512-4096, KV 8k-128k on and off the tile;
    # 417 train cells plus v1's 197, fitted under the criterion; a 70-cell holdout over groups 4-14 scored once: 45 changed at
    # 0.31-0.39, worst 1.07). v1 failed at group 12 on a train grid of groups 8 / 16. MiMo-V2's SWA decode (this geometry) carries a
    # sink, another class; step5 SWA is d192 x 192.
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (192, 128), "q1", "paged", "window", False, "gqa", False): Entry(
        "le", 6.5, "sm120_paged_window_decode_d192_v2_20261117", "tiles"
    ),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (192, 128), "q1", "paged", "window", False, "gqa", True): Entry(
        "le", 6.5, "sm120_paged_window_decode_d192_v2_20261117", "tiles"
    ),
    # study sm120_window_prefill_20261117 (RTX PRO 6000 Server, cuDNN 9.24.0.43, FE c59994a55; dense bottom-right chunks and squares,
    # d256 window 1024 and d512 MQA window 128; 221 train cells, a served-pair and an other-group holdout, each scored once: 0.34-0.58
    # geomean, worst 0.70). The hand rule trailed every windowed square; DSv4 d512 SWA squares ran 0.31-0.34 of the backend. The d256
    # twin is retired: an edge supplement at S_q 5-193 (edge_supp_20261117) lost 1.125 off the tile on a b 16 square of 33; the d512
    # twin passed it (0.484, worst 0.792). The d256 class is re-studied over its whole S_q range (sm120_window_d256_v2_20261117).
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (512, 512), "prefill", "dense", "window", False, "gqa", False): Entry(
        "always", None, "sm120_window_prefill_20261117"
    ),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (512, 512), "prefill", "dense", "window", False, "gqa", True): Entry(
        "always", None, "sm120_window_prefill_20261117"
    ),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (128, 128), "q1", "paged", "none", False, "gqa", False): Entry("never", None, "sm120_decode_20261112"),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (128, 128), "q1", "paged", "none", False, "gqa", True): Entry("never", None, "sm120_decode_20261112"),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (256, 256), "q1", "paged", "none", False, "gqa", False): Entry("never", None, "sm120_decode_20261112"),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (256, 256), "q1", "paged", "none", False, "gqa", True): Entry("never", None, "sm120_decode_20261112"),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (576, 512), "q1", "dense", "none", False, "gqa", False): Entry("always", None, "sm120_decode_20261112"),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (576, 512), "q1", "dense", "none", False, "gqa", True): Entry("always", None, "sm120_decode_20261112"),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (64, 64), "mtp", "paged", "window", True, "gqa", False): Entry("ge", 5.0, "sm120_decode_20261112", "tiles"),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (64, 64), "mtp", "paged", "window", True, "gqa", True): Entry("ge", 28.501, "sm120_decode_20261112"),
    # study sm107_decode_q1_20261112 (208-SM Rubin, cuDNN 9.30.0.4; benchmark/attention_inference's generation graph, s_q == 1:
    # dense d256 GQA 4-16 and paged d512 / D576xV512 absorbed MLA over one KV head, page 64 / 32; b 1-128, S_kv 2k-131k on and off
    # the tile; 309 train cells + a 48-cell seeded holdout scored once per class (new / measured default 0.20-0.48 geomean, worst
    # 0.64); the d512 / D576 entries also cover h_kv 2, checked separately. d128 is not included: no holdout cell moved)
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "q1", "dense", "none", False, "gqa", False): Entry("ge", 25.506, "sm107_decode_q1_20261112"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "q1", "dense", "none", False, "gqa", True): Entry("ge", 5.5, "sm107_decode_q1_20261112", "tiles"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (512, 512), "q1", "paged", "none", False, "gqa", False): Entry("always", None, "sm107_decode_q1_20261112"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (512, 512), "q1", "paged", "none", False, "gqa", True): Entry("always", None, "sm107_decode_q1_20261112"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (576, 512), "q1", "paged", "none", False, "gqa", False): Entry("always", None, "sm107_decode_q1_20261112"),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (576, 512), "q1", "paged", "none", False, "gqa", True): Entry("always", None, "sm107_decode_q1_20261112"),
    # study sm107_dense_chunk_20261117 (Rubin dense d256 chunks, trained at S_q >= 256): its two `always` entries are retired. The
    # class starts at S_q 5, and an edge supplement there (edge_supp_20261117) lost 1.58x off the tile on a b 8 square of 64. The
    # class is re-studied over its whole S_q range (sm107_dense_d256_v2_20261117, below).
    # study sm107_dense_d256_v2_20261117 (208-SM Rubin, cuDNN 9.30.0.4, FE 65ce81615; benchmark/attention_inference's context graph:
    # dense d256 bottom-right chunks S_q 5-2048 to KV 128k and squares 32-2048, on and off the tile, groups 5-16, b 1-16; 200 train
    # cells). Lead from KV 2^11.5 (changed train 0.546 / 0.408, worst 1.050); holdout at fresh heads, scored once, 0.583 / 0.529
    # (worst 1.099).
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "prefill", "dense", "causal_br", False, "gqa", False): Entry(
        "ge", 11.5, "sm107_dense_d256_v2_20261117", "kv"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (256, 256), "prefill", "dense", "causal_br", False, "gqa", True): Entry(
        "ge", 11.501, "sm107_dense_d256_v2_20261117", "kv"
    ),
    # study sm100_d576_prefill_20261117, Rubin part (208-SM Rubin, cuDNN 9.30.0.4, FE 8014a2c75): the B200 study's dense cells
    # (160 train). FROST leads everywhere: changed train cells 0.115 (worst 0.310); holdout 0.091-0.104 (worst 0.251). Paged
    # D576 prefill has no FROST plan on Rubin and no entry. Only h_kv 1 was measured.
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (576, 512), "prefill", "dense", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm100_d576_prefill_20261117"
    ),
    ("sdpa_fwd_prefill_sm107", (10, 7), "bf16", (576, 512), "prefill", "dense", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm100_d576_prefill_20261117"
    ),
    # study sm120_window_sink_prefill_20261117 (dense window-128 sink d192 x 128, trained at S_q >= 256): its two entries are retired.
    # An edge supplement at S_q 5-192 lost up to 2.94x (on the tile, S_q 5) and 1.75x (off it) against the hand rule's lead.
    # study sm120_window_d256_v2_20261117 (RTX PRO 6000 Server, cuDNN 9.24.0.43, FE 65ce81615; benchmark/attention_inference's context
    # graph: dense d256 window 1024, bottom-right chunks S_q 5-2048 to KV 128k and squares 32-2048, on and off the tile, groups 2-5,
    # b 1-16; 200 train cells). Lead from KV 2^7.5 (changed train 0.563 / 0.650, worst 0.919); holdout at fresh heads, scored once,
    # 0.634 / 0.585 (worst 0.806).
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (256, 256), "prefill", "dense", "window", False, "gqa", False): Entry(
        "ge", 7.5, "sm120_window_d256_v2_20261117", "kv"
    ),
    ("sdpa_fwd_prefill_sm120", (12, 0), "bf16", (256, 256), "prefill", "dense", "window", False, "gqa", True): Entry(
        "ge", 7.508, "sm120_window_d256_v2_20261117", "kv"
    ),
    # study sm100_swa_prefill_20261031 (cuDNN 9.29.0.18; B100 850 W, cc 10.0; the sm103 study's grid and seeds: 320 train + 64 random
    # holdout cells; 90 training cells timed while another process held the GPU were re-timed, 32 of them moved > 5%)
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (128, 128), "prefill", "dense", "window", False, "gqa", False): Entry(
        "always", None, "sm100_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (128, 128), "prefill", "dense", "window", False, "gqa", True): Entry(
        "always", None, "sm100_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (128, 128), "prefill", "paged", "window", False, "gqa", False): Entry(
        "always", None, "sm100_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (128, 128), "prefill", "paged", "window", False, "gqa", True): Entry(
        "always", None, "sm100_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (256, 256), "prefill", "dense", "window", False, "gqa", False): Entry("le", 41.0, "sm100_swa_prefill_20261031"),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (256, 256), "prefill", "dense", "window", False, "gqa", True): Entry(
        "always", None, "sm100_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (256, 256), "prefill", "paged", "window", False, "gqa", False): Entry(
        "always", None, "sm100_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (256, 256), "prefill", "paged", "window", False, "gqa", True): Entry(
        "always", None, "sm100_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (64, 64), "prefill", "dense", "window", False, "gqa", False): Entry(
        "always", None, "sm100_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (64, 64), "prefill", "dense", "window", False, "gqa", True): Entry(
        "always", None, "sm100_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (64, 64), "prefill", "dense", "window", True, "gqa", False): Entry(
        "always", None, "sm100_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (64, 64), "prefill", "dense", "window", True, "gqa", True): Entry("always", None, "sm100_swa_prefill_20261031"),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (64, 64), "prefill", "paged", "window", False, "gqa", False): Entry(
        "always", None, "sm100_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (64, 64), "prefill", "paged", "window", False, "gqa", True): Entry(
        "always", None, "sm100_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (64, 64), "prefill", "paged", "window", True, "gqa", False): Entry(
        "always", None, "sm100_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 0), "bf16", (64, 64), "prefill", "paged", "window", True, "gqa", True): Entry("always", None, "sm100_swa_prefill_20261031"),
    # study sm103_swa_prefill_20261031 (cuDNN 9.29.0.18; B300; sliding-window context, dense and paged, d64 with and without
    # a sink, d128 / d256 without, windows 128 / 4096 in training; 320 train + 64 random + 32 on-tile holdout cells)
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (128, 128), "prefill", "dense", "window", False, "gqa", False): Entry(
        "always", None, "sm103_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (128, 128), "prefill", "dense", "window", False, "gqa", True): Entry(
        "always", None, "sm103_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (128, 128), "prefill", "paged", "window", False, "gqa", False): Entry(
        "always", None, "sm103_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (128, 128), "prefill", "paged", "window", False, "gqa", True): Entry(
        "always", None, "sm103_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (256, 256), "prefill", "dense", "window", False, "gqa", False): Entry("le", 41.0, "sm103_swa_prefill_20261031"),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (256, 256), "prefill", "dense", "window", False, "gqa", True): Entry(
        "always", None, "sm103_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (256, 256), "prefill", "paged", "window", False, "gqa", False): Entry(
        "always", None, "sm103_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (256, 256), "prefill", "paged", "window", False, "gqa", True): Entry(
        "always", None, "sm103_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (64, 64), "prefill", "dense", "window", False, "gqa", False): Entry(
        "always", None, "sm103_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (64, 64), "prefill", "dense", "window", False, "gqa", True): Entry(
        "always", None, "sm103_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (64, 64), "prefill", "dense", "window", True, "gqa", False): Entry(
        "always", None, "sm103_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (64, 64), "prefill", "dense", "window", True, "gqa", True): Entry("always", None, "sm103_swa_prefill_20261031"),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (64, 64), "prefill", "paged", "window", False, "gqa", False): Entry(
        "always", None, "sm103_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (64, 64), "prefill", "paged", "window", False, "gqa", True): Entry(
        "always", None, "sm103_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (64, 64), "prefill", "paged", "window", True, "gqa", False): Entry(
        "always", None, "sm103_swa_prefill_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (64, 64), "prefill", "paged", "window", True, "gqa", True): Entry("always", None, "sm103_swa_prefill_20261031"),
    # study sm103_d256_bwd_v2_20261031, THD half (B300 SXM6 1100 W, cuDNN 9.28.0.38; 129 train cases: S 128-32k, b 1-128, GQA 16/2 /
    # 32/4 / 32/1 / 64/2 and MHA 16/16; 24 off-lattice holdout cases). The dense classes have no entry: they lost on small-work
    # holdout cases there, and their refit (study sm103_d256_bwd_v3_20261031) lost again on its own holdout.
    ("sdpa_bwd_sm100_d256", (10, 3), "bf16", (256, 256), "prefill", "thd", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm103_d256_bwd_v2_20261031"
    ),
    ("sdpa_bwd_sm100_d256", (10, 3), "bf16", (256, 256), "prefill", "thd", "causal_br", False, "mha", False): Entry("ge", 28.0, "sm103_d256_bwd_v2_20261031"),
    ("sdpa_bwd_sm100_d256", (10, 3), "bf16", (256, 256), "prefill", "thd", "none", False, "gqa", False): Entry("ge", 28.5, "sm103_d256_bwd_v2_20261031"),
    ("sdpa_bwd_sm100_d256", (10, 3), "bf16", (256, 256), "prefill", "thd", "none", False, "mha", False): Entry("ge", 28.5, "sm103_d256_bwd_v2_20261031"),
    ("sdpa_bwd_sm100_d256", (10, 3), "bf16", (256, 256), "prefill", "thd_ragged", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm103_d256_bwd_v2_20261031"
    ),
    ("sdpa_bwd_sm100_d256", (10, 3), "bf16", (256, 256), "prefill", "thd_ragged", "causal_br", False, "mha", False): Entry(
        "ge", 28.5, "sm103_d256_bwd_v2_20261031"
    ),
    ("sdpa_bwd_sm100_d256", (10, 3), "bf16", (256, 256), "prefill", "thd_ragged", "none", False, "gqa", False): Entry(
        "always", None, "sm103_d256_bwd_v2_20261031"
    ),
    ("sdpa_bwd_sm100_d256", (10, 3), "bf16", (256, 256), "prefill", "thd_ragged", "none", False, "mha", False): Entry("ge", 28.5, "sm103_d256_bwd_v2_20261031"),
    # study sm103_d256_bwd_thd_offtile_20261031, the off-tile twins of the classes above (B300 SXM6 1100 W, cuDNN 9.28.0.38;
    # 80 train cases: S 200-30000 off the 128-key tile, b 1-164, GQA 16/2 / 8/1 and MHA 16/16 / 6/6; 24 off-lattice holdout cases)
    ("sdpa_bwd_sm100_d256", (10, 3), "bf16", (256, 256), "prefill", "thd", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm103_d256_bwd_thd_offtile_20261031"
    ),
    ("sdpa_bwd_sm100_d256", (10, 3), "bf16", (256, 256), "prefill", "thd", "causal_br", False, "mha", True): Entry(
        "always", None, "sm103_d256_bwd_thd_offtile_20261031"
    ),
    ("sdpa_bwd_sm100_d256", (10, 3), "bf16", (256, 256), "prefill", "thd", "none", False, "gqa", True): Entry(
        "always", None, "sm103_d256_bwd_thd_offtile_20261031"
    ),
    ("sdpa_bwd_sm100_d256", (10, 3), "bf16", (256, 256), "prefill", "thd", "none", False, "mha", True): Entry(
        "ge", 30.374, "sm103_d256_bwd_thd_offtile_20261031"
    ),
    ("sdpa_bwd_sm100_d256", (10, 3), "bf16", (256, 256), "prefill", "thd_ragged", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm103_d256_bwd_thd_offtile_20261031"
    ),
    ("sdpa_bwd_sm100_d256", (10, 3), "bf16", (256, 256), "prefill", "thd_ragged", "causal_br", False, "mha", True): Entry(
        "always", None, "sm103_d256_bwd_thd_offtile_20261031"
    ),
    ("sdpa_bwd_sm100_d256", (10, 3), "bf16", (256, 256), "prefill", "thd_ragged", "none", False, "gqa", True): Entry(
        "always", None, "sm103_d256_bwd_thd_offtile_20261031"
    ),
    ("sdpa_bwd_sm100_d256", (10, 3), "bf16", (256, 256), "prefill", "thd_ragged", "none", False, "mha", True): Entry(
        "ge", 30.374, "sm103_d256_bwd_thd_offtile_20261031"
    ),
    # study sm103_d128_fwd_chunk_20261031, GQA half (B300 SXM6 1100 W, cuDNN 9.28.0.38; dense K/V, bottom-right causal chunks
    # s_q < s_kv: 84 train cases, s_q 128-2048 over S_kv 2048-32768 on the 128-key tile and 3000 / 10000 off it, b 1 / 4, GQA 32/8 /
    # 16/2 / 64/8; 12 off-lattice holdout cases). FROST ran 0.076-0.994 of the backend's time on every one, so past
    # fwd/placement's CHUNKED_MAX_Q_TILES / CHUNKED_MIN_KV_TOKENS bounds (B200) as well. Its MHA classes have no entry: on the
    # tile MHA 16/16 lost a holdout case (1.78x at b 3, s_q 768, S_kv 16384).
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (128, 128), "prefill", "dense", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm103_d128_fwd_chunk_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (128, 128), "prefill", "dense", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm103_d128_fwd_chunk_20261031"
    ),
    # study sm103_d128_fwd_dense_20261031, causal GQA half (B300 SXM6 1100 W, cuDNN 9.28.0.38; dense BSHD squares, 39 train
    # cases: S 512-16384 on the 128-key tile and 700-12000 off it, b 1-64, GQA 32/8 / 16/2 / 64/4; 8 off-lattice holdout cases).
    # Its mask-free classes lost on short-S large-batch holdout cases (S 768: 1.34x GQA 48/6, 1.24x MHA 20/20) and its MHA
    # classes are thin and non-uniform: no entry, so those graphs keep fwd/placement's hand rules.
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (128, 128), "prefill", "dense", "causal_tl", False, "gqa", False): Entry(
        "always", None, "sm103_d128_fwd_dense_20261031"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (128, 128), "prefill", "dense", "causal_tl", False, "gqa", True): Entry(
        "always", None, "sm103_d128_fwd_dense_20261031"
    ),
    # study sm103_dense_prefill_bf16_20261107 (B300 SXM6 1100 W, cuDNN 9.27.0; dense BSHD prefill-phase graphs, s_q >= 5, no sink: d64 / d128 / d192 x no mask /
    # causal / causal + left window x MHA / GQA x S_kv on and off the 128-key tile, less the classes already keyed on cc 10.3; 360 train + 120 off-lattice holdout
    # cases). Only the classes gen.py --margin 0.15 fits as always on and off the tile: at 12 train cases per class thresholds did not hold under
    # cross-validation, so the rest keep the hand rules. The f16 twin is not keyed: its fresh B300 holdout (2026-11-08) had three short graphs
    # beyond noise, so f16 keeps the hand rules.
    # Withdraw both d192 window GQA classes after the 2026-11-09 retime confirmed the q19/kv45248 holdout regression
    # (PR #2346). The other 12 generated entries are unchanged; withdrawn classes retain the prior default route.
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (192, 192), "prefill", "dense", "causal_br", False, "gqa", False): Entry(
        "always", None, "sm103_dense_prefill_bf16_20261107"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (192, 192), "prefill", "dense", "causal_br", False, "gqa", True): Entry(
        "always", None, "sm103_dense_prefill_bf16_20261107"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (192, 192), "prefill", "dense", "causal_br", False, "mha", False): Entry(
        "always", None, "sm103_dense_prefill_bf16_20261107"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (192, 192), "prefill", "dense", "causal_br", False, "mha", True): Entry(
        "always", None, "sm103_dense_prefill_bf16_20261107"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (192, 192), "prefill", "dense", "none", False, "gqa", False): Entry(
        "always", None, "sm103_dense_prefill_bf16_20261107"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (192, 192), "prefill", "dense", "none", False, "gqa", True): Entry(
        "always", None, "sm103_dense_prefill_bf16_20261107"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (192, 192), "prefill", "dense", "none", False, "mha", False): Entry(
        "always", None, "sm103_dense_prefill_bf16_20261107"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (192, 192), "prefill", "dense", "none", False, "mha", True): Entry(
        "always", None, "sm103_dense_prefill_bf16_20261107"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (192, 192), "prefill", "dense", "window", False, "mha", False): Entry(
        "always", None, "sm103_dense_prefill_bf16_20261107"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (192, 192), "prefill", "dense", "window", False, "mha", True): Entry(
        "always", None, "sm103_dense_prefill_bf16_20261107"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (64, 64), "prefill", "dense", "window", False, "mha", False): Entry(
        "always", None, "sm103_dense_prefill_bf16_20261107"
    ),
    ("sdpa_fwd_prefill_sm100", (10, 3), "bf16", (64, 64), "prefill", "dense", "window", False, "mha", True): Entry(
        "always", None, "sm103_dense_prefill_bf16_20261107"
    ),
    # study sm121_bwd_bf16_20261107 (GB10, cuDNN 9.27.0.42; dense BSHD squares, 288 train cases: d64 / d128, no mask / causal / causal + left window,
    # MHA / GQA, S on and off the 128-key tile, 128-16k; 96 off-lattice holdout cases: FROST / backend geomean 0.75, 0.47-0.98).
    ("sdpa_bwd_sm120", (12, 1), "bf16", (128, 128), "prefill", "dense", "causal_br", False, "gqa", False): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (128, 128), "prefill", "dense", "causal_br", False, "gqa", True): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (128, 128), "prefill", "dense", "causal_br", False, "mha", False): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (128, 128), "prefill", "dense", "causal_br", False, "mha", True): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (128, 128), "prefill", "dense", "none", False, "gqa", False): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (128, 128), "prefill", "dense", "none", False, "gqa", True): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (128, 128), "prefill", "dense", "none", False, "mha", False): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (128, 128), "prefill", "dense", "none", False, "mha", True): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (128, 128), "prefill", "dense", "window", False, "gqa", False): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (128, 128), "prefill", "dense", "window", False, "gqa", True): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (128, 128), "prefill", "dense", "window", False, "mha", False): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (128, 128), "prefill", "dense", "window", False, "mha", True): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (64, 64), "prefill", "dense", "causal_br", False, "gqa", False): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (64, 64), "prefill", "dense", "causal_br", False, "gqa", True): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (64, 64), "prefill", "dense", "causal_br", False, "mha", False): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (64, 64), "prefill", "dense", "causal_br", False, "mha", True): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (64, 64), "prefill", "dense", "none", False, "gqa", False): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (64, 64), "prefill", "dense", "none", False, "gqa", True): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (64, 64), "prefill", "dense", "none", False, "mha", False): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (64, 64), "prefill", "dense", "none", False, "mha", True): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (64, 64), "prefill", "dense", "window", False, "gqa", False): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (64, 64), "prefill", "dense", "window", False, "gqa", True): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (64, 64), "prefill", "dense", "window", False, "mha", False): Entry("always", None, "sm121_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "bf16", (64, 64), "prefill", "dense", "window", False, "mha", True): Entry("always", None, "sm121_bwd_bf16_20261107"),
    # study sm121_bwd_fp16_20261107 (GB10, cuDNN 9.27.0.42; dense BSHD squares, 288 train cases: d64 / d128, no mask / causal / causal + left window,
    # MHA / GQA, S on and off the 128-key tile, 128-16k; 96 off-lattice holdout cases: FROST / backend geomean 0.74, 0.37-1.00).
    ("sdpa_bwd_sm120", (12, 1), "f16", (128, 128), "prefill", "dense", "causal_br", False, "gqa", False): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (128, 128), "prefill", "dense", "causal_br", False, "gqa", True): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (128, 128), "prefill", "dense", "causal_br", False, "mha", False): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (128, 128), "prefill", "dense", "causal_br", False, "mha", True): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (128, 128), "prefill", "dense", "none", False, "gqa", False): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (128, 128), "prefill", "dense", "none", False, "gqa", True): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (128, 128), "prefill", "dense", "none", False, "mha", False): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (128, 128), "prefill", "dense", "none", False, "mha", True): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (128, 128), "prefill", "dense", "window", False, "gqa", False): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (128, 128), "prefill", "dense", "window", False, "gqa", True): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (128, 128), "prefill", "dense", "window", False, "mha", False): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (128, 128), "prefill", "dense", "window", False, "mha", True): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (64, 64), "prefill", "dense", "causal_br", False, "gqa", False): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (64, 64), "prefill", "dense", "causal_br", False, "gqa", True): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (64, 64), "prefill", "dense", "causal_br", False, "mha", False): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (64, 64), "prefill", "dense", "causal_br", False, "mha", True): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (64, 64), "prefill", "dense", "none", False, "gqa", False): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (64, 64), "prefill", "dense", "none", False, "gqa", True): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (64, 64), "prefill", "dense", "none", False, "mha", False): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (64, 64), "prefill", "dense", "none", False, "mha", True): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (64, 64), "prefill", "dense", "window", False, "gqa", False): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (64, 64), "prefill", "dense", "window", False, "gqa", True): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (64, 64), "prefill", "dense", "window", False, "mha", False): Entry("always", None, "sm121_bwd_fp16_20261107"),
    ("sdpa_bwd_sm120", (12, 1), "f16", (64, 64), "prefill", "dense", "window", False, "mha", True): Entry("always", None, "sm121_bwd_fp16_20261107"),
    # study sm120_bwd_{bf16,f16}_20261107 (RTX 5090, cuDNN 9.27.0; dense BSHD backward, s_q == s_kv, 288 train cases per dtype: d64 / d128, mask-free / causal /
    # causal + left window, MHA / GQA, S on and off the 128-key tile). gen.py --margin 0.05 on train, keeping only `always` classes whose on/off-tile twin is also
    # `always`. The key is cc 12.0, so these also apply to other cc 12.0 parts; only the RTX 5090 was timed.
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill", "dense", "causal_br", False, "gqa", False): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill", "dense", "causal_br", False, "gqa", True): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill", "dense", "causal_br", False, "mha", False): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill", "dense", "causal_br", False, "mha", True): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill", "dense", "none", False, "gqa", False): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill", "dense", "none", False, "gqa", True): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill", "dense", "none", False, "mha", False): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill", "dense", "none", False, "mha", True): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill", "dense", "window", False, "gqa", False): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill", "dense", "window", False, "gqa", True): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill", "dense", "window", False, "mha", False): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill", "dense", "window", False, "mha", True): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill", "dense", "causal_br", False, "gqa", False): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill", "dense", "causal_br", False, "gqa", True): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill", "dense", "causal_br", False, "mha", False): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill", "dense", "causal_br", False, "mha", True): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill", "dense", "none", False, "gqa", False): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill", "dense", "none", False, "gqa", True): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill", "dense", "none", False, "mha", False): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill", "dense", "none", False, "mha", True): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill", "dense", "window", False, "gqa", False): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill", "dense", "window", False, "gqa", True): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill", "dense", "window", False, "mha", False): Entry("always", None, "sm120_bwd_bf16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill", "dense", "window", False, "mha", True): Entry("always", None, "sm120_bwd_bf16_20261107"),
    # study sm80_det_bwd_20261117 (A100-SXM4 80GB, cuDNN 9.24.0.43, FE ba0b35370; deterministic backward, dense BSHD squares, d64 / d128,
    # MHA and GQA groups 3-8, no mask / causal / 128- and 1024-key windows, S 1k-8k; 192 train cells and a 48-cell holdout scored
    # once: 19 changed at 0.17-0.64, worst 1.02). On A100 FROST's deterministic body leads only windowed graphs from S 4096 (d128
    # MHA on cost); every unwindowed class measured `never` and keeps the hand rule. gpt-oss d64 SWA backward ran engine 7 at 9.1x.
    ("sdpa_bwd_sm80", (8, 0), "bf16", (64, 64), "prefill_det", "dense", "window", False, "gqa", False): Entry("ge", 11.5, "sm80_det_bwd_20261117", "kv"),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (64, 64), "prefill_det", "dense", "window", False, "gqa", True): Entry("ge", 11.5, "sm80_det_bwd_20261117", "kv"),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (128, 128), "prefill_det", "dense", "window", False, "gqa", False): Entry("ge", 11.5, "sm80_det_bwd_20261117", "kv"),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (128, 128), "prefill_det", "dense", "window", False, "gqa", True): Entry("ge", 11.5, "sm80_det_bwd_20261117", "kv"),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (128, 128), "prefill_det", "dense", "window", False, "mha", False): Entry("ge", 34.5, "sm80_det_bwd_20261117"),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (128, 128), "prefill_det", "dense", "window", False, "mha", True): Entry("ge", 34.5, "sm80_det_bwd_20261117"),
    # study sm80_det_bwd_d64mha_v3_20261117 (A100, cuDNN 9.24.0.43, FE 1d8a0bfeb; gpt-oss's d64 windowed MHA, heads 16-128, windows
    # 128 / 1024, S 1k-9k with its own off-tile lattice; 112 train cells, a 36-cell holdout scored once: 14 changed at 0.26 / 0.35,
    # worst 0.69). The backend's deterministic engine 7 is timed twice per cell and reads 0.63-1.60x apart, so both the fit and the
    # criterion use the mean of the two timings.
    ("sdpa_bwd_sm80", (8, 0), "bf16", (64, 64), "prefill_det", "dense", "window", False, "mha", False): Entry(
        "ge", 11.79, "sm80_det_bwd_d64mha_v3_20261117", "kv"
    ),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (64, 64), "prefill_det", "dense", "window", False, "mha", True): Entry(
        "ge", 11.92, "sm80_det_bwd_d64mha_v3_20261117", "kv"
    ),
    # study sm80_det_bwd_narrow_window_20261117 (A100, cuDNN 9.24.0.43, FE 7c63bf4ff; deterministic backward at windows of up to ~384
    # keys, d64 / d128, MHA and GQA; the earlier A100 studies' narrow-window cells plus 128 short-S cells (S 256-1024) as train, fitted
    # under the criterion; a 56-cell holdout scored once). Narrow windows lead from S ~900 (the `window` class from S ~3k), so
    # gpt-oss's d64 SWA backward at S 2048 leaves engine 7 (2.61x on the a100 nightly).
    ("sdpa_bwd_sm80", (8, 0), "bf16", (128, 128), "prefill_det", "dense", "window_narrow", False, "gqa", False): Entry(
        "ge", 9.792, "sm80_det_bwd_narrow_window_20261117", "kv"
    ),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (128, 128), "prefill_det", "dense", "window_narrow", False, "gqa", True): Entry(
        "ge", 10.682, "sm80_det_bwd_narrow_window_20261117", "kv"
    ),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (64, 64), "prefill_det", "dense", "window_narrow", False, "gqa", False): Entry(
        "ge", 9.792, "sm80_det_bwd_narrow_window_20261117", "kv"
    ),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (64, 64), "prefill_det", "dense", "window_narrow", False, "gqa", True): Entry(
        "ge", 32.449, "sm80_det_bwd_narrow_window_20261117", "cost"
    ),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (64, 64), "prefill_det", "dense", "window_narrow", False, "mha", False): Entry(
        "ge", 9.792, "sm80_det_bwd_narrow_window_20261117", "kv"
    ),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (64, 64), "prefill_det", "dense", "window_narrow", False, "mha", True): Entry(
        "ge", 9.521, "sm80_det_bwd_narrow_window_20261117", "kv"
    ),
    # d128 MHA narrow windows failed the narrow study's holdout (1.38x off the tile), so they keep sm80_det_bwd_20261117's `window` rule.
    ("sdpa_bwd_sm80", (8, 0), "bf16", (128, 128), "prefill_det", "dense", "window_narrow", False, "mha", False): Entry("ge", 34.5, "sm80_det_bwd_20261117"),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (128, 128), "prefill_det", "dense", "window_narrow", False, "mha", True): Entry("ge", 34.5, "sm80_det_bwd_20261117"),
    # study sm120_det_bwd_20261117 (RTX PRO 6000 Server, cuDNN 9.24.0.43, FE 9bdf9a8ab; deterministic backward, dense BSHD squares,
    # d64 / d128, MHA and GQA 4 / 8, no mask, causal and 128 / 1024-key windows, S 1k-8k; 128 train cells and a 40-cell holdout
    # scored once: 36 changed cells in 16 classes at 0.14-0.92 geomean, worst 0.93; the two class pairs no holdout cell exercised (d128 causal MHA, d64 no-mask GQA) have no entry). FROST's deterministic body (two kernels) runs 0.1-0.9 of the backend's deterministic engine;
    # gpt-oss d64 SWA backward ran engine 7 at 9.8x. Squares only (bwd/heuristics.SQUARE_ONLY_STUDIES).
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill_det", "dense", "causal_br", False, "gqa", False): Entry("always", None, "sm120_det_bwd_20261117"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill_det", "dense", "causal_br", False, "gqa", True): Entry("always", None, "sm120_det_bwd_20261117"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill_det", "dense", "causal_br", False, "mha", False): Entry("always", None, "sm120_det_bwd_20261117"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill_det", "dense", "causal_br", False, "mha", True): Entry("always", None, "sm120_det_bwd_20261117"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill_det", "dense", "none", False, "mha", False): Entry("always", None, "sm120_det_bwd_20261117"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill_det", "dense", "none", False, "mha", True): Entry("always", None, "sm120_det_bwd_20261117"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill_det", "dense", "window", False, "gqa", False): Entry("always", None, "sm120_det_bwd_20261117"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill_det", "dense", "window", False, "gqa", True): Entry("always", None, "sm120_det_bwd_20261117"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill_det", "dense", "window", False, "mha", False): Entry("always", None, "sm120_det_bwd_20261117"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (64, 64), "prefill_det", "dense", "window", False, "mha", True): Entry("always", None, "sm120_det_bwd_20261117"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill_det", "dense", "causal_br", False, "gqa", False): Entry("always", None, "sm120_det_bwd_20261117"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill_det", "dense", "causal_br", False, "gqa", True): Entry("always", None, "sm120_det_bwd_20261117"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill_det", "dense", "none", False, "gqa", False): Entry("always", None, "sm120_det_bwd_20261117"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill_det", "dense", "none", False, "gqa", True): Entry("always", None, "sm120_det_bwd_20261117"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill_det", "dense", "none", False, "mha", False): Entry("always", None, "sm120_det_bwd_20261117"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill_det", "dense", "none", False, "mha", True): Entry("always", None, "sm120_det_bwd_20261117"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill_det", "dense", "window", False, "gqa", False): Entry("ge", 10.5, "sm120_det_bwd_20261117", "kv"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill_det", "dense", "window", False, "gqa", True): Entry("ge", 10.5, "sm120_det_bwd_20261117", "kv"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill_det", "dense", "window", False, "mha", False): Entry("ge", 10.5, "sm120_det_bwd_20261117", "kv"),
    ("sdpa_bwd_sm120", (12, 0), "bf16", (128, 128), "prefill_det", "dense", "window", False, "mha", True): Entry("ge", 10.5, "sm120_det_bwd_20261117", "kv"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (128, 128), "prefill", "dense", "causal_br", False, "gqa", False): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (128, 128), "prefill", "dense", "causal_br", False, "gqa", True): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (128, 128), "prefill", "dense", "causal_br", False, "mha", False): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (128, 128), "prefill", "dense", "causal_br", False, "mha", True): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (128, 128), "prefill", "dense", "none", False, "gqa", False): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (128, 128), "prefill", "dense", "none", False, "gqa", True): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (128, 128), "prefill", "dense", "none", False, "mha", False): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (128, 128), "prefill", "dense", "none", False, "mha", True): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (128, 128), "prefill", "dense", "window", False, "gqa", False): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (128, 128), "prefill", "dense", "window", False, "gqa", True): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (128, 128), "prefill", "dense", "window", False, "mha", False): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (128, 128), "prefill", "dense", "window", False, "mha", True): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (64, 64), "prefill", "dense", "causal_br", False, "gqa", False): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (64, 64), "prefill", "dense", "causal_br", False, "gqa", True): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (64, 64), "prefill", "dense", "causal_br", False, "mha", False): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (64, 64), "prefill", "dense", "causal_br", False, "mha", True): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (64, 64), "prefill", "dense", "none", False, "gqa", False): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (64, 64), "prefill", "dense", "none", False, "gqa", True): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (64, 64), "prefill", "dense", "none", False, "mha", False): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (64, 64), "prefill", "dense", "none", False, "mha", True): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (64, 64), "prefill", "dense", "window", False, "gqa", False): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (64, 64), "prefill", "dense", "window", False, "gqa", True): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (64, 64), "prefill", "dense", "window", False, "mha", False): Entry("always", None, "sm120_bwd_f16_20261107"),
    ("sdpa_bwd_sm120", (12, 0), "f16", (64, 64), "prefill", "dense", "window", False, "mha", True): Entry("always", None, "sm120_bwd_f16_20261107"),
    # studies sm80_bwd_square_{bf16,f16}_20261108 (A100, cuDNN 9.27.0; dense BSHD square backward under a bottom-right left window, d64 / d128, MHA / GQA, on and off the tile,
    # 128 train cases per dtype; gen.py --margin 0.05, keeping a class only if it and its tile twin are `always`). bwd.heuristics._measured_lead lifts only square graphs of this row.
    ("sdpa_bwd_sm80", (8, 0), "bf16", (128, 128), "prefill", "dense", "window", False, "gqa", False): Entry("always", None, "sm80_bwd_square_bf16_20261108"),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (128, 128), "prefill", "dense", "window", False, "gqa", True): Entry("always", None, "sm80_bwd_square_bf16_20261108"),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (128, 128), "prefill", "dense", "window", False, "mha", False): Entry("always", None, "sm80_bwd_square_bf16_20261108"),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (128, 128), "prefill", "dense", "window", False, "mha", True): Entry("always", None, "sm80_bwd_square_bf16_20261108"),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "window", False, "gqa", False): Entry("always", None, "sm80_bwd_square_bf16_20261108"),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "window", False, "gqa", True): Entry("always", None, "sm80_bwd_square_bf16_20261108"),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "window", False, "mha", False): Entry("always", None, "sm80_bwd_square_bf16_20261108"),
    ("sdpa_bwd_sm80", (8, 0), "bf16", (64, 64), "prefill", "dense", "window", False, "mha", True): Entry("always", None, "sm80_bwd_square_bf16_20261108"),
    ("sdpa_bwd_sm80", (8, 0), "f16", (128, 128), "prefill", "dense", "window", False, "gqa", False): Entry("always", None, "sm80_bwd_square_f16_20261108"),
    ("sdpa_bwd_sm80", (8, 0), "f16", (128, 128), "prefill", "dense", "window", False, "gqa", True): Entry("always", None, "sm80_bwd_square_f16_20261108"),
    ("sdpa_bwd_sm80", (8, 0), "f16", (128, 128), "prefill", "dense", "window", False, "mha", False): Entry("always", None, "sm80_bwd_square_f16_20261108"),
    ("sdpa_bwd_sm80", (8, 0), "f16", (128, 128), "prefill", "dense", "window", False, "mha", True): Entry("always", None, "sm80_bwd_square_f16_20261108"),
    ("sdpa_bwd_sm80", (8, 0), "f16", (64, 64), "prefill", "dense", "window", False, "gqa", False): Entry("always", None, "sm80_bwd_square_f16_20261108"),
    ("sdpa_bwd_sm80", (8, 0), "f16", (64, 64), "prefill", "dense", "window", False, "gqa", True): Entry("always", None, "sm80_bwd_square_f16_20261108"),
    ("sdpa_bwd_sm80", (8, 0), "f16", (64, 64), "prefill", "dense", "window", False, "mha", False): Entry("always", None, "sm80_bwd_square_f16_20261108"),
    ("sdpa_bwd_sm80", (8, 0), "f16", (64, 64), "prefill", "dense", "window", False, "mha", True): Entry("always", None, "sm80_bwd_square_f16_20261108"),
}
