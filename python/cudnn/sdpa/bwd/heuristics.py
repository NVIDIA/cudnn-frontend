# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDPA-backward family heuristics: list each eligible engine's plan WITH the
tiles it will build, so a recorded ``(engine_id, knobs)`` pins the kernel.

``(kind, facts, offered) -> [PlanConfig]``, backend-blind and graph-blind like
``sdpa.fwd.heuristics.recommend``. The backward engines have no runner-up
candidates: every eligible row contributes exactly one entry, for ``"A"`` and
``"FALLBACK"`` alike. What the entry spells is what the lowering would pick
when handed no knobs:

- a row whose kernel takes a tile choice (``Capabilities.tile_ms`` /
  ``tile_ns`` non-empty) gets its default pair as ``TILE_M`` / ``TILE_N`` --
  the SM120 kernel's per-head-dim table, or the sole point of a fixed domain;
- a row with no tile axis lists no knobs: its geometry is not a tuning
  decision, so ``{}`` is the complete record.

Replaying the listed knobs through ``create_execution_plan`` builds the same
kernel the automatic pick builds today, and keeps building it if the default
table moves in a later release.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterator, List, Optional, Tuple

from cudnn.engines.base import PlanConfig

from .engines import ENGINE_SPECS, EngineSpec, SdpaBwdKnobs, mismatch


def _sm120_default_tiles(facts) -> Optional[Tuple[int, int]]:
    """The SM120 f16 kernel's default (q_tile, kv_tile) for these head dims
    (``config_sm120.DEFAULT_TILES``, keyed by the padded d_qk the kernel runs at)."""
    from .config_sm120 import DEFAULT_TILES, padded_head_dims

    if facts.thd:
        from .sm120_thd_contract import native_tiles

        return native_tiles(int(facts.d_qk), int(facts.d_v))
    pads = padded_head_dims(int(facts.d_qk), int(facts.d_v))
    return None if pads is None else DEFAULT_TILES[pads[0]]


# Rows whose kernel resolves an unset tile from a table of its own. A row with a
# single-point tile domain needs no entry (the sole point is the default); a row
# without a tile axis needs none either. test_sdpa_bwd_heuristics guards that
# every row with a wider domain is listed here.
_DEFAULT_TILE_RESOLVERS: Dict[str, Callable[[object], Optional[Tuple[int, int]]]] = {
    "sdpa_bwd_sm120": _sm120_default_tiles,
}


def default_knobs(spec: EngineSpec, facts) -> Optional[SdpaBwdKnobs]:
    """The knobs the lowering would resolve for ``facts`` when handed none:
    ``None`` for a row with no tile axis, else a complete ``SdpaBwdKnobs``."""
    caps = spec.capabilities
    if not caps.tile_ms and not caps.tile_ns:
        return None
    resolver = _DEFAULT_TILE_RESOLVERS.get(spec.name)
    if resolver is not None:
        tiles = resolver(facts)
        if tiles is None:
            return None
        return SdpaBwdKnobs(tile_m=tiles[0], tile_n=tiles[1])
    if len(caps.tile_ms) == 1 and len(caps.tile_ns) == 1:
        return SdpaBwdKnobs(tile_m=next(iter(caps.tile_ms)), tile_n=next(iter(caps.tile_ns)))
    raise NotImplementedError(f"{spec.name} advertises a tile domain but no default-tile resolver is registered for it")


def _eligible(facts, offered: Dict[str, int]) -> Iterator[Tuple[int, EngineSpec]]:
    """(engine_id, spec) for each offered row whose capability row admits ``facts``."""
    for spec in ENGINE_SPECS:
        engine_id = offered.get(spec.name)
        if engine_id is not None and mismatch(spec.capabilities, facts, None) is None:
            yield engine_id, spec


def _leads(spec: EngineSpec, facts) -> bool:
    """Whether a default row's plans go ahead of the backend's block.

    ``sdpa_bwd_sm100_d128`` (measured 2026-10-26 against cuDNN 9.28.0.38, 216 cases at 32k tokens: d64/d128, GQA
    32/8 16/2 and MHA, S 1k-16k, causal / mask-free / SWA1024, BSHD and THD at equal and half-to-full lengths): THD at
    d128 wins 0.79-0.95 (one 1.07) -> ahead; dense BSHD d128 is parity (0.93-1.27), a window loses ~2x and dense D64
    loses 1.7-2.3x -> behind the backend.  Measured on cc 10.0; cc 10.3 runs the same rule (the sm103 admission,
    2026-10-28); cc 10.7 (Rubin, admitted 2026-11-03 with the same bodies compiled for sm_107a) runs the same rule on the
    evidence of 2026-11-03 (a Rubin class node, 212 SMs, A/B/A x3 x100 both build orders): the backend offers NO plan for
    d128 THD backward on cc 10.7 with cuDNN 9.24.0.43 or 9.27.0.42, so the FROST row is the only candidate there; dense MHA d128 vs the backend's engine (9.27.0.42) is +1.6 % at 16K and inside noise at 8K = the parity band of
    the cc 10.0 rule -> dense stays behind the backend there too.

    ``sdpa_bwd_sm100_d256`` (measured 2026-10-27 on a full-power B200 against cuDNN 9.28.0.38, 108 cases at 32k tokens:
    d256 GQA 16/2, 32/4 and MHA 16/16, D192x128 MHA 16-128 heads, S 1k-16k, causal / mask-free, BSHD and THD at equal and
    half-to-full lengths): d256 wins 0.73-0.98 -> ahead, except a mask-free ragged THD declaration (declared total below
    b * S_q) at S_q >= 4096 (0.95-1.10); D192x128 loses 1.08-1.91 everywhere -> behind. Only cc 10.0 was measured, and
    deterministic / bias / dropout / window / sink graphs were not. Every other row keeps ours-first. On cc 10.3 the d256
    row's bf16 THD graphs are decided by the placement table (studies sm103_d256_bwd_v2_20261031 on the 128-key tile and
    sm103_d256_bwd_thd_offtile_20261031 off it); its dense graphs, which lost on small-work holdout cases in two studies,
    still trail the backend there.

    A measured entry in ``cudnn.sdpa.placement_table`` answers first; the rules above decide every graph without one."""
    from cudnn.sdpa import placement_table

    measured = placement_table.lookup(spec.name, facts)
    if measured is not None:
        return measured
    if spec.name == "sdpa_bwd_sm100_d128":
        return facts.device_cc in D128_LEAD_CCS and facts.thd and facts.window_left is None and (facts.d_qk, facts.d_v) == (128, 128)
    if spec.name == "sdpa_bwd_sm100_d256":
        return _d256_leads(facts)
    return spec.name not in TRAIL_ONLY_ROWS


TRAIL_ONLY_ROWS = frozenset({"sdpa_bwd_sm80", "sdpa_bwd_sm120", "sdpa_bwd_sm107", "sdpa_bwd_sm107_d512"})
"""Default-offered rows with no hand-written lead shard: ranked after the backend's block, so they serve a graph only when the
backend declines it (a correctness fallback, not a performance claim) -- except where a measured ``placement_table`` entry
leads the graph's class (:func:`_measured_lead`)."""


SQUARE_ONLY_STUDIES = ("sm80_bwd_square_", "sm120_bwd_")
"""Study name prefixes whose train and holdout timed square graphs only. ``class_key`` excludes squareness (Rule S12: a feature the key does not
hold), so a non-square graph that one of their entries answers for keeps trailing. A100 under a left window: every square holdout case ran faster
than the backend, while q < kv ran up to 1.45x slower in the same classes (b12 40/8 d64 q128 kv8960, 7.0 vs 4.8 ms). RTX 5090 (cc 12.0): the
studies timed s_q == s_kv only, and an earlier single-timing sweep had 2 of 200 non-square d64 / d128 cases above 1.05 of the backend. The GB10
(cc 12.1) studies are not listed: their entries keep answering for non-square graphs as merged."""


def _measured_lead(spec: EngineSpec, facts) -> bool:
    """A trail-only row whose class a ``placement_table`` study measured as a lead goes ahead of the backend like any row. The
    studies time full-length graphs, and ``class_key`` does not see runtime lengths, so a graph with them keeps trailing. The
    same goes for a non-square graph whose entry comes from a study in :data:`SQUARE_ONLY_STUDIES`."""
    from cudnn.sdpa import placement_table

    entry = placement_table.matched_entry(spec.name, facts)
    if entry is not None and facts.s_q != facts.s_kv and entry.study.startswith(SQUARE_ONLY_STUDIES):
        return False
    return not facts.padded and not facts.seq_q_trim and placement_table.lookup(spec.name, facts) is True


D128_LEAD_CCS = ((10, 0), (10, 3), (10, 7))
"""The parts where ``sdpa_bwd_sm100_d128``'s lead box (unwindowed d128 THD) applies: cc 10.0 measured (2026-10-26), cc 10.3
carried with the sm103 admission, cc 10.7 measured 2026-11-03 on a Rubin class node -- the backend serves no d128 THD backward
plan there (cuDNN 9.24.0.43 and 9.27.0.42), so the FROST row is the only candidate in the box; its dense d128 graphs read
parity against the backend (+1.6 % at 16K, noise at 8K) and trail, as on cc 10.0."""


SM100_D256_BWD_RAGGED_MASKFREE_MAX_S_Q = 4095  # mask-free ragged THD d256 past this length: 0.95-1.10 of the backend


def _d256_leads(facts) -> bool:
    """The measured d256 lead box for ``sdpa_bwd_sm100_d256`` (see :func:`_leads`)."""
    if (facts.d_qk, facts.d_v) != (256, 256) or facts.device_cc != (10, 0):
        return False
    if facts.deterministic or facts.has_bias or facts.has_dropout or facts.has_sink or facts.window_left is not None:
        return False
    ragged = facts.thd and facts.max_total_seq_len_q is not None and facts.max_total_seq_len_q < facts.b * facts.s_q
    return not (ragged and not facts.causal and facts.s_q > SM100_D256_BWD_RAGGED_MASKFREE_MAX_S_Q)


def propose(kind: str, facts, offered: Dict[str, int]) -> List[PlanConfig]:
    """The manifest hook: :func:`recommend`'s plans, the backend's block placed after the rows that lead
    (:func:`_leads`) and before the rest. ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES`` set keeps ours first everywhere."""
    ours = recommend(kind, facts, offered)
    from cudnn.engines.manifest import opt_in_engines_enabled

    if not ours or opt_in_engines_enabled():
        return ours
    from cudnn.engines.heuristics import BACKEND, LAST_RESORT

    spec_of = {offered[s.name]: s for s in ENGINE_SPECS if s.name in offered}
    last = [p for p in ours if spec_of[p.engine_id].name in TRAIL_ONLY_ROWS and not _measured_lead(spec_of[p.engine_id], facts)]
    lead = [p for p in ours if p not in last and _leads(spec_of[p.engine_id], facts)]
    trail = [p for p in ours if p not in last and p not in lead]
    if last:
        return lead + [BACKEND] + trail + [LAST_RESORT] + last
    return lead + [BACKEND] + trail if trail else lead


def _lists_last(spec: EngineSpec, facts) -> bool:
    """Rows listed after the other eligible rows: the d192 half row on a dense graph, where the d256 row's native
    192/128 geometry serves the same declaration."""
    return spec.name == "sdpa_bwd_sm100_d192" and not facts.thd


def recommend(kind: str, facts, offered: Dict[str, int]) -> List[PlanConfig]:
    """One complete entry per eligible row, in ``ENGINE_SPECS`` order except for :func:`_lists_last`; ``kind``
    does not change the answer (no runner-up candidates to thin out)."""
    out: List[Tuple[EngineSpec, PlanConfig]] = []
    for engine_id, spec in _eligible(facts, offered):
        knobs = default_knobs(spec, facts)
        if knobs is None or mismatch(spec.capabilities, facts, knobs) is None:
            out.append((spec, PlanConfig(engine_id, knobs)))
    out.sort(key=lambda entry: _lists_last(entry[0], facts))
    return [plan for _, plan in out]
