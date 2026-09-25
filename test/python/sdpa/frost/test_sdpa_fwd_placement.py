# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDPA-forward proposal contracts, independent of performance-policy rankings.

Synthetic recommendations and placement verdicts exercise the hook's marker,
ordering and override behavior. Performance measurements belong in offline
validation; tuning a real workload's ranking must not change these contracts.
"""

from unittest.mock import Mock

import pytest

import cudnn
from cudnn.engines import manifest
from cudnn.engines.base import PlanConfig
from cudnn.engines.heuristics import BACKEND, is_backend_block
from cudnn.sdpa.fwd import heuristics, placement
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

_FAMILY = next(f for f in manifest.MANIFEST if f.name == "frost_sdpa_fwd")
_SM100 = "sdpa_fwd_prefill_sm100"
_SM120 = "sdpa_fwd_prefill_sm120"
_OFFERED = {name: _FAMILY.engine_id + slot.slot for name, slot in _FAMILY.slots.items()}


def _facts(**over):
    base = dict(
        b=1,
        h_q=8,
        h_kv=1,
        s_q=1,
        s_kv=8192,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        causal=True,
        bottom_right=True,
        device_cc=(10, 0),
        device_sm_count=148,
    )
    base.update(over)
    return SdpaGraphFacts(**base)


@pytest.fixture(autouse=True)
def _no_opt_in_flag(monkeypatch):
    """Override the suite's autouse opt-in so each test controls the flag."""
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)


@pytest.fixture
def recommendations(monkeypatch):
    # Opaque knobs deliberately carry no real performance-policy assumptions.
    plans = [PlanConfig(_OFFERED[_SM100], object()), PlanConfig(_OFFERED[_SM100], object())]
    recommend = Mock(return_value=plans)
    monkeypatch.setattr(heuristics, "recommend", recommend)
    return plans, recommend


@pytest.mark.L0
def test_known_default_and_opt_in_rows_are_offered(monkeypatch):
    default_rows = {_SM100, _SM120}
    opt_in_rows = {"sdpa_fwd_prefill_sm80", "sdpa_fwd_prefill_sm100_fp8"}
    offered = _FAMILY.offered_ids()
    assert default_rows <= offered.keys()
    assert opt_in_rows.isdisjoint(offered)
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    assert default_rows | opt_in_rows <= _FAMILY.offered_ids().keys()


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
@pytest.mark.parametrize("verdict", [placement.LEAD, placement.TRAIL])
def test_propose_preserves_recommendations_and_places_one_marker(monkeypatch, recommendations, kind, verdict):
    ours, recommend = recommendations
    original = tuple(ours)
    facts, offered = _facts(), {_SM100: _OFFERED[_SM100]}
    monkeypatch.setattr(placement, "place", Mock(return_value=verdict))
    plans = heuristics.propose(kind, facts, offered)
    expected = ours + [BACKEND] if verdict == placement.LEAD else [BACKEND] + ours
    assert plans == expected
    assert sum(is_backend_block(plan) for plan in plans) == 1
    assert tuple(ours) == original, "propose must not mutate recommend's list"
    recommend.assert_called_once_with(kind, facts, offered)


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
def test_propose_opt_in_overrides_placement(monkeypatch, recommendations, kind):
    ours, _ = recommendations
    monkeypatch.setattr(placement, "place", Mock(return_value=placement.TRAIL))
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    assert heuristics.propose(kind, _facts(), {_SM100: _OFFERED[_SM100]}) == ours + [BACKEND]


@pytest.mark.L0
@pytest.mark.parametrize("opt_in", ["0", "1"])
def test_propose_keeps_an_empty_recommendation_empty(monkeypatch, opt_in):
    monkeypatch.setattr(heuristics, "recommend", Mock(return_value=[]))
    monkeypatch.setattr(placement, "place", Mock(return_value=placement.LEAD))
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", opt_in)
    assert heuristics.propose("A", _facts(), {_SM100: _OFFERED[_SM100]}) == []


@pytest.mark.L0
def test_propose_is_empty_when_nothing_is_eligible():
    assert heuristics.propose("A", _facts(device_cc=(9, 0), device_sm_count=132), {_SM100: _OFFERED[_SM100]}) == []
    assert heuristics.propose("A", _facts(), {}) == []


@pytest.mark.L0
def test_marker_never_reaches_a_ranked_list():
    from cudnn.engines.heuristics import _assemble

    backend = [PlanConfig(-1, None), PlanConfig(7, {"k": 1}, cpp_index=0, mode=cudnn.heur_mode.A)]
    for hook in (lambda kind: [BACKEND, PlanConfig(20511, "x")], lambda kind: [PlanConfig(20511, "x"), BACKEND], lambda kind: [BACKEND]):
        for modes in ([cudnn.heur_mode.A], [cudnn.heur_mode.OPENSOURCE], [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]):
            assert not any(is_backend_block(p) for p in _assemble(modes, hook, backend))


@pytest.mark.L0
@pytest.mark.parametrize("d,s_q", [(128, 4), (256, 4), (512, 4), (512, 64)])
@pytest.mark.parametrize("min_kv", [512, 4096])
def test_short_query_placement_respects_configured_domain(monkeypatch, d, s_q, min_kv):
    """Exercise the domain guard without pinning a measured threshold or winner."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    monkeypatch.setattr(placement, "SHORT_QUERY_MIN_KV_TOKENS", min_kv, raising=False)
    spec = next(spec for spec in ENGINE_SPECS if spec.name == _SM100)
    assert placement.place(spec, _facts(d_qk=d, d_v=d, s_q=s_q, s_kv=min_kv - 1)) == placement.TRAIL
