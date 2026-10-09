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
    default_rows = {_SM100, _SM120, "sdpa_fwd_prefill_sm107", "sdpa_fwd_prefill_sm90", "sdpa_fwd_prefill_sm100_fp8", "sdpa_fwd_prefill_sm107_mxfp8"}
    opt_in_rows = {"sdpa_fwd_prefill_sm80", "sdpa_fwd_prefill_sm100_mxfp8", "sdpa_fwd_prefill_sm107_fp8"}
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
@pytest.mark.parametrize("d,s_q", [(512, 64), (512, 256)])
@pytest.mark.parametrize("min_kv", [512, 4096])
def test_short_query_placement_respects_configured_domain(monkeypatch, d, s_q, min_kv):
    """Exercise the d512 prefill domain guard without pinning a measured threshold or winner."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    monkeypatch.setattr(placement, "SHORT_QUERY_MIN_KV_TOKENS", min_kv, raising=False)
    spec = next(spec for spec in ENGINE_SPECS if spec.name == _SM100)
    assert placement.place(spec, _facts(d_qk=d, d_v=d, s_q=s_q, s_kv=min_kv - 1)) == placement.TRAIL


@pytest.mark.L0
@pytest.mark.parametrize("d", [128, 256, 512])
@pytest.mark.parametrize("s_q", [2, 4, 16])
@pytest.mark.parametrize("s_kv", [64, 1000, 2047, 8192])
def test_decode_shaped_rows_lead_at_every_kv_length(d, s_q, s_kv):
    """Multi-token decode rows (2 <= s_q <= 16) lead the backend whatever the cache
    length: the backend serves them with a prefill-class engine, the row with its
    decode tile, and the short-cache gap is the largest (see the module docstring)."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == _SM100)
    facts = _facts(b=32, h_q=64, h_kv=4, d_qk=d, d_v=d, s_q=s_q, s_kv=s_kv)
    assert placement.place(spec, facts) == placement.LEAD


@pytest.mark.L0
def test_paged_d512_decode_trails_until_its_decode_tile():
    """Paged d512 at s_q == 1 runs the role-split prefill tile (no d512 decode tile) and
    measured behind the backend's paged decode engine, so it TRAILS by default, while the
    same launch over dense K/V keeps the measured dense-d512 rule (LEAD at these KV tokens
    in flight); multi-token paged d512 keeps the decode-shaped LEAD and paged d512 prefill
    stays backend-first like every other paged prefill (the paged d256 THD shard excepted)."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == _SM100)
    paged = dict(d_qk=512, d_v=512, b=8, h_q=64, h_kv=1, s_kv=4096, has_paged_kv=True, padded=True, page_size=16, causal=False, bottom_right=False)
    assert placement.place(spec, _facts(s_q=1, **paged)) == placement.TRAIL
    assert placement.place(spec, _facts(s_q=1, **{**paged, "has_paged_kv": False, "page_size": 0})) == placement.LEAD
    assert placement.place(spec, _facts(s_q=4, **paged)) == placement.LEAD
    assert placement.place(spec, _facts(s_q=128, **paged)) == placement.TRAIL


@pytest.mark.L0
@pytest.mark.parametrize(
    "outside",
    [
        None,
        {"has_paged_kv": False},
        {"wants_stats": True},
        {"dtype": cudnn.data_type.HALF},
        {"d_qk": 192},
        {"s_q": 63},
        {"s_kv": 257},
        {"b": 3},
        {"device_cc": (10, 3)},
        {"page_size": 16},
        {"h_q": 8},
        {"bottom_right": False},
        {"window_left": 128},
    ],
)
def test_paged_prefill_placement_stays_inside_configured_domain(monkeypatch, outside):
    """Synthetic shard bounds check isolation, not measured workload winners."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    monkeypatch.setattr(placement, "PAGED_D256_PREFILL_HEADS", frozenset({(6, 2)}), raising=False)
    monkeypatch.setattr(placement, "PAGED_D256_PREFILL_PAGE_SIZES", frozenset({32}), raising=False)
    monkeypatch.setattr(placement, "PAGED_D256_PREFILL_MIN_Q", 64, raising=False)
    monkeypatch.setattr(placement, "PAGED_D256_PREFILL_MAX_KV", 256, raising=False)
    monkeypatch.setattr(placement, "PAGED_D256_PREFILL_MAX_BATCH", 2, raising=False)
    spec = next(spec for spec in ENGINE_SPECS if spec.name == _SM100)
    values = dict(h_q=6, h_kv=2, s_q=128, s_kv=256, d_qk=256, d_v=256, has_paged_kv=True, thd=True, page_size=32)
    values.update(outside or {})
    assert placement.place(spec, _facts(**values)) == (placement.TRAIL if outside else placement.LEAD)


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16])
@pytest.mark.parametrize(
    "chooser,decision",
    [
        ("_prefer_paged_d256_lpt", True),
        ("paged_d256_prefix_launch", object()),
        ("nonpaged_thd_split_choice", (3, True)),
        ("paged_thd_split_choice", (3, False)),
    ],
)
def test_sm107_placement_consumes_qualified_choices(monkeypatch, dtype, chooser, decision):
    """A qualified chooser result reaches public placement; no workload winner is fixed."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm107")
    for name, empty in (
        ("_prefer_paged_d256_lpt", False),
        ("paged_d256_prefix_launch", None),
        ("nonpaged_thd_split_choice", (1, False)),
        ("paged_thd_split_choice", (1, False)),
    ):
        monkeypatch.setattr(heuristics, name, lambda *args, value=empty: value)
    monkeypatch.setattr(heuristics, chooser, lambda *args: decision)
    facts = _facts(device_cc=(10, 7), dtype=dtype)
    assert placement.place(spec, facts) == placement.LEAD
    # An architecture-specific choice cannot promote this row on another GPU.
    assert placement.place(spec, _facts(device_cc=(10, 0), dtype=dtype)) == placement.TRAIL


@pytest.mark.L0
@pytest.mark.parametrize(
    "outside",
    [
        None,
        dict(device_cc=(10, 0)),
        dict(thd=True, padded=True),
        dict(has_paged_kv=True, page_size=16, padded=True),
        dict(attn_scale_prefolded=True),
        dict(has_sink=False),
        dict(window_left=128),
        dict(right_band_widening=True, right_bound=8),
        dict(s_q=17),
        dict(b=1),
        dict(s_kv=512),
        dict(s_kv=65536),
        dict(shape_overrides=True),
        dict(h_q=8, h_kv=8),  # MHA: not in the measured family
        dict(h_q=16, h_kv=8),  # GQA2: not measured
        dict(h_q=96, h_kv=8),  # 12 does not divide the tile (runs unpacked): not measured
        dict(h_q=64, h_kv=2),  # GQA32: not measured
        dict(h_q=36, h_kv=8),  # a partial group
        dict(d_qk=256, d_v=256),  # the d256 flavor has no shared dense leg
        dict(causal=False, bottom_right=False, s_q=4),  # plain (mask-free) multi-token: not measured
    ],
    ids=[
        "inside",
        "sm100",
        "thd",
        "paged",
        "prefolded",
        "no_sink",
        "swa",
        "right_band",
        "s_q_17",
        "b1",
        "kv512",
        "kv64k",
        "overrides",
        "mha",
        "g2",
        "g12",
        "g32",
        "partial_group",
        "d256",
        "nomask_q4",
    ],
)
def test_sm107_sink_decode_shard_stays_inside_its_configured_domain(monkeypatch, outside):
    """The measured dense d128 sink decode / verify shard LEADs exactly inside its configured band (issue #1472): the
    constants are monkeypatched, so the contract under test is the shard's SHAPE -- the shared dense d128 half leg
    (engines.rubin_dense_d128_shared_leg), a sink, bottom-right verify rows or plain decode, a measured GQA group,
    units and cache inside the band -- never the measured numbers (those live in the module docstring)."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    monkeypatch.setattr(placement, "SM107_SINK_DECODE_MAX_S_Q", 16)
    monkeypatch.setattr(placement, "SM107_SINK_DECODE_GROUPS", (4, 8, 16))
    monkeypatch.setattr(placement, "SM107_SINK_DECODE_MIN_UNITS", 16)
    monkeypatch.setattr(placement, "SM107_SINK_DECODE_KV_TOKENS", (1024, 16384))
    spec = next(spec for spec in ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm107")
    values = dict(
        b=4,
        h_q=32,
        h_kv=8,
        s_q=8,
        s_kv=2048,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        causal=True,
        bottom_right=True,
        has_sink=True,
        device_cc=(10, 7),
        device_sm_count=216,
    )
    values.update(outside or {})
    assert placement.place(spec, _facts(**values)) == (placement.TRAIL if outside else placement.LEAD)
    if outside is None:
        # The plain-decode arm of the same band (s_q == 1 without a mask), the 256-row packed head (16 rows x GQA16: the
        # packed cga2 prefill body rather than the decode tile), f16, the d64 and d96 envelopes, the 64-unit / 16k-cache
        # corner, and the 32-unit bound at G = 4 with other head counts (16/4 b8, 4/1 b32) -- every one a measured cell
        # (the module docstring's band and its review-fix corner pass).
        assert placement.place(spec, _facts(**dict(values, causal=False, bottom_right=False, s_q=1))) == placement.LEAD
        assert placement.place(spec, _facts(**dict(values, s_q=16, h_q=64, h_kv=4))) == placement.LEAD
        assert placement.place(spec, _facts(**dict(values, dtype=cudnn.data_type.HALF))) == placement.LEAD
        assert placement.place(spec, _facts(**dict(values, d_qk=64, d_v=64))) == placement.LEAD
        assert placement.place(spec, _facts(**dict(values, d_qk=96, d_v=96))) == placement.LEAD
        assert placement.place(spec, _facts(**dict(values, b=8, s_kv=16384))) == placement.LEAD
        assert placement.place(spec, _facts(**dict(values, b=8, h_q=16, h_kv=4, s_q=4))) == placement.LEAD
        assert placement.place(spec, _facts(**dict(values, b=32, h_q=4, h_kv=1, s_q=4, s_kv=1024))) == placement.LEAD


@pytest.mark.L0
@pytest.mark.parametrize("group, expected", [(8, placement.LEAD), (16, placement.TRAIL)], ids=["gqa8", "gqa16"])
def test_sm107_paged_packed_prefill_shard_keeps_its_measured_groups(group, expected):
    """The paged packed-GQA prefill shard (Q 64-128, page 16, no sink) was timed on GQA 4 / 8; GQA16 now packs by default
    on cc 10.7 paged THD (issue #1472's paged table) but keeps the backend first here until that band is measured."""
    from types import SimpleNamespace

    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm107")
    hnd_pool = SimpleNamespace(get_stride=lambda: (16 * 128, 16 * 128 * 8, 128, 1))  # [pages, H, page, D]: head stride above the page stride
    facts = _facts(
        b=16,
        h_q=64,
        h_kv=64 // group,
        s_q=128,
        s_kv=4096,
        thd=True,
        padded=True,
        has_paged_kv=True,
        page_size=16,
        k_t=hnd_pool,
        device_cc=(10, 7),
        device_sm_count=216,
    )
    assert placement.place(spec, facts) == expected


@pytest.mark.L0
@pytest.mark.parametrize(
    "over",
    [
        dict(),
        dict(s_q=1),
        dict(s_q=1, has_sink=True),
        dict(d_qk=512, d_v=512),
        dict(d_qk=256, d_v=256, wants_stats=True, causal=False),
        dict(d_qk=192, d_v=128, dtype=cudnn.data_type.FP8_E5M2),
    ],
)
def test_sm107_mxfp8_placement_leads_on_exact_cc107_only(over):
    """The cc 10.7 MXFP8 row leads the backend on exact cc 10.7 for every graph it admits and trails on any other
    device (its arch range reaches cc 11.9; only cc 10.7 is qualified).  A contract, not a workload winner."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107_mxfp8")
    quant = dict(b=2, h_q=8, h_kv=2, s_q=4096, s_kv=4096, dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16, is_mxfp8=True)
    quant.update(over)
    assert placement.place(spec, _facts(device_cc=(10, 7), device_sm_count=216, **quant)) == placement.LEAD
    for cc in ((10, 0), (10, 3), (10, 8), (11, 0)):
        assert placement.place(spec, _facts(device_cc=cc, **quant)) == placement.TRAIL, cc


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
def test_propose_ranks_the_rubin_mxfp8_row_first_without_the_flag(monkeypatch, kind):
    """The arm is wired into place(): with recommend mocked, propose puts the row ahead of the backend block on
    cc 10.7 and behind it elsewhere, with the flag deleted."""
    name = "sdpa_fwd_prefill_sm107_mxfp8"
    plans = [PlanConfig(_OFFERED[name], object())]
    monkeypatch.setattr(heuristics, "recommend", Mock(return_value=plans))
    quant = dict(dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16, is_mxfp8=True, s_q=4096)
    assert heuristics.propose(kind, _facts(device_cc=(10, 7), **quant), {name: _OFFERED[name]}) == plans + [BACKEND]
    assert heuristics.propose(kind, _facts(device_cc=(10, 0), **quant), {name: _OFFERED[name]}) == [BACKEND] + plans
