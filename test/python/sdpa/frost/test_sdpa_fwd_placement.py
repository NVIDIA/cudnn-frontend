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
    default_rows = {
        _SM100,
        _SM120,
        "sdpa_fwd_prefill_sm107",
        "sdpa_fwd_sm90_d64_one_wg",
        "sdpa_fwd_sm90_f16_three_wg",
        "sdpa_fwd_sm90_d128_multi_q",
        "sdpa_fwd_prefill_sm100_fp8",
        "sdpa_fwd_prefill_sm107_fp8",
        "sdpa_fwd_prefill_sm107_mxfp8",
        "sdpa_fwd_prefill_sm80",
        "sdpa_fwd_prefill_sm120_fp8",
        "sdpa_fwd_prefill_sm120_mxfp8",
        "sdpa_fwd_sm100_d256_mxfp8_single_q",
        "sdpa_fwd_sm100_d128_mxfp8_multi_q",
        "sdpa_fwd_sm100_d192_mxfp8_multi_q",
        "sdpa_fwd_sm100_d512_mxfp8_slice_recompute",
        "sdpa_fwd_sm100_d512_mxfp8_retained_p",
    }
    opt_in_rows = {"sdpa_fwd_prefill_sm100_mxfp8"}
    offered = _FAMILY.offered_ids()
    assert default_rows <= offered.keys()
    assert opt_in_rows.isdisjoint(offered)
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    assert default_rows | opt_in_rows <= _FAMILY.offered_ids().keys()


@pytest.mark.L0
def test_trail_only_rows_rank_behind_the_backend():
    """A default row with no measured lead shard serves only graphs the backend declines (a correctness fallback)."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    assert placement.TRAIL_ONLY_ROWS <= _FAMILY.offered_ids().keys()
    for spec in ENGINE_SPECS:
        if spec.name in placement.TRAIL_ONLY_ROWS:
            assert placement.place(spec, _facts(device_cc=(8, 0))) == placement.TRAIL, spec.name
    from cudnn.engines.heuristics import LAST_RESORT, _assemble

    # Across modes: a backend that offers only FALLBACK entries still precedes a last-resort row (review on #1962).
    frost = PlanConfig(_OFFERED[_SM100], object())
    proposals = lambda kind: [BACKEND, LAST_RESORT, frost]
    backend_fb = [PlanConfig(7, "fb", mode=cudnn.heur_mode.FALLBACK)]
    final = _assemble([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK], proposals, backend_fb)
    assert [p.engine_id for p in final] == [7, frost.engine_id]
    assert [p.engine_id for p in _assemble([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK], proposals, [])] == [frost.engine_id]
    assert [p.engine_id for p in _assemble([cudnn.heur_mode.OPENSOURCE], proposals, backend_fb)] == [frost.engine_id]


_SM100_MXFP8_OWNERS = (
    "sdpa_fwd_sm100_d256_mxfp8_single_q",
    "sdpa_fwd_sm100_d128_mxfp8_multi_q",
    "sdpa_fwd_sm100_d192_mxfp8_multi_q",
    "sdpa_fwd_sm100_d512_mxfp8_slice_recompute",
    "sdpa_fwd_sm100_d512_mxfp8_retained_p",
)


@pytest.mark.L0
@pytest.mark.parametrize("flag", [None, "0"])
@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
def test_sm100_mxfp8_owners_trail_the_backend_by_default(monkeypatch, flag, cc):
    """The SM100 MXFP8 explicit owners (default since 2026-11-16) and their opt-in row rank after the backend's block and the
    LAST_RESORT marker with the flag unset or 0; the flag puts them first."""
    from cudnn.engines.heuristics import LAST_RESORT, is_last_resort
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    if flag is not None:
        monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", flag)
    specs = {s.name: s for s in ENGINE_SPECS}
    facts = _facts(s_q=2048, s_kv=2048, h_kv=8, dtype=cudnn.data_type.FP8_E4M3, is_mxfp8=True, device_cc=cc)
    tag = lambda p: "backend" if is_backend_block(p) else "last_resort" if is_last_resort(p) else "frost"  # noqa: E731
    for name in _SM100_MXFP8_OWNERS:
        assert name in _FAMILY.offered_ids() and placement.place(specs[name], facts) == placement.TRAIL, name
        ours = [PlanConfig(_OFFERED[name], None)]
        monkeypatch.setattr(heuristics, "recommend", lambda kind, facts, offered, ours=ours: list(ours))
        assert [tag(p) for p in heuristics.propose("A", facts, _OFFERED)] == ["backend", "last_resort", "frost"], name
    assert placement.place(specs["sdpa_fwd_prefill_sm100_mxfp8"], facts) == placement.TRAIL
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    assert [tag(p) for p in heuristics.propose("A", facts, _OFFERED)] == ["frost", "backend"]
    assert LAST_RESORT not in heuristics.propose("A", facts, _OFFERED)


@pytest.mark.L0
@pytest.mark.parametrize("entry", [None, "always", "never"])
def test_a_trail_only_row_leads_where_a_generated_entry_says_so(monkeypatch, entry):
    """SM80 plans go after the last-resort marker unless a placement_table entry leads the graph; then they lead the backend."""
    from cudnn.engines.heuristics import LAST_RESORT
    from cudnn.sdpa import placement_table
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    name = "sdpa_fwd_prefill_sm80"
    spec = next(s for s in ENGINE_SPECS if s.name == name)
    facts = _facts(s_q=2048, s_kv=2048, h_q=32, h_kv=8, device_cc=(8, 0), device_sm_count=108)
    table = {} if entry is None else {placement_table.class_key(name, facts): placement_table.Entry(entry, None, "synthetic")}
    monkeypatch.setattr(placement_table, "TABLE", table)
    ours = [PlanConfig(_OFFERED[name], object())]
    monkeypatch.setattr(heuristics, "recommend", Mock(return_value=ours))
    plans = heuristics.propose("A", facts, {name: _OFFERED[name]})
    assert plans == (ours + [BACKEND] if entry == "always" else [BACKEND, LAST_RESORT] + ours)
    assert placement.place(spec, facts) == (placement.LEAD if entry == "always" else placement.TRAIL)


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131073], ids=["on_tile", "off_tile"])
def test_sm80_d128_gqa_long_kv_chunks_lead(s_kv):
    """Studies sm80_dense_d128_gqa_v4_split_20261117 and sm80_dense_d128_split_key_v2_20261122 (A100): bf16 dense d128 GQA causal
    prefill leads with few query rows / tiles over a long cache; many rows, f16 or another part trail. A short cache that k_prefetch
    still splits 4 ways is the split class's and leads too."""
    from cudnn.sdpa import placement_table
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    row, impl = (next(s for s in ENGINE_SPECS if s.name == n) for n in ("sdpa_fwd_prefill_sm80", "sdpa_fwd_sm80_f16_k_prefetch"))

    def f(**over):
        return _facts(**{**dict(s_q=256, s_kv=s_kv, has_paged_kv=False, device_cc=(8, 0), device_sm_count=108), **over})

    assert placement.place(row, f()) == placement.LEAD
    assert placement.place(impl, f()) == placement.LEAD
    assert placement.place(row, f(b=8, h_q=64, h_kv=8, s_q=2048)) == placement.TRAIL  # many query rows
    short = f(s_kv=1024 + s_kv % 2)
    assert placement_table.matched_entry("sdpa_fwd_prefill_sm80", short).study == "sm80_dense_d128_split_key_v2_20261122"
    assert placement.place(row, short) == placement.LEAD
    assert placement.place(row, f(dtype=cudnn.data_type.HALF)) == placement.TRAIL
    assert placement.place(row, f(device_cc=(8, 6))) == placement.TRAIL


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
@pytest.mark.parametrize("h_q", [12, 20, 24, 48], ids=["g3", "g5", "g6", "g12"])
@pytest.mark.parametrize("b", [1, 4, 16])
def test_sm120_d256_decode_of_a_group_the_tile_does_not_divide_trails(h_q, b):
    """SM120 packs a group the tile does not divide (tile_q % G != 0), yet at d256 s_q == 1 its plan still trails
    the backend (1.00-1.44x over b 1-16, KV 4k-128k; module docstring), so the backend leads at every KV unit count;
    a power-of-two group leads from 12 KV units."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == _SM120)
    facts = dict(b=b, h_kv=4, d_qk=256, d_v=256, s_kv=65536, causal=False, bottom_right=False, device_cc=(12, 0), device_sm_count=188)
    assert placement.place(spec, _facts(h_q=h_q, **facts)) == placement.TRAIL
    assert placement.place(spec, _facts(h_q=16, **facts)) == (placement.LEAD if b * 4 >= placement.SM120_SQ1_D256_MIN_KV_UNITS else placement.TRAIL)


@pytest.mark.L0
@pytest.mark.parametrize("b, s_kv", [(1, 2048), (1, 131072), (12, 30000), (128, 131072)])
def test_sm120_d512_decode_wide_group_leads_when_it_fills_the_tile(b, s_kv):
    """SM120 d512 s_q == 1 over one KV head: a 128-wide group packs into one 128-row tile and leads at every batch and
    KV length (0.16-0.70 of the backend, module docstring); 96 and 112 leave tile rows idle and keep the backend
    (112/1 b96 ran 1.27-1.30x); 64 and below keep the one-KV-head rule (b1 at 2k KV trails)."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == _SM120)
    facts = dict(b=b, h_kv=1, s_q=1, s_kv=s_kv, d_qk=512, d_v=512, causal=False, bottom_right=False, device_cc=(12, 0), device_sm_count=188)
    assert placement.place(spec, _facts(h_q=128, **facts)) == placement.LEAD
    assert placement.place(spec, _facts(h_q=112, **facts)) == placement.TRAIL
    assert placement.place(spec, _facts(h_q=96, **facts)) == placement.TRAIL
    if (b, s_kv) == (1, 2048):
        assert placement.place(spec, _facts(h_q=64, **facts)) == placement.TRAIL


@pytest.mark.L0
@pytest.mark.parametrize("d_qk", [512, 576], ids=["d512", "d576"])
@pytest.mark.parametrize("b", [1, 16, 128])
def test_sm120_paged_d512_decode_follows_the_group(b, d_qk):
    """Study sm120_paged_d512_q1_v2_20261117 (DSv4 paged decode, page 64) and its D576 transfer check (Kimi-K3, page 32 / 64): on
    paged K/V the group decides at any batch -- 40-64 query heads lead, 32 and below keep the backend (1.06-1.75x), a wider group
    leads only as a power of two."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS
    from cudnn.sdpa import placement_table as pt

    spec = next(spec for spec in ENGINE_SPECS if spec.name == _SM120)
    paged = dict(b=b, h_kv=1, s_q=1, s_kv=65536, d_qk=d_qk, d_v=512, causal=False, bottom_right=False, device_cc=(12, 0), device_sm_count=188)
    paged.update(has_paged_kv=True, page_size=64, padded=True)
    for h_q, verdict in ((8, placement.TRAIL), (16, placement.TRAIL), (32, placement.TRAIL), (40, placement.LEAD), (64, placement.LEAD)):
        assert placement.place(spec, _facts(h_q=h_q, **paged)) == verdict, h_q
    assert placement.place(spec, _facts(h_q=96, **paged)) == placement.TRAIL
    assert placement.place(spec, _facts(h_q=128, **paged)) == placement.LEAD
    assert pt.matched_entry("sdpa_fwd_prefill_sm120", _facts(h_q=16, **paged)) is None


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [32768, 32769], ids=["on_tile", "off_tile"])
def test_sm120_paged_d128_mtp_follows_the_group(s_kv):
    """Studies sm120_paged_d128_mtp_v6 / _v7_20261117 (llama paged MTP, page 16): FROST's time is flat in the GQA group and the
    backend's grows with it, so from group 4 FROST leads at any batch; groups 2 / 3 lead up to 2^5.5 KV units and trail above; f16
    and another part keep the hand rules."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == _SM120)
    mtp = dict(h_kv=8, s_q=2, s_kv=s_kv, d_qk=128, d_v=128, causal=True, bottom_right=True, device_cc=(12, 0), device_sm_count=188)
    mtp.update(has_paged_kv=True, page_size=16, padded=True)
    for b, h_q in ((1, 64), (128, 64), (128, 40), (128, 32), (4, 16)):  # groups 8 / 8 / 5 / 4 at any batch; 32 units at group 2
        assert placement.place(spec, _facts(b=b, h_q=h_q, **mtp)) == placement.LEAD, (b, h_q)
    for b, h_q in ((128, 24), (16, 16)):  # groups 3 / 2 past 2^5.5 units
        assert placement.place(spec, _facts(b=b, h_q=h_q, **mtp)) == placement.TRAIL, (b, h_q)
    body = dict(mtp, h_kv=4, s_q=4)  # group 16: 64 rows run FROST's prefill body, whose `gqa_body` entries decide
    assert placement.place(spec, _facts(b=128, h_q=64, **body)) == placement.LEAD
    assert placement.place(spec, _facts(b=1, h_q=64, **body)) == placement.TRAIL
    assert placement.place(spec, _facts(b=128, h_q=32, dtype=cudnn.data_type.HALF, **mtp)) == placement.LEAD  # f16: no class, hand rule


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131071], ids=["on_tile", "off_tile"])
def test_sm120_paged_d192_window_decode_leads_on_few_tiles(s_kv):
    """Study sm120_paged_window_decode_d192x192_v2_20261117: bf16 paged d192 x 192 windowed decode (step5 SWA ran the backend at
    2.49x) leads up to 2^6.5 KV-head tiles; above it the hand rule decides (powers of two lead, other groups trail), and
    d192 x V128 has its own table entries."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == _SM120)
    dec = dict(h_q=64, h_kv=4, s_q=1, s_kv=s_kv, d_qk=192, d_v=192, causal=True, bottom_right=True, window_left=511)
    dec.update(has_paged_kv=True, page_size=16, padded=True, device_cc=(12, 0), device_sm_count=188)
    for b, h_q, h_kv in ((1, 64, 4), (16, 40, 4), (8, 96, 8)):  # 2^2 / 2^6 / 2^6 tiles; groups 16 / 10 / 12
        assert placement.place(spec, _facts(**dict(dec, b=b, h_q=h_q, h_kv=h_kv))) == placement.LEAD, (b, h_q, h_kv)
    assert placement.place(spec, _facts(**dict(dec, b=64, h_q=40, h_kv=4))) == placement.TRAIL  # 2^8 tiles, group 10: the hand rule
    assert placement.place(spec, _facts(**dict(dec, b=64, h_q=64, h_kv=4))) == placement.LEAD  # 2^8 tiles, group 16: the hand rule
    assert placement.place(spec, _facts(**dict(dec, b=1, window_left=None))) == placement.TRAIL  # unwindowed: not this class


@pytest.mark.L0
@pytest.mark.parametrize("s_q", [8, 16, 128, 2048])
def test_sm120_paged_prefill_body_trails(s_q):
    """SM120 paged K/V on the prefill body (no 16 / 32-row decode tile) keeps the backend first: it ran 15-27x behind on the RTX PRO
    6000 (module docstring); the same graph on dense K/V still leads, and a paged graph that fits the decode tile keeps its rules."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == _SM120)
    shape = dict(b=1, h_q=8, h_kv=1, s_q=s_q, s_kv=8192, d_qk=128, d_v=128, causal=True, bottom_right=True, device_cc=(12, 0), device_sm_count=188)
    assert placement.place(spec, _facts(has_paged_kv=True, page_size=16, padded=True, **shape)) == placement.TRAIL
    assert placement.place(spec, _facts(**shape)) == placement.LEAD
    decode_tile = dict(shape, h_q=32, h_kv=8, s_q=8)  # 8 x group 4 = 32 rows
    assert placement.place(spec, _facts(has_paged_kv=True, page_size=16, padded=True, **decode_tile)) == placement.LEAD


@pytest.mark.L0
@pytest.mark.parametrize("d_qk", [512, 576])
@pytest.mark.parametrize("h_q", [8, 16, 128])
@pytest.mark.parametrize("s_q", [2, 4])
def test_sm120_paged_wide_mla_mtp_leads(d_qk, h_q, s_q):
    """The paged prefill-body TRAIL covers head dims the decode tile can take (d <= 256): paged d512 / D576xV512 MTP over one
    KV head leads, 0.04-0.40 of the backend over 108 nightly cells (deepseek_v4, kimi_k3; module docstring)."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == _SM120)
    shape = dict(b=1, h_q=h_q, h_kv=1, s_q=s_q, s_kv=131072, d_qk=d_qk, d_v=512, causal=True, bottom_right=True, device_cc=(12, 0), device_sm_count=188)
    assert placement.place(spec, _facts(has_paged_kv=True, page_size=64, padded=True, **shape)) == placement.LEAD


@pytest.mark.L0
def test_paged_d512_decode_follows_the_generated_entry_and_trails_without_one():
    """Paged d512 at s_q == 1 on cc 10.0 follows the generated entry (study sm100_d512_paged_decode_b100_20261030): it
    leads past the cost threshold and trails tiny launches. Without an entry (an SM100-row part no study or transfer check covers) the hand rule's
    TRAIL stands. Dense K/V keeps the dense-d512 rule; multi-token paged d512 keeps the decode-shaped LEAD; a mask-free
    paged prefill has no entry and stays backend-first."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == _SM100)
    paged = dict(d_qk=512, d_v=512, b=8, h_q=64, h_kv=1, s_kv=4096, has_paged_kv=True, padded=True, page_size=16, causal=False, bottom_right=False)
    assert placement.place(spec, _facts(s_q=1, **paged)) == placement.LEAD  # measured neighbours win 0.26-0.81
    assert placement.place(spec, _facts(s_q=1, **{**paged, "b": 1, "h_q": 8})) == placement.TRAIL  # measured loss 2.7x
    assert placement.place(spec, _facts(s_q=1, device_cc=(10, 1), **paged)) == placement.TRAIL
    assert placement.place(spec, _facts(s_q=1, **{**paged, "has_paged_kv": False, "page_size": 0})) == placement.LEAD
    assert placement.place(spec, _facts(s_q=4, **paged)) == placement.LEAD
    assert placement.place(spec, _facts(s_q=128, **paged)) == placement.TRAIL


@pytest.mark.L0
def test_paged_d576_mla_decode_leads_past_its_work_floor():
    """Paged bf16 D576x512 MLA decode leads at s_q == 1 once log2(h_q * b * S_kv) passes the generated table's threshold
    (study sm100_d576_decode_20261027, cc 10.0; log2_work is the table's cost proxy); dense K/V, a window, a sink or f16 keep the backend's decode engine;
    multi-token decode keeps the decode-shaped LEAD; graphs outside the table (an unmeasured part such as cc 10.1) keep the #1933 hand floor."""
    import math

    from cudnn.sdpa import placement_table
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == _SM100)
    mla = dict(d_qk=576, d_v=512, h_kv=1, has_paged_kv=True, padded=True, page_size=64, causal=False, bottom_right=False)
    entry = placement_table.TABLE[(_SM100, (10, 0), "bf16", (576, 512), "q1", "paged", "none", False, "gqa", False)]
    assert entry.direction == "ge"
    for h_q in (16, 32, 64, 128):
        for b in (1, 8, 32, 128):
            for s_kv in (2048, 8192, 32768, 131072):
                facts = _facts(s_q=1, h_q=h_q, b=b, s_kv=s_kv, **mla)
                want = placement.LEAD if placement_table.log2_work(facts) >= entry.log2_work else placement.TRAIL
                assert placement.place(spec, facts) == want, (h_q, b, s_kv)
    lead = dict(h_q=64, b=64, s_kv=8192)
    assert placement.place(spec, _facts(s_q=1, **lead, **mla)) == placement.LEAD
    dense = {**mla, "has_paged_kv": False, "page_size": 0}
    assert placement.place(spec, _facts(s_q=1, **lead, **dense)) == placement.LEAD  # study sm100_d576_dense_decode_20261030
    assert placement.place(spec, _facts(s_q=1, device_cc=(10, 1), **lead, **dense)) == placement.TRAIL  # no entry: hand floor (paged only)
    assert placement.place(spec, _facts(s_q=1, dtype=cudnn.data_type.HALF, **lead, **mla)) == placement.TRAIL
    assert placement.place(spec, _facts(s_q=1, window_left=128, **lead, **mla)) == placement.TRAIL
    assert placement.place(spec, _facts(s_q=1, has_sink=True, **lead, **mla)) == placement.TRAIL
    # 2**19: v2 timed it as a loss (backend 37.4 us, its split-KV reduction uncounted); re-timed 51.0 us vs FROST 41.6 us
    assert placement.place(spec, _facts(s_q=1, h_q=16, b=1, s_kv=32768, **mla)) == placement.LEAD
    assert placement.place(spec, _facts(s_q=4, h_q=16, b=1, s_kv=2048, **mla)) == placement.LEAD
    # Outside the table (another SM100-row part, a bounded declaration) the #1933 hand floor still decides.
    assert placement.place(spec, _facts(s_q=1, device_cc=(10, 1), **lead, **mla)) == placement.LEAD
    assert placement.place(spec, _facts(s_q=1, shape_overrides=True, **lead, **mla)) == placement.LEAD
    assert placement.place(spec, _facts(s_q=1, device_cc=(10, 1), h_q=16, b=1, s_kv=32768, **mla)) == placement.TRAIL


@pytest.mark.L0
@pytest.mark.parametrize(
    "outside",
    [
        None,
        {"has_paged_kv": False},
        {"wants_stats": True},
        {"dtype": cudnn.data_type.FP8_E4M3},
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
    """Synthetic shard bounds check isolation, not measured workload winners: the generated table is emptied so only the
    hand-written shard decides."""
    from cudnn.sdpa import placement_table
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    monkeypatch.setattr(placement_table, "TABLE", {})

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
        dict(has_sink=False, causal=False, bottom_right=False, s_q=1),  # sink-free s_q 2-16 leads by the dense decode-shaped arm
        dict(window_left=128),
        dict(right_band_widening=True, right_bound=8),
        dict(s_q=17, b=64),  # past the band; b=64 keeps it off the one-decode-tile-wave prefill arm
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
        # The d64 envelope selects the row's native 64-wide prefill tile, not the shared decode tile; study
        # sm107_d64_sink_prefill_20261112 measured it (verify rows 0.04-0.11 of the backend) and leads it.
        assert placement.place(spec, _facts(**dict(values, d_qk=64, d_v=64))) == placement.LEAD
        assert placement.place(spec, _facts(**dict(values, d_qk=96, d_v=96))) == placement.LEAD
        assert placement.place(spec, _facts(**dict(values, b=8, s_kv=16384))) == placement.LEAD
        assert placement.place(spec, _facts(**dict(values, b=8, h_q=16, h_kv=4, s_q=4))) == placement.LEAD
        assert placement.place(spec, _facts(**dict(values, b=32, h_q=4, h_kv=1, s_q=4, s_kv=1024))) == placement.LEAD


@pytest.mark.L0
@pytest.mark.parametrize("group, expected", [(8, placement.LEAD), (16, placement.TRAIL)], ids=["gqa8", "gqa16"])
def test_sm107_paged_packed_prefill_shard_keeps_its_measured_groups(group, expected):
    """The paged packed-GQA prefill shard (Q 64-128, page 16, no sink) was timed on GQA 4 / 8; GQA16 now packs by default
    on cc 10.7 paged THD (issue #1472's paged table) but the shard keeps the backend first there. An exact bf16 graph is
    answered first by study sm107_thd_prefill_20261111, which leads both groups."""
    from types import SimpleNamespace

    from cudnn.sdpa import placement_table
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
    assert placement._place_sm107_f16(spec.capabilities, facts) == expected
    assert placement.place(spec, facts) == placement.LEAD
    assert placement_table.matched_entry(spec.name, facts).study == "sm107_thd_prefill_20261111"


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


_SM107_FP8_DENSE = dict(b=1, h_q=16, h_kv=2, s_q=4096, s_kv=4096, d_qk=256, d_v=256, causal=True, bottom_right=True)
_SM107_FP8_QUANT = dict(dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16, is_fp8=True, device_sm_count=216)


@pytest.mark.L0
@pytest.mark.parametrize(
    "over",
    [dict(), dict(d_qk=512, d_v=512), dict(causal=False, bottom_right=False), dict(bottom_right=False, wants_stats=True), dict(s_q=1, causal=False)],
)
def test_sm107_fp8_placement_leads_d256_d512_on_exact_cc107(over):
    """The cc 10.7 per-tensor FP8 row leads at exact d256 / d512 on exact cc 10.7, and trails on any other device."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107_fp8")
    values = {**_SM107_FP8_DENSE, **_SM107_FP8_QUANT, **over}
    assert placement.place(spec, _facts(device_cc=(10, 7), **values)) == placement.LEAD
    for cc in ((10, 8), (11, 0)):
        assert placement.place(spec, _facts(device_cc=cc, **values)) == placement.TRAIL, cc


@pytest.mark.L0
@pytest.mark.parametrize(
    "over",
    [
        dict(d_qk=128, d_v=128),
        dict(d_qk=192, d_v=128),
        dict(dtype=cudnn.data_type.FP8_E5M2),
        dict(thd=True),
        dict(has_paged_kv=True),
        dict(padded=True),
        dict(window_left=127),
        dict(right_band_widening=True, right_bound=16),
        dict(has_sink=True),
        dict(o_block_scale=32),
        dict(has_epilogue_gate=True),
    ],
)
def test_sm107_fp8_placement_trails_outside_the_measured_domain(over):
    """d128 / d192 and every graph feature the A/B did not cover keep the backend first."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107_fp8")
    values = {**_SM107_FP8_DENSE, **_SM107_FP8_QUANT, **over}
    assert placement.place(spec, _facts(device_cc=(10, 7), **values)) == placement.TRAIL


@pytest.mark.L0
def test_sm107_fp8_d64_sink_decode_placement_structure():
    """cc 10.7 FP8 d64 decode with a sink: s_q 2-4 leads; s_q == 1 leads from the module's KV-unit floor, and below it over a
    whole cache its plan splits; a short window leads only for a power-of-two group from the same floor; a longer window, paging,
    or no sink keeps the backend. Bounds come from the module, so this pins the gate's structure, not a measured boundary."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107_fp8")
    units = placement.SM107_FP8_D64_SINK_DECODE_MIN_UNITS
    base = {**_SM107_FP8_QUANT, "d_qk": 64, "d_v": 64, "h_q": 64, "h_kv": 8, "s_kv": 131072, "has_sink": True, "device_cc": (10, 7)}
    mtp = {**base, "b": 1, "s_q": 4, "causal": True, "bottom_right": True}
    q1 = {**base, "s_q": 1, "causal": False, "bottom_right": False}
    assert placement.place(spec, _facts(**mtp)) == placement.LEAD
    assert placement.place(spec, _facts(**{**q1, "b": units // 8})) == placement.LEAD
    assert placement.place(spec, _facts(**{**q1, "b": units // 8 - 1})) == placement.LEAD  # the whole cache splits
    # an unaligned dense KV tail keeps the unsplit plan (heuristics._synth_kv_padding)
    assert placement.place(spec, _facts(**{**q1, "b": units // 8 - 1, "s_kv": 131071})) == placement.TRAIL
    for over in (dict(window_left=127), dict(has_paged_kv=True), dict(has_sink=False), dict(device_cc=(10, 0))):
        assert placement.place(spec, _facts(**{**mtp, **over})) == placement.TRAIL, over
    window = placement.SM107_FP8_D64_SINK_MAX_WINDOW
    for shape in (mtp, q1):
        swa = {**shape, "b": units // 8, "window_left": window}
        assert placement.place(spec, _facts(**swa)) == placement.LEAD
        assert placement.place(spec, _facts(**{**swa, "b": units // 8 - 1})) == placement.TRAIL
        assert placement.place(spec, _facts(**{**swa, "h_q": 48})) == placement.TRAIL  # group 6 does not pack whole
        assert placement.place(spec, _facts(**{**swa, "b": 128, "window_left": window + 1})) == placement.TRAIL


@pytest.mark.L0
def test_sm107_fp8_placement_trails_one_kv_tile_with_few_q_tiles():
    """The launch-bound floor reads its bounds from the module, so it pins the gate's structure, not a measured boundary."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107_fp8")
    kv, tiles = placement.SM107_FP8_TINY_MAX_KV, placement.SM107_FP8_TINY_MAX_Q_TILES
    values = {**_SM107_FP8_DENSE, **_SM107_FP8_QUANT, "b": 1, "h_q": tiles, "h_kv": 1, "s_q": 32, "s_kv": kv, "causal": False, "bottom_right": False}
    assert placement.place(spec, _facts(device_cc=(10, 7), **values)) == placement.TRAIL
    assert placement.place(spec, _facts(device_cc=(10, 7), **{**values, "s_kv": kv + 1})) == placement.LEAD
    assert placement.place(spec, _facts(device_cc=(10, 7), **{**values, "h_q": tiles + 1, "h_kv": 1})) == placement.LEAD


@pytest.mark.L0
def test_sm107_fp8_d128_chunk_placement_structure():
    """The d128 chunk arm reads every bound from the module, so this pins the gate's structure, not a measured boundary."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107_fp8")
    long_kv, max_q = placement.SM107_FP8_D128_LONG_KV, placement.SM107_FP8_D128_CAUSAL_MAX_S_Q
    chunk = {**_SM107_FP8_DENSE, **_SM107_FP8_QUANT, "d_qk": 128, "d_v": 128, "b": 1, "h_q": 64, "h_kv": 8, "s_q": max_q, "s_kv": long_kv}
    place = lambda **over: placement.place(spec, _facts(device_cc=(10, 7), **{**chunk, **over}))
    assert place() == placement.LEAD
    assert place(s_q=max_q + 1) == placement.TRAIL
    assert place(s_kv=long_kv - 128) == placement.TRAIL  # below the long cache and under the KV-per-Q ratio
    ratio_q = placement.SM107_FP8_D128_MIN_KV // placement.SM107_FP8_D128_KV_PER_Q
    assert place(s_q=ratio_q, s_kv=placement.SM107_FP8_D128_MIN_KV) == placement.LEAD
    assert place(s_q=ratio_q, s_kv=placement.SM107_FP8_D128_MIN_KV - 128) == placement.TRAIL
    assert place(s_q=long_kv, s_kv=long_kv) == placement.TRAIL  # square
    assert place(h_q=48, h_kv=8) == placement.TRAIL  # GQA group 6 cannot pack
    assert place(bottom_right=False) == placement.TRAIL  # top-left causal chunks were not measured
    free = {"causal": False, "bottom_right": False, "s_q": placement.SM107_FP8_D128_MASKFREE_MAX_S_Q}
    assert place(**free) == placement.LEAD
    assert place(**{**free, "s_kv": long_kv + 1}) == placement.TRAIL  # off the KV tile: the mask-free graph cannot split
    assert place(**{**free, "s_q": placement.SM107_FP8_D128_MASKFREE_MAX_S_Q + 1}) == placement.TRAIL
    # MHA: one 2-SM cluster wave of units, causal only, within one 512-row Q tile
    mha = {"h_q": 108, "h_kv": 108, "s_q": placement.SM107_FP8_D128_MHA_MAX_S_Q}
    assert place(**mha) == placement.LEAD
    assert place(**{**mha, "b": 2}) == placement.TRAIL
    assert place(**{**mha, "device_sm_count": None}) == placement.TRAIL
    assert place(**{**mha, "s_q": placement.SM107_FP8_D128_MHA_MAX_S_Q + 1}) == placement.TRAIL
    assert place(**{**mha, **free}) == placement.TRAIL


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
def test_propose_ranks_the_rubin_fp8_row_first_without_the_flag(monkeypatch, kind):
    """The arm is wired into place(): with recommend mocked and the flag deleted, propose puts the row ahead of the backend
    block on cc 10.7 at d256 and behind it at d128."""
    name = "sdpa_fwd_prefill_sm107_fp8"
    plans = [PlanConfig(_OFFERED[name], object())]
    monkeypatch.setattr(heuristics, "recommend", Mock(return_value=plans))
    values = {**_SM107_FP8_DENSE, **_SM107_FP8_QUANT}
    offered = {name: _OFFERED[name]}
    assert heuristics.propose(kind, _facts(device_cc=(10, 7), **values), offered) == plans + [BACKEND]
    assert heuristics.propose(kind, _facts(device_cc=(10, 7), **{**values, "d_qk": 128, "d_v": 128}), offered) == [BACKEND] + plans


_SM120_FP8 = "sdpa_fwd_prefill_sm120_fp8"
_SM120_FP8_DENSE = dict(b=1, h_q=32, h_kv=8, s_q=4096, s_kv=4096, d_qk=128, d_v=128, causal=True, bottom_right=False)
_SM120_FP8_QUANT = dict(dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16, is_fp8=True, device_cc=(12, 0), device_sm_count=188)


def _place_sm120(name, **over):
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == name)
    return placement.place(spec, _facts(**{**_SM120_FP8_DENSE, **_SM120_FP8_QUANT, **over}))


@pytest.mark.L0
@pytest.mark.parametrize(
    "over",
    [
        dict(),
        dict(d_qk=64, d_v=64, s_q=512, s_kv=512),
        dict(d_qk=256, d_v=256, s_q=512, s_kv=512),
        dict(s_q=8, s_kv=512, bottom_right=True),
        dict(s_q=1, s_kv=512, causal=False),
        dict(window_left=1023),
        dict(device_cc=(12, 1), device_sm_count=48),
    ],
)
def test_sm120_fp8_placement_leads_outside_the_short_cache_d128_band(over):
    """d64, d > 128, decode-shaped graphs and d128 on a long cache lead on every cc 12.x part."""
    assert _place_sm120(_SM120_FP8, **over) == placement.LEAD


@pytest.mark.L0
def test_sm120_fp8_short_cache_d128_band_structure():
    """The band reads its bounds from the module, so this pins the gate's structure, not a measured boundary."""
    short, maskfree = placement.SM120_FP8_SHORT_KV, placement.SM120_FP8_MASKFREE_SHORT_KV
    assert _place_sm120(_SM120_FP8, s_q=short, s_kv=short) == placement.TRAIL
    assert _place_sm120(_SM120_FP8, s_q=short + 1, s_kv=short + 1) == placement.LEAD
    assert _place_sm120(_SM120_FP8, s_q=maskfree, s_kv=maskfree, causal=False) == placement.TRAIL
    assert _place_sm120(_SM120_FP8, s_q=maskfree + 1, s_kv=maskfree + 1, causal=False) == placement.LEAD
    assert _place_sm120(_SM120_FP8, d_qk=96, d_v=96, s_q=short, s_kv=short) == placement.TRAIL
    assert _place_sm120(_SM120_FP8, d_qk=64, d_v=64, s_q=short, s_kv=short) == placement.LEAD
    decode = placement.DECODE_SHAPED_MAX_S_Q
    assert _place_sm120(_SM120_FP8, s_q=decode, s_kv=short, bottom_right=True) == placement.LEAD
    assert _place_sm120(_SM120_FP8, s_q=decode + 1, s_kv=short, bottom_right=True) == placement.TRAIL


@pytest.mark.L0
@pytest.mark.parametrize(
    "over",
    [
        dict(thd=True),
        dict(padded=True),
        dict(right_band_widening=True, right_bound=16),
        dict(has_sink=True),
        dict(has_sink=True, s_q=8, s_kv=4096, bottom_right=True, window_left=511),
        dict(has_sink=True, d_qk=64, d_v=64, s_q=512, s_kv=4096, bottom_right=False),
        dict(has_sink=True, d_qk=64, d_v=64, s_q=8, s_kv=4096, causal=False, bottom_right=False),
        dict(has_sink=True, d_qk=96, d_v=96, s_q=1, s_kv=4096, causal=False),
        dict(has_sink=True, d_qk=64, d_v=64, s_q=1, s_kv=8192, causal=True, bottom_right=False),
        dict(has_sink=True, s_q=1, s_kv=8192, causal=True, bottom_right=False),
        dict(o_block_scale=16),
        dict(has_epilogue_gate=True),
        dict(dtype=cudnn.data_type.FP8_E5M2),
    ],
)
def test_sm120_fp8_placement_trails_outside_the_measured_domain(over):
    """Every graph feature the benchmark did not cover keeps the backend first."""
    assert _place_sm120(_SM120_FP8, **over) == placement.TRAIL


@pytest.mark.L0
@pytest.mark.parametrize(
    "over",
    [
        dict(d_qk=64, d_v=64, s_q=1, s_kv=131072, causal=False),
        dict(d_qk=64, d_v=64, s_q=1, s_kv=131072, bottom_right=True, window_left=127),
        dict(d_qk=64, d_v=64, s_q=2048, s_kv=8192, bottom_right=True),
        dict(d_qk=64, d_v=64, s_q=4, s_kv=8192, bottom_right=True, window_left=127),
        dict(d_qk=64, d_v=64, s_q=2048, s_kv=8192, bottom_right=True, window_left=1023),
        dict(d_qk=64, d_v=64, s_q=512, s_kv=512, bottom_right=True),
        dict(s_q=1, s_kv=8192, causal=False),
        dict(s_q=16, s_kv=8192, bottom_right=True),
        dict(s_q=4, s_kv=512, bottom_right=True),
    ],
)
def test_sm120_fp8_sink_leads_in_the_measured_classes(over):
    """With a sink: exact d64 (outside short windows past the MTP rows) and exact d128 decode-shaped graphs without a window lead."""
    assert _place_sm120(_SM120_FP8, has_sink=True, **over) == placement.LEAD


@pytest.mark.L0
def test_sm120_fp8_sink_band_structure():
    """The d128 decode band reads DECODE_SHAPED_MAX_S_Q, so this pins the gate's structure, not a measured boundary."""
    decode = placement.DECODE_SHAPED_MAX_S_Q
    sink = dict(has_sink=True, s_kv=8192, bottom_right=True)
    assert _place_sm120(_SM120_FP8, s_q=decode, **sink) == placement.LEAD
    assert _place_sm120(_SM120_FP8, s_q=decode + 1, **sink) == placement.TRAIL
    assert _place_sm120(_SM120_FP8, d_qk=64, d_v=64, s_q=decode + 1, **sink) == placement.LEAD
    assert _place_sm120(_SM120_FP8, has_sink=True, padded=True, d_qk=64, d_v=64, s_q=1, causal=False) == placement.TRAIL
    short, mtp = placement.SM120_FP8_SINK_SHORT_WINDOW, placement.SM120_FP8_SINK_SHORT_WINDOW_MAX_S_Q
    d64 = dict(sink, d_qk=64, d_v=64)
    assert _place_sm120(_SM120_FP8, s_q=mtp + 1, window_left=short - 1, **d64) == placement.TRAIL
    assert _place_sm120(_SM120_FP8, s_q=mtp, window_left=short - 1, **d64) == placement.LEAD
    assert _place_sm120(_SM120_FP8, s_q=mtp + 1, window_left=short, **d64) == placement.LEAD


@pytest.mark.L0
@pytest.mark.parametrize("over", [dict(), dict(d_qk=64, d_v=64, s_q=512, s_kv=512, causal=False), dict(s_q=1, causal=False), dict(device_cc=(12, 1))])
def test_sm120_mxfp8_placement_leads_everywhere(over):
    """The backend has no MXFP8 SDPA on cc 12.x, so the row leads every graph it admits."""
    assert _place_sm120("sdpa_fwd_prefill_sm120_mxfp8", **{"is_fp8": False, "is_mxfp8": True, **over}) == placement.LEAD


@pytest.mark.L0
@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
def test_propose_ranks_the_sm120_fp8_row_per_its_band_without_the_flag(monkeypatch, kind):
    """The arm is wired into place(): with recommend mocked and the flag deleted, propose puts the row ahead of the backend
    block on a long d128 cache and behind it on a short one."""
    plans = [PlanConfig(_OFFERED[_SM120_FP8], object())]
    monkeypatch.setattr(heuristics, "recommend", Mock(return_value=plans))
    offered = {_SM120_FP8: _OFFERED[_SM120_FP8]}
    values = {**_SM120_FP8_DENSE, **_SM120_FP8_QUANT}
    assert heuristics.propose(kind, _facts(**values), offered) == plans + [BACKEND]
    assert heuristics.propose(kind, _facts(**{**values, "s_q": 512, "s_kv": 512}), offered) == [BACKEND] + plans


@pytest.mark.L0
@pytest.mark.parametrize("arch, cc, sms", [("sm107", (10, 7), 216), ("sm100", (10, 0), 148)])
@pytest.mark.parametrize("sink", [False, True])
def test_dense_d128_one_decode_tile_wave_leads(monkeypatch, arch, cc, sms, sink):
    """Past the decode-shaped band a causal dense d64/d128 graph, sink or not, leads exactly when its decode-tile launch fits
    one wave (heuristics.d128_dense_waves, issue #1518); the wave counts are patched, so no measured boundary is pinned."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == f"sdpa_fwd_prefill_{arch}")
    values = dict(
        b=2, h_q=32, h_kv=8, s_q=384, s_kv=384, d_qk=128, d_v=128, dtype=cudnn.data_type.BFLOAT16, causal=True, has_sink=sink, device_cc=cc, device_sm_count=sms
    )
    for waves, causal, expected in (((1, 1), True, placement.LEAD), ((2, 1), True, placement.TRAIL), ((1, 1), False, placement.TRAIL)):
        monkeypatch.setattr(heuristics, "d128_dense_waves", lambda caps, facts, pack_gqa=None, waves=waves: waves)
        assert placement.place(spec, _facts(**dict(values, causal=causal))) == expected, (waves, causal)
    monkeypatch.undo()
    if not sink:
        assert placement.place(spec, _facts(**dict(values, s_q=8, s_kv=32768, b=64))) == placement.LEAD  # decode-shaped: every unit count


def _hnd_pool(h_kv, page_size, d=256):
    from types import SimpleNamespace

    return SimpleNamespace(
        get_stride=lambda: (h_kv * page_size * d, page_size * d, d, 1)
    )  # [pages, H_kv, page, D] head-major: each head's page contiguous (vLLM's cache; measured on the 24/2 chunk)


def _nhd_pool(h_kv, page_size, d=256):
    from types import SimpleNamespace

    return SimpleNamespace(get_stride=lambda: (page_size * h_kv * d, d, h_kv * d, 1))  # [pages, H_kv, page, D] token-major in the page: the band's pool


def _sm107_paged_d256_decode_facts(**over):
    """FlashInfer's cuDNN decode graph shape on cc 10.7: a dense Q over a page pool with the padding mask (per-request KV lengths), no
    mask / sink / Stats (the pool strided as FlashInfer's benchmark strides it: token-major in the page -- the band's pool; the head-major
    pool is the measured twin: the 24/2 chunk)."""
    values = dict(
        b=4,
        h_q=24,
        h_kv=2,
        s_q=1,
        s_kv=8192,
        d_qk=256,
        d_v=256,
        dtype=cudnn.data_type.BFLOAT16,
        causal=False,
        bottom_right=False,
        has_paged_kv=True,
        padded=True,
        page_size=16,
        device_cc=(10, 7),
        device_sm_count=212,
    )
    values.update(over)
    values.setdefault("k_t", _nhd_pool(values["h_kv"], values["page_size"]))
    return _facts(**values)


_SM107_PAGED_D256_DECODE_CORNERS = [
    # (heads, b, declared KV, verdict): the measured frontier of the Rubin paged d256 decode-tile band -- units = b * h_kv,
    # LEAD from 4k declared KV at any unit count, from 32 units at any declared KV, and from 4 units at 2**16 KV tokens in flight
    # (re-measured 2026-11-17); TRAIL at 4-31 units below 2**16 with <= 2k declared KV, at 1-3 units below 4k, past the measured
    # 512 units / 128k declared KV, below 1k.
    ("24/2", 4, 8192, placement.LEAD),
    ("24/2", 4, 4096, placement.LEAD),
    ("24/2", 32, 1024, placement.LEAD),
    ("24/2", 16, 1024, placement.LEAD),
    ("24/2", 16, 2048, placement.LEAD),
    ("24/2", 8, 2048, placement.TRAIL),
    ("24/2", 8, 4096, placement.LEAD),
    ("24/2", 2, 32768, placement.LEAD),
    ("24/2", 2, 8192, placement.LEAD),
    ("24/2", 1, 131072, placement.LEAD),
    ("8/1", 1, 4096, placement.LEAD),
    ("8/1", 3, 4096, placement.LEAD),
    ("8/1", 3, 2048, placement.TRAIL),
    ("8/1", 16, 2048, placement.TRAIL),
    ("8/1", 8, 1024, placement.TRAIL),
    ("8/1", 31, 1024, placement.TRAIL),
    ("24/2", 128, 131072, placement.LEAD),
    ("24/2", 32, 262144, placement.TRAIL),
    ("24/2", 64, 512, placement.TRAIL),
    ("32/2", 2, 32768, placement.LEAD),
    ("32/2", 2, 8192, placement.LEAD),
    ("16/4", 128, 131072, placement.LEAD),
    ("16/4", 256, 131072, placement.TRAIL),
    ("16/4", 1, 32768, placement.LEAD),
    ("16/4", 1, 8192, placement.LEAD),
    ("24/4", 2, 8192, placement.LEAD),
    ("24/4", 2, 4096, placement.LEAD),
    ("8/1", 8, 8192, placement.LEAD),
    ("8/1", 8, 4096, placement.LEAD),
    ("8/1", 4, 8192, placement.LEAD),
    ("12/1", 16, 4096, placement.LEAD),
    ("12/1", 16, 2048, placement.TRAIL),
    ("16/1", 64, 1024, placement.LEAD),
    ("16/1", 32, 1024, placement.LEAD),
    # group 32 (64/2, the 32-column tile in one unit): LEAD from 4 units and 2**17 KV tokens in flight (the 2**16 step holds the 8 x 8k eager
    # parity cell), the measured 256-unit / 128k edges, 2 units only at the 128k endpoint (not admitted)
    ("64/2", 32, 8192, placement.LEAD),
    ("64/2", 2, 32768, placement.LEAD),
    ("64/2", 2, 8192, placement.TRAIL),
    ("64/2", 4, 8192, placement.TRAIL),
    ("64/2", 4, 32768, placement.LEAD),
    ("64/2", 8, 4096, placement.TRAIL),
    ("64/2", 8, 8192, placement.LEAD),
    ("64/2", 16, 2048, placement.TRAIL),
    ("64/2", 16, 4096, placement.LEAD),
    ("64/2", 32, 1024, placement.TRAIL),
    ("64/2", 32, 2048, placement.LEAD),
    ("64/2", 64, 1024, placement.LEAD),
    ("64/2", 128, 131072, placement.LEAD),
    ("64/2", 256, 131072, placement.TRAIL),
    ("64/2", 1, 131072, placement.TRAIL),
    ("64/2", 32, 262144, placement.TRAIL),
    ("32/1", 64, 8192, placement.TRAIL),  # the same group 32 on an unmeasured geometry (one KV head): the backend stays first
]

_SM107_PAGED_D256_MTP_CORNERS = [
    # (heads, s_q, b, declared KV, dtype, page, verdict): the MTP rows -- FlashInfer's decode graph at q_len_per_req 2-4 (the bottom-right
    # causal diagonal + the padding mask) -- are the generated table's (study sm107_paged_mtp_20261110: bf16 GQA 4-32, b 1-256, page
    # 16 / 32 / 64, declared KV 16 to 1M). FROST leads from 2**7.5 KV tokens on the 128-key tile and from 2**7.01 off it. The old hand
    # box (24/2 and 32/2, page 16, b 1-128, KV 1k-128k) is inside; f16 has no entry and keeps the backend first.
    ("24/2", 2, 1, 1024, cudnn.data_type.BFLOAT16, 16, placement.LEAD),
    ("24/2", 3, 4, 8192, cudnn.data_type.BFLOAT16, 16, placement.LEAD),
    ("24/2", 4, 128, 131072, cudnn.data_type.BFLOAT16, 16, placement.LEAD),
    ("32/2", 2, 32, 4096, cudnn.data_type.BFLOAT16, 16, placement.LEAD),
    ("32/2", 4, 1, 131072, cudnn.data_type.BFLOAT16, 16, placement.LEAD),
    ("24/2", 2, 1, 512, cudnn.data_type.BFLOAT16, 16, placement.LEAD),
    ("24/2", 2, 1, 128, cudnn.data_type.BFLOAT16, 16, placement.TRAIL),  # one KV tile on the tile: below the 2**7.5 threshold
    ("24/2", 2, 1, 125, cudnn.data_type.BFLOAT16, 16, placement.TRAIL),  # off the tile below 2**7.01 KV tokens
    ("24/2", 2, 1, 133, cudnn.data_type.BFLOAT16, 16, placement.LEAD),
    ("24/2", 2, 32, 262144, cudnn.data_type.BFLOAT16, 16, placement.LEAD),
    ("24/2", 2, 256, 8192, cudnn.data_type.BFLOAT16, 16, placement.LEAD),
    ("16/4", 2, 32, 8192, cudnn.data_type.BFLOAT16, 16, placement.LEAD),
    ("64/2", 2, 32, 8192, cudnn.data_type.BFLOAT16, 16, placement.LEAD),
    ("12/1", 2, 64, 8192, cudnn.data_type.BFLOAT16, 16, placement.LEAD),
    ("16/1", 3, 64, 8192, cudnn.data_type.BFLOAT16, 16, placement.LEAD),
    ("24/2", 2, 32, 8192, cudnn.data_type.BFLOAT16, 32, placement.LEAD),
    ("24/2", 2, 32, 8192, cudnn.data_type.HALF, 16, placement.LEAD),  # f16 borrows the bf16 entry (f16_transfer_rubin_20261127)
]


@pytest.mark.L0
@pytest.mark.parametrize("replay", [False, True], ids=["eager", "replay"])
@pytest.mark.parametrize(
    "heads, b, s_kv, verdict", _SM107_PAGED_D256_DECODE_CORNERS, ids=[f"{h}-b{b}-kv{kv}" for h, b, kv, _ in _SM107_PAGED_D256_DECODE_CORNERS]
)
def test_sm107_paged_d256_decode_tile_leads_on_its_measured_corner_cells(heads, b, s_kv, verdict, replay):
    """The paged dense-Q d256 decode-tile shard of ``_place_sm107_f16`` (cudnn-frontend 1.32.0, the release this default ships in --
    FlashInfer's ``(10, 7)`` cuDNN decode admission keys on it): the REAL constants pin each verdict (``placement.LEAD`` or ``placement.TRAIL``) at the measured
    frontier cells of the band (module docstring, "PAGED dense-Q d256 half decode"; 212-SM cc 10.7 perf class, cuDNN 9.27.0, CUDA-graph
    replay and eager), the same verdict in both regimes -- the frontier the two regimes share.  Before this arm every one of these graphs
    kept the backend first (the RED half of this test on the pre-arm tree)."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    assert tuple(int(x) for x in cudnn.__version__.split(".")[:3]) >= (1, 32, 0)
    spec = next(spec for spec in ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm107")
    h_q, h_kv = (int(x) for x in heads.split("/"))
    facts = _sm107_paged_d256_decode_facts(h_q=h_q, h_kv=h_kv, b=b, s_kv=s_kv, cuda_graph_replay=replay)
    assert placement.place(spec, facts) == verdict


@pytest.mark.L0
@pytest.mark.parametrize("replay", [False, True], ids=["eager", "replay"])
@pytest.mark.parametrize(
    "heads, s_q, b, s_kv, dtype, page, verdict",
    _SM107_PAGED_D256_MTP_CORNERS,
    ids=[f"{h}-q{q}-b{b}-kv{kv}-{'f16' if dt == cudnn.data_type.HALF else 'bf16'}-p{p}" for h, q, b, kv, dt, p, _ in _SM107_PAGED_D256_MTP_CORNERS],
)
def test_sm107_paged_d256_mtp_rows_follow_the_generated_table(heads, s_q, b, s_kv, dtype, page, verdict, replay):
    """FlashInfer's paged decode graph at ``q_len_per_req`` 2-4 (the bottom-right causal diagonal with the padding mask) on cc 10.7 is
    placed by study sm107_paged_mtp_20261110 (f16 through its transfer, study f16_transfer_rubin_20261127), the same verdict in both
    regimes; without the diagonal the graph is another class with no entry and keeps the backend first. With a window (no sink) the bf16
    graph is study sm107_window_gaps_20261118's class, which f16 does not borrow."""
    from cudnn.sdpa import placement_table
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm107")
    h_q, h_kv = (int(x) for x in heads.split("/"))
    shape = dict(h_q=h_q, h_kv=h_kv, b=b, s_q=s_q, s_kv=s_kv, dtype=dtype, page_size=page, cuda_graph_replay=replay)
    facts = _sm107_paged_d256_decode_facts(causal=True, bottom_right=True, **shape)
    assert placement.place(spec, facts) == verdict
    entry = placement_table.matched_entry(spec.name, facts)
    assert entry is not None and entry.study == "sm107_paged_mtp_20261110"
    assert placement.place(spec, _sm107_paged_d256_decode_facts(**shape)) == placement.TRAIL
    windowed = _sm107_paged_d256_decode_facts(causal=True, bottom_right=True, window_left=128, **shape)
    off_tile = s_kv % 128 != 0  # the off-tile class leads from cost 2**29: the 125 / 133-key rows trail
    assert placement.place(spec, windowed) == (placement.TRAIL if dtype == cudnn.data_type.HALF or off_tile else placement.LEAD)
    if dtype != cudnn.data_type.HALF:
        assert placement_table.matched_entry(spec.name, windowed).study == "sm107_window_gaps_20261118"


@pytest.mark.L0
def test_sm107_dense_d128_chunks_lead_up_to_256_tiles():
    """Study sm107_dense_d128_chunk_tiles_20261117: Rubin dense bf16 d128 chunks from a 4k cache lead up to 256 launch tiles
    (llama3.1 64/8 b 1 at S_q 512 ran the backend at 1.4-1.6x); past it, and for f16 or other widths past 128, the backend stays."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm107")
    base = dict(b=1, h_q=64, h_kv=8, s_q=512, s_kv=131072, d_qk=128, d_v=128, causal=True, bottom_right=True, device_cc=(10, 7), device_sm_count=208)
    base.update(has_paged_kv=False, padded=False, page_size=0)
    place = lambda **over: placement.place(spec, _facts(**{**base, **over}))
    assert placement.SM107_D128_CHUNKED_MAX_Q_TILES == 256
    assert place() == placement.LEAD  # 256 tiles
    assert place(h_q=32, h_kv=4, s_q=1024) == placement.LEAD  # 256 tiles
    assert place(s_q=1024) == placement.LEAD  # 512 tiles: group 8 packs whole (sm107_dense_d128_longkv_v2_20261118)
    assert place(h_q=56, s_q=1024) == placement.TRAIL  # 448 tiles: group 7 runs unpacked and keeps the cap
    assert place(dtype=cudnn.data_type.HALF) == placement.TRAIL  # f16 keeps 128 tiles
    assert place(h_q=32, h_kv=4) == placement.LEAD  # 128 tiles, every width
    assert place(b=2, h_q=56, h_kv=8, s_q=192, s_kv=16384) == placement.TRAIL  # 224 tiles below S_q 256: the edge supplement lost 1.37x


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131073], ids=["on_tile", "off_tile"])
def test_sm107_dense_d256_long_cache_chunks_lead_past_the_tile_cap(s_kv):
    """Study sm107_dense_d256_longkv_20261117: Rubin bf16 dense d256 GQA causal chunks past the 128-tile cap lead from a 32k cache (qwen3.5
    32/2 at b 1, S_q 1024 over 128k ran the backend at 2.3-2.95x). Study rubin_dense_thd_prefill_20261127 extends the lead to any d256
    chunk over a cache of at least 8 x S_q (shorter S_q and MHA included); wider chunks and f16 keep the tile cap."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm107")
    base = dict(b=1, h_q=32, h_kv=2, s_q=1024, s_kv=s_kv, d_qk=256, d_v=256, causal=True, bottom_right=True, device_cc=(10, 7), device_sm_count=208)
    base.update(has_paged_kv=False, padded=False, page_size=0)
    place = lambda **over: placement.place(spec, _facts(**{**base, **over}))
    assert placement.SM107_D256_LONG_CHUNK_MIN_KV == 32768
    assert place() == placement.LEAD  # 256 tiles over a long cache
    assert place(b=8, h_q=64, h_kv=8) == placement.LEAD  # 4096 tiles
    assert place(s_kv=4096 + s_kv % 2) == placement.TRAIL  # a 4 x S_q cache, past the tile cap
    assert place(s_kv=16384 + s_kv % 2) == placement.LEAD  # below the 32k floor, but 16 x S_q (rubin_dense_thd_prefill_20261127)
    assert place(s_q=192, b=8, s_kv=65536 + s_kv % 2) == placement.LEAD  # below S_q 256, likewise
    assert place(h_kv=32, s_kv=65536 + s_kv % 2) == placement.LEAD  # MHA, likewise
    assert place(dtype=cudnn.data_type.HALF) == placement.TRAIL


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131073], ids=["on_tile", "off_tile"])
@pytest.mark.parametrize("heads", [(64, 64), (128, 8)], ids=["mha", "gqa"])
def test_sm107_dense_d192x128_long_cache_chunks_lead_past_the_tile_cap(s_kv, heads):
    """Study sm107_dense_d192x128_long_cache_20261118: Rubin bf16 dense d192 x 128 causal chunks, MHA and GQA, lead from S_q 512 over a
    32k cache (kimi_k2 / deepseek_v3 / mimo chunks over 64k-128k ran the backend at 2.0-2.4x); 256-query chunks and short caches keep the
    hand rule's tile cap, and f16 was not measured."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm107")
    base = dict(b=1, h_q=heads[0], h_kv=heads[1], s_q=1024, s_kv=s_kv, d_qk=192, d_v=128, causal=True, bottom_right=True, device_cc=(10, 7))
    base.update(device_sm_count=224, has_paged_kv=False, padded=False, page_size=0)
    place = lambda **over: placement.place(spec, _facts(**{**base, **over}))
    assert (placement.SM107_D192_LONG_CHUNK_MIN_S_Q, placement.SM107_D192_LONG_CHUNK_MIN_KV) == (512, 32768)
    assert place() == placement.LEAD
    assert place(s_q=512, s_kv=65536 + s_kv % 2) == placement.LEAD
    assert place(s_q=256, b=4) == placement.TRAIL  # below S_q 512, past the tile cap
    assert place(s_kv=16384 + s_kv % 2, b=4) == placement.TRAIL  # below the cache floor, past the tile cap
    assert place(dtype=cudnn.data_type.HALF) == placement.TRAIL


@pytest.mark.L0
@pytest.mark.parametrize("s_kv", [131072, 131073], ids=["on_tile", "off_tile"])
def test_sm107_dense_d128_whole_pack_chunks_lead_to_512_tiles(s_kv):
    """Study sm107_dense_d128_longkv_v2_20261118: Rubin bf16 dense d128 GQA causal chunks whose group the row packs whole lead up to 512
    launch tiles from an 8k cache (llama3.1 chunks at b 1 over 8k-128k ran the backend at 1.12-1.52x); groups the row runs unpacked keep
    the 256-tile cap (they lost up to 2.0x past it). Past the cap, a chunk the row splits over KV is the split class's
    (dense_d128_split_key_bw_20261123), which leads."""
    from cudnn.sdpa import placement_table
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm107")
    base = dict(b=1, h_q=64, h_kv=8, s_q=1024, s_kv=s_kv, d_qk=128, d_v=128, causal=True, bottom_right=True, device_cc=(10, 7))
    base.update(device_sm_count=224, has_paged_kv=False, padded=False, page_size=0)
    place = lambda **over: placement.place(spec, _facts(**{**base, **over}))
    assert (placement.SM107_D128_WHOLE_PACK_MAX_Q_TILES, placement.SM107_D128_WHOLE_PACK_MIN_KV) == (512, 8192)
    assert place() == placement.LEAD  # 512 tiles, group 8 packs whole
    assert place(h_q=32, h_kv=4, s_q=2048) == placement.LEAD  # llama3.1-tp2
    assert place(s_q=4096) == placement.TRAIL  # 2048 tiles, unsplit
    split = _facts(**{**base, "s_q": 2048})  # 1024 tiles, a 2-way KV split: llama3.1 b 1 over a 128k cache
    assert placement_table.class_key("sdpa_fwd_prefill_sm107", split)[8] == "gqa_split"
    assert placement.place(spec, split) == placement.LEAD
    assert place(h_q=48) == placement.TRAIL  # group 6 runs unpacked: 384 tiles past the 256-tile cap
    assert place(s_kv=4096 + s_kv % 2, s_q=1024) == placement.TRAIL  # below the cache floor
    assert place(dtype=cudnn.data_type.HALF) == placement.TRAIL


@pytest.mark.L0
def test_sm107_d128_window_unpacked_groups_lead_at_few_units_over_wide_windows():
    """Study sm107_d128_window_unpacked_v2_20261118: Rubin bf16 paged d128 windowed MTP / verify whose group the row does not pack whole
    leads at b x h_kv <= 4 with a window of at least 8 KV tiles (ripple_swa-tp8 10/1 b 1 window 1024 ran the backend at 1.73x)."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm107")
    base = dict(b=1, h_q=10, h_kv=1, s_q=4, s_kv=131072, d_qk=128, d_v=128, causal=True, bottom_right=True, window_left=1024)
    base.update(device_cc=(10, 7), device_sm_count=224, has_paged_kv=True, padded=True, page_size=16)
    place = lambda **over: placement.place(spec, _facts(**{**base, **over}))
    assert (placement.SM107_D128_WINDOW_UNPACKED_MAX_UNITS, placement.SM107_D128_WINDOW_UNPACKED_MIN_TILES) == (4, 8)
    assert place() == placement.LEAD  # ripple_swa-tp8 MTP
    assert place(s_q=8) == placement.LEAD  # ... and verify
    assert place(b=8) == placement.TRAIL  # 8 units
    assert place(window_left=512) == placement.TRAIL  # 4-tile window


@pytest.mark.L0
def test_sm107_d128_window_mtp_leads_outside_two_long_cache_bands():
    """Study sm107_d128_window_mtp_v4_20261118: Rubin bf16 paged d128 windowed MTP on the tile leads, except over caches of 2^16 keys
    or more with 32-95 units and a window of at most 8 KV tiles, or a window of at most 4 tiles at S_q >= 4 below 96 units.
    step37_swa b 128 (window 512, 128k keys) ran the backend at 5.8-6.1x FROST's time."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm107")
    base = dict(b=128, h_q=64, h_kv=8, s_q=4, s_kv=131072, d_qk=128, d_v=128, causal=True, bottom_right=True, window_left=512)
    base.update(device_cc=(10, 7), device_sm_count=224, has_paged_kv=True, padded=True, page_size=16)
    place = lambda **over: placement.place(spec, _facts(**{**base, **over}))
    assert place() == placement.LEAD  # step37_swa b 128
    assert place(b=8, h_q=32) == placement.TRAIL  # 64 units, 4-tile window, 128k keys
    assert place(b=1) == placement.TRAIL  # 8 units at S_q 4 with a 4-tile window: within the boards' gap
    assert place(b=1, s_q=2) == placement.LEAD
    assert place(b=8, h_q=32, window_left=2048) == placement.LEAD  # 16-tile window
    assert place(b=8, h_q=32, s_kv=16384) == placement.LEAD  # short cache


@pytest.mark.L0
def test_sm107_dense_d256_decode_leads_from_32_planned_ctas():
    """Study sm107_dense_d256_q1_cta_v2_20261118: Rubin bf16 dense d256 GQA single-row decode leads when the decode tile's eager plan
    launches at least 32 CTAs (b x h_q / pack x split). FROST lost only unsplit launches under 32 CTAs; qwen3.5 b 1 over 8k-32k keys,
    split 64-128, ran 0.55-0.75 of the backend that the retired cost entry chose."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm107")
    base = dict(b=1, h_q=8, h_kv=1, s_q=1, s_kv=8192, d_qk=256, d_v=256, causal=False, bottom_right=False, device_cc=(10, 7))
    base.update(device_sm_count=224, has_paged_kv=False, padded=False, page_size=0)
    place = lambda **over: placement.place(spec, _facts(**{**base, **over}))
    assert placement.SM107_D256_DECODE_MIN_CTAS == 32
    assert place() == placement.LEAD  # split 64
    assert place(s_kv=32768) == placement.LEAD
    assert place(b=4, s_kv=2048) == placement.TRAIL  # unsplit, 4 CTAs
    assert place(b=32, s_kv=2048) == placement.LEAD  # unsplit, 32 CTAs
    assert place(b=31, s_kv=2048) == placement.TRAIL
    assert place(b=32, s_kv=2049) == placement.LEAD  # off the tile dense KV runs unsplit
    assert place(b=16, s_kv=2049) == placement.TRAIL
    assert place(s_kv=8193) == placement.TRAIL  # one CTA
    assert place(b=2, s_kv=2048, cuda_graph_replay=True) == placement.TRAIL  # the captured split (16) was not timed
    assert place(dtype=cudnn.data_type.HALF) == placement.TRAIL  # f16 is not in the class


@pytest.mark.L0
@pytest.mark.parametrize("d", [(256, 256), (192, 128)])
@pytest.mark.parametrize("s_q", [5, 8, 16])
def test_sm107_paged_d256_past_the_mtp_rows_follows_the_generated_table(s_q, d):
    """Past the MTP rows (s_q >= 5) FlashInfer's paged decode graph with the bottom-right diagonal is the paged prefill class, which
    study sm107_paged_prefill_20261109 leads: on this dense-Q graph FROST measured 0.03-0.61 of the backend's plan at s_q 5 / 8 / 16 (216
    cells, d256 and D192xV128, 208-SM cc 10.7, cuDNN 9.30.0.2), as on the THD prefill graph the study fit. Without the diagonal the
    graph is the mask-free paged prefill class: study sm107_thd_prefill_20261111 leads d256 there and has no D192 entry (the backend
    failed to build it), which keeps the backend first. With a window, d256 is study sm107_window_gaps_20261118's class (LEAD) and
    D192xV128 has no entry (the backend first)."""
    from cudnn.sdpa import placement_table
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm107")
    shape = dict(b=32, s_q=s_q, s_kv=8192, d_qk=d[0], d_v=d[1])
    assert placement.place(spec, _sm107_paged_d256_decode_facts(causal=True, bottom_right=True, **shape)) == placement.LEAD
    mask_free = _sm107_paged_d256_decode_facts(**shape)
    assert placement.place(spec, mask_free) == (placement.LEAD if d == (256, 256) else placement.TRAIL)
    assert (placement_table.matched_entry(spec.name, mask_free) is None) == (d != (256, 256))
    windowed = _sm107_paged_d256_decode_facts(causal=True, bottom_right=True, window_left=128, **shape)
    assert placement.place(spec, windowed) == (placement.LEAD if d == (256, 256) else placement.TRAIL)
    assert (placement_table.matched_entry(spec.name, windowed) is None) == (d != (256, 256))


@pytest.mark.L0
@pytest.mark.parametrize(
    "outside",
    [
        None,
        dict(device_cc=(10, 0)),
        dict(padded=False),  # a paged decode WITHOUT per-request KV lengths: every band cell carried the padding mask -> unmeasured
        dict(s_q=2),  # two query tokens WITHOUT the diagonal: not FlashInfer's MTP graph, unmeasured (the MTP band keys on the diagonal)
        dict(h_q=64, b=4, s_kv=8192),  # group 32 one step below its band (2**16 tokens in flight holds the parity cell)
        dict(h_q=64, page_size=32),  # group 32 at page 32: measured on 24/2 only, not on 64/2
        dict(h_q=4),  # group 2: not measured
        dict(h_q=2),  # MHA: not measured
        dict(h_q=36, h_kv=8),  # a partial group
        dict(d_qk=128, d_v=128),  # the d128 flavor has no paged decode tile on cc 10.7
        dict(d_qk=192, d_v=128),
        dict(dtype=cudnn.data_type.FLOAT),
        dict(wants_stats=True),
        dict(has_sink=True),
        dict(causal=True, bottom_right=True),
        dict(right_band_widening=True, right_bound=8),
        dict(attn_scale_prefolded=True),
        dict(has_epilogue_gate=True),
        dict(page_size=128),
        dict(thd=True),  # ragged queries never ride the dense-Q decode tile
    ],
    ids=[
        "inside",
        "sm100",
        "unpadded",
        "q2_no_diagonal",
        "g32_below_band",
        "g32_page32",
        "g2",
        "mha",
        "partial_group",
        "d128",
        "d192",
        "fp32",
        "stats",
        "sink",
        "causal",
        "right_band",
        "prefolded",
        "gate",
        "page128",
        "thd",
    ],
)
def test_sm107_paged_d256_decode_tile_shard_stays_inside_its_measured_shape(outside):
    """Inside the band the shard's SHAPE is FlashInfer's decode graph on the Rubin d256 decode tile (``engines.d256_decode_tile_selected``
    with the whole group packed; a page pool of a measured page size; no mask / window / sink / Stats / gate); every variant outside it
    keeps the backend first.  The leading cell (24/2, b 32, declared KV 8k = 64 units x 8k) holds for f16, pages 32 / 64, the CUDA-graph
    replay hint, execute-time shape overrides and both in-page pool layouts alike -- the token-major one the band measured (FlashInfer's
    benchmark pool) and the head-major one (each head's page contiguous, vLLM's cache) measured on the 24/2 chunk: the tile faster on every
    in-band cell in both regimes -- so the arm reads no pool stride at all (a pool without readable strides is placed like any other)."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(spec for spec in ENGINE_SPECS if spec.name == "sdpa_fwd_prefill_sm107")
    values = dict(b=32)
    values.update(outside or {})
    assert placement.place(spec, _sm107_paged_d256_decode_facts(**values)) == (placement.TRAIL if outside else placement.LEAD)
    if outside is None:
        assert placement.place(spec, _sm107_paged_d256_decode_facts(b=32, s_q=2, causal=True, bottom_right=True)) == placement.LEAD  # the MTP rows (table)
        # a window leaves the shard: study sm107_window_gaps_20261118 (table) leads windowed paged d256 decode
        assert placement.place(spec, _sm107_paged_d256_decode_facts(b=32, window_left=128, causal=True, bottom_right=True)) == placement.LEAD
        # dense K/V leaves the shard: the dense d256 decode arm leads this cell (64 planned CTAs)
        assert placement.place(spec, _sm107_paged_d256_decode_facts(b=32, has_paged_kv=False, page_size=0, padded=False)) == placement.LEAD
        assert placement.place(spec, _sm107_paged_d256_decode_facts(b=32, h_q=64)) == placement.LEAD  # group 32 inside its band (64 units x 8k)
        assert placement.place(spec, _sm107_paged_d256_decode_facts(b=32, h_q=64, dtype=cudnn.data_type.HALF)) == placement.LEAD  # its f16 twin
        assert placement.place(spec, _sm107_paged_d256_decode_facts(b=32, dtype=cudnn.data_type.HALF)) == placement.LEAD
        assert placement.place(spec, _sm107_paged_d256_decode_facts(b=32, page_size=32)) == placement.LEAD
        assert placement.place(spec, _sm107_paged_d256_decode_facts(b=32, page_size=64)) == placement.LEAD
        assert placement.place(spec, _sm107_paged_d256_decode_facts(b=32, cuda_graph_replay=True)) == placement.LEAD
        assert placement.place(spec, _sm107_paged_d256_decode_facts(b=32, shape_overrides=True)) == placement.LEAD
        assert placement.place(spec, _sm107_paged_d256_decode_facts(b=32, k_t=_hnd_pool(2, 16))) == placement.LEAD  # the head-major pool: the 24/2 chunk
        assert placement.place(spec, _sm107_paged_d256_decode_facts(b=32, k_t=None)) == placement.LEAD  # no pool strides to read: the layout is not a key
        assert (
            placement.place(spec, _sm107_paged_d256_decode_facts(h_q=8, h_kv=1, b=64, k_t=_hnd_pool(1, 16))) == placement.LEAD
        )  # one KV head: one in-page order
        assert placement.place(spec, _sm107_paged_d256_decode_facts(h_q=8, h_kv=1, b=64, k_t=_nhd_pool(1, 16))) == placement.LEAD


@pytest.mark.L0
@pytest.mark.parametrize(
    "row, cc, sms, quant",
    [
        ("sdpa_fwd_sm100_d512_2x2", (10, 0), 148, {}),
        ("sdpa_fwd_sm100_d512_fp8_role_split", (10, 0), 148, _SM107_FP8_QUANT),
        ("sdpa_fwd_sm90_f16_three_wg", (9, 0), 132, {}),
        ("sdpa_fwd_sm107_d512_2x2", (10, 7), 216, {}),
        ("sdpa_fwd_sm107_d512_fp8_role_split", (10, 7), 216, _SM107_FP8_QUANT),
    ],
)
def test_d512_sliding_window_leads_inside_the_measured_box(row, cc, sms, quant, monkeypatch):
    """Each d512 row admits a dense causal sliding window, proposes only admissible sets for it, and its window gate reads
    the module's bounds (gate structure, not a measured winner)."""
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS, mismatch

    if cc == (10, 7):
        # metadata-only: the synthetic Rubin rows must not depend on the installed DSL's target set
        from cudnn.frost import buffers

        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    spec = next(s for s in ENGINE_SPECS if s.name == row)
    swa = {"b": 2, "h_q": 64, "h_kv": 1, "s_q": 4096, "s_kv": 4096, "d_qk": 512, "d_v": 512, "causal": True, "window_left": 128, **quant}
    swa.update(device_cc=cc, device_sm_count=sms)

    def facts(**over):
        return _facts(**{**swa, **over})

    assert mismatch(spec.capabilities, facts()) is None
    offered = {row: _OFFERED[row]}
    plans = [p for p in heuristics.recommend("A", facts(), offered) if p.engine_id == _OFFERED[row]]
    assert plans and all(mismatch(spec.capabilities, facts(), p.knobs) is None for p in plans)
    assert placement._d512_swa(facts(s_q=placement.D512_SWA_MIN_S_Q, s_kv=32768, bottom_right=True))
    assert not placement._d512_swa(facts(s_q=placement.D512_SWA_MIN_S_Q - 1, s_kv=32768, bottom_right=True))
    assert not placement._d512_swa(facts(s_q=placement.D512_SWA_MIN_KV // 2, s_kv=placement.D512_SWA_MIN_KV // 2))


def _sm107_paged_facts(**over):
    from types import SimpleNamespace

    base = dict(
        b=3,
        h_q=8,
        h_kv=2,
        s_q=864,
        s_kv=16384,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        causal=True,
        bottom_right=True,
        thd=True,
        padded=True,
        has_paged_kv=True,
        page_size=16,
        k_t=SimpleNamespace(get_stride=lambda: (8192, 4096, 256, 1)),
        max_total_seq_len_q=3 * 864,
        device_cc=(10, 7),
        device_sm_count=216,
    )
    base.update(over)
    return _facts(**base)


@pytest.mark.L0
def test_sm107_paged_gqa_prefill_leads_inside_its_box():
    """Rubin paged bf16 d128 GQA 4/8/16 prefill chunks (exact or bounded) and full squares lead; MHA, group 2,
    short queries and out-of-box batches keep the backend. The box is the hand rule; an exact graph is answered first by
    study sm107_thd_prefill_20261111, so the box below is read through the row's hand rules."""
    from cudnn.sdpa import placement_table
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107")
    assert placement_table.matched_entry(spec.name, _sm107_paged_facts()).study == "sm107_thd_prefill_20261111"
    assert placement_table.matched_entry(spec.name, _sm107_paged_facts(shape_overrides=True)) is None
    place = lambda **over: placement._place_sm107_f16(spec.capabilities, _sm107_paged_facts(**over))
    assert place() == placement.LEAD
    assert place(h_q=32, h_kv=8, b=4, s_q=751, s_kv=21650, max_total_seq_len_q=4 * 751) == placement.LEAD
    assert place(h_q=64, h_kv=64, s_q=628, s_kv=28240) == placement.TRAIL
    assert place(h_q=64, h_kv=32) == placement.TRAIL
    assert place(h_q=32, h_kv=2) == placement.LEAD  # GQA16
    assert place(h_q=32, h_kv=2, shape_overrides=True) == placement.LEAD  # bounded GQA16
    assert place(h_q=32, h_kv=2, b=1, s_q=2048, s_kv=2048, bottom_right=False, max_total_seq_len_q=2048) == placement.LEAD  # GQA16 square
    assert place(h_q=32, h_kv=4, shape_overrides=True) == placement.LEAD  # bounded GQA8
    assert place(h_q=32, h_kv=4, b=1, s_q=2048, s_kv=2048, bottom_right=False, max_total_seq_len_q=2048) == placement.LEAD  # G8 square
    assert place(h_q=32, h_kv=2, b=1, s_q=1536, s_kv=1536, max_total_seq_len_q=1536) == placement.LEAD  # bottom-right square
    assert place(h_q=32, h_kv=2, b=1, s_q=1024, s_kv=1024, bottom_right=False, max_total_seq_len_q=1024) == placement.LEAD
    assert place(h_q=32, h_kv=8, b=1, s_q=2048, s_kv=2048, bottom_right=False, max_total_seq_len_q=2048) == placement.LEAD  # G4 square
    assert place(h_q=32, h_kv=4, b=1, s_q=8192, s_kv=8192, max_total_seq_len_q=8192) == placement.LEAD
    assert place(h_q=32, h_kv=8, b=2, s_q=4096, s_kv=4096, shape_overrides=True, max_total_seq_len_q=None) == placement.LEAD  # bounded square
    assert not placement._sm107_paged_gqa_prefill(_sm107_paged_facts(h_q=32, h_kv=8, b=1, s_q=512, s_kv=512, bottom_right=False))
    assert not placement._sm107_paged_gqa_prefill(_sm107_paged_facts(h_q=32, h_kv=4, b=1, s_q=12288, s_kv=12288))
    assert not placement._sm107_paged_gqa_prefill(_sm107_paged_facts(h_q=32, h_kv=16, b=1, s_q=2048, s_kv=2048))  # group 2 square
    assert placement._sm107_paged_gqa_prefill(_sm107_paged_facts(h_q=32, h_kv=8, shape_overrides=True))  # bounded GQA4
    assert place(h_q=32, h_kv=16, shape_overrides=True) == placement.TRAIL  # bounded group 2
    assert place(b=placement.SM107_PAGED_GQA_MAX_B + 1, max_total_seq_len_q=5 * 864) == placement.TRAIL
    assert place(dtype=cudnn.data_type.HALF) == placement.TRAIL
    assert not placement._sm107_paged_gqa_prefill(_sm107_paged_facts(s_kv=65536))  # the box ends at 32768 ...
    assert place(s_kv=65536) == placement.LEAD  # ... and the split leads past it (paged_thd_long_kv_split_rubin_v2_20261127)
    assert place(s_kv=131073) == placement.TRAIL
    # long-prefix chunks: the measured grid's corners lead for every group, its edges bound the box
    for h_q, h_kv in ((32, 8), (32, 4), (64, 4)):
        for b, s_q, s_kv in ((1, 1536, 6144), (4, 1536, 32768), (1, 4096, 6144), (4, 4096, 32768)):
            assert place(h_q=h_q, h_kv=h_kv, b=b, s_q=s_q, s_kv=s_kv, max_total_seq_len_q=b * s_q) == placement.LEAD, (h_q, h_kv, b, s_q, s_kv)
    long = dict(b=1, s_q=2048, max_total_seq_len_q=2048)
    assert place(h_q=32, h_kv=8, s_kv=placement.SM107_PAGED_GQA_LONG_CHUNK_MIN_KV - 1, **long) == placement.TRAIL
    assert place(h_q=32, h_kv=8, b=1, s_q=4097, s_kv=8192, max_total_seq_len_q=4097) == placement.TRAIL
    assert place(h_q=32, h_kv=16, s_kv=8192, **long) == placement.TRAIL  # group 2
    assert place(h_q=32, h_kv=32, s_kv=8192, **long) == placement.TRAIL  # MHA


@pytest.mark.L0
@pytest.mark.parametrize("cc, sms", [((10, 0), 148), ((10, 3), 160)], ids=["b200", "b300"])
def test_sm100_paged_mixed_full_squares_lead(cc, sms):
    """Study sm100_paged_full_prefill_20261112: mixed-length full prefills (Q = KV) lead for d128 GQA 4 / 8 to b 32, S 512-16384,
    pages 16 / 64, and for d256 MHA to b 8; d128 MHA / group 2 and d64 keep the backend."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm100")
    base = dict(b=8, s_q=2048, s_kv=2048, bottom_right=False, max_total_seq_len_q=None, device_cc=cc, device_sm_count=sms)
    place = lambda **over: placement.place(spec, _sm107_paged_facts(**{**base, **over}))
    for h_q, h_kv, over in ((32, 8, {}), (64, 8, dict(b=32, s_q=512, s_kv=512, page_size=64)), (32, 8, dict(b=16, s_q=16384, s_kv=16384))):
        assert place(h_q=h_q, h_kv=h_kv, **over) == placement.LEAD, (h_q, h_kv, over)
    assert place(h_q=16, h_kv=16, d_qk=256, d_v=256) == placement.LEAD
    assert place(h_q=16, h_kv=16, d_qk=256, d_v=256, b=9) == placement.TRAIL
    assert place(h_q=32, h_kv=8, b=33) == placement.TRAIL
    assert place(h_q=32, h_kv=8, page_size=32) == placement.TRAIL
    for h_q, h_kv, d in ((40, 40, 128), (32, 16, 128), (32, 8, 64)):
        assert not placement._sm100_paged_full_square(_sm107_paged_facts(**{**base, "h_q": h_q, "h_kv": h_kv, "d_qk": d, "d_v": d})), (h_q, h_kv, d)


@pytest.mark.L0
def test_sm107_dense_chunks_lead_at_any_launch_size():
    """Study rubin_dense_thd_prefill_20261127: Rubin dense bf16 causal bottom-right chunks lead past the 128-tile cap at d64, at d128 GQA
    within one Q tile and at d256 over a cache of at least 8 x S_q (FROST 0.03-0.89 of the backend there). Longer d128 chunks, d128 MHA
    and d256 chunks near the square keep the tile cap; the box bounds the rest."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107")
    base = dict(dtype=cudnn.data_type.BFLOAT16, causal=True, bottom_right=True, device_cc=(10, 7), device_sm_count=208)
    f = lambda **over: _facts(**{**base, **over})
    place = lambda **over: placement.place(spec, f(**over))
    rule = lambda **over: placement._sm107_dense_chunk(f(**over))
    d64, d128, d256 = dict(d_qk=64, d_v=64), dict(d_qk=128, d_v=128), dict(d_qk=256, d_v=256)
    assert place(**d64, b=16, h_q=64, h_kv=64, s_q=128, s_kv=16384) == placement.LEAD  # 1024 launch tiles: past the cap
    assert place(**d128, b=64, h_q=64, h_kv=8, s_q=128, s_kv=2048) == placement.LEAD
    assert place(**d256, b=32, h_q=32, h_kv=2, s_q=128, s_kv=65536) == placement.LEAD
    assert place(**d256, b=8, h_q=16, h_kv=16, s_q=2048, s_kv=32768) == placement.LEAD  # MHA, 16 x S_q
    assert rule(**d128, b=64, h_q=64, h_kv=8, s_q=placement.SM107_DENSE_D128_CHUNK_MAX_S_Q, s_kv=2048)
    assert not rule(**d128, b=64, h_q=64, h_kv=8, s_q=placement.SM107_DENSE_D128_CHUNK_MAX_S_Q + 1, s_kv=2048)
    assert not rule(**d128, b=64, h_q=16, h_kv=16, s_q=128, s_kv=2048)  # d128 MHA
    assert rule(**d256, b=16, h_q=64, h_kv=8, s_q=256, s_kv=8 * 256)
    assert not rule(**d256, b=16, h_q=64, h_kv=8, s_q=1024, s_kv=2048)  # near the square: 1.20 of the backend
    for over in (
        dict(thd=True),
        dict(has_paged_kv=True),
        dict(has_sink=True),
        dict(window_left=127),
        dict(dtype=cudnn.data_type.HALF),
        dict(device_cc=(10, 0), device_sm_count=148),
        dict(bottom_right=False),
        dict(s_q=16384),  # S_q == S_kv
        dict(s_q=16),  # the decode-shaped band's
        dict(s_kv=placement.SM107_DENSE_CHUNK_KV[0] - 1, s_q=64),
        dict(s_kv=placement.SM107_DENSE_CHUNK_KV[1] + 1),
        dict(b=placement.SM107_DENSE_CHUNK_MAX_B + 1),
        dict(h_q=placement.SM107_DENSE_CHUNK_MAX_HEADS + 8, h_kv=placement.SM107_DENSE_CHUNK_MAX_HEADS + 8),
    ):
        assert not rule(**{**d64, "b": 16, "h_q": 64, "h_kv": 64, "s_q": 128, "s_kv": 16384, **over}), over


@pytest.mark.L0
def test_sm100_paged_full_square_leads_inside_its_box():
    """cc 10.0 and cc 10.3 paged bf16 d128 full prefill squares lead for GQA 4/8/16, 1024 <= S <= 8192; MHA / group 2 keep the
    backend."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm100")
    base = dict(b=2, s_q=1536, s_kv=1536, bottom_right=False, max_total_seq_len_q=None, device_cc=(10, 0), device_sm_count=148)
    place = lambda **over: placement.place(spec, _sm107_paged_facts(**{**base, **over}))
    for h_q, h_kv in ((16, 4), (8, 1), (64, 4)):
        assert place(h_q=h_q, h_kv=h_kv) == placement.LEAD, (h_q, h_kv)
    for h_q, h_kv in ((8, 8), (16, 8)):
        assert not placement._sm100_paged_full_square(_sm107_paged_facts(**{**base, "h_q": h_q, "h_kv": h_kv}))
    assert place(h_q=8, h_kv=1, b=1, s_q=8192, s_kv=8192) == placement.LEAD
    # A square is placed as its bottom-right twin, so the packed full-prefill box (Q from 512, cc 10.0 / 10.3) leads it too.
    assert place(h_q=16, h_kv=4, s_q=512, s_kv=512) == placement.LEAD
    assert place(h_q=16, h_kv=4, device_cc=(10, 3), device_sm_count=160) == placement.LEAD
    assert place(h_q=8, h_kv=1, b=1, s_q=16384, s_kv=16384) == placement.LEAD  # study sm100_paged_full_prefill_20261112
    assert place(h_q=8, h_kv=1, b=1, s_q=16385, s_kv=16385) == placement.TRAIL
    assert place(h_q=16, h_kv=8, device_cc=(10, 3), device_sm_count=160) == placement.TRAIL  # group 2 on cc 10.3 too


@pytest.mark.L0
def test_sm100_paged_gqa_prefill_leads_inside_its_box():
    """cc 10.0 and cc 10.3 paged bf16 d128 GQA prefill leads for GQA16 (exact or bounded) and bounded GQA 4 / 8; exact G8 leads
    as a prefix up to S_q 2048 (the prefix rule); MHA, G2 and out-of-box shapes keep the backend."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm100")
    base = dict(h_q=32, h_kv=2, device_cc=(10, 0), device_sm_count=148)
    place = lambda **over: placement.place(spec, _sm107_paged_facts(**{**base, **over}))
    assert place() == placement.LEAD
    assert place(b=1, s_q=4096, s_kv=4096, bottom_right=False, max_total_seq_len_q=4096) == placement.LEAD
    assert place(bottom_right=False) == placement.TRAIL  # top-left non-square chunk: outside the measured scope
    assert place(h_q=32, h_kv=4) == placement.LEAD  # exact G8 prefix with a total-Q: study sm100_paged_d128_prefix_declq_20261117
    assert place(h_q=32, h_kv=4, s_q=3000, max_total_seq_len_q=3 * 3000) == placement.TRAIL  # exact G8 past the prefix rule's S_q
    assert place(h_q=32, h_kv=4, max_total_seq_len_q=None) == placement.LEAD  # exact G8, no total-Q declaration
    assert place(h_q=32, h_kv=32, max_total_seq_len_q=None) == placement.TRAIL
    assert place(h_q=32, h_kv=4, shape_overrides=True) == placement.LEAD  # bounded G8
    assert place(h_q=32, h_kv=8, shape_overrides=True) == placement.LEAD  # bounded G4
    assert place(shape_overrides=True) == placement.LEAD  # bounded G16
    assert place(h_q=32, h_kv=32) == placement.TRAIL
    assert place(h_q=32, h_kv=32, shape_overrides=True) == placement.TRAIL
    assert place(h_q=32, h_kv=16, shape_overrides=True) == placement.TRAIL
    assert place(device_cc=(10, 3), device_sm_count=160) == placement.LEAD  # study sm103_paged_d128_prefill_transfer_20261031
    assert place(h_q=32, h_kv=4, max_total_seq_len_q=None, device_cc=(10, 3), device_sm_count=160) == placement.LEAD
    assert place(h_q=32, h_kv=4, device_cc=(10, 3), device_sm_count=160) == placement.LEAD  # the B300 transfer qualified it
    past_box = dict(b=placement.SM100_PAGED_GQA16_MAX_B + 1, max_total_seq_len_q=5 * 864)
    assert not placement._sm100_paged_gqa_prefill(_sm107_paged_facts(**{**base, **past_box}))
    assert place(**past_box) == placement.LEAD  # the prefix rule answers past the box


@pytest.mark.L0
@pytest.mark.parametrize("cc", [(10, 0), (10, 3)], ids=["b200", "b300"])
def test_sm100_paged_mha_prefix_leads(cc):
    """Study paged_thd_mha_placement_20261127: a cc 10.0 / 10.3 paged MHA prefix with THD Q leads at d256 and at d64 within one Q tile,
    with an exact, ragged or no total-Q declaration. d64 past one Q tile and d128 keep their rules; the box bounds the rest."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm100")
    base = dict(h_q=8, h_kv=8, b=2, s_q=512, s_kv=81920, max_total_seq_len_q=2 * 512, device_cc=cc, device_sm_count=148)
    d256, d64 = dict(d_qk=256, d_v=256), dict(d_qk=64, d_v=64)
    place = lambda **over: placement.place(spec, _sm107_paged_facts(**{**base, **over}))
    rule = lambda **over: placement._sm100_paged_mha_prefill(_sm107_paged_facts(**{**base, **over}))
    for total in (2 * 512, 700, None):
        assert place(**d256, max_total_seq_len_q=total) == placement.LEAD, total
    assert place(**d64, s_q=100, max_total_seq_len_q=200) == placement.LEAD
    assert place(**d64, s_q=100, max_total_seq_len_q=None) == placement.LEAD
    assert place(**d64) == placement.TRAIL  # past one Q tile: FROST ran 0.91-1.09 of the backend
    assert not rule()  # d128 keeps its rules
    assert rule(**d256, s_q=placement.SM100_PAGED_MHA_PREFILL_MAX_Q, max_total_seq_len_q=None)
    assert rule(**d64, s_q=placement.SM100_PAGED_MHA_PREFILL_D64_MAX_Q, max_total_seq_len_q=None)
    assert rule(
        **d256, b=placement.SM100_PAGED_MHA_PREFILL_MAX_B, h_q=placement.SM100_PAGED_MHA_PREFILL_MAX_HEADS, h_kv=placement.SM100_PAGED_MHA_PREFILL_MAX_HEADS
    )
    for over in (
        dict(s_q=placement.SM100_PAGED_MHA_PREFILL_MAX_Q + 1, max_total_seq_len_q=None),
        dict(s_q=81920),  # a full square: _sm100_paged_full_square's
        dict(s_kv=placement.SM100_PAGED_MHA_PREFILL_KV[1] + 1),
        dict(s_kv=placement.SM100_PAGED_MHA_PREFILL_KV[0] - 1, s_q=100),
        dict(b=placement.SM100_PAGED_MHA_PREFILL_MAX_B + 1),
        dict(h_q=placement.SM100_PAGED_MHA_PREFILL_MAX_HEADS + 8, h_kv=placement.SM100_PAGED_MHA_PREFILL_MAX_HEADS + 8),
        dict(h_q=16),  # GQA 2
        dict(page_size=32),
        dict(window_left=127),
        dict(has_sink=True),
        dict(bottom_right=False),
        dict(shape_overrides=True),
        dict(thd=False),
        dict(device_cc=(10, 7)),
    ):
        assert not rule(**d256, **over), over


@pytest.mark.L0
def test_sm107_paged_mha_split_keys_apart(monkeypatch):
    """Study paged_thd_mha_placement_20261127: a Rubin paged THD MHA prefix the row splits keys `mha_split`, which no table entry
    answers, so the split leads (FROST 0.05-0.85 of the backend at d128) instead of sm107_thd_prefill_20261111's unsplit MHA threshold.
    Graphs the row does not split keep the `mha` class and its entry."""
    from cudnn.frost import buffers
    from cudnn.sdpa import placement_table as pt
    from cudnn.sdpa.fwd.engines import sm100_legacy_policy_spec

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    spec = sm100_legacy_policy_spec("sdpa_fwd_prefill_sm107")
    base = dict(h_q=4, h_kv=4, b=1, s_q=64, s_kv=90112, max_total_seq_len_q=64, device_sm_count=208)
    f = lambda **over: _sm107_paged_facts(**{**base, **over})
    split = f()
    assert heuristics.paged_thd_split_choice(spec.capabilities, split)[0] > 1
    assert pt.class_key(spec.name, split)[8] == "mha_split"
    assert pt.matched_entry(spec.name, split) is None
    assert placement.place(spec, split) == placement.LEAD
    unsplit = f(b=64, h_q=8, h_kv=8, s_q=1024, s_kv=65536, max_total_seq_len_q=64 * 1024)
    assert heuristics.paged_thd_split_choice(spec.capabilities, unsplit)[0] == 1
    assert pt.class_key(spec.name, unsplit)[8] == "mha"
    assert pt.matched_entry(spec.name, unsplit).study == "sm107_thd_prefill_20261111"
    assert pt.class_key(spec.name, f(h_q=16, h_kv=4))[8] != "mha_split"  # GQA keeps its classes
    assert pt.class_key("sdpa_fwd_prefill_sm100", f(device_cc=(10, 0), device_sm_count=148))[8] == "mha"  # Rubin's row only


@pytest.mark.L0
def test_sm107_paged_unpacked_cga_tile_keys_apart(monkeypatch):
    """Study rubin_paged_unpacked_prefill_20261127: Rubin paged THD d128 prefill whose plan runs unpacked and unsplit, inside one 2-CTA
    Q tile per head, keys `prefill_cga_tile`. Its entries lead small launches and trail large ones, monotone in the launch. Packed
    groups, split MHA, windows, other rows and S_q outside the band keep their classes."""
    from cudnn.frost import buffers
    from cudnn.sdpa import placement_table as pt
    from cudnn.sdpa.fwd.engines import sm100_legacy_policy_spec

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    spec = sm100_legacy_policy_spec("sdpa_fwd_prefill_sm107")
    lo, hi = pt.SM107_PAGED_CGA_TILE_S_Q
    base = dict(b=1, h_q=24, h_kv=8, s_q=hi, s_kv=8192, page_size=64, max_total_seq_len_q=hi, device_sm_count=208)
    f = lambda **over: _sm107_paged_facts(**{**base, **over})
    for h_q, h_kv in ((24, 8), (40, 8), (48, 8), (28, 4), (32, 16), (32, 32)):  # groups 3, 5, 6, 7, 2 and MHA run unpacked
        assert pt.class_key(spec.name, f(h_q=h_q, h_kv=h_kv))[4] == "prefill_cga_tile"
        assert pt.matched_entry(spec.name, f(h_q=h_q, h_kv=h_kv)).study == "rubin_paged_unpacked_prefill_20261127"
    for b in (1, 64):
        assert pt.class_key(spec.name, f(b=b, s_kv=8192 + 1, max_total_seq_len_q=b * hi))[9]  # the off-tile twin is answered too
        assert pt.matched_entry(spec.name, f(b=b, s_kv=8192 + 1, max_total_seq_len_q=b * hi)).study == "rubin_paged_unpacked_prefill_20261127"
    for h_q, h_kv in ((24, 8), (32, 32)):
        verdicts = [placement.place(spec, f(h_q=h_q, h_kv=h_kv, b=b, max_total_seq_len_q=b * hi)) for b in (1, 2, 4, 8, 16, 32, 64)]
        assert verdicts[0] == placement.LEAD and verdicts[-1] == placement.TRAIL
        assert verdicts == sorted(verdicts, key=lambda v: v == placement.TRAIL)  # once trailing, a larger launch trails
    assert pt.class_key(spec.name, f(s_q=lo, max_total_seq_len_q=lo))[4] == "prefill"
    assert pt.class_key(spec.name, f(s_q=lo + 1, max_total_seq_len_q=lo + 1))[4] == "prefill_cga_tile"
    assert pt.class_key(spec.name, f(s_q=hi + 1, max_total_seq_len_q=hi + 1))[4] == "prefill"
    assert pt.class_key(spec.name, f(h_q=32, h_kv=8))[4] == "prefill"  # the row packs group 4
    split = f(h_q=4, h_kv=4, s_q=192, s_kv=90112, page_size=16, max_total_seq_len_q=192)
    assert heuristics.paged_thd_split_choice(spec.capabilities, split)[0] > 1
    assert pt.class_key(spec.name, split)[4::4] == ("prefill", "mha_split")
    assert pt.class_key(spec.name, f(window_left=128))[4] == "prefill"
    assert pt.class_key("sdpa_fwd_prefill_sm100", f(device_cc=(10, 0), device_sm_count=148))[4] == "prefill"  # Rubin's row only


@pytest.mark.L0
def test_sm107_thd_d128_split_keys_apart_and_leads(monkeypatch):
    """Study rubin_thd_d128_longkv_20261127: Rubin exact nonpaged THD d128 prefill whose plan splits over KV keys `prefill_split`, has no
    entry, and leads through the row's split rule (FROST's shared single-Q body ran 0.10-1.00 of the backend across caches up to
    1024 x S_q). An unsplit graph keeps its class and study; bounded, paged and d256 graphs and the SM100 row keep their classes."""
    from cudnn.frost import buffers
    from cudnn.sdpa import placement_table as pt
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.engines import sm100_legacy_policy_spec

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    spec = sm100_legacy_policy_spec("sdpa_fwd_prefill_sm107")
    base = dict(b=1, h_q=16, h_kv=1, s_q=160, s_kv=24576, thd=True, has_paged_kv=False, max_total_seq_len_q=118, device_cc=(10, 7), device_sm_count=208)
    f = lambda **over: _facts(**{**base, **over})
    for dtype in (cudnn.data_type.BFLOAT16, cudnn.data_type.HALF):
        split = f(dtype=dtype)
        assert heuristics.nonpaged_thd_split_choice(spec.capabilities, split)[0] > 1
        assert pt.class_key(spec.name, split)[4] == "prefill_split" and pt.matched_entry(spec.name, split) is None
        assert placement.place(spec, split) == placement.LEAD, dtype
    unsplit = f(b=16, h_q=20, h_kv=20, s_q=96, s_kv=57344, max_total_seq_len_q=16 * 96)
    assert heuristics.nonpaged_thd_split_choice(spec.capabilities, unsplit)[0] == 1
    assert pt.class_key(spec.name, unsplit)[4] == "prefill" and pt.matched_entry(spec.name, unsplit).study == "sm107_thd_te_prefill_20261122"
    for over in (dict(shape_overrides=True), dict(d_qk=256, d_v=256), dict(has_paged_kv=True, page_size=16), dict(device_cc=(10, 0), device_sm_count=148)):
        row = "sdpa_fwd_prefill_sm100" if over.get("device_cc") == (10, 0) else spec.name
        assert pt.class_key(row, f(**over))[4] != "prefill_split", over


@pytest.mark.L0
def test_sm103_fp8_d192_long_causal_chunks_lead():
    """cc 10.3 per-tensor FP8 d192x128 causal chunks from SM103_FP8_D192_LONG_CHUNK_MIN_S_Q query tokens lead past
    FP8_D192_MAX_Q_TILES inside the qualified bottom-right E4M3 MHA band. Other graphs keep the tile bound."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm100_fp8")
    base = dict(b=4, h_q=128, h_kv=128, s_q=1024, s_kv=8192, d_qk=192, d_v=128, causal=True, bottom_right=True, device_cc=(10, 3), device_sm_count=148)

    def place(**over):
        return placement.place(spec, _facts(**{**base, **_SM107_FP8_QUANT, "dtype_o": cudnn.data_type.FP8_E4M3, **over}))

    assert place() == placement.LEAD  # 4096 Q tiles, s_q 1024
    assert place(s_q=placement.SM103_FP8_D192_LONG_CHUNK_MIN_S_Q, b=1) == placement.LEAD
    assert place(s_q=512, s_kv=2048, b=1, h_q=32, h_kv=32) == placement.TRAIL
    for s_q in (511, 512, 513):
        for s_kv in (4095, 4096, 4097):
            expected = placement.LEAD if s_q >= 512 and s_kv >= 4096 else placement.TRAIL
            assert place(s_q=s_q, s_kv=s_kv, b=1, h_q=32, h_kv=32) == expected
    assert place(s_q=512, s_kv=2048, b=1, h_q=16, h_kv=16) == placement.LEAD  # original 64-tile lead is unchanged
    assert place(bottom_right=False) == placement.TRAIL  # top-left was not part of the long-chunk study
    assert place(h_q=40, h_kv=8, b=1, s_q=768, s_kv=6144) == placement.TRAIL  # do not extend MHA qualification to GQA
    assert place(h_kv=32) == placement.TRAIL
    assert place(dtype=cudnn.data_type.FP8_E5M2) == placement.TRAIL
    assert place(dtype_o=cudnn.data_type.BFLOAT16) == placement.TRAIL
    assert place(s_q=256, b=2) == placement.TRAIL  # 512 tiles below the long-chunk bound: B200 tile bound
    assert place(causal=False, bottom_right=False) == placement.TRAIL  # mask-free: mixed, not admitted
    assert place(device_cc=(10, 0)) == placement.TRAIL  # cc 10.0 keeps FP8_D192_MAX_Q_TILES
    assert place(s_q=128, b=1, h_q=16, h_kv=16) == placement.LEAD  # 16 tiles: the B200 tile bound still leads


@pytest.mark.L0
def test_sm100_fp8_decode_shaped_gqa_leads_inside_the_measured_box():
    """SM100 per-tensor FP8 decode-shaped GQA (module docstring): 2 <= s_q <= 16 leads at d128 / d256 and d64 with a sink,
    s_q == 1 only for the d64 sink graph over a whole cache it splits, under a window, or from 32 KV units; outside the measured
    box the backend leads."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm100_fp8")
    base = dict(b=32, h_q=16, h_kv=2, s_q=4, s_kv=8192, d_qk=128, d_v=128, causal=True, bottom_right=True, device_cc=(10, 0), device_sm_count=148)

    def place(**over):
        return placement.place(spec, _facts(**{**base, **_SM107_FP8_QUANT, "dtype_o": cudnn.data_type.FP8_E4M3, **over}))

    sink = dict(d_qk=64, d_v=64, h_q=64, h_kv=8, has_sink=True)
    assert place() == placement.LEAD
    assert place(has_paged_kv=True, padded=True, page_size=16) == placement.LEAD
    assert place(d_qk=256, d_v=256, h_q=32, s_q=2) == placement.LEAD
    assert place(**sink, s_q=16) == placement.LEAD
    paged_swa = {**sink, "h_q": 8, "h_kv": 1, "window_left": 128, "has_paged_kv": True, "padded": True, "page_size": 16}
    assert place(**{**paged_swa, "s_q": 4, "b": 1}) == placement.TRAIL  # one KV unit: 1.09-1.49x
    assert place(**{**paged_swa, "s_q": 4, "b": 32}) == placement.LEAD
    assert place(**sink, s_q=1, causal=False, bottom_right=False, b=4) == placement.LEAD  # 32 KV units
    assert place(**sink, s_q=1, causal=False, bottom_right=False, b=1, window_left=128) == placement.LEAD
    assert place(**sink, s_q=1, causal=False, bottom_right=False, b=1) == placement.LEAD  # 8 units over a full cache it splits
    # an unaligned dense KV tail keeps the unsplit plan (heuristics._synth_kv_padding), which loses below 32 units
    assert place(**sink, s_q=1, causal=False, bottom_right=False, b=1, s_kv=8191) == placement.TRAIL
    assert place(s_q=1, causal=False, bottom_right=False) == placement.TRAIL  # d128 decode still loses
    assert place(h_q=2) == placement.TRAIL  # MHA not measured
    assert place(d_qk=64, d_v=64, h_q=64, h_kv=8) == placement.TRAIL  # d64 without a sink not measured
    assert place(window_left=128) == placement.TRAIL  # a window off the d64 sink graph not measured
    assert place(dtype_o=cudnn.data_type.BFLOAT16) == placement.TRAIL  # measured with E4M3 O only
    assert place(s_kv=512) == placement.TRAIL  # below the measured caches
    assert place(dtype=cudnn.data_type.FP8_E5M2) == placement.TRAIL  # E5M2 inputs not measured
    assert place(right_band_widening=True, right_bound=4) == placement.TRAIL  # widened band not measured


@pytest.mark.L0
def test_sm100_fp8_sink_lead_follows_the_proposed_split():
    """Below 32 KV units the whole-cache d64 FP8 sink graph leads only on a split plan: a proposal that runs it unsplit
    (mode B's fallback knobs) keeps the backend first."""
    facts = _facts(**{**_SM107_FP8_QUANT, "dtype_o": cudnn.data_type.FP8_E4M3, "device_sm_count": 148}, d_qk=64, d_v=64, has_sink=True, s_q=4, s_kv=32768)
    lead = heuristics.propose("A", facts, _OFFERED)[0]
    assert not is_backend_block(lead) and (lead.knobs.split_kv or 1) >= 2, lead
    plans = heuristics.propose("B", facts, _OFFERED)
    assert is_backend_block(plans[0]) and all((p.knobs.split_kv or 1) == 1 for p in plans if not is_backend_block(p)), plans


@pytest.mark.L0
def test_sm107_fp8_sink_lead_follows_the_proposed_split():
    """cc 10.7: below 32 KV units the single-query d64 FP8 sink graph leads only on a split plan; mode B's unsplit fallback
    knobs keep the backend first."""
    quant = {**_SM107_FP8_QUANT, "dtype_o": cudnn.data_type.FP8_E4M3}
    facts = _facts(**quant, device_cc=(10, 7), d_qk=64, d_v=64, has_sink=True, s_q=1, s_kv=131072, causal=False, bottom_right=False)
    lead = heuristics.propose("A", facts, _OFFERED)[0]
    assert not is_backend_block(lead) and (lead.knobs.split_kv or 1) >= 2, lead
    assert is_backend_block(heuristics.propose("B", facts, _OFFERED)[0])


@pytest.mark.L0
def test_sm107_fp8_d192_decode_gap_follows_the_study():
    """Study sm107_fp8_d192_decode_20261118 (module docstring): on cc 10.7, dense FP8 d192 x 128 verify / MTP leads where the row packs
    the whole GQA group, with the full cache always and with a window + sink from 3208 KV tokens in flight. Unpacked groups, S_q 1 and
    graphs past the verify depth keep the existing rule (TRAIL for d192)."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107_fp8")
    base = dict(b=1, h_q=64, h_kv=4, s_q=8, s_kv=131072, d_qk=192, d_v=128, causal=True, bottom_right=True, device_cc=(10, 7))

    def place(**over):
        return placement.place(spec, _facts(**{**base, **_SM107_FP8_QUANT, "dtype_o": cudnn.data_type.FP8_E4M3, **over}))

    for h_q, h_kv in ((64, 4), (16, 1), (128, 8)):  # mimo full cache: group 16
        assert place(h_q=h_q, h_kv=h_kv) == placement.LEAD
        assert place(h_q=h_q, h_kv=h_kv, b=128, s_q=2) == placement.LEAD
    assert place(h_q=36, h_kv=3) == placement.TRAIL  # group 12 runs unpacked
    assert place(s_q=1, causal=False, bottom_right=False) == placement.TRAIL  # not measured (the backend has no S_q 1 plan here)
    assert place(s_q=32) == placement.TRAIL  # past the verify depth
    sink = dict(window_left=127, has_sink=True, h_q=64, h_kv=8)  # 136 keys per row at S_q 8
    assert place(**sink, b=128) == placement.LEAD
    assert place(**sink, b=3) == placement.LEAD  # 3 x 8 x 136 = 3264 tokens
    assert place(**sink, b=2) == placement.TRAIL  # 2176 tokens: the b 1-2 rows (~11 us) the backend wins
    assert place(**{**sink, "b": 128, "h_q": 36, "h_kv": 3}) == placement.TRAIL  # unpacked: no rule shipped


@pytest.mark.L0
def test_sm100_fp8_decode_gap_classes_follow_the_study():
    """Study sm100_fp8_decode_gaps_20261118 (module docstring): paged d64 (any window, S_q 1-16), windowed paged d256, dense d192 x 128
    and its windowed sink twin lead where the row packs the whole GQA group (powers of two); unpacked groups lead only inside their fitted
    bounds, windowed paged d128 unpacked only on cc 10.0. Windowed paged d128 packed follows sm100_fp8_decode_v3_20261122 (a KV-token
    floor). Outside the measured box the hand rules answer as before."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm100_fp8")
    paged = dict(has_paged_kv=True, padded=True, page_size=16)
    base = dict(b=128, h_q=128, h_kv=16, s_q=4, s_kv=131072, d_qk=64, d_v=64, causal=True, bottom_right=True, device_cc=(10, 0), **paged)

    def place(**over):
        return placement.place(spec, _facts(**{**base, **_SM107_FP8_QUANT, "dtype_o": cudnn.data_type.FP8_E4M3, **over}))

    # d64 paged, packed (muse_spark 128/16 and 16/2): every S_q, with or without a window, both parts
    for s_q in (1, 4, 8):
        for cc in ((10, 0), (10, 3)):
            assert place(s_q=s_q, device_cc=cc) == placement.LEAD
            assert place(s_q=s_q, device_cc=cc, h_q=16, h_kv=2, b=1, window_left=2047) == placement.LEAD
    unpacked = dict(h_q=36, h_kv=3)  # group 12 runs unpacked
    assert place(**unpacked) == placement.TRAIL
    assert place(**unpacked, window_left=2047, s_kv=16384) == placement.LEAD
    assert place(**unpacked, window_left=2047, s_kv=placement.SM100_FP8_D64_WIN_UNPACKED_MAX_KV + 1) == placement.TRAIL
    assert place(**unpacked, window_left=2047, s_q=1, causal=True, s_kv=16384) == placement.TRAIL
    # windowed paged d128: packed leads from a KV-token floor, both parts (step37 b 128 leads, step37-tp4 b 1 keeps the backend);
    # unpacked leads on cc 10.0 up to S_q 12, and on cc 10.3 from enough KV units at any S_q (sm103_fp8_d128_win_unpacked_20261122)
    d128 = dict(d_qk=128, d_v=128, window_left=511)
    for cc in ((10, 0), (10, 3)):
        assert place(**d128, h_q=64, h_kv=8, device_cc=cc) == placement.LEAD
        assert place(**d128, h_q=16, h_kv=2, b=1, device_cc=cc) == placement.TRAIL
    assert place(**d128, h_q=80, h_kv=8, s_q=8) == placement.LEAD
    assert place(**d128, h_q=80, h_kv=8, s_q=16) == placement.TRAIL
    for s_q in (8, 16):  # ripple_swa b 128 leads on cc 10.3, its b 1 row keeps the backend
        assert place(**d128, h_q=80, h_kv=8, s_q=s_q, device_cc=(10, 3)) == placement.LEAD
        assert place(**d128, h_q=80, h_kv=8, s_q=s_q, b=1, device_cc=(10, 3)) == placement.TRAIL
    # dense d192 x 128 (mimo): packed leads; its windowed sink twin leads from enough KV tokens in flight
    d192 = dict(d_qk=192, d_v=128, has_paged_kv=False, padded=False, page_size=0, s_q=8)
    assert place(**d192, h_q=64, h_kv=4) == placement.LEAD
    assert place(**d192, h_q=36, h_kv=3) == placement.TRAIL  # unpacked failed its holdout
    sink = dict(d192, window_left=127, has_sink=True, h_q=64, h_kv=8)
    assert place(**sink) == placement.LEAD
    assert place(**sink, b=1) == placement.TRAIL  # 8 units x 136 keys = 1088 tokens, below the packed floor
    sink_unpacked = {**sink, "b": 1, "h_q": 36, "h_kv": 3, "window_left": 511}  # 3 units x 520 keys = 1560 tokens
    assert place(**sink_unpacked) == placement.LEAD
    assert place(**sink_unpacked, device_cc=(10, 3)) == placement.TRAIL  # failed the B300 holdout
    # windowed paged d256 (gemma4): packed every group it serves
    assert place(d_qk=256, d_v=256, h_q=16, h_kv=8, window_left=1023) == placement.LEAD
    # outside the measured box
    assert place(page_size=64) == placement.TRAIL
    assert place(dtype_o=cudnn.data_type.BFLOAT16) == placement.TRAIL
    assert place(s_kv=512) == placement.TRAIL
    assert place(h_q=256, h_kv=8) == placement.TRAIL  # group 32 not measured
    assert place(padded=False) == placement.TRAIL  # paged without per-request lengths


@pytest.mark.L0
@pytest.mark.parametrize("device_cc", [(10, 0), (10, 3)])
def test_sm100_fp8_paged_d192_spec_decode_follows_its_study(device_cc):
    """Study sm100_fp8_decode_v3_20261122 (module docstring): paged d192 x 128 spec decode (mimo, S_q 2-16) leads where the row packs
    the whole GQA group; its windowed sink twin leads packed from enough KV units, unpacked from enough KV tokens on cc 10.0 only.
    The full-cache unpacked class, THD Q and single-row decode keep the hand rules."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm100_fp8")
    base = dict(b=1, h_q=16, h_kv=1, s_q=8, s_kv=131072, d_qk=192, d_v=128, causal=True, bottom_right=True, device_cc=device_cc)
    base.update(has_paged_kv=True, padded=True, page_size=16)

    def place(**over):
        return placement.place(spec, _facts(**{**base, **_SM107_FP8_QUANT, "dtype_o": cudnn.data_type.FP8_E4M3, **over}))

    # full cache: packed leads at any grid, unpacked (group 12) keeps the backend
    for b, h_q, h_kv in ((1, 16, 1), (1, 128, 8), (128, 64, 4)):
        assert place(b=b, h_q=h_q, h_kv=h_kv) == placement.LEAD
        assert place(b=b, h_q=h_q, h_kv=h_kv, s_q=2) == placement.LEAD
    assert place(h_q=36, h_kv=3) == placement.TRAIL
    # window + sink: packed leads at b 128, not at b 1 (both verdicts in the class)
    sink = dict(window_left=127, has_sink=True)
    for h_q, h_kv in ((16, 2), (128, 8)):
        assert place(**sink, b=128, h_q=h_q, h_kv=h_kv) == placement.LEAD
        assert place(**sink, b=1, h_q=h_q, h_kv=h_kv) == placement.TRAIL
    # window + sink, unpacked: leads from enough tokens on cc 10.0, never on cc 10.3
    assert place(**{**sink, "window_left": 511}, b=16, h_q=36, h_kv=3) == (placement.LEAD if device_cc == (10, 0) else placement.TRAIL)
    assert place(**sink, b=1, h_q=36, h_kv=3) == placement.TRAIL
    # outside the measured box
    assert place(thd=True) == placement.TRAIL
    assert place(page_size=64) == placement.TRAIL


@pytest.mark.L0
def test_sm100_fp8_d512_mtp_leads_outside_its_small_work_floor():
    """SM100 per-tensor FP8 d512 MTP rows (2 <= s_q <= 16 under the bottom-right diagonal, dense unpadded E4M3 KV, MLA-absorbed or GQA)
    lead except one request of <= 32 query heads over a short cache (module docstring). Before the rule every one trailed."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm100_fp8")
    base = dict(b=128, h_q=128, h_kv=1, s_q=2, s_kv=131072, d_qk=512, d_v=512, causal=True, bottom_right=True, device_cc=(10, 0), device_sm_count=148)

    def place(**over):
        return placement.place(spec, _facts(**{**base, **_SM107_FP8_QUANT, "dtype_o": cudnn.data_type.FP8_E4M3, **over}))

    assert place() == placement.LEAD  # a DSv4 MTP row of the nightly: the backend default 1.71 s, FROST 4 ms
    assert place(b=1, h_q=8, s_q=16, s_kv=2048) == placement.LEAD
    assert place(b=1, h_q=32, h_kv=4, s_q=16, s_kv=2048) == placement.LEAD  # GQA 0.89
    assert place(b=1, h_q=64, s_kv=256) == placement.LEAD  # 0.63
    assert place(b=8, h_q=8, s_kv=256) == placement.LEAD  # 0.65
    assert place(b=1, h_q=8, s_kv=256) == placement.TRAIL  # 1.21
    assert place(b=1, h_q=32, s_q=16, s_kv=1024) == placement.TRAIL  # 1.47
    assert place(b=1, h_q=44, s_kv=1447) == placement.TRAIL
    assert place(b=1, h_q=45, s_kv=1447) == placement.LEAD
    assert place(b=1, h_q=44, s_kv=1448) == placement.LEAD
    assert place(causal=False, bottom_right=False) == placement.TRAIL  # mask-free MTP not measured
    paged = dict(has_paged_kv=True, padded=True, page_size=64)
    assert place(**paged) == placement.LEAD  # paged d512 MTP (#2372): the same floor
    assert place(**paged, b=1, h_q=16, s_q=16, s_kv=1024) == placement.TRAIL  # 1.50
    assert place(**paged, b=1, h_q=16, s_q=1, s_kv=1024, causal=False, bottom_right=False) == placement.LEAD  # no backend plan
    assert place(has_paged_kv=True, padded=False, page_size=64) == placement.TRAIL  # paged without per-request lengths
    assert place(padded=True) == placement.TRAIL  # per-request lengths over a dense cache not measured
    assert place(window_left=128) == placement.TRAIL
    assert place(h_kv=128) == placement.TRAIL  # MHA not measured
    assert place(dtype_o=cudnn.data_type.BFLOAT16) == placement.TRAIL


@pytest.mark.L0
def test_sm107_fp8_paged_d512_decode_leads():
    """cc 10.7 per-tensor FP8 paged D512 decode / MTP over E4M3 page pools with a dense Q (#2374) leads: FROST 0.001-0.89 of
    the backend at s_q 2-16 and the only plan for one row (module docstring). Before the arm every paged graph trailed."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm107_fp8")
    base = dict(b=128, h_q=128, h_kv=1, s_q=2, s_kv=65536, d_qk=512, d_v=512, causal=True, bottom_right=True, has_paged_kv=True, padded=True)
    base.update(page_size=64, device_cc=(10, 7), device_sm_count=208)

    def place(**over):
        return placement.place(spec, _facts(**{**base, **_SM107_FP8_QUANT, "dtype_o": cudnn.data_type.FP8_E4M3, **over}))

    assert place() == placement.LEAD
    assert place(b=1, h_q=16, s_q=16, s_kv=1024, page_size=16) == placement.LEAD  # 0.89, the closest cell
    assert place(s_q=1, causal=False, bottom_right=False) == placement.LEAD  # no backend plan
    assert place(h_q=32, h_kv=4, b=1, s_q=16, s_kv=2048) == placement.LEAD
    assert place(padded=False) == placement.TRAIL  # paged without per-request lengths not measured
    assert place(thd=True, max_total_seq_len_q=256) == placement.TRAIL  # THD Q not measured
    assert place(causal=False, bottom_right=False) == placement.TRAIL  # mask-free MTP not measured
    assert place(window_left=128) == placement.TRAIL
    assert place(h_kv=128) == placement.TRAIL  # MHA not measured
    assert place(d_qk=256, d_v=256) == placement.TRAIL  # only D512 measured paged
    assert place(dtype_o=cudnn.data_type.BFLOAT16) == placement.TRAIL
    assert place(s_q=17) == placement.TRAIL  # past the decode-shaped rows


@pytest.mark.L0
def test_square_causal_places_the_same_from_either_corner():
    """A square causal mask covers the same tiles anchored top-left or bottom-right, so both get one verdict."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm100")
    place = lambda **over: placement.place(spec, _sm107_paged_facts(**{"device_cc": (10, 0), "device_sm_count": 148, **over}))
    paged_d512 = dict(h_q=64, h_kv=1, d_qk=512, d_v=512, page_size=64, max_total_seq_len_q=None)
    for over in (dict(paged_d512, s_q=2048, s_kv=2048), dict(paged_d512, s_q=8192, s_kv=8192), dict(h_q=32, h_kv=2, s_q=4096, s_kv=4096)):
        assert place(bottom_right=False, **over) == place(bottom_right=True, **over), over
    assert place(bottom_right=False, s_q=8192, s_kv=8192, **paged_d512) == placement.LEAD


@pytest.mark.L0
@pytest.mark.parametrize(
    "over, want",
    [
        pytest.param(dict(h_q=32, h_kv=8, s_q=1024, s_kv=1024), placement.LEAD, id="gqa-causal-s1024"),
        pytest.param(dict(h_q=24, h_kv=1, s_q=640, s_kv=640), placement.LEAD, id="mqa-causal-s640"),
        pytest.param(dict(h_q=32, h_kv=32, s_q=1024, s_kv=1024), placement.LEAD, id="mha-causal-s1024"),
        pytest.param(dict(h_q=32, h_kv=32, s_q=640, s_kv=640, causal=False, bottom_right=False), placement.LEAD, id="mha-dense-s640"),
        pytest.param(dict(h_q=32, h_kv=32, s_q=640, s_kv=640), placement.TRAIL, id="mha-causal-s640"),
        pytest.param(dict(h_q=32, h_kv=32, s_q=768, s_kv=768, bottom_right=False), placement.TRAIL, id="mha-causal-tl-s768"),
        pytest.param(dict(h_q=32, h_kv=32, s_q=896, s_kv=896), placement.LEAD, id="mha-causal-s896"),
        pytest.param(dict(h_q=24, h_kv=24, s_q=128, s_kv=8192), placement.TRAIL, id="mha-sq128"),
        pytest.param(dict(h_q=24, h_kv=24, s_q=128, s_kv=8192, causal=False, bottom_right=False), placement.TRAIL, id="mha-sq128-dense"),
        pytest.param(dict(h_q=24, h_kv=24, s_q=384, s_kv=3072), placement.LEAD, id="mha-chunked-sq384"),
        pytest.param(dict(h_q=40, h_kv=8, s_q=128, s_kv=8192), placement.LEAD, id="gqa-sq128"),
        pytest.param(dict(h_q=32, h_kv=8, s_q=1, s_kv=8192), placement.TRAIL, id="decode-s_q1"),
        pytest.param(dict(h_q=32, h_kv=8, s_q=1024, s_kv=1024, has_sink=True), placement.TRAIL, id="sink"),
    ],
)
def test_sm90_d128_follows_its_measurement(over, want):
    """The SM90 row's exact dense d128 graphs (#378's D128 template, module docstring): every GQA and MQA graph the
    template serves leads, and MHA leads outside its two measured regions (s_q <= 128; causal with at most 3.5 KV tiles
    per Q tile, square S <= 768 from either corner); a graph the template does not serve (decode, a sink) trails, as
    the 64 x 64 tile runs it."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_sm90_f16_three_wg")
    shape = dict(b=8, d_qk=128, d_v=128, dtype=cudnn.data_type.BFLOAT16, device_cc=(9, 0), device_sm_count=132, causal=True, bottom_right=True)
    assert placement.place(spec, _facts(**{**shape, **over})) == want


@pytest.mark.L0
@pytest.mark.parametrize(
    "over, want",
    [
        pytest.param(dict(h_q=32, h_kv=8, s_q=2048, s_kv=1024), placement.TRAIL, id="bottom-right-sq-over-skv"),
    ],
)
def test_sm90_d128_declined_graphs_trail(monkeypatch, over, want):
    """A d128 graph the D128 template declines runs the 64 x 64 tile, which no study of these classes timed: the backend
    stays first. Default placement: the FROST flag unset in the body (Rule S12)."""
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_sm90_f16_three_wg")
    shape = dict(b=8, d_qk=128, d_v=128, dtype=cudnn.data_type.BFLOAT16, device_cc=(9, 0), device_sm_count=132, causal=True, bottom_right=True)
    assert placement.place(spec, _facts(**{**shape, **over})) == want


@pytest.mark.L0
def test_sm90_d128_default_placement_reads_its_study(monkeypatch):
    """Study sm90_d128_h100_refit_20261114: the SM90 row's exact dense d128 graphs the D128 template serves are placed by their class's generated
    entry (placement_table answers first), square causal from either corner alike; a class without an entry (mtp/gqa_unpacked/causal_br/off, mtp/gqa_unpacked/causal_br/on, mtp/mha/none/off, mtp/mha/none/on, prefill/gqa/causal_br/off, prefill/gqa/causal_br/on, prefill/gqa/none/off, prefill/gqa/none/on, prefill/mha/causal_br/off, prefill/mha/causal_br/on, prefill/mha/causal_tl/off, prefill/mha/causal_tl/on, prefill/mha/none/off, prefill/mha/none/on, verify/gqa/causal_br/off, verify/gqa/causal_br/on, verify/gqa/none/off, verify/gqa/none/on)
    keeps the backend first. Default placement: the FROST flag unset in the body (Rule S12)."""
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    from cudnn.sdpa import placement_table as pt
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_sm90_f16_three_wg")
    row = "sdpa_fwd_prefill_sm90"
    shape = dict(d_qk=128, d_v=128, dtype=cudnn.data_type.BFLOAT16, device_cc=(9, 0), device_sm_count=132)
    for over in (
        dict(b=8, h_q=32, h_kv=8, s_q=1024, s_kv=1024, causal=True, bottom_right=True),
        dict(b=8, h_q=32, h_kv=32, s_q=1024, s_kv=1024, causal=True, bottom_right=True),
        dict(b=8, h_q=32, h_kv=32, s_q=640, s_kv=640, causal=False, bottom_right=False),
        dict(b=16, h_q=24, h_kv=24, s_q=128, s_kv=8192, causal=True, bottom_right=True),
        dict(b=16, h_q=40, h_kv=8, s_q=384, s_kv=3072, causal=True, bottom_right=False),
    ):
        f = _facts(**{**shape, **over})
        answer = pt.lookup(row, f)
        if answer is None and pt.class_key(row, f) in getattr(placement, "SM90_D128_HAND_RULE_CLASSES", frozenset()):
            want = placement._place_sm90_d128(f)  # a NON-MONOTONE class: its hand rule decides
        else:
            want = placement.TRAIL if answer is None else (placement.LEAD if answer else placement.TRAIL)
        assert placement.place(spec, f) == want, over
        if over["causal"] and over["s_q"] == over["s_kv"]:
            flipped = _facts(**{**shape, **over, "bottom_right": not over["bottom_right"]})
            assert placement.place(spec, flipped) == placement.place(spec, f)


@pytest.mark.L0
@pytest.mark.parametrize("s_q, paged", [(1, placement.LEAD), (5, placement.LEAD), (1024, placement.TRAIL)])
def test_sm90_paged_d512_follows_its_measurement(s_q, paged):
    """The SM90 row's dense d512 lead holds at every s_q. Paged d512 leads where studies measured it -- s_q 1-4
    (sm90_decode_20261112, incl. GQA over 2 KV heads at page 16: 0.26-0.94 at s_q 1) and verify rows (sm90_decode_gaps_20261117:
    0.05 of the backend) -- and keeps the backend first where nothing measured it (prefill)."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_sm90_f16_three_wg")
    shape = dict(b=64, h_q=8, h_kv=2, s_q=s_q, s_kv=4096, d_qk=512, d_v=512, dtype=cudnn.data_type.BFLOAT16, device_cc=(9, 0), device_sm_count=132)
    assert placement.place(spec, _facts(**shape)) == placement.LEAD
    assert placement.place(spec, _facts(has_paged_kv=True, page_size=16, padded=True, **shape)) == paged
