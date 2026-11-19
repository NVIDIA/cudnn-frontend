# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLUSTER_K_REDUCE on the SM100 D256 swap-AB decode tile: the knob, the heuristics rule, the declines and the binding.

No device and no kernel build: the rule is a pure function of graph facts, so every cell here pins POLICY on synthetic
facts. The device cells are in ``sdpa_decode_cluster_reduce_sm100.py``.
"""

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

import cudnn
from cudnn.engines.manifest import MANIFEST
from cudnn.sdpa.fwd import engines, heuristics as heur, placement
from cudnn.sdpa.fwd import prepared as prep
from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
from cudnn.sdpa.fwd.kernels.sm100.d256_f16_contract import LOOKAHEAD, SWAP_AB
from cudnn.sdpa.fwd.kernels.sm100.d256_f16_swap_ab import contract as swap_ab
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0

SPECS = {s.name: s for s in engines.ENGINE_SPECS}
OFFERED = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd").offered_ids()
NAME_OF = {engine_id: name for name, engine_id in OFFERED.items()}


def facts(b=1, s_q=1, s_kv=2048, sm=148, replay=False, cc=(10, 0), heads=(24, 4), **overrides):
    """A paged D256 bf16 decode graph (24 / 4 heads unless named); a multi-token step carries the bottom-right causal mask."""
    kw = dict(causal=True, bottom_right=True) if s_q > 1 else {}
    kw.update(overrides)
    return SdpaGraphFacts(
        **dict(
            dict(
                b=b,
                h_q=heads[0],
                h_kv=heads[1],
                s_q=s_q,
                s_kv=s_kv,
                d_qk=256,
                d_v=256,
                dtype=cudnn.data_type.BFLOAT16,
                device_cc=cc,
                device_sm_count=sm,
                padded=True,
                has_paged_kv=True,
                page_size=64,
                cuda_graph_replay=replay,
            ),
            **kw,
        )
    )


def _plans(f):
    return [(NAME_OF[p.engine_id], p.knobs) for p in heur.recommend("A", f, OFFERED)]


def _lead_swap_ab(f):
    return next((k for name, k in _plans(f) if name == SWAP_AB), None)


GRID = [
    (b, s_q, s_kv, sm, replay, heads)
    for heads in ((24, 4), (32, 8), (8, 1))
    for sm in (148, 152)
    for replay in (False, True)
    for s_q in (1, 2, 3, 4, 6, 8)
    for b in (1, 2, 3, 4, 8, 9, 10, 15, 16, 64)
    for s_kv in (128, 512, 1024, 1536, 1792, 2048, 3072, 4096, 8192, 32768)
]
# The recorded rows of the two parts, and a 152-SM part's one-wave cluster counts (its rows are exactly those).
ROWS = {(148, 2): 16, (148, 4): 16, (148, 8): 8, (152, 2): 76, (152, 4): 36, (152, 8): 15}


def _cell(cell, **overrides):
    b, s_q, s_kv, sm, replay, heads = cell
    return facts(b, s_q, s_kv, sm, replay, heads=heads, **overrides)


def _clusters(f, knobs):
    """Thread-block clusters of a packed set: one per KV-head group and token unit of every sequence."""
    group = f.h_q // f.h_kv
    return f.b * (f.h_q // group) * heur.decode_d256_q_units(f.s_q, group, knobs.tile_m)


def test_knob_is_a_public_zero_one_knob_that_an_unset_record_does_not_carry():
    kt = cudnn.knob_type
    assert SdpaFwdKnobs().cluster_reduce is None and kt.CLUSTER_K_REDUCE not in SdpaFwdKnobs(split_kv=8).to_public()
    knobs = SdpaFwdKnobs(tile_m=16, tile_n=128, cga=1, sched_policy=0, pack_gqa=True, split_kv=8, cluster_reduce=True)
    public = knobs.to_public()
    assert public[kt.CLUSTER_K_REDUCE] == 1 and SdpaFwdKnobs.from_public(public) == knobs
    assert SdpaFwdKnobs.from_public({int(k): v for k, v in public.items()}) == knobs
    assert SdpaFwdKnobs.from_public({kt.CLUSTER_K_REDUCE: 0}).cluster_reduce is False
    assert SdpaFwdKnobs.from_public({kt.CLUSTER_K_REDUCE: True}).cluster_reduce is True
    with pytest.raises(ValueError, match="CLUSTER_K_REDUCE must be 0 or 1"):
        SdpaFwdKnobs.from_public({kt.CLUSTER_K_REDUCE: 2})
    with pytest.raises(ValueError, match="must be an int"):
        SdpaFwdKnobs.from_public({kt.CLUSTER_K_REDUCE: "1"})


def test_only_the_swap_ab_row_carries_the_form():
    """The capability domain: one row has True, and every other row declines a request with the same reason."""
    with_form = [s.name for s in engines.ENGINE_SPECS if True in s.capabilities.cluster_reduces]
    assert with_form == [SWAP_AB]
    assert all(False in s.capabilities.cluster_reduces for s in engines.ENGINE_SPECS)
    request = SdpaFwdKnobs(split_kv=4, cluster_reduce=True)
    for spec in engines.ENGINE_SPECS:
        if spec.name != SWAP_AB:
            assert "is a form of the SM100 D256 swap-AB split decode tile" in engines.mismatch(spec.capabilities, facts(), request), spec.name


@pytest.mark.parametrize(
    "overrides,knobs,reason",
    [
        ({}, dict(split_kv=1), "needs 2 to 8 KV chunks"),
        ({}, dict(split_kv=16), "needs 2 to 8 KV chunks"),
        ({}, dict(split_kv=2, tile_m=32), "16-column D256 swap-AB rendering"),
        (dict(cc=(10, 3)), dict(split_kv=8), "verified on compute capability 10.0 only"),
        (dict(has_stats_log2=True), dict(split_kv=4), "does not carry a sink, base-2 Stats"),
    ],
    ids=["unsplit", "split16", "tile32", "cc103", "log2-stats"],
)
def test_row_declines_a_request_outside_the_form(overrides, knobs, reason):
    caps = SPECS[SWAP_AB].capabilities
    request = SdpaFwdKnobs(**{**dict(tile_m=16, tile_n=128, cga=1, sched_policy=0, pack_gqa=True, cluster_reduce=True), **knobs})
    f = facts(**overrides)
    assert reason in engines.mismatch(caps, f, request)
    # only the request is declined: the same set without the knob is served
    assert engines.mismatch(caps, f, dataclasses.replace(request, cluster_reduce=None)) is None


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        (dict(thd=True), "does not carry"),
        (dict(has_sink=True), "does not carry"),
        (dict(gated=True), "does not carry"),
        (dict(o_matches_qkv=False), "does not carry"),
        (dict(use_pdl=True), "programmatic-dependent-launch"),
        (dict(d_shape=(128, 128)), "16-column D256"),
        (dict(split_kv=9), "2 to 8"),
        (dict(q_rows=17), "one token unit per head group"),
        (dict(d_v=200), "multiples of 32"),
        (dict(d_v=136), "multiples of 32"),
    ],
)
def test_contract_predicate_rows(kwargs, reason):
    base = dict(device_cc=(10, 0), d_shape=(256, 256), tile=16, split_kv=8)
    assert swap_ab.cluster_reduce_decline(**base) is None
    assert all(swap_ab.cluster_reduce_decline(**{**base, "split_kv": s}) is None for s in range(2, 9))
    assert swap_ab.cluster_reduce_decline(**{**base, "q_rows": 16}) is None
    assert all(swap_ab.cluster_reduce_decline(**{**base, "d_v": d}) is None for d in (160, 192, 224, 256))
    assert reason in swap_ab.cluster_reduce_decline(**{**base, **kwargs})


def test_rule_adds_removes_and_reorders_nothing(monkeypatch):
    """Clearing the knob in every listed set gives the list of a tree without the rule, entry for entry: no set appears,
    disappears or moves, and no split count changes. On both parts with recorded rows, and on each of them something is
    marked, so the comparison is not empty there."""
    marked = {}
    for cell in GRID:
        marked[cell] = _plans(_cell(cell))
    monkeypatch.setattr(heur, "fused_selected", lambda *_: False)
    count = {148: 0, 152: 0}
    for cell in GRID:
        plain = _plans(_cell(cell))
        assert all(not k.cluster_reduce for _, k in plain)
        cleared = [(name, dataclasses.replace(k, cluster_reduce=None)) for name, k in marked[cell]]
        assert cleared == plain, cell
        count[cell[3]] += sum(bool(k.cluster_reduce) for _, k in marked[cell])
    assert count[148] > 0 and count[152] > 0, count


def test_rule_is_pure_and_every_marked_set_is_honoured(monkeypatch):
    """No device query on the planning path, and a set the rule marks is a set the row admits."""
    from cudnn.frost import occupancy

    def refuse(*_args, **_kwargs):
        raise AssertionError("the planning path queried the device")

    monkeypatch.setattr(occupancy, "occupancy_map", refuse)
    monkeypatch.setattr(occupancy, "max_active_clusters", refuse)
    caps = SPECS[SWAP_AB].capabilities
    seen = set()
    for cell in GRID:
        f = _cell(cell)
        for name, knobs in _plans(f):
            if knobs.cluster_reduce:
                assert name == SWAP_AB and knobs.pack_gqa and knobs.tile_m == 16 and 2 <= knobs.split_kv <= 8, cell
                assert engines.mismatch(caps, f, knobs) is None, cell
                # one tile holds the packed rows, and the launch is inside the part's recorded cluster count for this split
                assert f.s_q * (f.h_q // f.h_kv) <= 16, cell
                assert 0 < _clusters(f, knobs) <= heur.FUSED_CLUSTER_BOUND[(f.device_sm_count, knobs.split_kv)], cell
                seen.add((f.device_sm_count, _clusters(f, knobs), knobs.split_kv))
    # the recorded bounds are reached, and nothing past them is marked
    assert {(148, 16, 4), (148, 8, 8)} <= seen and all(split in (4, 8) for sm, _, split in seen if sm == 148)
    # 152 SMs: the rows of 4 and 8 chunks are reached exactly; the existing rules split 2 ways only below that row
    assert {(152, 36, 4), (152, 15, 8)} <= seen and {split for sm, _, split in seen if sm == 152} == {2, 4, 8}
    assert max(n for sm, n, split in seen if (sm, split) == (152, 2)) == 72


@pytest.mark.parametrize(
    "b,s_q,s_kv,replay,split,reduced",
    [
        # one user, one or two tokens: unsplit up to 1,024 declared keys, then today's 4 / 8 chunks reduced in the launch
        (1, 1, 512, False, 1, False),
        (1, 1, 1024, False, 1, False),
        (1, 1, 1536, False, 4, True),
        (1, 2, 2048, False, 8, True),
        (1, 1, 3072, False, 8, True),
        (1, 1, 4096, False, 16, False),  # 16 chunks are past one cluster: two launches as before
        (1, 2, 1024, True, 4, True),  # a replay-declared graph splits earlier
        # two and four users inside the conservative one-wave bound
        (2, 1, 1536, False, 4, True),
        (2, 2, 6144, False, 8, True),
        (2, 1, 8192, False, 16, False),
        (4, 2, 1536, False, 4, True),
        (4, 1, 2560, False, 4, True),
        (4, 1, 3072, False, 8, False),  # 16 clusters of 8 are past the recorded 8
        # three users: 12 clusters of 4 are inside the recorded 16, 12 clusters of 8 are past the recorded 8
        (3, 1, 1792, False, 4, True),
        (3, 2, 2048, False, 8, False),
        # eight and nine users split 2 ways into 32 and 36 clusters, past the recorded 16
        (8, 1, 1792, False, 2, False),
        (9, 1, 1792, False, 2, False),
        (9, 2, 2048, False, 2, False),
        (8, 1, 2048, False, 4, False),
        (16, 1, 8192, False, 2, False),
        (64, 1, 2048, False, 1, False),
    ],
)
def test_rule_table_on_148_sms(b, s_q, s_kv, replay, split, reduced):
    lead = _lead_swap_ab(facts(b, s_q, s_kv, 148, replay))
    assert lead.pack_gqa and (lead.split_kv, bool(lead.cluster_reduce)) == (split, reduced)


@pytest.mark.parametrize(
    "b,s_q,s_kv,replay,split,reduced",
    [
        # one user: the splits of the existing rules, reduced in the launch as on the other part
        (1, 1, 1024, False, 1, False),
        (1, 1, 1536, False, 4, True),
        (1, 2, 2048, False, 8, True),
        (1, 1, 3072, False, 8, True),
        (1, 1, 4096, False, 16, False),  # 16 chunks are past one cluster: two launches as before
        (1, 2, 1024, True, 4, True),  # a replay-declared graph splits earlier
        # 8 chunks: one wave holds 15 clusters, so two and three users are marked and four (16 clusters) are not
        (2, 2, 2048, False, 8, True),
        (3, 1, 2048, False, 8, True),
        (3, 2, 2048, False, 8, True),
        (4, 1, 2048, False, 8, False),
        (4, 2, 2048, False, 8, False),
        (4, 1, 3072, False, 8, False),
        # 4 chunks: one wave holds 36 clusters, so four to nine users are marked and ten (40 clusters) are not
        (4, 2, 1536, False, 4, True),
        (4, 1, 2560, False, 4, True),
        (5, 1, 2048, False, 4, True),
        (8, 1, 2048, False, 4, True),
        (9, 1, 2048, False, 4, True),
        (9, 2, 2048, False, 4, True),
        (10, 1, 8192, False, 4, False),
        # 2 chunks: one wave holds 76 clusters, and the existing rules split 2 ways up to 72
        (9, 1, 1792, False, 2, True),
        (10, 1, 2048, False, 2, True),
        (16, 1, 8192, False, 2, True),
        (18, 1, 8192, False, 2, True),
        # unsplit plans have nothing to reduce
        (12, 1, 2048, False, 1, False),
        (64, 1, 2048, False, 1, False),
    ],
)
def test_rule_table_on_152_sms(b, s_q, s_kv, replay, split, reduced):
    """The measured part: each row is the count of clusters of that size one wave holds, so the marked plans are the ones
    whose clusters are all resident at once, and one cluster more keeps two launches."""
    lead = _lead_swap_ab(facts(b, s_q, s_kv, 152, replay))
    assert lead.pack_gqa and (lead.split_kv, bool(lead.cluster_reduce)) == (split, reduced)


@pytest.mark.parametrize(
    "sm,heads,b,s_q,s_kv,split,reduced",
    [
        # the rule is about packed rows in one tile and clusters, not about one head ratio
        (148, (32, 8), 1, 4, 2048, 8, True),  # 16 rows of four tokens, 8 clusters of 8
        (148, (32, 8), 2, 3, 2048, 4, True),  # 16 clusters of 4
        (148, (32, 8), 2, 4, 4096, 8, False),  # 16 clusters of 8 are past the recorded 8
        (148, (32, 8), 3, 3, 2048, 4, False),  # 24 clusters of 4 are past the recorded 16
        (148, (16, 4), 1, 3, 2048, 8, True),  # 12 rows of three tokens, 4 clusters of 8
        (148, (8, 1), 1, 1, 2048, 8, True),  # 8 rows, 1 cluster of 8
        (148, (8, 1), 8, 2, 2048, 8, True),  # 16 rows of two tokens, 8 clusters of 8
        (148, (8, 1), 9, 1, 2048, 8, False),  # 9 clusters of 8
        (148, (8, 1), 16, 1, 2048, 4, True),  # 16 clusters of 4
        (148, (8, 1), 17, 1, 2048, 4, False),  # 17 clusters of 4
        # 152 SMs: 15 clusters of 8, 36 of 4 and 76 of 2 are one wave
        (152, (32, 8), 1, 4, 2048, 8, True),  # 8 clusters of 8
        (152, (32, 8), 2, 1, 2048, 8, False),  # 16 clusters of 8, one past the wave
        (152, (32, 8), 2, 3, 2048, 4, True),  # 16 clusters of 4
        (152, (32, 8), 5, 1, 8192, 4, False),  # 40 clusters of 4
        (152, (32, 8), 9, 1, 8192, 2, True),  # 72 clusters of 2
        (152, (16, 4), 1, 3, 2048, 8, True),  # 12 rows of three tokens, 4 clusters of 8
        (152, (16, 4), 9, 3, 2048, 4, True),  # 36 clusters of 4, the row
        (152, (8, 1), 1, 1, 2048, 8, True),  # 1 cluster of 8
        (152, (8, 1), 15, 1, 2048, 8, True),  # 15 clusters of 8, the row
        (152, (8, 1), 15, 2, 2048, 8, True),  # the same with 16 rows of two tokens
        (152, (8, 1), 16, 1, 2048, 8, False),  # one cluster more
        (152, (8, 1), 36, 1, 2048, 4, True),  # 36 clusters of 4, the row
        (152, (8, 1), 37, 1, 2560, 4, False),  # one cluster more
        (152, (8, 1), 64, 1, 4096, 2, True),  # 64 clusters of 2
    ],
)
def test_rule_table_for_other_head_ratios(sm, heads, b, s_q, s_kv, split, reduced):
    lead = _lead_swap_ab(facts(b, s_q, s_kv, sm, heads=heads))
    assert lead.pack_gqa and (lead.split_kv, bool(lead.cluster_reduce)) == (split, reduced)


def test_unpacked_sets_are_not_marked():
    """24 / 4 heads at three to eight tokens do not fit one packed tile: the first plan stays the prefill kernel, and the decode
    tile's unpacked set keeps two launches unless a caller sets the knob, which the row then honours. No unpacked set of any
    cell is marked."""
    for s_q in (3, 4, 6, 8):
        f = facts(1, s_q, 2048)
        plans = _plans(f)
        assert plans[0][0] == LOOKAHEAD
        unpacked = [k for name, k in plans if name == SWAP_AB and not k.pack_gqa]
        assert unpacked and all(not k.cluster_reduce for k in unpacked)
        assert engines.mismatch(SPECS[SWAP_AB].capabilities, f, dataclasses.replace(unpacked[0], cluster_reduce=True)) is None
    for cell in GRID:
        assert all(k.pack_gqa for _, k in _plans(_cell(cell)) if k.cluster_reduce), cell


def test_row_declines_more_than_one_token_unit_with_the_knob():
    """Tokens times packed heads past one tile launch several token units per head group: the row serves that set in two
    launches and declines the knob on it (the in-launch form has no device cell for it)."""
    caps = SPECS[SWAP_AB].capabilities
    for heads, s_q, pack in (((24, 4), 3, True), ((32, 2), 2, True), ((24, 4), 17, False)):
        f = facts(1, s_q, 2048, heads=heads)
        knobs = SdpaFwdKnobs(tile_m=16, tile_n=128, cga=1, sched_policy=0, pack_gqa=pack, split_kv=8)
        assert engines.mismatch(caps, f, knobs) is None, (heads, s_q)
        assert "one token unit per head group" in engines.mismatch(caps, f, dataclasses.replace(knobs, cluster_reduce=True)), (heads, s_q)
    one_unit = facts(1, 4, 2048, heads=(32, 8))
    assert engines.mismatch(caps, one_unit, SdpaFwdKnobs(tile_m=16, tile_n=128, cga=1, sched_policy=0, pack_gqa=True, split_kv=8, cluster_reduce=True)) is None


def test_form_needs_a_head_width_of_whole_warps():
    """In the in-launch form the epilogue's tensor-memory load sits behind its store's range guard, and that load is
    warp-collective: the guard must be one value per warp, which holds for head widths that are multiples of 32. Another
    width of the envelope (200) keeps its two-launch plans: the row declines the knob, the rule does not mark the set, and
    the kernel refuses to build the form."""
    caps = SPECS[SWAP_AB].capabilities
    knobs = SdpaFwdKnobs(tile_m=16, tile_n=128, cga=1, sched_policy=0, pack_gqa=True, split_kv=8)
    for width, served in ((256, True), (224, True), (192, True), (200, False), (232, False)):
        f = facts(1, 1, 2048, d_qk=width, d_v=width)
        assert engines.mismatch(caps, f, knobs) is None, width
        reason = engines.mismatch(caps, f, dataclasses.replace(knobs, cluster_reduce=True))
        assert (reason is None) if served else ("multiples of 32" in reason), width
        lead = _lead_swap_ab(f)
        assert lead.split_kv == 8 and bool(lead.cluster_reduce) is served, width
    source = (Path(prep.__file__).parent / "kernels/sm100/d256_f16_swap_ab/kernel.py").read_text()
    supported = _function(ast.parse(source), "cluster_combine_supported")
    assert "d_v % 32 == 0" in ast.unparse(supported)
    assert swap_ab.CLUSTER_REDUCE_WIDTH_STEP == 32


def test_recorded_cluster_bound_steers_the_rule(monkeypatch):
    """A part or a split count without a recorded row is never marked; a row admits up to its count and nothing else moves."""
    assert heur.FUSED_CLUSTER_BOUND == ROWS
    assert [heur.fused_cluster_bound(148, s) for s in (2, 3, 4, 8, 16)] == [16, 0, 16, 8, 0]
    assert [heur.fused_cluster_bound(152, s) for s in (2, 3, 4, 8, 16)] == [76, 0, 36, 15, 0]
    assert [heur.fused_cluster_bound(sm, 8) for sm in (132, 1000)] == [0, 0]
    for sm in (132, 1000):
        for cell in GRID:
            f = _cell(cell[:3] + (sm,) + cell[4:])
            assert all(not k.cluster_reduce for _, k in _plans(f)), (sm, cell)
    # 152 SMs: a set at the row's count is marked and the same set with one cluster more is not, for each split of the form
    for split, row in ((2, 76), (4, 36), (8, 15)):
        knobs = SdpaFwdKnobs(tile_m=16, tile_n=128, cga=1, sched_policy=0, pack_gqa=True, split_kv=split)
        assert [heur.fused_selected(facts(b, 1, 2048, 152, heads=(8, 1)), knobs) for b in (1, row, row + 1)] == [True, True, False], split
    assert not heur.fused_selected(facts(1, 1, 2048, 152), SdpaFwdKnobs(tile_m=16, tile_n=128, cga=1, sched_policy=0, pack_gqa=True, split_kv=3))
    # a raised row admits what the recorded one holds back
    assert (_lead_swap_ab(facts(4, 1, 3072)).split_kv, _lead_swap_ab(facts(4, 1, 3072)).cluster_reduce) == (8, None)
    monkeypatch.setattr(heur, "FUSED_CLUSTER_BOUND", {**heur.FUSED_CLUSTER_BOUND, (148, 8): 16})
    lead = _lead_swap_ab(facts(4, 1, 3072))
    assert (lead.split_kv, lead.cluster_reduce) == (8, True)
    assert _lead_swap_ab(facts(8, 1, 4096)).cluster_reduce is None
    # a removed row keeps the split and its two launches
    monkeypatch.setattr(heur, "FUSED_CLUSTER_BOUND", {(148, 4): 16})
    assert (_lead_swap_ab(facts(1, 1, 2048)).split_kv, _lead_swap_ab(facts(1, 1, 2048)).cluster_reduce) == (8, None)
    assert _lead_swap_ab(facts(1, 1, 1536)).cluster_reduce is True


def test_rule_marks_nothing_on_an_extension_without_the_reduced_launch(monkeypatch):
    """An extension that predates the reduced launch over pages and lengths keeps every listed set as it was: the two-launch
    plan of the same split stays in the list, instead of a marked set the adapter would decline."""
    with_form = {cell: _plans(_cell(cell)) for cell in GRID[:480]}
    assert any(k.cluster_reduce for plans in with_form.values() for _, k in plans)
    monkeypatch.setattr(cudnn._pybind_module, "_SdpaDenseBinder", SimpleNamespace())
    for cell, plans in with_form.items():
        older = _plans(_cell(cell))
        assert all(not k.cluster_reduce for _, k in older), cell
        assert older == [(name, dataclasses.replace(k, cluster_reduce=None)) for name, k in plans], cell


def test_rule_proposes_nothing_off_the_verified_part():
    for cell in GRID:
        f = _cell(cell, cc=(10, 3))
        assert all(not k.cluster_reduce for _, k in _plans(f)), cell
    assert not heur.fused_selected(facts(sm=0), SdpaFwdKnobs(tile_m=16, pack_gqa=True, split_kv=8))


def test_rows_of_152_sms_mark_nothing_on_another_compute_capability():
    """152 SMs is also the SM count of compute capability 10.3 parts, and the table's key is (SM count, split). The form's
    own domain carries the capability: every set the rows mark on a 10.0 device stays unmarked on a 10.3 device with the
    same SM count, the rule answers no for it, and the row declines the knob there by name."""
    caps = SPECS[SWAP_AB].capabilities
    marked = set()
    for cell in GRID:
        if cell[3] != 152:
            continue
        lead = _lead_swap_ab(_cell(cell))
        if lead is None or not lead.cluster_reduce:
            continue
        marked.add(lead.split_kv)
        other = _cell(cell, cc=(10, 3))
        assert other.device_sm_count == 152 and all(not k.cluster_reduce for _, k in _plans(other)), cell
        assert not heur.fused_selected(other, dataclasses.replace(lead, cluster_reduce=None)), cell
        assert "verified on compute capability 10.0 only" in engines.mismatch(caps, other, lead), cell
        # the two-launch set of the same split is still served there
        assert engines.mismatch(caps, other, dataclasses.replace(lead, cluster_reduce=None)) is None, cell
    assert marked == {2, 4, 8}


def _function(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_cluster_barrier_is_one_ordered_pair_behind_every_role():
    """The reduction reads the peers' partials behind ONE ordered cluster barrier that every thread of every chunk passes.
    Pinned in the kernel's source: the arrive and the wait occur once each, consecutively, at the top level of the kernel
    function, behind the role chain and under the in-launch form's compile-time condition only; no role function returns
    (an early return would let threads leave without arriving); and the host entry launches the split axis as one
    thread-block cluster under the same condition, without which the pair would order nothing across chunks."""
    path = Path(prep.__file__).parent / "kernels/sm100/d256_f16_swap_ab/kernel.py"
    source = path.read_text()
    tree = ast.parse(source)
    arrive, wait = "barrier_cluster_arrive_aligned", "barrier_cluster_wait_aligned"
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and "barrier_cluster" in n.func.attr]
    assert sorted(calls) == sorted([arrive, wait]), "exactly one ordered cluster-barrier pair, and no relaxed form"
    kernel = _function(tree, "_kernel")
    roles = ("_softmax_warp_group", "_mma_warp_group", "_tmaldg_warp_group")
    chain = [
        i
        for i, node in enumerate(kernel.body)
        if isinstance(node, ast.If) and any(isinstance(c, ast.Call) and getattr(c.func, "id", "") in roles for c in ast.walk(node))
    ]
    assert len(chain) == 1, "one role chain at the top level of the kernel"
    called = {c.func.id for c in ast.walk(kernel.body[chain[0]]) if isinstance(c, ast.Call) and getattr(c.func, "id", "") in roles}
    assert called == set(roles)
    reduce_arm = kernel.body[chain[0] + 1]
    assert isinstance(reduce_arm, ast.If) and ast.unparse(reduce_arm.test) == "cutlass.const_expr(final_o_tensor is not None)"
    assert not reduce_arm.orelse and kernel.body[-1] is reduce_arm
    first, second = reduce_arm.body[0], reduce_arm.body[1]
    assert ast.unparse(first) == f"nvvm.{arrive}()" and ast.unparse(second) == f"nvvm.{wait}()"
    for role in roles:
        assert not any(isinstance(n, ast.Return) for n in ast.walk(_function(tree, role))), f"{role} must not return early"
    host = _function(tree, "_host")
    clusters = [k for n in ast.walk(host) if isinstance(n, ast.Call) for k in n.keywords if k.arg == "cluster"]
    assert len(clusters) == 1 and ast.unparse(clusters[0].value) == "(1, 1, SPLIT_KV) if args[-2] is not None else None"
    # in this form the tensor-memory load in front of the release is issued only where its thread stores from it
    softmax = _function(tree, "_softmax_warp_group")
    guards = [n for n in ast.walk(softmax) if isinstance(n, ast.If) and ast.unparse(n.test) == "cutlass.const_expr(reduce_follows)"]
    assert len(guards) == 1 and ast.unparse(guards[0].body[0]) == "o_loaded = o_loaded & (d_idx < cutlass.Int32(D_V))"


def test_form_holds_the_split_counts_of_the_lean_group():
    """The contract's split range is the lean group of the separate pass (read from its source: the module needs the DSL)."""
    combine = ast.parse((Path(prep.__file__).parent / "kernels/sm100/split_combine.py").read_text())
    lean = next(n.value.value for n in combine.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "G_LEAN_SMALL")
    assert swap_ab.CLUSTER_REDUCE_SPLITS == (2, lean)


@pytest.mark.parametrize("s_kv", [128, 2048, 16384])
def test_default_placement_is_unchanged(monkeypatch, s_kv):
    """With the engines' opt-in flag unset: one user x one token still ranks the backend first up to 16,384 declared keys
    (the marked plan does not change its class), two tokens lead as before, and six tokens lead with the prefill kernel."""
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    spec = SPECS[SWAP_AB]
    one = facts(1, 1, s_kv)
    assert placement.place(spec, one, _lead_swap_ab(one)) == placement.TRAIL
    two = facts(1, 2, s_kv)
    assert placement.place(spec, two, _lead_swap_ab(two)) == placement.LEAD
    six = _plans(facts(1, 6, s_kv))
    assert six[0][0] == LOOKAHEAD


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("requested", [False, True])
def test_adapter_passes_the_keywords_only_for_a_requested_plan(stats, requested):
    """The Blackwell-line tile reduces in the launch on request, never by itself; without the request the compile
    call is the one of a tree without the form (no extra compile key per Stats presence)."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    def compile(*, has_lse=True, lse_kind="dense", d_qk=256, d_v=256, paged_hnd=False, use_pdl=False, cluster_combine=False, has_final_lse=True):  # noqa: A001
        raise AssertionError("host-only selection must not compile")

    km = SimpleNamespace(compile=compile, cluster_combine_supported=lambda dq, dv: True, CLUSTER_COMBINE_BY_KNOB=True)
    api = SimpleNamespace(
        packed_thd_split=False,
        thd=False,
        thd_decode_leg=False,
        _k_mod=km,
        lse_desc=object() if stats else None,
        paged=True,
        _paged_pool_stride=lambda desc: (1, 2, 3, 4),
        k_desc=None,
        use_pdl=False,
        stats_log2=False,
        gate_desc=None,
        _device_cc=(10, 0),
        s_q_max=2,
        head_dim_qk=256,
        head_dim_v=256,
        split_kv=8,
        cluster_reduce=requested,
        _not_implemented_error_if=lambda cond, msg: (_ for _ in ()).throw(NotImplementedError(msg)) if cond else None,
    )
    result = SdpaFwdDslSm100._explicit_compile_kwargs(api)
    assert api._cluster_combine is requested
    if requested:
        assert result["cluster_combine"] is True and result["has_final_lse"] is stats
    else:
        assert "cluster_combine" not in result and "has_final_lse" not in result
    assert result["has_lse"] is True
    km.cluster_combine_supported = lambda dq, dv: False
    if requested:
        with pytest.raises(NotImplementedError, match="no in-launch reduction"):
            SdpaFwdDslSm100._explicit_compile_kwargs(api)


@pytest.mark.parametrize(
    "change,reason",
    [
        (None, None),
        ("old_extension", "rebuild the extension"),
        ("no_cluster", "cannot run a thread-block cluster of 8 blocks"),
        ("prefill", "does not lower onto it"),
        ("rubin", "does not lower onto it"),
        ("pdl", "programmatic-dependent-launch"),
        ("sink", "does not carry a sink"),
        ("split16", "2 to 8 KV chunks"),
    ],
)
def test_standalone_adapter_declines_like_the_row_and_adds_the_installation_checks(monkeypatch, change, reason):
    """The adapter's twin of the row decline, on a stand-in plan: the same contract predicate, then the two answers only an
    installation can give -- an extension that binds the reduced launch over pages and lengths, and a device that can launch
    a cluster of that size (a device query at support, never on the planning path)."""
    from cudnn.frost import occupancy
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    binder = SimpleNamespace() if change == "old_extension" else SimpleNamespace(supports_fused_split_paged=True)
    monkeypatch.setattr(cudnn._pybind_module, "_SdpaDenseBinder", binder)
    table = [148, 74, 45, 33, 26, 22, 15, 0 if change == "no_cluster" else 15] + [0] * 8
    monkeypatch.setattr(occupancy, "occupancy_map", lambda device=None: table)
    half = SimpleNamespace(dtype="bfloat16", device=SimpleNamespace(index=0))
    api = SimpleNamespace(
        _fp8=False,
        _mixed_fp8_kv=False,
        _device_cc=(10, 7) if change == "rubin" else (10, 0),
        _decode_q_tile=lambda: 0 if change == "prefill" else 16,
        flavor=(256, 256),
        s_q_max=1,
        h_q=24,
        h_kv=4,
        head_dim_v=256,
        pack_gqa=True,
        split_kv=16 if change == "split16" else 8,
        thd=False,
        has_sink=change == "sink",
        stats_log2=False,
        gate_desc=None,
        o_desc=half,
        q_desc=half,
        use_pdl=change == "pdl",
    )
    got = SdpaFwdDslSm100._cluster_reduce_decline(api)
    assert got is None if reason is None else reason in got


def _paged_fused_fixture(stats):
    """A split-8 paged D256 spec with per-batch KV and Q lengths whose host order is this tile's own, bound fused."""
    from test_sdpa_native_split_binding import _split_fixture

    spec, f, frames, combined = _split_fixture(d=256, splits=8, paged=True, hnd=True, lengths=True, lse=stats)
    path = Path(prep.__file__).parent / "kernels/sm100/d256_f16_swap_ab/kernel.py"
    host = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_host")
    values = dict(zip(spec.order, spec.template))
    spec.order = [arg.arg for arg in host.args.args if "Constexpr" not in ast.unparse(arg.annotation)]
    assert spec.order[-5:] == ["final_o_ptr", "final_lse_ptr", "final_o_strides", "final_lse_strides", "stream"]
    spec.index = {name: i for i, name in enumerate(spec.order)}
    spec.template = [values.get(name) for name in spec.order]
    spec.fused_split = True
    spec.native = cudnn._pybind_module._SdpaDenseBinder(spec)
    return spec, f, frames, combined


@pytest.mark.parametrize("stats", [False, True])
def test_native_and_python_frames_agree_over_pages_and_lengths(stats):
    """The reduced launch binds the final O / Stats beside the partial slabs over a paged cache with per-batch lengths, the
    native binder and the Python frame agree slot for slot, and exactly one launch is issued."""
    import sdpa_binding_reference as reference
    from test_sdpa_native_dense_binding import _pack

    if not getattr(cudnn._pybind_module._SdpaDenseBinder, "supports_fused_split_paged", False):
        pytest.skip("the installed extension predates the reduced launch over pages and lengths")
    spec, f, frames, combined = _paged_fused_fixture(stats)
    assert spec.paged and spec.seq_kv_present and spec.seq_q_present
    for offset, stream in [(0, 17), (0x200000, 29)]:
        current = {role: fact._replace(ptr=fact.ptr + offset) for role, fact in f.items()}
        workspace = 2**40 + offset
        expected, combine = reference.bind_dense_split(spec, current, workspace, stream, stream)
        for name, value in {
            "final_o_ptr": current["o"].ptr,
            "final_lse_ptr": current["lse"].ptr if stats else None,
            "final_o_strides": combine[6][:3],
            "final_lse_strides": combine[7],
        }.items():
            expected[spec.index[name]] = value
        actual = spec.native.bind_split(_pack(current), prep._NATIVE_DENSE_INDICES, workspace, stream)
        python_frame, python_combine = prep.bind_dense_split(spec, current, workspace, stream, stream)
        assert list(actual[0]) == python_frame == expected
        assert actual[1] == python_combine == combine
        spec.native.execute(_pack(current), prep._NATIVE_DENSE_INDICES, stream, workspace=workspace)
        assert frames[-1] == tuple(expected)
    assert len(frames) == 2 and combined == []
    # the reduction reads the partial rows as 16-byte groups: both frames refuse an unaligned workspace, as for every split
    with pytest.raises(ValueError, match="16-byte aligned"):
        prep.bind_dense_split(spec, f, 2**40 + 8, 17, 17)
    with pytest.raises(ValueError, match="16-byte aligned"):
        spec.native.bind_split(_pack(f), prep._NATIVE_DENSE_INDICES, 2**40 + 8, 17)
    with pytest.raises(ValueError, match="overlaps"):
        spec.native.execute(_pack(f), prep._NATIVE_DENSE_INDICES, 17, workspace=f["o"].ptr)


def test_reduced_frame_still_refuses_a_sink():
    from test_sdpa_native_split_binding import _split_fixture

    spec, _, _, _ = _split_fixture(d=256, splits=8, lengths=False, lse=True)
    spec.fused_split, spec.has_sink = True, True
    with pytest.raises(ValueError, match="without a sink"):
        cudnn._pybind_module._SdpaDenseBinder(spec)
