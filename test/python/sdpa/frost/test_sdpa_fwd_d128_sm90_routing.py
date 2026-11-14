# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the SM90 per-flavor plumbing (issue #378): no SM90 device needed.

The SM90 row lowers two templates for exact (128, 128) graphs, chosen by the tile knob: the D512
body (#2426) on its 64 x 64 tile, and the dedicated D128 template on a 128 x 128 tile that serves
a dense subset. The row's, the heuristics' and config_sm90's seams are checked here, on every FROST
lane: no CI lane runs Hopper. Most tests read the row spelled out field by field
(``_two_flavor_row``); ``test_the_two_flavor_row_is_the_shipped_row`` pins it equal to the shipped
row, so each of them is a test of the shipped row, and the tests that name ``_caps()`` read the
shipped row directly. The adapter's seams (its flavor pick, that flavor's tile, the template it
loads) ask the real device and are pinned in the cc 9.0 suite, test_sdpa_fwd_dsl_sm90.py; the D128
kernel's numerics and graph route are test_sdpa_fwd_d128_sm90.py's.
"""

import dataclasses
import itertools

import pytest

from frost_test_utils import requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]

_ROW = "sdpa_fwd_prefill_sm90"
_D512, _D128 = (512, 512), (128, 128)


def _caps(name=_ROW):
    from cudnn.sdpa.fwd import engines

    return {s.name: s.capabilities for s in engines.ENGINE_SPECS}[name]


def _two_flavor_row(**over):
    """The SM90 row with the D128 template's 128 x 128 tile spelled out, as #378 spells it: the
    (128, 128) d-shape's Q and KV tile domains hold 64 (the D512 body) and 128 (D128). Equal to the
    shipped row (test_the_two_flavor_row_is_the_shipped_row); ``over`` varies one field."""
    row = dataclasses.replace(
        _caps(),
        tile_ms_by_d_shape=((_D128, frozenset({64, 128})),),
        tile_ns_by_d_shape=((_D128, frozenset({64, 128})),),
    )
    return dataclasses.replace(row, **over)


def _facts(d=128, **kw):
    """Half facts for a Hopper graph at head dims ``d`` (an int for d_qk == d_v, else a pair)."""
    import cudnn
    from cudnn.sdpa import graph_analyzer as ga

    d_qk, d_v = (d, d) if isinstance(d, int) else d
    base = dict(b=2, h_q=8, h_kv=8, s_q=512, s_kv=512, d_qk=d_qk, d_v=d_v, dtype=cudnn.data_type.HALF, dtype_o=cudnn.data_type.HALF)
    base.update(device_cc=(9, 0), device_sm_count=132)
    base.update(kw)
    return ga.SdpaGraphFacts(**base)


def test_sm90_d128_set_leads_only_from_its_query_row_floor():
    """heuristics._sm90_d128_knob_sets offers the D128 template's 128 x 128 set on an exact dense (128, 128) graph the
    template serves only from config_sm90.D128_LEAD_MIN_SQ query rows: one row below, the set is empty and the D512
    body's sets (its packed and split decode runners among them) keep the SM90 row's order; at the floor and above it
    leads with one set at the 128 x 128 tile. The floor's value is a measured tuning choice and is read, not pinned
    (test/AGENTS.md). Decode and multi-token shapes, MHA and GQA, mask-free and bottom-right causal."""
    from cudnn.sdpa.fwd import engines, heuristics
    from cudnn.sdpa.fwd.config_sm90 import D128_LEAD_MIN_SQ

    spec = next(s for s in engines.ENGINE_SPECS if s.name == _ROW)
    floor = D128_LEAD_MIN_SQ
    for h_q, h_kv in ((16, 16), (32, 8)):
        for mask in (dict(), dict(causal=True, bottom_right=True)):
            for s_q in sorted({2, 4, 16, floor - 1}):
                facts = _facts(h_q=h_q, h_kv=h_kv, s_q=s_q, s_kv=4096, **mask)
                assert heuristics._sm90_d128_knob_sets(spec, facts) == [], (h_q, h_kv, mask, s_q)
            for s_q in (floor, max(floor, 512)):
                sets = heuristics._sm90_d128_knob_sets(spec, _facts(h_q=h_q, h_kv=h_kv, s_q=s_q, s_kv=4096, **mask))
                assert len(sets) == 1 and (sets[0].tile_m, sets[0].tile_n) == (128, 128), (h_q, h_kv, mask, s_q, sets)


def test_sm90_d128_sched_proposals_stay_in_the_legal_domain():
    """The graph route's scheduler proposal on the SM90 row (heuristics._sched_points), with and without the D128
    template's ``d128`` flag, is a non-empty, duplicate-free subset of the row's legal scheduler domain
    (engines.effective_sched_policies) on causal D128 graphs whose record is the paired walk, the NATURAL stride or one
    tile per CTA, at 132 and 114 SMs, on a causal D512 graph and on mask-free graphs. Which order comes first is the
    record's own choice, which test_sm90_d128_proposal_mirrors_the_record_check_support_picks checks against
    check_support; the order itself is a tuning choice and is not pinned here (test/AGENTS.md)."""
    from cudnn.sdpa.fwd import engines, heuristics

    row = _two_flavor_row()
    graphs = [
        (128, dict(causal=True, b=8, h_q=16, h_kv=16, s_q=1024, s_kv=1024)),
        (128, dict(causal=True, bottom_right=True, b=16, h_q=32, h_kv=8, s_q=256, s_kv=2048)),
        (128, dict(causal=True, h_q=32, h_kv=8, s_q=4096, s_kv=4096)),
        (128, dict(causal=True)),
        (128, dict(causal=True, b=1, h_q=8, h_kv=8, s_q=2048, s_kv=131072, bottom_right=True)),
        (128, dict()),
        (512, dict(causal=True)),
        (512, dict()),
    ]
    for d, kw in graphs:
        for ctas in (132, 114):
            facts = _facts(d, device_sm_count=ctas, **kw)
            domain = set(engines.effective_sched_policies(row, facts))
            for d128 in (False, True):
                points = heuristics._sched_points(row, facts, d128=d128)
                assert points and len(points) == len(set(points)) and set(points) <= domain, (d, kw, ctas, d128, points)


def test_sm90_d128_proposal_mirrors_the_record_check_support_picks(monkeypatch):
    """The graph route's first order for a causal D128 graph is the one check_support gives that graph's record without
    a pin, under every setting of api_dsl's call-time switches, read from the module: NATURAL where the record is the
    walk's (api_dsl._sm90_d128_persistence's causal_walk, less what api_dsl._sm90_d128_causal_route routes while
    D128_CAUSAL_WALK_ROUTE is on), else api_dsl._causal_sched_policy. Shapes across both route branches, both alignments,
    GQA and MHA, at 132 and 114 SMs. With api_dsl not loaded the proposal is the shipped switches' one."""
    import sys

    from cudnn.frost.tile_dsl.constants import SCHED_NATURAL
    from cudnn.sdpa.fwd import api_dsl, heuristics

    row = _two_flavor_row()
    shapes = [
        (b, h_q, h_kv, s_q, s_kv, br)
        for (b, h_q, h_kv) in ((1, 8, 8), (2, 8, 8), (4, 32, 8), (16, 32, 8), (2, 64, 64))
        for (s_q, s_kv) in ((128, 128), (512, 512), (1024, 1024), (256, 2048), (2048, 8192), (4096, 4096), (1024, 16384))
        for br in (False, True)
        if s_q <= s_kv
    ]

    def expected(b, h_q, h_kv, s_q, s_kv, br, ctas):
        _, walk = api_dsl._sm90_d128_persistence(api_dsl._SM90_D128_FLAVOR, True)
        routed = api_dsl.D128_CAUSAL_WALK_ROUTE and api_dsl._sm90_d128_causal_route((b, h_q, h_kv, s_q, s_kv), br, ctas)
        return SCHED_NATURAL if walk and not routed else api_dsl._causal_sched_policy(s_kv, 128, 128, elem_bytes=2)

    def proposals(ctas):
        return [
            heuristics._sched_points(row, _facts(128, causal=True, b=b, h_q=hq, h_kv=hk, s_q=sq, s_kv=sk, bottom_right=br, device_sm_count=ctas), d128=True)[0]
            for b, hq, hk, sq, sk, br in shapes
        ]

    settings = (
        {},
        {"D128_PERSISTENT_CAUSAL_WALK": 0},
        {"D128_PERSISTENT_CAUSAL_WALK": 1},
        {"D128_CAUSAL_WALK_ROUTE": False},
        {"D128_PERSISTENT": False},
        {"D128_INTRA_WG_OVERLAP": False},
        # The claim counter hands out only records check_support keeps persistent and NATURAL: the proposal stays, with
        # the counter off and with its gate at 0.
        {"D128_CAUSAL_CLAIM": 0},
        {"D128_CLAIM_MIN_ROUNDS": 0},
    )
    for setting in settings:
        with monkeypatch.context() as m:
            for name, value in setting.items():
                m.setattr(api_dsl, name, value)
            for ctas in (132, 114):
                want = [expected(*shape, ctas) for shape in shapes]
                assert proposals(ctas) == want, (setting, ctas)
    shipped = {ctas: proposals(ctas) for ctas in (132, 114)}
    monkeypatch.delitem(sys.modules, "cudnn.sdpa.fwd.api_dsl")
    assert {ctas: proposals(ctas) for ctas in (132, 114)} == shipped


def test_sm90_d128_shipped_switches_are_api_dsls():
    """heuristics reads api_dsl's walk switches from the module once it is loaded and from _SM90_D128_SHIPPED_SWITCHES
    before: those must be the values api_dsl ships with, or a graph proposed before the adapter loads would carry another
    order than one proposed after."""
    from cudnn.sdpa.fwd import api_dsl, heuristics

    assert heuristics._SM90_D128_SHIPPED_SWITCHES == tuple((name, getattr(api_dsl, name)) for name, _ in heuristics._SM90_D128_SHIPPED_SWITCHES)
    assert [name for name, _ in heuristics._SM90_D128_SHIPPED_SWITCHES] == [
        "D128_INTRA_WG_OVERLAP",
        "D128_PERSISTENT",
        "D128_PERSISTENT_CAUSAL_WALK",
        "D128_CAUSAL_WALK_ROUTE",
    ]


def test_the_two_flavor_row_is_the_shipped_row():
    """The spelled-out row is the shipped one, so every test below that reads ``_two_flavor_row()``
    tests what ``engines._sm90_spec`` ships. A tile entry spelled in ``_two_flavor_row`` that the row
    drops or changes fails here first (the tile overrides are tuples: their order counts). Row-wide
    fields are copied from the shipped row, so this pin cannot see them;
    test_the_shipped_row_proposes_each_flavors_own_geometry and the cc 9.0 suite's
    test_sm90_every_candidate_is_valid read them on the shipped row."""
    assert _two_flavor_row() == _caps()


# --- The row against config_sm90 (rule 8b': row and adapter read the same constants) -------

# Each D128_NOT_SERVED key as an exact (128, 128) graph declares it (facts) or a knob set asks for it (knobs).
_NOT_SERVED_GRAPHS = {
    "thd": (dict(thd=True, padded=True), {}),
    "paged": (dict(has_paged_kv=True, page_size=64, padded=True), {}),
    "split_kv": ({}, dict(split_kv=2)),
    "sink": (dict(has_sink=True), {}),
    "window_left": (dict(causal=True, window_left=16), {}),
    "window_right": (dict(right_band_widening=True, right_bound=4), {}),
    "lengths": (dict(padded=True), {}),
    "pack_gqa": (dict(h_kv=2), dict(pack_gqa=True)),
}


def test_the_row_mirrors_config_sm90_feature_by_feature():
    """Lockstep with the adapter's table: every D128_NOT_SERVED key, declared on an exact (128, 128)
    graph, is declined on the 128 x 128 tile with config_sm90.d128_decline's label and served on the
    64 x 64 tile (the D512 body), so the row offers the dedicated template exactly what the adapter
    compiles on it; a key added to the table without a case here fails. S_q == 1 is a ROW-ONLY
    decline of the tile: config_sm90 has no decode key, since the D128 kernel runs S_q == 1 on the
    direct route, but decode is outside the slice #378 accepted."""
    from cudnn.sdpa.fwd import config_sm90 as c, engines

    row = _caps()
    assert set(_NOT_SERVED_GRAPHS) == set(c.D128_NOT_SERVED), "a NOT_SERVED key without a case (or the reverse)"
    for key, (declared, asked) in _NOT_SERVED_GRAPHS.items():
        facts = _facts(128, **declared)
        assert engines.mismatch(row, facts, engines.SdpaFwdKnobs(tile_m=64, tile_n=64, **asked)) is None, key
        reason = engines.mismatch(row, facts, engines.SdpaFwdKnobs(tile_m=128, tile_n=128, **asked))
        assert reason == f"SM90's 128 x 128 tile (the D128 template) does not serve {c.D128_NOT_SERVED[key]}", (key, reason)
    assert "decode" not in c.D128_NOT_SERVED
    decode = _facts(128, s_q=1)
    assert engines.mismatch(row, decode, engines.SdpaFwdKnobs(tile_m=64, tile_n=64)) is None
    assert (engines.mismatch(row, decode, engines.SdpaFwdKnobs(tile_m=128, tile_n=128)) or "").startswith("s_q == 1 (decode) is outside")
    assert engines.effective_tile_ms(row, _facts(128)) == engines.effective_tile_ns(row, _facts(128)) == {c.TILE_M, *c.FLAVOR_TILES[c.D128_FLAVOR]}


def test_row_and_picker_offer_the_128_tile_at_the_same_head_dims():
    """Lockstep: the row admits the 128 x 128 tile on a dense graph exactly where the adapter's picker
    takes D128 with nothing declared (pick_flavor over the full pool: exact (128, 128), FLAVOR_FLOORS),
    and every head pair the row admits keeps its 64 x 64 plan. Drift would rank a D128 plan the adapter
    compiles on the D512 body, or the reverse."""
    from cudnn.sdpa.fwd import config_sm90 as c, engines

    row, d128, d512 = _caps(), engines.SdpaFwdKnobs(tile_m=128, tile_n=128), engines.SdpaFwdKnobs(tile_m=64, tile_n=64)
    for d_qk, d_v in itertools.product(range(8, 513, 8), repeat=2):
        facts = _facts((d_qk, d_v))
        if engines.mismatch(row, facts) is None:
            assert engines.mismatch(row, facts, d512) is None, (d_qk, d_v)
        assert (engines.mismatch(row, facts, d128) is None) == (c.pick_flavor(d_qk, d_v, c.FLAVORS) == c.D128_FLAVOR), (d_qk, d_v)


# --- Outer-stride alignment on the row (the #2038 row check, with config_sm90.dense_bind_strides) -------


class _Port:
    """A half graph tensor as mismatch() reads one: dims, strides and data type, no graph needed."""

    def __init__(self, dim, stride):
        import cudnn

        self._dim, self._stride, self._dtype = tuple(dim), tuple(stride), cudnn.data_type.HALF

    def get_dim(self):
        return list(self._dim)

    def get_stride(self):
        return list(self._stride)

    def get_data_type(self):
        return self._dtype


def _bshd_ports(d, pitch=None, role=None, b=2, h=8, s=512):
    """Q/K/V/O descriptors at (b, h, s, d), BSHD-physical; ``role`` gets a token pitch of ``pitch`` elements instead of d."""
    ports = {}
    for name in ("q", "k", "v", "o"):
        p = pitch if name == role else d
        ports[name + "_t"] = _Port((b, h, s, d), (s * h * p, p, h * p, 1))
    return ports


@pytest.mark.parametrize("d", [128, 512], ids=["d128", "d512"])
@pytest.mark.parametrize("role", ["q", "k", "v", "o"])
def test_the_row_declines_outer_strides_off_16_bytes(d, role):
    """The SM90 adapter binds Q/K/V/O as TMA maps and refuses, in check_support, a dense pitch that is not a 16-byte
    multiple. The row now declines it too, on both flavors, through the predicate the SM90 prepared launch uses
    (config_sm90.dense_bind_strides), as the SM100, SM107 and SM120 rows do since #2038. An 8-byte pad (d + 4 half
    elements) is declined by name; a 16-byte pad (d + 8) is served."""
    from cudnn.sdpa.fwd import engines

    reason = engines.mismatch(_caps(), _facts(d, **_bshd_ports(d, pitch=d + 4, role=role)))
    assert (reason or "").startswith(
        f"{role}: native dense forward requires contiguous D, non-overlapping B/H/S strides and 16-byte outer-stride alignment"
    ), reason
    assert engines.mismatch(_caps(), _facts(d, **_bshd_ports(d, pitch=d + 8, role=role))) is None


def test_the_row_stride_check_reads_only_declared_dense_ports():
    """Controls: a unit batch axis's stride is never stepped, so an odd one is served (config_sm90 pins it to D, as the
    adapter does); a pure-fact probe with no descriptors is not judged; a THD graph keeps its own packed-layout checks."""
    from cudnn.sdpa.fwd import engines

    ports = _bshd_ports(128, b=1)
    ports["q_t"] = _Port((1, 8, 512, 128), (12345, 128, 8 * 128, 1))
    assert engines.mismatch(_caps(), _facts(128, b=1, **ports)) is None
    assert engines.mismatch(_caps(), _facts(128)) is None
    thd = _bshd_ports(512, pitch=516, role="q")
    assert "16-byte outer-stride" not in (engines.mismatch(_caps(), _facts(512, thd=True, padded=True, **thd)) or "")


# --- Tile domains per flavor (tile_ms_by_d_shape / tile_ns_by_d_shape) -------


def test_effective_tiles_read_the_selected_flavor():
    """An override entry REPLACES the row-wide domain for the d-shape the row selects (through its
    d_envelope_floors, so (120, 72) reads the (128, 128) entry too). A d-shape without an entry reads
    the row-wide domain; the default () reads it everywhere."""
    from cudnn.sdpa.fwd import engines

    row, shipped = _two_flavor_row(), _caps()
    bare = dataclasses.replace(shipped, tile_ms_by_d_shape=(), tile_ns_by_d_shape=())
    for read in (engines.effective_tile_ms, engines.effective_tile_ns):
        assert read(row, _facts(128)) == read(row, _facts((120, 72))) == frozenset({64, 128})
        assert read(row, _facts(512)) == read(row, _facts((264, 504))) == read(row, _facts(256)) == read(row, _facts(64)) == frozenset({64})
        assert read(shipped, _facts(128)) == frozenset({64, 128}), "the shipped row offers 128 at (128, 128)"
        assert read(bare, _facts(512)) == read(bare, _facts(128)) == frozenset({64}), "() reads the row-wide domain"


@pytest.mark.parametrize(
    ("d", "tile", "why"),
    [
        (128, (128, 128), None),
        (128, (64, 64), None),
        (128, (128, 64), "SM90's D128 template runs a 128 x 128 tile; requested tile_m=128/tile_n=64"),
        ((120, 72), (128, 128), "SM90's 128 x 128 tile (the D128 template) serves exact (128, 128) head dims; graph has D_QK=120/D_V=72"),
        (512, (64, 64), None),
        (512, (128, 128), "requested tile_m=128 is outside this engine's domain [64]"),
        (256, (128, 128), "requested tile_m=128 is outside this engine's domain [64]"),
        ((264, 504), (128, 128), "requested tile_m=128 is outside this engine's domain [64]"),
    ],
    ids=["d128-128", "d128-64", "d128-mixed", "d120x72-128", "d512-own", "d512-pins-128", "d256-pins-128", "envelope-pins-128"],
)
def test_tile_knobs_are_judged_per_flavor(d, tile, why):
    """mismatch() judges a pinned tile against the SELECTED d-shape's domain: (128, 128) admits both
    tiles (each names its template), every other d-shape only the D512 body's 64, so no plan is
    ranked at a tile its kernel would refuse at build. Inside (128, 128) the 128 tile names the D128
    template: both axes must ask for it, and only exact (128, 128) head dims ride it."""
    from cudnn.sdpa.fwd import engines

    reason = engines.mismatch(_two_flavor_row(), _facts(d), engines.SdpaFwdKnobs(tile_m=tile[0], tile_n=tile[1]))
    assert reason == why, reason


@pytest.mark.parametrize(
    ("d", "s_q", "s_kv", "declared", "why"),
    [
        pytest.param(128, 512, 256, {}, "unpadded bottom-right S_q > S_kv is not yet supported on SM90", id="d128"),
        pytest.param(128, 256, 512, {}, None, id="d128-s_q-below-s_kv"),
        pytest.param(128, 512, 512, {}, None, id="d128-square"),
        pytest.param(128, 512, 256, dict(padded=True), "SM90's 128 x 128 tile (the D128 template) does not serve per-batch lengths", id="d128-padded"),
        pytest.param(512, 512, 256, {}, None, id="d512"),
        pytest.param((264, 504), 512, 256, {}, None, id="envelope"),
    ],
)
def test_unpadded_bottom_right_above_s_kv_is_declined_at_d128(d, s_q, s_kv, declared, why):
    """The SM90 adapter declines unpadded bottom-right causal with S_q > S_kv at build on every flavor (its pending case;
    the cc 9.0 suite pins that side). On the 128 x 128 tile the shipped row declines it too, with the adapter's message, so
    no D128 plan is ranked that cannot build; a padded graph meets the lengths decline first. The 64 x 64 tile, at D512,
    its envelope and (128, 128) alike, still ranks it and leaves the decline to the adapter, and on the 128 x 128 tile
    bottom-right with S_q <= S_kv and top-left causal at S_q > S_kv are served."""
    from cudnn.sdpa.fwd import engines

    knobs = engines.SdpaFwdKnobs(tile_m=128, tile_n=128) if d == 128 else None
    reason = engines.mismatch(_caps(), _facts(d, s_q=s_q, s_kv=s_kv, causal=True, bottom_right=True, **declared), knobs)
    assert reason is None if why is None else reason is not None and reason.startswith(why), reason
    top_left = engines.mismatch(_caps(), _facts(d, s_q=s_q, s_kv=s_kv, causal=True, **declared), knobs)
    assert top_left is None if not declared else top_left == reason, top_left
    if d == 128:
        small = engines.SdpaFwdKnobs(tile_m=64, tile_n=64)
        assert engines.mismatch(_caps(), _facts(d, s_q=s_q, s_kv=s_kv, causal=True, bottom_right=True, **declared), small) is None


def test_every_flavor_spelled_out_equals_the_defaults_on_every_shipped_row(monkeypatch):
    """() means "the row-wide domain at every d-shape". Spelling that out (a tile override
    restating the row-wide domain for each d-shape that has none) changes no verdict and no message
    on any shipped half row: a reader that matched raw dims instead of the selected d-shape would
    decline the envelope graphs below, and one that skipped the override would differ at a pinned
    tile. Entries a row sets (the SM90 row's, and the SM100 and SM107 rows' tile_ns_by_d_shape) keep
    their values, so each row is checked on its default-valued entries alone and the bound still
    counts five rows."""
    from cudnn.frost import buffers
    from cudnn.sdpa.fwd import engines

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)  # model the cc 10.7 target on any DSL build
    declared = [
        {},
        dict(causal=True),
        dict(causal=True, bottom_right=True),
        dict(causal=True, window_left=16),
        dict(right_band_widening=True, right_bound=4),
        dict(padded=True),
        dict(has_sink=True),
        dict(thd=True, padded=True),
        dict(s_q=1),
    ]
    dims = [64, 128, (192, 128), 256, (264, 504), 512, (120, 128), (96, 128)]
    knob_sets = [
        None,
        engines.SdpaFwdKnobs(),
        engines.SdpaFwdKnobs(tile_m=64, tile_n=64),
        engines.SdpaFwdKnobs(tile_m=128, tile_n=128),
        engines.SdpaFwdKnobs(pack_gqa=True),
    ]

    def verdict(caps, facts, knobs):
        try:
            return engines.mismatch(caps, facts, knobs)
        except Exception as exc:  # noqa: BLE001 (a combination that raises must raise identically on both rows)
            return f"raised {exc!r}"

    checked = 0
    for spec in engines.ENGINE_SPECS:
        caps = spec.capabilities
        if caps.is_fp8 or caps.is_mxfp8:
            continue  # half facts stop at those rows' quantization check
        ms_by, ns_by = dict(caps.tile_ms_by_d_shape), dict(caps.tile_ns_by_d_shape)  # a row's own entries stay as they are
        spelled = dataclasses.replace(
            caps,
            tile_ms_by_d_shape=tuple((shape, ms_by.get(shape, caps.tile_ms)) for shape in sorted(caps.d_shapes)),
            tile_ns_by_d_shape=tuple((shape, ns_by.get(shape, caps.tile_ns)) for shape in sorted(caps.d_shapes)),
        )
        for decl, d, knobs in itertools.product(declared, dims, knob_sets):
            facts = _facts(d, h_kv=2, device_cc=divmod(caps.sm_lo, 10), **decl)
            assert verdict(spelled, facts, knobs) == verdict(caps, facts, knobs), (spec.name, decl, d, knobs)
            checked += 1
    assert checked >= 5 * len(declared) * len(dims) * len(knob_sets), checked


# --- Proposals (heuristics) -------


@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
def test_the_shipped_row_proposes_each_flavors_own_geometry(kind):
    """recommend() through the real ENGINE_SPECS, no stand-in row: the every-lane copy of the cc 9.0
    suite's test_sm90_every_candidate_is_valid. Every set is admissible, deduplicated, bounded and
    carries a tile its template runs; on an exact (128, 128) graph the D128 template serves, its
    128 x 128 sets lead (unsplit and unpacked) and the D512 body's 64 x 64 sets follow; the fallback
    is the 64 x 64 plan everywhere. Validity and the tile order only: no winner or set count."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
    from cudnn.sdpa.fwd import engines, heuristics

    row = _caps()
    for d, tiles in ((128, {(128, 128), (64, 64)}), (512, {(64, 64)}), ((264, 504), {(64, 64)})):
        facts = _facts(d, h_kv=2, s_q=1024, s_kv=8192, causal=True)
        offered = heuristics.recommend(kind, facts, {_ROW: 20517})
        assert offered and len({p.knobs for p in offered}) == len(offered) <= heuristics._MAX_SETS_PER_ENGINE, (d, offered)
        assert {p.engine_id for p in offered} == {20517}, offered
        lead = (offered[0].knobs.tile_m, offered[0].knobs.tile_n)
        assert lead == ((128, 128) if kind == "A" and (128, 128) in tiles else (64, 64)), (d, kind, lead)
        for knobs in (p.knobs for p in offered):
            assert engines.mismatch(row, facts, knobs) is None, (d, knobs)
            assert (knobs.tile_m, knobs.tile_n) in tiles and knobs.cga == 1, (d, knobs)
            if (knobs.tile_m, knobs.tile_n) == (128, 128):
                assert knobs.split_kv == 1 and knobs.pack_gqa is False, (d, knobs)
            assert knobs.sched_policy in {SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2} and knobs.pack_gqa in {False, True}, (d, knobs)


# --- config_sm90: flavors, tiles and the template backstop -------


def test_pick_flavor_without_features_takes_d128_at_exactly_128():
    """Over the default pool (D512 alone) every declaration selects D512. Over the full pool, with no
    declared feature, D128 is picked at exactly (128, 128), its floor of 127 as the row will declare
    it, and D512 everywhere else."""
    from cudnn.sdpa.fwd import config_sm90 as c

    pairs = [(a, b) for a in range(8, 513, 8) for b in range(8, 513, 8)]
    assert {c.pick_flavor(*pair) for pair in pairs} == {c.D512_FLAVOR}
    assert {pair for pair in pairs if c.pick_flavor(*pair, c.FLAVORS) == c.D128_FLAVOR} == {(128, 128)}
    assert c.pick_flavor(520, 512, c.FLAVORS) == c.D512_FLAVOR, "total: head_dims_mismatch, not the picker, declines it"
    assert c.FLAVORS == (c.D128_FLAVOR, c.D512_FLAVOR)
    assert c.FLAVOR_TILES == {c.D512_FLAVOR: (c.TILE_M, c.TILE_N), c.D128_FLAVOR: (128, 128)}


@pytest.mark.parametrize("feature", ["thd", "paged", "split_kv", "sink", "window_left", "window_right", "lengths", "pack_gqa"])
def test_pick_flavor_keeps_each_not_served_feature_on_d512(feature):
    """Decided once, up front: a (128, 128) declaration naming a feature D128 does not serve selects
    D512 directly, with or without other features; one D128 serves selects D128."""
    from cudnn.sdpa.fwd import config_sm90 as c

    assert feature in c.D128_NOT_SERVED and c.NOT_SERVED[c.D128_FLAVOR] is c.D128_NOT_SERVED
    assert c.pick_flavor(128, 128, c.FLAVORS, frozenset({feature})) == c.D512_FLAVOR
    assert c.pick_flavor(128, 128, c.FLAVORS, frozenset({feature, "lengths"})) == c.D512_FLAVOR
    assert c.pick_flavor(128, 128, c.FLAVORS, frozenset({"a_feature_d128_serves"})) == c.D128_FLAVOR


def test_declared_features_spells_each_flag_once():
    """Absent bounds and plain causal (a right bound of 0) name nothing; each other flag names itself."""
    from cudnn.sdpa.fwd import config_sm90 as c

    assert c.declared_features() == c.declared_features(window_left=None, window_right=0, split_kv=1) == frozenset()
    assert c.declared_features(window_left=0) == {"window_left"}
    assert c.declared_features(window_right=1) == {"window_right"}
    named = c.declared_features(thd=True, sink=True, window_left=4, window_right=2, lengths=True, pack_gqa=True, paged=True, split_kv=2)
    assert named == set(c.D128_NOT_SERVED)


def test_d128_refusal_and_the_picker_read_one_table():
    """validate_params(flavor=D128) refuses a specialization for a feature exactly when pick_flavor
    keeps the same declaration on D512, so the D128 template's backstop never fires on the
    adapter's path and never lets through what the picker would keep away."""
    from cudnn.sdpa.fwd import config_sm90 as c

    flags = dict(
        thd_varlen=(False, True),
        has_sink=(False, True),
        window_left=(None, 16),
        window_right=(None, 4),
        seq_kv_lens_present=(False, True),
        pack_gqa=(False, True),
        page_size=(0, 64),
        split_kv=(1, 2),
    )
    for values in itertools.product(*flags.values()):
        params = dataclasses.replace(c.TemplateParams(), **dict(zip(flags, values)))
        declared = c.declared_features(
            thd=params.thd_varlen,
            sink=params.has_sink,
            window_left=params.window_left,
            window_right=params.window_right,
            lengths=params.seq_q_lens_present or params.seq_kv_lens_present,
            pack_gqa=params.pack_gqa,
            paged=params.page_size != 0,
            split_kv=params.split_kv,
        )
        try:
            c.validate_params(params, flavor=c.D128_FLAVOR)
            refused = False
        except ValueError as exc:
            refused = str(exc).startswith("SM90 D128 does not serve ")
        assert refused == (c.pick_flavor(128, 128, c.FLAVORS, declared) == c.D512_FLAVOR), (values, declared)


def test_d128_template_refuses_at_load_what_the_picker_keeps_away():
    """The template module validates its injected specialization on import: loading it with a sink
    raises before any trace, so a caller that bypasses the adapter cannot run D128 on a declaration
    it does not serve."""
    from cudnn.sdpa.fwd import config_sm90 as c
    from cudnn.sdpa.fwd.api_dsl import _load_kernel_template

    with pytest.raises(ValueError, match="^SM90 D128 does not serve attention sinks"):
        _load_kernel_template("sm90/prefill_d128_f16.py", dataclasses.replace(c.TemplateParams(), has_sink=True), tag="refusal_probe_sm90_d128")


def test_d512_template_params_messages_are_unchanged():
    """validate_params' flavor argument defaults to D512, and D512's messages keep their spelling
    (the SM90 suite pins them by substring on cc 9.0; here they are pinned whole on any GPU)."""
    from cudnn.sdpa.fwd import config_sm90 as c

    cases = {
        "SM90 D512 requires uniform FP16 or BF16 Q/K/V/O": dict(dtype_qkv=0),
        "SM90 D512 PackGQA requires a positive head ratio sharing a factor with tile_m=64": dict(pack_gqa=True, qh_per_kh=3),
        "SM90 D512 THD uses the natural single-tile work decoder": dict(thd_varlen=True, seq_kv_lens_present=True, sched_policy=1),
        "SM90 D512 window_left must be an integer in [0, 2**30)": dict(window_left=-1),
        "SM90 D512 bottom_right repeats the alignment causal implies; the canonical spelling is None": dict(causal=True, bottom_right=True),
    }
    for message, over in cases.items():
        for kw in ({}, dict(flavor=c.D512_FLAVOR)):
            with pytest.raises(ValueError) as raised:
                c.validate_params(dataclasses.replace(c.TemplateParams(), **over), **kw)
            assert str(raised.value) == message


@pytest.mark.parametrize(
    ("over", "feature"),
    [
        (dict(thd_varlen=True, seq_kv_lens_present=True), "THD"),
        (dict(has_sink=True), "attention sinks"),
        (dict(window_left=16), "a sliding window"),
        (dict(window_right=4), "a widened right band"),
        (dict(seq_kv_lens_present=True), "per-batch lengths"),
        (dict(pack_gqa=True, qh_per_kh=2), "PackGQA"),
        (dict(page_size=64, seq_kv_lens_present=True), "paged KV"),
        (dict(split_kv=2), "split-KV"),
    ],
    ids=["thd", "sink", "swa", "right-band", "lengths", "pack-gqa", "paged", "split"],
)
def test_d128_template_params_refuse_what_d512_keeps(over, feature):
    """validate_params(flavor=D128), one case per refusal; each is a specialization D512 accepts."""
    from cudnn.sdpa.fwd import config_sm90 as c

    params = dataclasses.replace(c.TemplateParams(), **over)
    c.validate_params(params)
    with pytest.raises(ValueError, match=f"^SM90 D128 does not serve {feature}"):
        c.validate_params(params, flavor=c.D128_FLAVOR)


def test_d128_template_params_accept_the_dense_subset_under_their_own_name():
    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2
    from cudnn.sdpa.fwd import config_sm90 as c

    for over in ({}, dict(causal=True), dict(causal=True, bottom_right=False), dict(has_lse=True, stats_log2=True), dict(scale_mode=c.SCALE_NEGATIVE)):
        c.validate_params(dataclasses.replace(c.TemplateParams(), **over), flavor=c.D128_FLAVOR)
    c.validate_params(dataclasses.replace(c.TemplateParams(), qh_per_kh=8, sched_policy=SCHED_LPT_L2), flavor=c.D128_FLAVOR)
    with pytest.raises(ValueError, match=r"^SM90 D128 scale mode must be positive, zero, or negative$"):
        c.validate_params(dataclasses.replace(c.TemplateParams(), scale_mode=2), flavor=c.D128_FLAVOR)
    with pytest.raises(ValueError, match=r"^SM90 has no \(256, 256\) template"):
        c.validate_params(c.TemplateParams(), flavor=(256, 256))


# --- The shared SM90 prepared host (read from source: no trace, no device) -------


def _sm90_host_sources():
    """The D128 and D512 templates and the prepared host they share, parsed from source."""
    import ast
    from pathlib import Path

    from cudnn.sdpa.fwd import config_sm90 as c

    root = Path(c.__file__).parent / "kernels" / "sm90"
    return [ast.parse((root / name).read_text(encoding="utf-8")) for name in ("prefill_d128_f16.py", "prefill_d512_f16.py", "prepared_host.py")]


def test_d128_template_fits_the_shared_sm90_prepared_host():
    """D128 launches through D512's prepared host, so that host's contract is D128's too: both templates import the same
    host entries and export the same PREPARED_* values, D128's ``__call__`` takes D512's parameters (the host passes
    them positionally), D128's ``compile`` takes D512's plus at most ``persistent_ctas``, and each template sets every
    ``kernel.<attribute>`` the host reads (in its constructor or its compile path). Read from source, so a change to the
    shared host that D128 does not follow (as #1894's five THD origin arguments and ``kernel.thd_origins``) fails on
    every FROST lane, not at the first Hopper compile."""
    import ast

    d128, d512, host = _sm90_host_sources()

    def top(tree, kind, name):
        return next(n for n in tree.body if isinstance(n, kind) and n.name == name)

    def method(tree, name):
        return top(top(tree, ast.ClassDef, "SM90FusedMultiHeadAttentionForward"), ast.FunctionDef, name)

    def host_imports(tree):
        module = "cudnn.sdpa.fwd.kernels.sm90.prepared_host"
        return sorted((a.name, a.asname) for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == module for a in n.names)

    def exports(tree):
        assigns = [n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)]
        return {n.targets[0].id: ast.literal_eval(n.value) for n in assigns if n.targets[0].id.startswith("PREPARED_")}

    def call_args(tree):
        return [a.arg for a in method(tree, "__call__").args.args]

    def compile_args(tree):
        args = top(tree, ast.FunctionDef, "compile").args
        return [a.arg for a in args.args + args.kwonlyargs]

    def set_attrs(tree):
        """Attributes the constructor sets on self, and those the template's compile path sets on its kernel object."""

        def assigned(root, owner):
            assigns = [n for n in ast.walk(root) if isinstance(n, (ast.Assign, ast.AnnAssign))]
            targets = [t for n in assigns for t in (n.targets if isinstance(n, ast.Assign) else [n.target])]
            flat = [e for t in targets for e in (t.elts if isinstance(t, ast.Tuple) else [t])]
            return {e.attr for e in flat if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id == owner}

        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        return assigned(method(tree, "__init__"), "self").union(*(assigned(f, "kernel") for f in functions))

    assert ("host", "_host") in host_imports(d512) and host_imports(d128) == host_imports(d512), (host_imports(d128), host_imports(d512))
    assert exports(d512) and exports(d128) == exports(d512), (exports(d128), exports(d512))
    assert call_args(d128) == call_args(d512), (call_args(d128), call_args(d512))
    head, extra = compile_args(d128)[: len(compile_args(d512))], compile_args(d128)[len(compile_args(d512)) :]
    assert head == compile_args(d512) and set(extra) <= {"persistent_ctas"}, (compile_args(d128), compile_args(d512))
    reads = {n.attr for n in ast.walk(host) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "kernel"}
    assert reads and reads <= set_attrs(d512), sorted(reads - set_attrs(d512))
    assert reads <= set_attrs(d128), f"the prepared host reads kernel.{sorted(reads - set_attrs(d128))}, which the D128 template never sets"


# --- D128 switches: intra_wg_overlap and persistent -------

_SWITCHES = ("intra_wg_overlap", "persistent")
_SWITCH_RECORDS = [dict(zip(_SWITCHES, values)) for values in itertools.product((False, True), repeat=len(_SWITCHES))]


def _record(**over):
    from cudnn.sdpa.fwd import config_sm90 as c

    return dataclasses.replace(c.TemplateParams(), **over)


def _admitted(fields):
    """Whether validate_params(flavor=D128) admits the switch record ``fields`` (the rule itself, not a hand list)."""
    from cudnn.sdpa.fwd import config_sm90 as c

    try:
        c.validate_params(_record(**fields), flavor=c.D128_FLAVOR)
    except ValueError:
        return False
    return True


def test_d128_switch_fields_are_appended_last():
    """The overlap switch, the persistent switch, the causal walk, then the claim counter are appended after
    thd_origins, defaulting to False, False, 0 (the stride by the grid) and 0 (no counter): every earlier specialization
    keeps its defaults (the append-only rule). The segment is pinned from its own first field, so a field appended after
    it later still passes."""
    from cudnn.sdpa.fwd import config_sm90 as c

    fields = dataclasses.fields(c.TemplateParams)
    names = [f.name for f in fields]
    start = names.index("intra_wg_overlap")
    assert start > names.index("thd_origins"), names
    expected = [("intra_wg_overlap", False, bool), ("persistent", False, bool), ("causal_walk", 0, int), ("causal_claim", 0, int)]
    assert [(f.name, f.default, type(f.default)) for f in fields[start : start + len(expected)]] == expected


@pytest.mark.parametrize(
    ("switch", "value", "needs", "why"),
    [
        pytest.param("intra_wg_overlap", True, {}, "SM90 D512 has no intra-warpgroup overlap switch", id="overlap"),
        pytest.param("persistent", True, dict(intra_wg_overlap=True), "SM90 D512 has no persistent switch", id="persistent"),
        pytest.param("causal_walk", 1, dict(causal=True, intra_wg_overlap=True, persistent=True), "SM90 D512 has no causal walk switch", id="causal-walk"),
        pytest.param("causal_claim", 1, dict(causal=True, intra_wg_overlap=True, persistent=True), "SM90 D512 has no causal claim counter", id="causal-claim"),
    ],
)
def test_d512_refuses_the_d128_switches(switch, value, needs, why):
    """Every switch is a D128 template field: D512 refuses it alone (the overlap message keeps its spelling), and D128
    takes it with the switches it requires."""
    from cudnn.sdpa.fwd import config_sm90 as c

    with pytest.raises(ValueError, match=f"^{why}"):
        c.validate_params(_record(**{switch: value}))
    c.validate_params(_record(**needs, **{switch: value}), flavor=c.D128_FLAVOR)


def test_d128_persistent_refuses_a_record_without_overlap():
    """Persistence rides the overlap loop (v1 stays the non-persistent reference). The template validates on import, so
    loading such a record raises before a trace."""
    from cudnn.sdpa.fwd import config_sm90 as c
    from cudnn.sdpa.fwd.api_dsl import _load_kernel_template

    why = "^SM90 D128 persistent requires intra_wg_overlap"
    with pytest.raises(ValueError, match=why):
        c.validate_params(_record(persistent=True), flavor=c.D128_FLAVOR)
    with pytest.raises(ValueError, match=why):
        _load_kernel_template("sm90/prefill_d128_f16.py", _record(persistent=True), tag="persistent_refusal_probe_sm90_d128")


_CAUSAL_PERSISTENT = dict(causal=True, intra_wg_overlap=True, persistent=True)


@pytest.mark.parametrize(
    ("fields", "why"),
    [
        pytest.param(dict(causal=True, intra_wg_overlap=True, causal_walk=1), "^SM90 D128 causal_walk=1 requires persistent and causal", id="one-tile"),
        pytest.param(dict(intra_wg_overlap=True, persistent=True, causal_walk=1), "^SM90 D128 causal_walk=1 requires persistent and causal", id="dense"),
        pytest.param(dict(_CAUSAL_PERSISTENT, causal_walk=2), "^SM90 D128 causal_walk must be 0", id="two"),
        pytest.param(dict(_CAUSAL_PERSISTENT, causal_walk=-1), "^SM90 D128 causal_walk must be 0", id="negative"),
        pytest.param(dict(_CAUSAL_PERSISTENT, causal_walk=True), "^SM90 D128 causal_walk must be 0", id="bool"),
    ],
)
def test_d128_causal_walk_is_the_int_0_or_1_on_a_persistent_causal_record(fields, why):
    """causal_walk names the walk of a persistent causal record: D128 refuses the paired walk (1) on a one-tile record
    and on a dense one, and refuses any value but the ints 0 and 1 (a bool True included, though it equals 1); a
    persistent causal record takes either walk."""
    from cudnn.sdpa.fwd import config_sm90 as c

    with pytest.raises(ValueError, match=why):
        c.validate_params(_record(**fields), flavor=c.D128_FLAVOR)
    for walk in (0, 1):
        c.validate_params(_record(**_CAUSAL_PERSISTENT, causal_walk=walk), flavor=c.D128_FLAVOR)


@pytest.mark.parametrize(
    ("fields", "why"),
    [
        pytest.param(dict(causal=True, intra_wg_overlap=True, causal_claim=1), "^SM90 D128 causal_claim requires persistent and causal", id="one-tile"),
        pytest.param(dict(intra_wg_overlap=True, persistent=True, causal_claim=1), "^SM90 D128 causal_claim requires persistent and causal", id="dense"),
        pytest.param(dict(_CAUSAL_PERSISTENT, causal_walk=1, causal_claim=1), "^SM90 D128 causal_claim requires .* causal_walk 0", id="with-walk"),
        pytest.param(dict(_CAUSAL_PERSISTENT, sched_policy=2, causal_claim=1), "^SM90 D128 causal_claim requires .* sched_policy NATURAL", id="lpt-l2"),
        pytest.param(dict(_CAUSAL_PERSISTENT, causal_claim=2), "^SM90 D128 causal_claim must be 0", id="two"),
        pytest.param(dict(_CAUSAL_PERSISTENT, causal_claim=3), "^SM90 D128 causal_claim must be 0", id="three"),
        pytest.param(dict(_CAUSAL_PERSISTENT, causal_claim=-1), "^SM90 D128 causal_claim must be 0", id="negative"),
        pytest.param(dict(_CAUSAL_PERSISTENT, causal_claim=True), "^SM90 D128 causal_claim must be 0", id="bool"),
    ],
)
def test_d128_causal_claim_is_the_int_0_or_1_on_a_persistent_causal_natural_record(fields, why):
    """causal_claim hands out a persistent causal record's items from the workspace counter: D128 refuses it on a
    one-tile record, a dense one, beside the paired walk and on a pinned LPT/LPT_L2 order (the counter numbers its items
    itself), and refuses any value but the ints 0 and 1 (2, a heaviest-first numbering no longer built, and a bool True
    included); D512 refuses it outright. A persistent causal NATURAL record without the walk takes each value."""
    from cudnn.sdpa.fwd import config_sm90 as c

    with pytest.raises(ValueError, match=why):
        c.validate_params(_record(**fields), flavor=c.D128_FLAVOR)
    for claim in (c.D128_CLAIM_OFF, c.D128_CLAIM_NATURAL):
        c.validate_params(_record(**_CAUSAL_PERSISTENT, causal_claim=claim), flavor=c.D128_FLAVOR)
    with pytest.raises(ValueError, match="^SM90 D512 has no causal claim counter"):
        c.validate_params(_record(causal_claim=1))
    assert [f.name for f in dataclasses.fields(c.TemplateParams)][-2:] == ["causal_walk", "causal_claim"]
    assert c.TemplateParams().causal_claim == c.D128_CLAIM_OFF == 0 and c.D128_CLAIM_WORKSPACE_BYTES == 128


def test_d128_claim_coords_number_every_item_once_group_major():
    """config_sm90.d128_claim_coords (work_of's decode of a claim-counter record, in host form) visits every (Q tile,
    head, batch) exactly once for every item count, a group's T items consecutive (batch, then head, then Q tile) with
    its Q tiles ascending (NATURAL)."""
    from cudnn.sdpa.fwd import config_sm90 as c

    for q_tiles, heads, b in itertools.product((1, 2, 3, 6, 8, 9, 17), (1, 3, 8), (1, 2, 5)):
        coords = [c.d128_claim_coords(item, q_tiles, heads) for item in range(q_tiles * heads * b)]
        assert sorted(coords) == sorted(itertools.product(range(q_tiles), range(heads), range(b))), (q_tiles, heads, b)
        for group in range(heads * b):
            run = coords[group * q_tiles : (group + 1) * q_tiles]
            assert {(h, bb) for _, h, bb in run} == {(group % heads, group // heads)}
            assert [t for t, _, _ in run] == list(range(q_tiles))


def test_d128_claim_hand_out_runs_every_item_once_and_leaves_the_item_count():
    """The claim counter's hand-out on plain ints, as the template runs it: CTA c's first item is c, every CTA claims its
    next item (the CTA count plus the counter's old value) at the start of each item and stops after the item whose
    claim reached the item count, in any interleaving of the CTAs. Every item runs exactly once, every CTA claims exactly
    one end marker, and the counter reads the item count afterwards (the GPU tests read that word back)."""
    import random

    for items, ctas in ((1, 1), (5, 5), (6, 5), (40, 2), (264, 132), (1000, 132), (38976, 132)):
        rng = random.Random(items * 1009 + ctas)
        n = min(items, ctas)
        counter, ran, ends = 0, [], 0
        current = {cta: cta for cta in range(n)}
        while current:
            cta = rng.choice(sorted(current))
            ran.append(current[cta])
            nxt = n + counter
            counter += 1
            if nxt >= items:
                ends += 1
                del current[cta]
            else:
                current[cta] = nxt
        assert sorted(ran) == list(range(items)) and ends == n and counter == items, (items, ctas)


def test_d128_claim_counter_contract_is_one_in_python_and_the_binder():
    """The claim counter's workspace contract has one value on every side: config_sm90.D128_CLAIM_WORKSPACE_BYTES (the
    adapter's scratch_workspace_bytes), the native dense binder's kClaimWorkspaceBytes (its ordered path's check and
    ordered_workspace_bytes), and the binder's roles the Python side probes (supports_claim_counter) or the plan reads
    (ordered_workspace_bytes); the prepared spec carries the flag (DenseLaunchSpec.claim_counter) and the binder's reset
    is the driver memset resolved natively, never cuda-python's. The switch and its gate are typed module constants
    (api_dsl's comment gives the measurement behind their values)."""
    import pathlib
    import re

    from cudnn.sdpa.fwd import config_sm90 as c
    from cudnn.sdpa.fwd import prepared

    root = pathlib.Path(__file__).resolve().parents[4]  # the source tree this test file sits in
    binder = (root / "python" / "pygraph" / "sdpa_dense_binding.cpp").read_text()
    assert re.search(r"constexpr int64_t kClaimWorkspaceBytes = (\d+);", binder).group(1) == str(c.D128_CLAIM_WORKSPACE_BYTES) == "128"
    for name in ('"supports_claim_counter"', '"ordered_workspace_bytes"', 'get_driver_entry_point("cuMemsetD32Async")', 'flag("claim_counter")'):
        assert name in binder, name
    reset = binder[binder.index("reset_claim_counter(int64_t workspace") :]
    assert "fill_word_" not in reset[: reset.index("first_scalar_role")]
    assert "claim_counter" in prepared.DenseLaunchSpec.__slots__
    api_src = (root / "python" / "cudnn" / "sdpa" / "fwd" / "api_dsl.py").read_text()
    assert re.search(r"^D128_CAUSAL_CLAIM: int = [01]$", api_src, re.M) and re.search(r"^D128_CLAIM_MIN_ROUNDS: int = \d+$", api_src, re.M)


def test_each_causal_claim_is_its_own_module_and_digest():
    """A persistent causal record's two claim values are two modules with two digests (two compile keys), each carrying
    its value as CAUSAL_CLAIM; the record's default (0) is its static stride module."""
    from cudnn.sdpa.fwd.api_dsl import _load_kernel_template

    mods = [
        _load_kernel_template("sm90/prefill_d128_f16.py", _record(**_CAUSAL_PERSISTENT, causal_claim=claim), tag="claim_probe_sm90_d128") for claim in (0, 1)
    ]
    assert [mod.CAUSAL_CLAIM for mod in mods] == [0, 1]
    assert len({id(m) for m in mods}) == len({m.FROST_SOURCE_DIGEST for m in mods}) == 2
    assert _load_kernel_template("sm90/prefill_d128_f16.py", _record(**_CAUSAL_PERSISTENT), tag="claim_probe_sm90_d128") is mods[0]


def test_each_causal_walk_is_its_own_module_and_digest():
    """A persistent causal record's two walks are two modules with two digests (two compile keys) in one process, each
    carrying its value as CAUSAL_WALK; the stride is the module the persistent causal record loads by default."""
    from cudnn.sdpa.fwd.api_dsl import _load_kernel_template

    mods = [_load_kernel_template("sm90/prefill_d128_f16.py", _record(**_CAUSAL_PERSISTENT, causal_walk=walk), tag="walk_probe_sm90_d128") for walk in (0, 1)]
    assert [mod.CAUSAL_WALK for mod in mods] == [0, 1]
    assert len({id(m) for m in mods}) == len({m.FROST_SOURCE_DIGEST for m in mods}) == 2
    assert _load_kernel_template("sm90/prefill_d128_f16.py", _record(**_CAUSAL_PERSISTENT), tag="walk_probe_sm90_d128") is mods[0]


def test_each_switch_combination_is_its_own_module_and_digest():
    """The loader keys modules on (path, record) and folds repr(record) into the digest, so the three switch records
    validate_params admits (v1, ov and ov + persistent) are three modules with three digests (three compile keys) in one
    process, with no hand-set digest (python/cudnn/sdpa/AGENTS.md); the default record and a spelled-out all-False one
    are one module."""
    from cudnn.sdpa.fwd import config_sm90 as c
    from cudnn.sdpa.fwd.api_dsl import _load_kernel_template

    admitted = [fields for fields in _SWITCH_RECORDS if _admitted(fields)]
    assert len(admitted) == 3, admitted
    mods = []
    for fields in admitted:
        mod = _load_kernel_template("sm90/prefill_d128_f16.py", _record(**fields), tag="switch_probe_sm90_d128")
        assert {name: getattr(mod, name.upper()) for name in _SWITCHES} == fields
        mods.append(mod)
    assert len({id(m) for m in mods}) == len({m.FROST_SOURCE_DIGEST for m in mods}) == 3
    assert _load_kernel_template("sm90/prefill_d128_f16.py", c.TemplateParams(), tag="switch_probe_sm90_d128") is mods[0]


def test_d128_compile_takes_a_cta_count_exactly_when_persistent():
    """compile()'s persistent_ctas is positive exactly when the record is persistent, checked before any trace: a
    one-tile-per-CTA key never names a device, and a persistent artifact always has its grid cap."""
    from cudnn.sdpa.fwd.api_dsl import _load_kernel_template

    q, kv = (1048576, 128, 1024, 1), (262144, 128, 256, 1)
    args = (2, 8, 2, 1024, 1024, q, kv, kv, q)

    def load(**over):
        return _load_kernel_template("sm90/prefill_d128_f16.py", _record(intra_wg_overlap=True, **over), tag="ctas_probe_sm90_d128")

    why = "persistent_ctas must be positive exactly when the record is persistent"
    for bad in (dict(persistent_ctas=132), dict(persistent_ctas=True), dict(persistent_ctas=-1)):
        with pytest.raises(ValueError, match=why):
            load().compile(*args, **bad)
    for bad in ({}, dict(persistent_ctas=0), dict(persistent_ctas=True)):
        with pytest.raises(ValueError, match=why):
            load(persistent=True).compile(*args, **bad)


def test_sm90_persistent_cta_count_is_the_sm_count_and_ignores_the_thd_overrides(monkeypatch):
    """SdpaFwdDslSm90._persistent_ctas is the device's SM count (_device_sm_count, cached per device), and the THD grid
    overrides FROST_THD_CTAS / FROST_THD_CTAS_PER_SM do not reach it (a forced THD count must neither resize nor re-key
    a D128 grid). Host-only: the cache answers."""
    import torch
    from cudnn.sdpa.fwd import api_dsl

    # Not 132 (an H100's count), so a hard-coded count cannot pass: only a read of the cached SM count returns 77.
    monkeypatch.setattr(api_dsl, "_THD_SMS_CACHE", {0: 77})
    monkeypatch.setattr(api_dsl, "_THD_CTAS_CACHE", {})
    monkeypatch.setenv("FROST_THD_CTAS", "3")
    monkeypatch.setenv("FROST_THD_CTAS_PER_SM", "2")
    assert api_dsl.SdpaFwdDslSm90._persistent_ctas(None, torch.device("cuda", 0)) == 77


def test_d128_ships_persistence_over_the_overlap_loop():
    """The shipped D128 switches, api_dsl's constants as committed: intra-warpgroup overlap and, over it, persistence,
    which check_support writes into every non-causal plan and, as the paired walk (D128_PERSISTENT_CAUSAL_WALK 2), into
    every causal plan the route keeps (test_d128_route_ships_on_without_cutoffs pins the route;
    test_d128_ships_the_persistent_loop_and_the_causal_route in test_sdpa_fwd_d128_sm90.py runs both masks on an SM90
    GPU). The measurements behind the defaults are D128_PERSISTENT's and D128_PERSISTENT_CAUSAL_WALK's comments;
    changing any value changes this test."""
    from cudnn.sdpa.fwd import api_dsl

    assert (api_dsl.D128_INTRA_WG_OVERLAP, api_dsl.D128_PERSISTENT, api_dsl.D128_PERSISTENT_CAUSAL_WALK) == (True, True, 2)


def test_d128_causal_plans_take_the_walk_the_constant_names(monkeypatch):
    """check_support's persistence decision (_sm90_d128_persistence) over every value of the three D128 switch
    constants and both masks: D128_PERSISTENT_CAUSAL_WALK 0 keeps a causal plan one tile per CTA, 1 makes it persistent
    on the stride by the grid (causal_walk 0) and 2 on the paired walk (causal_walk 1), each only under overlap and
    persistence; a non-causal plan ignores it, a D512 plan carries neither switch, every record it names validates, and
    a value outside 0, 1 and 2 is refused. Host-only: the decision reads the constants alone."""
    from cudnn.sdpa.fwd import api_dsl, config_sm90 as c

    for overlap, persistent, walk, causal in itertools.product((False, True), (False, True), (0, 1, 2), (False, True)):
        monkeypatch.setattr(api_dsl, "D128_INTRA_WG_OVERLAP", overlap)
        monkeypatch.setattr(api_dsl, "D128_PERSISTENT", persistent)
        monkeypatch.setattr(api_dsl, "D128_PERSISTENT_CAUSAL_WALK", walk)
        on = overlap and persistent and (not causal or walk > 0)
        got = api_dsl._sm90_d128_persistence(c.D128_FLAVOR, causal)
        assert got == (on, int(on and causal and walk == 2)), (overlap, persistent, walk, causal, got)
        c.validate_params(_record(causal=causal, intra_wg_overlap=overlap, persistent=got[0], causal_walk=got[1]), flavor=c.D128_FLAVOR)
        assert api_dsl._sm90_d128_persistence(c.D512_FLAVOR, causal) == (False, 0)
    for bad in (3, -1, True, 2.0):
        monkeypatch.setattr(api_dsl, "D128_PERSISTENT_CAUSAL_WALK", bad)
        with pytest.raises(ValueError, match="D128_PERSISTENT_CAUSAL_WALK must be 0, 1 or 2"):
            api_dsl._sm90_d128_persistence(c.D128_FLAVOR, True)


# --- Persistent walk: the Q protocol and source pins (host-only, no trace) -------


def _passes(completed, parity):
    """An mbarrier wait at ``parity`` passes once the phase of that parity has completed: with ``completed`` phases done,
    the phase in progress has parity completed & 1, so a wait passes iff its parity differs (a fresh barrier passes a
    wait at parity 1 at once, the Bars docstring of prefill_d128_f16.py)."""
    return (completed & 1) != parity


def _q_phase_twin(n_items, rule):
    """The persistent Q protocol of one CTA in plain Python, no GPU.

    Q has one slot, which item n (from 0) fills for the n-th time. The producer's EMPTY wait for item n must block until
    item n - 1 has released Q and pass once it has (the first item's passes at once); the consumers' FULL wait for item n
    must block until Q(n) lands and pass once it has. ``rule`` picks the consumers' FULL parity: "state" carries a phase
    flipped once per item (the kernel), "fixed" waits parity 0 every time (the one-tile kernel's literal phase, right for
    the one Q per CTA it loads). Returns ``(bad_full, bad_empty, drain_ok)``: items whose FULL wait passes before their Q
    lands or never passes, items whose EMPTY wait passes before the previous release or never passes, and whether the
    producer's closing EMPTY wait blocks until the last release and passes after it."""
    bad_full, bad_empty = [], []
    p_phase, c_phase = 1, 0  # producer EMPTY from phase 1, consumers' FULL from phase 0 (the Bars docstring)
    for n in range(n_items):
        # EMPTY: n - 1 releases are in before item n - 1's own, n after it. FULL: n Q loads have landed before Q(n).
        if (n > 0 and _passes(n - 1, p_phase)) or not _passes(n, p_phase):
            bad_empty.append(n)
        parity = c_phase if rule == "state" else 0
        if _passes(n, parity) or not _passes(n + 1, parity):
            bad_full.append(n)
        p_phase ^= 1
        c_phase ^= 1
    drain_ok = not _passes(n_items - 1, p_phase) and _passes(n_items, p_phase)
    return bad_full, bad_empty, drain_ok


def test_q_phase_twin_flips_the_carried_phase_once_per_item():
    """With one Q slot, a phase carried across items and flipped once per item never passes a Q FULL wait before its Q
    lands and always passes it after, never passes a Q EMPTY wait before the previous item's release and always passes it
    after, and the closing wait passes exactly once the last release is in, for 1 to 9 items per CTA. The one-tile
    kernel's literal phase 0 passes every odd item's FULL wait at once, before its Q lands."""
    for n_items in range(1, 10):
        assert _q_phase_twin(n_items, "state") == ([], [], True), n_items
    assert _q_phase_twin(6, "fixed")[0] == [1, 3, 5]


def _kernel_ast():
    import ast

    from cudnn.sdpa.fwd import config_sm90 as c
    from cudnn.sdpa.fwd.api_dsl import _load_kernel_template

    mod = _load_kernel_template("sm90/prefill_d128_f16.py", c.TemplateParams(), tag="source_pin_sm90_d128")
    with open(mod.__file__, encoding="utf-8") as f:
        return ast.parse(f.read())


def _parents(tree):
    import ast

    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _function(tree, name):
    import ast

    return next(f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name == name)


def test_q_release_follows_the_one_release_rule():
    """Q EMPTY is arrived at exactly the sites the one-release rule names: the one-tile-per-CTA epilogue, and under
    release_q the first tile (one KV tile) and the steady step at its last kv_step, each right after the wait that retires
    the item's last QK. A missing site hangs the producer, an extra one releases Q twice."""
    import ast

    sites = {}
    for fn in ast.walk(_kernel_ast()):
        if isinstance(fn, ast.FunctionDef):
            n = sum(
                1
                for node in ast.walk(fn)
                if isinstance(node, ast.Call) and ast.unparse(node.func).startswith("bars.mb_tma_q_empty") and ast.unparse(node.func).endswith(".arrive")
            )
            if n:
                sites[fn.name] = n
    assert sites == {"epilogue": 1, "overlap_first_tile": 1, "overlap_kv_step": 1}, sites


@pytest.mark.parametrize(
    ("role", "barrier", "phase", "start"),
    [
        pytest.param("walk_items", "q_full", "q_full_phase", 0, id="full"),
        pytest.param("tmaldg_persistent", "q_empty", "q_empty_phase", 1, id="empty"),
    ],
)
def test_persistent_q_waits_read_the_carried_phase(role, barrier, phase, start):
    """Each persistent role waits on Q with the phase it carries across items (the twin above): it starts at the Bars
    docstring's phase and flips once per item through next_q_phase, inside the item loop. A literal phase, the one-tile
    kernel's, passes the second item's FULL wait before its Q lands."""
    import ast

    fn = _function(_kernel_ast(), role)
    waits = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func) == f"bars.mb_tma_{barrier}.wait"]
    assert waits and [ast.unparse(w.args[0]) for w in waits] == [phase] * len(waits), [ast.unparse(w) for w in waits]
    assigned = [ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == phase]
    assert assigned == [f"{phase} = cutlass.Int32({start})", f"{phase} = self.next_q_phase({phase})"], assigned
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.While))
    flips = [n for n in ast.walk(loop) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == phase]
    assert len(flips) == 1, [ast.unparse(n) for n in flips]


def test_deferred_store_wait_sits_outside_the_row_guard():
    """The persistent epilogue's O store wait runs on every lane of warp 0, outside the row guard and outside elect_sync:
    stmatrix overwrites sO for every item, including one whose rows are all past q_output_limit, while the previous
    item's store may still read it. compute_persistent repeats the wait before the CTA exits. The epilogue's order is
    pinned too: the wait, then the persistent-only named barrier that holds warps 1 to 3 until it returns, then
    stmatrix; without that barrier, or with the wait below stmatrix, warps overwrite the half the previous store still
    reads."""
    import ast

    tree = _kernel_ast()
    parents = _parents(tree)

    def guards(node):
        out = []
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.If):
                out.append(ast.unparse(node.test))
        return out

    for name, deferred_count in (("epilogue", 1), ("compute_persistent", 1)):
        fn = _function(tree, name)
        waits = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func) == "tma_store_wait"]
        deferred = [w for w in waits if not any("elect_sync" in g for g in guards(w))]
        assert len(deferred) == deferred_count, (name, [guards(w) for w in waits])
        for w in deferred:
            assert not any("row_base" in g or "q_output_limit" in g for g in guards(w)), (name, guards(w))
            assert any("local_tid < cute.arch.WARP_SIZE" in g for g in guards(w)), (name, guards(w))
    calls = [n for n in ast.walk(_function(tree, "epilogue")) if isinstance(n, ast.Call)]
    waits = [c for c in calls if ast.unparse(c.func) == "tma_store_wait" and not any("elect_sync" in g for g in guards(c))]
    held = [c for c in calls if ast.unparse(c.func) == "warpgroup_barrier_sync" and "cutlass.const_expr(persistent)" in guards(c)]
    stmatrix = [c for c in calls if ast.unparse(c.func) == "store_fragment"]
    assert len(waits) == 1 and len(held) == 2 and len(stmatrix) == 1, ([w.lineno for w in waits], [b.lineno for b in held], [s.lineno for s in stmatrix])
    assert all(waits[0].lineno < b.lineno < stmatrix[0].lineno for b in held), "the deferred wait, then the barrier that holds warps 1 to 3, then stmatrix"


def test_persistent_record_requires_every_item_to_have_kv_work():
    """The persistent constructor refuses a shape with an item that has no KV tile (S_kv < 1; bottom-right causal with
    S_q > S_kv), so neither role needs a liveness branch; a served shape constructs, with one item per (Q tile, head,
    batch). Host-only: no trace."""
    import math

    from cudnn.sdpa.fwd.api_dsl import _load_kernel_template

    mod = _load_kernel_template("sm90/prefill_d128_f16.py", _record(intra_wg_overlap=True, persistent=True), tag="ctor_probe_sm90_d128")

    def make(shape, **kw):
        return mod.SM90FusedMultiHeadAttentionForward(shape, intra_wg_overlap=True, persistent=True, persistent_ctas=132, **kw)

    for shape, kw in (((1, 2, 2, 300, 200), dict(is_causal=True, bottom_right=True)), ((1, 2, 2, 128, 0), {})):
        with pytest.raises(ValueError, match="needs KV work in every item"):
            make(shape, **kw)
    for shape, kw in (((1, 2, 2, 200, 300), dict(is_causal=True, bottom_right=True)), ((1, 2, 2, 300, 200), dict(is_causal=True)), ((1, 2, 2, 300, 1), {})):
        assert make(shape, **kw).items == math.ceil(shape[3] / 128) * shape[1] * shape[0]


# --- The paired walk's plan-time rule (host-only, no trace) -------
#
# causal_walk_plan picks, from the declared shape and the CTA count alone, the light-row count light_rows of the paired
# walk (or keeps the stride by the grid). The references below build every unit and every CTA's load item by item from
# the CausalWalk docstring, independently of the rule's run-length arithmetic.

_PAIRED = dict(_CAUSAL_PERSISTENT, causal_walk=1)


def _walk_module():
    from cudnn.sdpa.fwd.api_dsl import _load_kernel_template

    return _load_kernel_template("sm90/prefill_d128_f16.py", _record(**_PAIRED), tag="walk_rule_sm90_d128")


def _walk_units(q_tiles, groups, light):
    """The paired walk's units in walk order, built item by item from the CausalWalk docstring: per unit, its
    ``(Q tile, group)`` items in order."""
    pairs, odd = divmod(q_tiles - light, 2)
    middle = light + pairs
    units = []
    for g in range(groups):
        units += [((q_tiles - 1 - j, g), (light + j, g)) for j in range(pairs)]
        if odd and g % 2:
            units.append(((middle, g - 1), (middle, g)))
        elif odd and g == groups - 1:
            units.append(((middle, g),))
    for tile in range(light - 1, -1, -1):
        units += [((tile, g),) for g in range(groups)]
    return units


def _deal(units, ctas, snake):
    """Every CTA's units: round k gives units k * n to k * n + n - 1 to CTAs 0 to n - 1, n = min(ctas, units), and odd
    rounds to CTAs n - 1 to 0 under snake."""
    n = min(ctas, len(units))
    per_cta = [[] for _ in range(n)]
    for u, unit in enumerate(units):
        k, i = divmod(u, n)
        per_cta[n - 1 - i if snake and k % 2 else i].append(unit)
    return per_cta


def _lpt_l2_coords(linear, q_h, batch, q_tiles, heads_per_kv, seqlen_kv, row_bytes, l2_bytes):
    """scheduler.lpt_l2_tile_coords line by line on Python ints (its selects as max and min)."""
    n_kh = q_h // heads_per_kv
    num_groups = n_kh * batch
    per_group = max(1, seqlen_kv * row_bytes)
    active = min(max(1, l2_bytes // per_group), num_groups)
    budget_blocks = (num_groups + active - 1) // active
    active = (num_groups + budget_blocks - 1) // budget_blocks
    tiles_per_blk = active * q_tiles * heads_per_kv
    num_blocks = (num_groups + active - 1) // active
    block, within = divmod(linear, tiles_per_blk)
    eff = num_groups - block * active if block + 1 == num_blocks else active
    row_rank, in_rank = divmod(within, eff * heads_per_kv)
    kv_group = in_rank % eff + block * active
    return q_tiles - 1 - row_rank, (kv_group % n_kh) * heads_per_kv + in_rank // eff, kv_group // n_kh


def _stride_tiles(shape, sched_policy, qh_per_kh, q_tiles):
    """The stride walk's Q tile per item in the one-tile grid's order (work_of), decoded item by item."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2, SCHED_NATURAL

    b, h_q, _, _, s_kv = shape
    row_bytes = 2 * 128 * 2
    groups = 50 * 1024 * 1024 // (s_kv * row_bytes)
    l2_bytes = (1 << groups.bit_length() - 1) * s_kv * row_bytes if groups > 1 else 0
    items = q_tiles * h_q * b
    if sched_policy == SCHED_NATURAL:
        return [i % q_tiles for i in range(items)]
    if sched_policy == SCHED_LPT_L2:
        return [_lpt_l2_coords(i, h_q, b, q_tiles, qh_per_kh, s_kv, row_bytes, l2_bytes)[0] for i in range(items)]
    return [q_tiles - 1 - i // (h_q * b) for i in range(items)]


def test_causal_kv_tiles_count_every_row_of_the_tile():
    """causal_kv_tiles, the rule's count of each Q tile's KV tiles, against the count over every row the tile launches
    (rows past S_q included, which the kernel walks too): the tile walks up to the KV tile of its rows' last visible
    key, min(S_kv - 1, row + diag), diag = S_kv - S_q under bottom-right and 0 under top-left."""
    mod = _walk_module()
    for s_q, s_kv, bottom_right in [
        (1024, 1024, True),
        (1000, 1100, True),
        (1536, 1024, False),
        (1024, 1600, True),
        (128, 4096, True),
        (1000, 1000, False),
        (1152, 1152, False),
        (300, 200, False),
        (1, 300, True),
        (129, 129, True),
    ]:
        diag = s_kv - s_q if bottom_right else 0
        want = [max(min(s_kv - 1, row + diag) for row in range(128 * t, 128 * t + 128)) // 128 + 1 for t in range(-(-s_q // 128))]
        assert mod.causal_kv_tiles((1, 1, 1, s_q, s_kv), bottom_right) == want, (s_q, s_kv, bottom_right)


# Twenty-six causal shapes (b, h_q, h_kv, S_q = S_kv) at 132 CTAs, an H100 SXM's SM count: b1 to b66 at S = 512 to
# 16384, MHA, GQA and MQA, items below and above the CTA count. test_causal_walk_rule_is_an_exhaustive_search checks the
# rule's choice against a brute force; this table checks the layout the rule builds, whatever light-row count it picks.
_RULE_SHAPES = [
    (16, 16, 4, 512),
    (8, 16, 4, 1024),
    (2, 32, 8, 2048),
    (1, 32, 8, 4096),
    (1, 16, 4, 8192),
    (33, 8, 8, 512),
    (33, 4, 4, 1024),
    (33, 2, 2, 2048),
    (33, 1, 1, 4096),
    (66, 4, 1, 512),
    (33, 4, 1, 1024),
    (33, 1, 1, 8192),
    (1, 16, 4, 1024),
    (11, 3, 3, 1024),
    (17, 2, 2, 1024),
    (23, 6, 6, 1024),
    (32, 16, 4, 1024),
    (33, 16, 4, 1024),
    (2, 32, 8, 4096),
    (2, 32, 8, 8192),
    (2, 32, 8, 16384),
    (2, 32, 32, 4096),
    (8, 32, 8, 4096),
    (4, 8, 8, 8192),
    (2, 16, 4, 16384),
    (1, 64, 8, 16384),
]


def test_causal_walk_rule_layout_is_consistent_on_26_causal_shapes():
    """The walk's own search (the T cut off, the stride dealt straight, LPT_L2) and the shipped constants' plan on the
    record's NATURAL order, on _RULE_SHAPES at 132 CTAs: each plan the rule returns has a light-row count between 0 and
    T, its layout's constants are consistent (T and the group count are the shape's, the CTA count is the units' or
    the grid's, and the units hold every item once), and a plan of at most one item per CTA takes light_rows = T. Which
    light-row count or plan the rule picks is not pinned (test/AGENTS.md); the exhaustive search checks the choice."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2, SCHED_NATURAL

    mod = _walk_module()
    for b, h_q, h_kv, s in _RULE_SHAPES:
        q_tiles, groups = s // 128, h_q * b
        searched = mod.causal_walk_plan((b, h_q, h_kv, s, s), True, SCHED_LPT_L2, h_q // h_kv, 132, 2, min_tiles=-1, deal="straight")
        shipped = mod.causal_walk_plan((b, h_q, h_kv, s, s), True, SCHED_NATURAL, h_q // h_kv, 132, 2)
        for walk in (searched, shipped):
            if walk is None:
                continue
            light = walk.light_rows
            assert 0 <= light <= q_tiles and (q_tiles * groups > 132 or light == q_tiles), ((b, h_q, h_kv, s), walk)
            assert (walk.q_tiles, walk.groups, walk.ctas) == (q_tiles, groups, min(walk.units, 132)), walk
            assert sum(map(len, _walk_units(q_tiles, groups, light))) == q_tiles * groups == 2 * walk.heavy_units - walk.lone_middle + groups * light


@pytest.mark.parametrize(
    ("shape", "bottom_right", "sched"),
    [
        pytest.param((4, 16, 4, 1152, 1152), False, "SCHED_LPT_L2", id="odd-q-tiles"),
        pytest.param((2, 16, 4, 1536, 1024), False, "SCHED_LPT_L2", id="sq-above-skv"),
        pytest.param((4, 16, 4, 1000, 1100), True, "SCHED_LPT", id="partial-ragged-lpt"),
        pytest.param((3, 5, 5, 1152, 1152), True, "SCHED_NATURAL", id="odd-groups-natural"),
        pytest.param((1, 24, 8, 1024, 1024), True, "SCHED_LPT_L2", id="192-items"),
        pytest.param((23, 6, 6, 1024, 1024), True, "SCHED_LPT_L2", id="l2-cap-binds"),
        pytest.param((8, 24, 8, 256, 2048), True, "SCHED_LPT_L2", id="stride-wins"),
        pytest.param((2, 8, 1, 100, 300), True, "SCHED_LPT_L2", id="one-q-tile"),
        pytest.param((4, 16, 4, 1000, 1000), False, "SCHED_LPT_L2", id="partial-q"),
        pytest.param((4, 16, 4, 1024, 1600), True, "SCHED_LPT_L2", id="diag-576"),
        pytest.param((1, 32, 8, 2048, 8192), True, "SCHED_LPT_L2", id="chunked"),
        pytest.param((3, 5, 5, 4224, 4224), True, "SCHED_LPT_L2", id="odd-t33-odd-groups"),
        pytest.param((2, 48, 8, 640, 640), True, "SCHED_LPT_L2", id="odd-t5"),
        pytest.param((5, 8, 8, 512, 2560), True, "SCHED_LPT_L2", id="deal-keeps-the-stride-at-132"),
        pytest.param((11, 4, 4, 1536, 1664), True, "SCHED_LPT_L2", id="deal-keeps-the-stride-at-64"),
        # Chunked prefill (bottom-right, S_q well below S_kv): every Q tile walks nearly the same KV tiles, so pairs are
        # a coarser grain than single items and the comparison with the stride decides.
        pytest.param((4, 24, 8, 512, 4096), True, "SCHED_NATURAL", id="chunked-b4-h24-sq512-skv4096"),
        pytest.param((8, 24, 8, 256, 2048), True, "SCHED_NATURAL", id="chunked-b8-h24-sq256-skv2048"),
        pytest.param((8, 32, 8, 2048, 16384), True, "SCHED_NATURAL", id="chunked-b8-h32-sq2048-skv16384"),
        pytest.param((4, 32, 8, 2048, 16384), True, "SCHED_NATURAL", id="chunked-b4-h32-sq2048-skv16384"),
    ],
)
def test_causal_walk_rule_is_an_exhaustive_search(shape, bottom_right, sched):
    """The rule against a brute force that visits every item, at 1, 3, 7, 64 and 132 CTAs: for each light_rows from 0 to
    the L2 cap, every CTA's load summed over the units _walk_units builds and _deal deals; for the stride, every CTA's
    load over the one-tile grid's order decoded item by item (NATURAL, LPT, lpt_l2_tile_coords on plain ints); each item
    its KV tiles plus CAUSAL_WALK_PHI. The plan is the light_rows with the smallest largest load (the smaller on a tie),
    and None exactly where the stride's is smaller under the deal asked for (straight, or the lighter of the straight
    and the snake deal under "auto"); a plan with at most one item per CTA takes light_rows = T. The walk's own search
    is checked at every T (min_tiles=-1; test_causal_walk_min_tiles_sends_few_q_tiles_to_the_stride checks the cut). The
    cap is recomputed from its definition: a round of single items reads up to min(ceil(CTAs / qh_per_kh) + 1, h_kv * b)
    KV heads, each to its row's last KV tile of 128 rows of K and V, within the LPT_L2 order's L2 budget (one KV group's
    bytes where that is 0)."""
    from cudnn.frost.tile_dsl import constants

    mod = _walk_module()
    sched = getattr(constants, sched)
    b, h_q, h_kv, _, s_kv = shape
    kv_tiles = mod.causal_kv_tiles(shape, bottom_right)
    q_tiles, groups = len(kv_tiles), h_q * b
    cost = [tiles + mod.CAUSAL_WALK_PHI for tiles in kv_tiles]
    row_bytes = 2 * 128 * 2
    l2_groups = 50 * 1024 * 1024 // (s_kv * row_bytes)
    budget = ((1 << l2_groups.bit_length() - 1) * s_kv * row_bytes if l2_groups > 1 else 0) or s_kv * row_bytes
    for ctas, deal in itertools.product((1, 3, 7, 64, 132), ("straight", "auto")):
        walk = mod.causal_walk_plan(shape, bottom_right, sched, h_q // h_kv, ctas, 2, min_tiles=-1, deal=deal)
        if q_tiles * groups <= ctas:
            assert walk is not None and walk.light_rows == q_tiles, (ctas, walk)
            continue
        heads = min(-(-ctas // (h_q // h_kv)) + 1, h_kv * b)
        cap = 0
        while cap < q_tiles and heads * kv_tiles[cap] * 128 * row_bytes <= budget:
            cap += 1
        loads = []
        for light in range(cap + 1):
            per_cta = _deal(_walk_units(q_tiles, groups, light), ctas, True)
            loads.append(max(sum(cost[t] for unit in units for t, _ in unit) for units in per_cta))
        best = loads.index(min(loads))
        n, tiles = min(ctas, q_tiles * groups), _stride_tiles(shape, sched, h_q // h_kv, q_tiles)
        deals = (False, True) if deal == "auto" else (False,)
        stride = min(max(sum(cost[tiles[i]] for i in _stride_of_cta(mod, n, len(tiles), cta, snake)) for cta in range(n)) for snake in deals)
        want = best if loads[best] <= stride else None
        assert (None if walk is None else walk.light_rows) == want, (ctas, deal, cap, loads, stride, walk)


def test_causal_walk_rule_honours_its_constants():
    """The rule's constants as compile-time arms: a forced light-row count is taken as is (0: pairs only), clamped to T,
    and never by a plan with at most one item per CTA (light_rows = T); the L2 cap off never lowers the light-row count
    the cap allows; the snake arm is carried on the walk it returns. Each arm is read with the others at their ablation
    values (the T cut off, the stride dealt straight). The shipped values are tuning choices (config_sm90's comments
    give the measurements) and are checked only for their domains."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2

    mod = _walk_module()

    def light(shape, **kw):
        kw = dict(dict(min_tiles=-1, deal="straight"), **kw)
        walk = mod.causal_walk_plan(shape, True, SCHED_LPT_L2, shape[1] // shape[2], 132, 2, **kw)
        return None if walk is None else walk.light_rows

    assert [light((1, 16, 4, 8192, 8192), light_rows=n) for n in (0, 5, 64, 1000)] == [0, 5, 64, 64]
    assert [light((1, 16, 4, 1024, 1024), light_rows=n) for n in (0, 3)] == [8, 8]
    for shape in ((23, 6, 6, 1024, 1024), (2, 32, 32, 4096, 4096)):
        capped, free = (light(shape, l2_cap=cap) for cap in (True, False))
        assert capped is not None and free is not None and capped <= free, (shape, capped, free)
    for on in (True, False):
        walk = mod.causal_walk_plan((1, 16, 4, 8192, 8192), True, SCHED_LPT_L2, 4, 132, 2, snake=on, min_tiles=-1, deal="straight")
        assert walk is None or walk.snake is on, (on, walk)
    assert isinstance(mod.CAUSAL_WALK_LIGHT_ROWS, int) and mod.CAUSAL_WALK_LIGHT_ROWS >= -1
    assert isinstance(mod.CAUSAL_WALK_L2_CAP, bool) and isinstance(mod.CAUSAL_WALK_SNAKE, bool)
    assert isinstance(mod.CAUSAL_WALK_PHI, int) and mod.CAUSAL_WALK_PHI >= 0
    assert isinstance(mod.CAUSAL_WALK_MIN_TILES, int) and mod.CAUSAL_WALK_MIN_TILES >= -1
    assert mod.STRIDE_DEAL in ("auto", "straight", "snake")


def test_causal_walk_rule_stays_cheap_on_large_plans():
    """The rule runs once per compile, in Python: on four large plans (up to 262,144 items, T = 512 Q tiles over 512
    groups with unequal pairs on the top-left S_q > S_kv one) it stays far below a compile's seconds. Measured at most
    0.1 s each in the offline compile container (2 CPUs); the bound leaves room for a slow host."""
    import time

    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2

    mod = _walk_module()
    for shape, bottom_right in [
        ((32, 32, 8, 32000, 32020), True),
        ((8, 64, 8, 65536, 32768), False),
        ((16, 32, 8, 16000, 16010), True),
        ((2, 16, 4, 4096, 2048), False),
    ]:
        start = time.perf_counter()
        mod.causal_walk_plan(shape, bottom_right, SCHED_LPT_L2, shape[1] // shape[2], 132, 2)
        assert time.perf_counter() - start < 2.0, shape


def test_causal_walk_record_carries_the_rule():
    """The persistent constructor of a causal_walk=1 record takes the rule's walk for its shape, CTA count and order
    (``walk``); a causal_walk=0 record carries none; causal_walk=1 is refused on a one-tile record and on a non-causal
    one. Host-only: no trace."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2, SCHED_NATURAL

    mod = _walk_module()

    def make(shape, **kw):
        return mod.SM90FusedMultiHeadAttentionForward(shape, intra_wg_overlap=True, **kw)

    for shape, bottom_right, sched, ctas in [((2, 8, 2, 1024, 1024), True, SCHED_LPT_L2, 3), ((3, 5, 5, 1152, 1300), True, SCHED_NATURAL, 7)]:
        kw = dict(is_causal=True, bottom_right=bottom_right, sched_policy=sched, qh_per_kh=shape[1] // shape[2], persistent=True, persistent_ctas=ctas)
        record = make(shape, causal_walk=1, **kw)
        assert record.walk == mod.causal_walk_plan(shape, bottom_right, sched, shape[1] // shape[2], ctas, 2)
        assert record.stride_snake is (record.walk is None and mod.causal_stride_snake(shape, bottom_right, sched, shape[1] // shape[2], ctas, 2))
        assert make(shape, **kw).walk is None and make(shape, **kw).stride_snake is False
    with pytest.raises(ValueError, match="orders a persistent causal record's walk"):
        make((1, 2, 2, 256, 256), is_causal=True, causal_walk=1)
    with pytest.raises(ValueError, match="orders a persistent causal record's walk"):
        make((1, 2, 2, 256, 256), persistent=True, persistent_ctas=2, causal_walk=1)


def test_causal_stride_record_carries_its_deal(monkeypatch):
    """A causal_walk=1 record whose rule keeps the stride carries the stride's deal (causal_stride_snake) as
    ``stride_snake``, which next_item reads; a record on the walk, and a causal_walk=0 record, deal straight."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2

    mod = _walk_module()
    monkeypatch.setattr(mod, "causal_stride_snake", lambda *args, **kwargs: True)
    kw = dict(is_causal=True, bottom_right=True, sched_policy=SCHED_LPT_L2, persistent=True, persistent_ctas=132)
    stride = mod.SM90FusedMultiHeadAttentionForward((8, 24, 8, 256, 2048), intra_wg_overlap=True, causal_walk=1, qh_per_kh=3, **kw)
    walk = mod.SM90FusedMultiHeadAttentionForward((2, 32, 8, 4096, 4096), intra_wg_overlap=True, causal_walk=1, qh_per_kh=4, **kw)
    plain = mod.SM90FusedMultiHeadAttentionForward((8, 24, 8, 256, 2048), intra_wg_overlap=True, qh_per_kh=3, **kw)
    assert (stride.walk, stride.stride_snake) == (None, True)
    assert walk.walk is not None and walk.stride_snake is False
    assert plain.stride_snake is False


# --- The stride by the grid's deal and the walk's Q-tile cut (host-only, no trace) -------
#
# stride_next deals the stride's rounds (odd ones in reverse under the snake), causal_stride_snake picks the deal from
# the declared shape and the CTA count (the lighter busiest CTA), and CAUSAL_WALK_MIN_TILES sends plans of few Q tiles
# per group to the stride. The references below walk each CTA's items one by one.

_STRIDE_GRIDS = (1, 2, 3, 7, 64, 131, 132)
_DEAL_SHAPES = [  # ((b, h_q, h_kv, S_q, S_kv), bottom-right): shapes where each deal won on H100 SXM, and others
    pytest.param((4, 32, 32, 2176, 2944), True, id="b4-h32-sq2176-skv2944-br"),
    pytest.param((5, 64, 64, 768, 1536), True, id="b5-h64-sq768-skv1536-br"),
    pytest.param((10, 16, 16, 1536, 2688), True, id="b10-h16-sq1536-skv2688-br"),
    pytest.param((16, 32, 32, 2048, 2048), False, id="b16-h32-s2048-tl"),
    pytest.param((15, 64, 64, 256, 1280), True, id="b15-h64-sq256-skv1280-br"),
    pytest.param((1, 6, 6, 3000, 3000), False, id="b1-h6-s3000-tl"),
    pytest.param((3, 40, 40, 1664, 2432), True, id="b3-h40-sq1664-skv2432-br"),
    pytest.param((7, 28, 4, 1024, 1024), False, id="gqa7"),
    pytest.param((9, 12, 3, 640, 900), True, id="gqa4-partial"),
    pytest.param((2, 64, 8, 4096, 4096), False, id="gqa8-long"),
]


def _stride_of_cta(mod, n, items, cta, snake):
    """CTA ``cta``'s items on the stride by the grid of ``n`` CTAs: its block index, then stride_next while below
    ``items``."""
    item, seen = cta, []
    while item < items:
        seen.append(item)
        item = mod.stride_next(item, n, snake)
        assert len(seen) <= items, "the stride does not end"
    return seen


def test_stride_next_visits_every_item_once_and_alternates_rounds():
    """stride_next on plain ints for grids of 1, 2, 3, 7, 64, 131 and 132 CTAs, item counts from one round to three and
    a part: with and without the snake every item is visited once over the grid, a CTA's k-th item lies in round k, and
    under the snake CTA c takes column c of the even rounds and column n - 1 - c of the odd ones (column c always
    without it)."""
    mod = _walk_module()
    for n in _STRIDE_GRIDS:
        for items in sorted({n, n + 1, 2 * n - 1, 2 * n, 3 * n + n // 2, 5000}):
            for snake in (False, True):
                seen = []
                for cta in range(n):
                    visited = _stride_of_cta(mod, n, items, cta, snake)
                    assert [item // n for item in visited] == list(range(len(visited))), (n, items, snake, cta)
                    assert all(item % n == (n - 1 - cta if snake and k % 2 else cta) for k, item in enumerate(visited)), (n, items, snake, cta)
                    seen += visited
                assert sorted(seen) == list(range(items)), (n, items, snake)


@pytest.mark.parametrize("shape, bottom_right", _DEAL_SHAPES)
def test_stride_deal_is_the_lighter_busiest_cta_of_a_brute_force(shape, bottom_right):
    """causal_stride_snake against a brute force under NATURAL, LPT and LPT_L2 at 1, 3, 7, 64 and 132 CTAs: every CTA's
    load summed over its items (_stride_of_cta), each item its Q tile's KV tiles plus CAUSAL_WALK_PHI, the tile decoded
    in the one-tile grid's order (_stride_tiles). "auto" takes the snake exactly where its busiest CTA is strictly
    lighter, "straight" and "snake" force a deal, and anything else is refused by name."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL

    mod = _walk_module()
    b, h_q, h_kv, _, _ = shape
    kv_tiles = mod.causal_kv_tiles(shape, bottom_right)
    q_tiles = len(kv_tiles)
    items = q_tiles * h_q * b
    for sched, ctas in itertools.product((SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2), (1, 3, 7, 64, 132)):
        n, tiles = min(ctas, items), _stride_tiles(shape, sched, h_q // h_kv, q_tiles)
        load = {
            snake: max(sum(kv_tiles[tiles[i]] + mod.CAUSAL_WALK_PHI for i in _stride_of_cta(mod, n, items, cta, snake)) for cta in range(n))
            for snake in (False, True)
        }
        args = (shape, bottom_right, sched, h_q // h_kv, ctas, 2)
        assert mod.causal_stride_snake(*args, deal="auto") is (load[True] < load[False]), (sched, ctas, load)
        assert (mod.causal_stride_snake(*args, deal="straight"), mod.causal_stride_snake(*args, deal="snake")) == (False, True)
    with pytest.raises(ValueError, match="STRIDE_DEAL"):
        mod.causal_stride_snake(shape, bottom_right, SCHED_NATURAL, h_q // h_kv, 132, 2, deal="spiral")


def test_causal_walk_min_tiles_sends_few_q_tiles_to_the_stride():
    """CAUSAL_WALK_MIN_TILES as a compile-time arm: below ``min_tiles`` Q tiles per group the rule keeps the stride
    (None), at or above it the walk's rule decides as with -1, and -1 cuts nothing; a plan with at most one item per CTA
    keeps L = T and a forced light-row count keeps its walk whatever the cut."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2

    mod = _walk_module()
    for shape in [(8, 16, 4, 1024, 1024), (29, 32, 8, 256, 2048), (2, 48, 8, 640, 640), (2, 16, 4, 1152, 1152), (1, 32, 8, 2048, 8192)]:
        q_tiles = len(mod.causal_kv_tiles(shape, True))
        args = (shape, True, SCHED_LPT_L2, shape[1] // shape[2], 132, 2)
        free = mod.causal_walk_plan(*args, min_tiles=-1)
        assert mod.causal_walk_plan(*args, min_tiles=9) == (None if q_tiles < 9 else free), (shape, q_tiles, free)
        assert mod.causal_walk_plan(*args, min_tiles=9, light_rows=0) == mod.causal_walk_plan(*args, min_tiles=-1, light_rows=0) is not None
    one = mod.causal_walk_plan((1, 4, 4, 512, 512), True, SCHED_LPT_L2, 1, 132, 2, min_tiles=9)
    assert one is not None and one.light_rows == one.q_tiles == 4


def test_causal_walk_cut_max_qh_keeps_the_walks_rule_above_eight_q_heads_per_kv_head():
    """CAUSAL_WALK_CUT_MAX_QH as a compile-time arm (given 8 here): the Q-tile cut sends a plan of fewer than ``min_tiles``
    Q tiles per group to the stride only where ``qh_per_kh`` is at most ``cut_max_qh``; above it the walk's rule decides
    as with the cut off (min_tiles -1). On the NATURAL order a causal record of the walk's rule carries, each plan below
    walks under the rule with the cut off: at 4 and 8 Q heads per KV head the cut keeps the stride (b8 h16/4 S 1024, T
    8; b16 h32/4 S_q 256 S_kv 2048, T 2), and at 16 the rule's walk stays (b3 h64/4 S_q 384 S_kv 3072, T 3, and b121
    h16/1 S_q 128 S_kv 4096, T 1, where a timing run on H100 SXM found the stride 1.1% and 1.4% slower than that walk).
    cut_max_qh 16 cuts the last two as well, and 3 lets the first keep its walk. A shape whose rule picks the stride
    even with the cut off cannot show the arm and is passed over."""
    from cudnn.frost.tile_dsl.constants import SCHED_NATURAL

    mod = _walk_module()
    for shape, cut in [
        ((8, 16, 4, 1024, 1024), True),
        ((16, 32, 4, 256, 2048), True),
        ((3, 64, 4, 384, 3072), False),
        ((121, 16, 1, 128, 4096), False),
    ]:
        qh_per_kh = shape[1] // shape[2]
        assert len(mod.causal_kv_tiles(shape, True)) < 9 and qh_per_kh in (4, 8, 16), shape
        args = (shape, True, SCHED_NATURAL, qh_per_kh, 132, 2)
        free = mod.causal_walk_plan(*args, min_tiles=-1)
        if free is None:
            continue
        assert free.units > 132, (shape, free)
        assert mod.causal_walk_plan(*args, min_tiles=9, cut_max_qh=8) == (None if cut else free), (shape, qh_per_kh)
        assert mod.causal_walk_plan(*args, min_tiles=9, cut_max_qh=16) is None, shape
        assert mod.causal_walk_plan(*args, min_tiles=9, cut_max_qh=3) == (None if qh_per_kh <= 3 else free), shape


# --- The route: a causal plan of at most one item per CTA, or of L at least _D128_ROUTE_MAX_L, goes one tile per CTA ---
#
# _sm90_d128_causal_route sends a causal plan of the walk's rule one tile per CTA where it has at most one item per CTA,
# or where L, the mean KV tiles per Q tile, is at least _D128_ROUTE_MAX_L, whatever kind the walk's rule picks. Its
# cutoff is a tuning value (config_sm90.D128_ROUTE_MAX_L gives the measurements behind it); these tests check the
# relation at each plan's L, not the value. An earlier route also sent strides at lower L; it is retired (api_dsl's
# D128_CAUSAL_WALK_ROUTE comment).


def _route(shape, ctas=132, bottom_right=True):
    """Whether the route sends a causal plan of ``shape`` one tile per CTA at ``ctas`` CTAs (132: an H100 SXM's SM
    count) under the mask alignment ``bottom_right``."""
    from cudnn.sdpa.fwd import api_dsl

    return api_dsl._sm90_d128_causal_route(shape, bottom_right, ctas)


def _plan(shape, ctas=132, bottom_right=True):
    """``(kind, L)`` of the record check_support builds for a causal bf16 plan of ``shape`` before the route, from the
    template's planner on the record's NATURAL order: "one item per CTA" (at most ``ctas`` items), "pairs" (the walk's
    rule deals pairs alone, at most one unit per CTA), "walk" (any other paired walk), "stride" or "stride-snake" (the
    walk's rule keeps the stride, dealt straight or snake), and L, the mean KV tiles per Q tile."""
    from cudnn.frost.tile_dsl.constants import SCHED_NATURAL

    mod = _walk_module()
    b, h_q, h_kv, _, _ = shape
    kv_tiles = mod.causal_kv_tiles(shape, bottom_right)
    mean = sum(kv_tiles) / len(kv_tiles)
    if len(kv_tiles) * h_q * b <= ctas:
        return "one item per CTA", mean
    args = (shape, bottom_right, SCHED_NATURAL, h_q // h_kv, ctas, 2)
    walk = mod.causal_walk_plan(*args)
    if walk is None:
        return ("stride-snake" if mod.causal_stride_snake(*args) else "stride"), mean
    return ("pairs" if walk.units <= ctas else "walk"), mean


def test_d128_route_counts_kv_tiles_as_the_template_does():
    """The route reads L without loading the template: api_dsl._sm90_d128_causal_kv_tiles is the template's
    causal_kv_tiles, tile for tile, on S_q and S_kv at, below and above tile multiples, S_q above and below S_kv, both
    alignments and a partial last Q tile; on the largest plan the route runs no planner (b1 h128/1 S_q 262144, 2048 Q
    tiles: S_kv 262144 bottom-right at L 1024.5 goes, S_kv 4096 top-left at L 31.8 stays)."""
    import inspect

    from cudnn.sdpa.fwd import api_dsl

    mod = _walk_module()
    for s_q, s_kv, bottom_right in itertools.product(
        (1, 100, 127, 128, 129, 256, 1100, 1152, 2048, 4223), (1, 64, 128, 300, 1024, 1200, 1728, 4096, 16384), (True, False)
    ):
        shape = (2, 4, 2, s_q, s_kv)
        assert api_dsl._sm90_d128_causal_kv_tiles(shape, bottom_right) == mod.causal_kv_tiles(shape, bottom_right), (s_q, s_kv, bottom_right)
    assert list(inspect.signature(api_dsl._sm90_d128_causal_route).parameters) == ["shape", "bottom_right", "ctas"]
    assert _route((1, 128, 1, 262144, 262144)) is True
    assert _route((1, 128, 1, 262144, 4096), bottom_right=False) is False


def test_the_d128_templates_planner_is_config_sm90s():
    """One rule for the record, the route and the graph route's proposal: the D128 template's planner names (the seven
    rule constants, CausalWalk and the host functions) are config_sm90's own objects, the route's KV tile count is the
    planner's, and the SM90 templates' LPT_L2 budget (kernels/sm90/_common_hopper) is config_sm90's."""
    from cudnn.sdpa.fwd import config_sm90
    from cudnn.sdpa.fwd.kernels.sm90 import _common_hopper

    mod = _walk_module()
    names = (
        "CAUSAL_WALK_LIGHT_ROWS",
        "CAUSAL_WALK_L2_CAP",
        "CAUSAL_WALK_PHI",
        "CAUSAL_WALK_SNAKE",
        "CAUSAL_WALK_MIN_TILES",
        "CAUSAL_WALK_CUT_MAX_QH",
        "STRIDE_DEAL",
        "CausalWalk",
        "causal_walk_layout",
        "causal_kv_tiles",
        "_rounds_makespan",
        "_walk_unit_costs",
        "_lpt_l2_bytes",
        "_stride_item_costs",
        "_stride_deal",
        "causal_walk_plan",
        "causal_stride_snake",
    )
    assert [name for name in names if getattr(mod, name) is not getattr(config_sm90, name)] == []
    assert config_sm90.d128_causal_kv_tiles is config_sm90.causal_kv_tiles
    assert _common_hopper.SCHED_L2_BUDGET_BYTES == config_sm90.SCHED_L2_BUDGET_BYTES == 50 * 1024 * 1024


def test_d128_route_sends_every_plan_of_at_most_one_item_per_cta():
    """The item branch at its boundary, at 132 CTAs: b1 h4/1 S 4224 (132 items, L 17.0) goes one tile per CTA and b1
    h4/1 S 4352 (136 items, L 17.5) is decided by L alone; a partial last Q tile is an item (S_q 4223 S_kv 4300: 132
    items; S_q 4225: 136); at 7 CTAs b1 h7/7 S_q 128 (7 items) goes and b1 h8/8 S_q 128 (8 items) is decided by L. It
    counts Q tiles, heads and batch whatever L, the KV head count or the mask: b1 h16/4 S 1024 (128 items, L 4.5), b1
    h4/1 S_q 4096 S_kv 135168 (128 items, L 1040.5) and b1 h4/4 S_q 4224 S_kv 8192 (132 items, L 48.0) go."""
    from cudnn.sdpa.fwd import api_dsl

    cutoff = api_dsl._D128_ROUTE_MAX_L
    for shape, ctas, one_item, mean in [
        ((1, 4, 1, 4224, 4224), 132, True, 17.0),
        ((1, 4, 1, 4352, 4352), 132, False, 17.5),
        ((1, 4, 1, 4223, 4300), 132, True, 18.0),
        ((1, 7, 7, 128, 1024), 7, True, 8.0),
        ((1, 8, 8, 128, 1024), 7, False, 8.0),
        ((1, 16, 4, 1024, 1024), 132, True, 4.5),
        ((1, 4, 1, 4096, 135168), 132, True, 1040.5),
        ((1, 4, 4, 4224, 8192), 132, True, 48.0),
    ]:
        kind, got = _plan(shape, ctas)
        assert (kind == "one item per CTA") is one_item and got == mean, (shape, ctas, kind, got)
        assert _route(shape, ctas) is (one_item or mean >= cutoff), (shape, ctas)
    kind, mean = _plan((1, 4, 1, 4225, 4300))
    assert kind != "one item per CTA" and _route((1, 4, 1, 4225, 4300)) is (mean >= cutoff)


def test_d128_route_sends_every_kind_from_its_l_cutoff():
    """The L branch on plans of more items than CTAs, for every kind the walk's rule picks (paired walks, pairs alone,
    the stride dealt straight or snake) around L 48 and well past it: a plan goes one tile per CTA exactly when its L is
    at least _D128_ROUTE_MAX_L, and L is the plan's mean KV tiles per Q tile (checked against the planner's count). The
    route reads the mask alignment: b8 h32/8 S_q 2048 S_kv 16384 is at L 120.5 bottom-right and at L 8.5 top-left."""
    from cudnn.sdpa.fwd import api_dsl

    cutoff = api_dsl._D128_ROUTE_MAX_L
    for shape, mean in [
        ((1, 16, 4, 1280, 6656), 47.5),
        ((1, 16, 4, 1152, 6656), 48.0),
        ((1, 16, 4, 2048, 7040), 47.5),
        ((1, 16, 16, 1152, 6656), 48.0),
        ((2, 32, 8, 1280, 6656), 47.5),
        ((2, 16, 16, 1152, 6656), 48.0),
        ((1, 32, 8, 768, 6400), 47.5),
        ((2, 32, 8, 384, 6272), 48.0),
        ((3, 4, 1, 1536, 34304), 262.5),
        ((2, 8, 8, 1152, 7296), 53.0),
        ((11, 3, 3, 1024, 6656), 48.5),
        ((8, 32, 8, 2048, 16384), 120.5),
    ]:
        kind, got = _plan(shape)
        assert kind != "one item per CTA" and got == mean and _route(shape) is (mean >= cutoff), (shape, kind, got)
    kind, mean = _plan((8, 32, 8, 2048, 16384), bottom_right=False)
    assert mean == 8.5 and _route((8, 32, 8, 2048, 16384), bottom_right=False) is (mean >= cutoff)


# The plans an earlier route sent one tile per CTA past one item per CTA (the routed shapes of its boundary and
# head-layout tests and the cc 9.0 suite's routed case, bottom-right, 132 CTAs), with each plan's L: paired walks and
# strides from L 10 to 144.5 on MHA, MQA and GQA ratios 2, 3, 4 and 8.
_CUTOFF_ROUTED = {
    (8, 32, 8, 1920, 7040): 48.0,
    (8, 32, 8, 2048, 7168): 48.5,
    (8, 32, 8, 2048, 16384): 120.5,
    (41, 32, 1, 1792, 2816): 15.5,
    (8, 32, 32, 2176, 2304): 10.0,
    (13, 16, 16, 1152, 1792): 10.0,
    (8, 32, 32, 2176, 2560): 12.0,
    (9, 16, 16, 1664, 2688): 15.0,
    (12, 16, 8, 1280, 2560): 15.5,
    (8, 24, 8, 256, 2048): 15.5,
    (6, 32, 8, 256, 2048): 15.5,
    (41, 64, 8, 768, 2304): 15.5,
    (8, 24, 8, 256, 2176): 16.5,
    (3, 16, 8, 3072, 6528): 39.5,
    (3, 32, 16, 384, 6272): 48.0,
    (12, 3, 3, 1024, 6656): 48.5,
    (3, 8, 8, 2048, 10240): 72.5,
    **{(8, h_q, h_kv, 2176, 3328): 18.0 for h_q, h_kv in ((32, 32), (32, 16), (24, 8), (32, 8), (32, 4))},
    **{(6, h_q, h_kv, 4096, 20480): 144.5 for h_q, h_kv in ((32, 32), (32, 16), (24, 8), (32, 8), (32, 4))},
}


def test_d128_route_decides_the_earlier_routes_plans_by_l_alone():
    """The 27 plans of _CUTOFF_ROUTED, each with more items than CTAs, go one tile per CTA exactly when their L is at
    least _D128_ROUTE_MAX_L: the earlier route's lower stride cutoffs leave no trace. No other _D128_ROUTE_ constant is
    left."""
    from cudnn.sdpa.fwd import api_dsl

    assert len(_CUTOFF_ROUTED) == 27
    for shape, mean in _CUTOFF_ROUTED.items():
        kind, got = _plan(shape)
        assert kind != "one item per CTA" and got == mean and _route(shape) is (mean >= api_dsl._D128_ROUTE_MAX_L), (shape, kind, got)
    assert [name for name in vars(api_dsl) if name.startswith("_D128_ROUTE_")] == ["_D128_ROUTE_MAX_L"]


def test_d128_route_runs_only_under_its_switch():
    """check_support calls the route at one place, under ``if causal_walk and D128_CAUSAL_WALK_ROUTE:``, so with the
    switch False no plan is routed and every causal record of the walk's rule keeps its record; where the route returns
    True the record becomes D128_PERSISTENT_CAUSAL_WALK 0's (persistent and causal_walk cleared) with the causal order
    the one-tile record takes (causal_sched), and nothing else changes. The route is given the declared shape, the mask
    alignment as the template reads it (TemplateParams' bottom_right, None meaning the alignment causal implies) and the
    device's SM count, the CTA count compile() gives the walk's record. The record's head ratio is the expression the
    planner reads (Q head h reads KV head h // qh_per_kh); a PackGQA declaration never reaches the D128 template (the
    per-flavor tests above). The cc 9.0 suite's test_d128_adapter_routes_causal_plans_one_tile_per_cta runs both
    settings through check_support at 132 CTAs, so these arguments are pinned here."""
    import ast

    from cudnn.sdpa.fwd import api_dsl

    with open(api_dsl.__file__, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_sm90_d128_causal_route"]
    assert len(calls) == 1, [call.lineno for call in calls]
    inner = parents[calls[0]]
    assert isinstance(inner, ast.If) and inner.test is calls[0] and not inner.orelse, ast.unparse(inner)
    assert [ast.unparse(node) for node in inner.body] == ["self.params = replace(self.params, persistent=False, causal_walk=0, sched_policy=causal_sched)"]
    outer = parents[inner]
    assert isinstance(outer, ast.If) and outer.body == [inner] and not outer.orelse, ast.unparse(outer)
    assert ast.unparse(outer.test) == "causal_walk and D128_CAUSAL_WALK_ROUTE"
    assert not calls[0].keywords and [ast.unparse(arg) for arg in calls[0].args] == [
        "(b, h_q, h_kv, s_q, s_kv)",
        "self.params.causal if self.params.bottom_right is None else self.params.bottom_right",
        "self._persistent_ctas(self.q_desc.device)",
    ], ast.unparse(calls[0])
    records = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Sm90TemplateParams"]
    assert [ast.unparse(kw.value) for record in records for kw in record.keywords if kw.arg == "qh_per_kh"] == ["h_q // h_kv"]


def test_d128_causal_walk_records_take_the_natural_order_unless_pinned():
    """check_support gives a causal record of the walk's rule (causal_walk 1) the NATURAL order unless the caller pinned
    one (sched_policy set): right after the persistence decision it keeps the order it chose (causal_sched, the one-tile
    record's: _causal_sched_policy's LPT or LPT_L2 on a causal plan) and replaces it, under ``if causal_walk and
    self.sched_policy is None:``, with SCHED_NATURAL, which the record's sched_policy then carries. A pinned LPT or
    LPT_L2 reaches the record as pinned, and the stride walks that order under the deal the planner picks for it; a
    record of D128_PERSISTENT_CAUSAL_WALK 1 (causal_walk 0) keeps the causal order. The cc 9.0 suite's adapter tests run
    it through check_support."""
    import ast

    from cudnn.sdpa.fwd import api_dsl

    with open(api_dsl.__file__, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    pins = [node for node in ast.walk(tree) if isinstance(node, ast.If) and ast.unparse(node.test) == "causal_walk and self.sched_policy is None"]
    assert len(pins) == 1 and [ast.unparse(node) for node in pins[0].body] == ["sched = SCHED_NATURAL"] and not pins[0].orelse
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    body = parents[pins[0]].body
    at = body.index(pins[0])
    assert ast.unparse(body[at - 1]) == "causal_sched = sched"
    assert ast.unparse(body[at - 2]) == "persistent, causal_walk = _sm90_d128_persistence(self.flavor, band['causal'])"
    record = body[at + 1]
    assert isinstance(record, ast.Assign) and ast.unparse(record.targets[0]) == "self.params"
    assert [ast.unparse(kw.value) for kw in record.value.keywords if kw.arg == "sched_policy"] == ["sched"]


def test_d128_route_switch_and_cutoff_are_declared():
    """The route's switch is a bool and _D128_ROUTE_MAX_L its one cutoff constant, a positive number; their values are
    tuning choices (api_dsl's and config_sm90's comments give the measurements) and are not pinned here."""
    from cudnn.sdpa.fwd import api_dsl

    assert isinstance(api_dsl.D128_CAUSAL_WALK_ROUTE, bool)
    consts = {name: getattr(api_dsl, name) for name in vars(api_dsl) if name.startswith("_D128_ROUTE")}
    assert list(consts) == ["_D128_ROUTE_MAX_L"] and consts["_D128_ROUTE_MAX_L"] > 0, consts


# --- The paired walk: the kernel's decode on plain ints (host-only, no trace) -------
#
# walk_unit, walk_live, walk_next and walk_item are the functions both roles of the kernel trace in place; here they run
# on plain ints, CTA by CTA, as the item loop runs them (first_item, live, next_item, work_of).

_WALK_Q_TILES = (1, 2, 3, 4, 5, 8, 9, 17, 33)
_WALK_LAYOUTS = ((1, 1), (1, 2), (1, 3), (2, 2), (3, 3), (1, 16), (5, 7))  # (b, h_q): 1, 2, 3, 4, 9, 16 and 35 groups
_WALK_CTAS = (1, 2, 3, 7, 131, 132)


def _walk_of_cta(mod, walk, cta):
    """CTA ``cta``'s items in the order the kernel's item loop visits them: position 0, then walk_next while walk_live,
    each item decoded by walk_item."""
    pos, items = 0, []
    while mod.walk_live(pos, cta, walk):
        unit = mod.walk_unit(pos, cta, walk)
        items.append(mod.walk_item(pos, unit, walk))
        pos = mod.walk_next(pos, unit, walk)
        assert len(items) <= 2 * walk.units, "the walk does not end"
    return items


def test_paired_walk_visits_every_item_once_in_the_dealt_order():
    """The kernel's decode against the units built item by item (_walk_units) and dealt round by round (_deal): for Q
    tile counts 1, 2, 3, 4, 5, 8, 9, 17 and 33, group counts 1, 2, 3, 4, 9, 16 and 35, light-row counts at every edge
    (0, 1, 2, T // 2, T - 2, T - 1, T: no pairs, middle tiles only, one pair, light rows with and without middles, odd
    and even group counts), CTA counts 1, 2, 3, 7, 131 and 132, with and without the snake, every CTA visits exactly its
    dealt units' items in order, every CTA has at least one unit, and the CTAs together visit every (Q tile, group)
    item once. A pair count of 0 (T - light_rows of 0 or 1) never reaches a divisor: on plain ints it would raise
    ZeroDivisionError here."""
    mod = _walk_module()
    for q_tiles, (b, h_q), ctas, snake in itertools.product(_WALK_Q_TILES, _WALK_LAYOUTS, _WALK_CTAS, (True, False)):
        groups = b * h_q
        for light in sorted({0, 1, 2, q_tiles // 2, q_tiles - 2, q_tiles - 1, q_tiles} & set(range(q_tiles + 1))):
            walk = mod.causal_walk_layout(q_tiles, groups, light, ctas, snake)
            dealt = _deal(_walk_units(q_tiles, groups, light), ctas, snake)
            assert walk.ctas == len(dealt) and walk.units == sum(map(len, dealt)), (walk, len(dealt))
            seen = []
            for cta in range(walk.ctas):
                visited = _walk_of_cta(mod, walk, cta)
                assert visited and visited == [item for unit in dealt[cta] for item in unit], (walk, cta)
                seen += visited
            assert sorted(seen) == [(t, g) for t in range(q_tiles) for g in range(groups)], walk


def test_paired_walk_record_launches_one_cta_per_unit():
    """A causal_walk=1 record on the paired walk launches walk.ctas CTAs, min(units, persistent_ctas), so every CTA
    has a unit (b1 h8/2 S 2048 at 64 CTAs, T 16: pairs alone, 64 units); where the rule keeps the stride it launches
    min(items, persistent_ctas), as a causal_walk=0 record does. Host-only: no trace."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2

    mod = _walk_module()

    def make(shape, ctas, walk):
        return mod.SM90FusedMultiHeadAttentionForward(
            shape,
            is_causal=True,
            bottom_right=True,
            sched_policy=SCHED_LPT_L2,
            qh_per_kh=shape[1] // shape[2],
            intra_wg_overlap=True,
            persistent=True,
            persistent_ctas=ctas,
            causal_walk=walk,
        )

    paired = make((1, 8, 2, 2048, 2048), 64, 1)
    assert paired.paired_walk and paired.n_ctas == paired.walk.ctas == 64 and paired.walk.units == 64, paired.walk
    stride = make((8, 24, 8, 256, 2048), 132, 1)
    assert stride.walk is None and not stride.paired_walk and stride.n_ctas == 132
    plain = make((1, 8, 2, 2048, 2048), 64, 0)
    assert not plain.paired_walk and plain.n_ctas == 64


def test_both_roles_walk_with_the_same_methods():
    """The producer (tmaldg_persistent) and the compute warpgroups (walk_items) start at first_item, loop while live,
    decode with work_of and step with next_item, so under either walk they visit the same items in the same order; each
    of those methods takes the paired walk exactly under the record's paired_walk constant."""
    import ast

    tree = _kernel_ast()
    for role in ("tmaldg_persistent", "walk_items"):
        fn = _function(tree, role)
        calls = [ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func).startswith("self.")]
        loops = [n for n in ast.walk(fn) if isinstance(n, ast.While)]
        assert len(loops) == 1 and ast.unparse(loops[0].test) == "self.live(item)", (role, [ast.unparse(n.test) for n in loops])
        assert {"self.first_item", "self.work_of", "self.next_item"} <= set(calls), (role, calls)
    for name in ("first_item", "live", "next_item", "work_of"):
        tests = [ast.unparse(n.test) for n in ast.walk(_function(tree, name)) if isinstance(n, ast.If)]
        assert tests[:1] == ["cutlass.const_expr(self.paired_walk)"], (name, tests)
