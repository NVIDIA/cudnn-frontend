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


# --- Step 7 switch (TemplateParams.intra_wg_overlap) -------


def test_step7_switch_fields_are_appended_last():
    """The switch is appended after thd_origins, defaulting to False: every earlier specialization keeps its
    defaults (the append-only rule the kit's hygiene check also enforces). The segment is pinned from its own first
    field, so a field appended after it later still passes."""
    from cudnn.sdpa.fwd import config_sm90 as c

    fields = dataclasses.fields(c.TemplateParams)
    names = [f.name for f in fields]
    start = names.index("intra_wg_overlap")
    assert start > names.index("thd_origins"), names
    expected = [("intra_wg_overlap", False, bool)]
    assert [(f.name, f.default, type(f.default)) for f in fields[start : start + len(expected)]] == expected


@pytest.mark.parametrize("over", [dict(intra_wg_overlap=True)], ids=["overlap"])
def test_d512_refuses_the_step7_switches(over):
    """The switch is a D128 template field: D512 refuses it, D128 takes it."""
    from cudnn.sdpa.fwd import config_sm90 as c

    params = dataclasses.replace(c.TemplateParams(), **over)
    with pytest.raises(ValueError, match="SM90 D512 has no intra-warpgroup overlap switch"):
        c.validate_params(params)
    c.validate_params(params, flavor=c.D128_FLAVOR)


def test_each_switch_combination_is_its_own_module_and_digest():
    """The loader keys modules on (path, record) and folds repr(record) into the digest, so the two switch
    values are two modules with two digests (two compile keys) in one process, with no hand-set digest
    (python/cudnn/sdpa/AGENTS.md); the default record and a spelled-out False are one module."""
    from cudnn.sdpa.fwd import config_sm90 as c
    from cudnn.sdpa.fwd.api_dsl import _load_kernel_template

    mods = {}
    for overlap in (False, True):
        params = dataclasses.replace(c.TemplateParams(), intra_wg_overlap=overlap)
        mod = _load_kernel_template("sm90/prefill_d128_f16.py", params, tag="switch_probe_sm90_d128")
        assert mod.INTRA_WG_OVERLAP == overlap
        mods[overlap] = mod
    assert len({id(m) for m in mods.values()}) == 2
    assert len({m.FROST_SOURCE_DIGEST for m in mods.values()}) == 2
    assert _load_kernel_template("sm90/prefill_d128_f16.py", c.TemplateParams(), tag="switch_probe_sm90_d128") is mods[False]
