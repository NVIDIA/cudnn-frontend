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
    """The overlap switch, the persistent switch, then the causal walk are appended after thd_origins, defaulting to
    False, False and 0 (the stride by the grid): every earlier specialization keeps its defaults (the append-only
    rule). The segment is pinned from its own first field, so a field appended after it later still passes."""
    from cudnn.sdpa.fwd import config_sm90 as c

    fields = dataclasses.fields(c.TemplateParams)
    names = [f.name for f in fields]
    start = names.index("intra_wg_overlap")
    assert start > names.index("thd_origins"), names
    expected = [("intra_wg_overlap", False, bool), ("persistent", False, bool), ("causal_walk", 0, int)]
    assert [(f.name, f.default, type(f.default)) for f in fields[start : start + len(expected)]] == expected


@pytest.mark.parametrize(
    ("switch", "value", "needs", "why"),
    [
        pytest.param("intra_wg_overlap", True, {}, "SM90 D512 has no intra-warpgroup overlap switch", id="overlap"),
        pytest.param("persistent", True, dict(intra_wg_overlap=True), "SM90 D512 has no persistent switch", id="persistent"),
        pytest.param("causal_walk", 1, dict(causal=True, intra_wg_overlap=True, persistent=True), "SM90 D512 has no causal walk switch", id="causal-walk"),
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
    every causal plan the route keeps (test_d128_route_ships_on_with_its_cutoffs pins the route;
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


# Twenty-six causal shapes (b, h_q, h_kv, S_q = S_kv) at 132 CTAs, an H100 SXM's SM count, under the adapter's LPT_L2
# order, and the light-row count light_rows the rule gives each (test_causal_walk_rule_is_an_exhaustive_search checks
# the rule itself): pairs alone where pairs fill the rounds evenly, a light tail on the long shapes and where the pairs
# leave a round short, single items only (light_rows = T) on b1 h16/4 and b17 h2/2 s1024.
_RULE_L = {
    (16, 16, 4, 512): 0,
    (8, 16, 4, 1024): 0,
    (2, 32, 8, 2048): 0,
    (1, 32, 8, 4096): 32,
    (1, 16, 4, 8192): 31,
    (33, 8, 8, 512): 0,
    (33, 4, 4, 1024): 0,
    (33, 2, 2, 2048): 0,
    (33, 1, 1, 4096): 0,
    (66, 4, 1, 512): 0,
    (33, 4, 1, 1024): 0,
    (33, 1, 1, 8192): 0,
    (1, 16, 4, 1024): 8,
    (11, 3, 3, 1024): 0,
    (17, 2, 2, 1024): 8,
    (23, 6, 6, 1024): 1,
    (32, 16, 4, 1024): 0,
    (33, 16, 4, 1024): 0,
    (2, 32, 8, 4096): 32,
    (2, 32, 8, 8192): 31,
    (2, 32, 8, 16384): 29,
    (2, 32, 32, 4096): 8,
    (8, 32, 8, 4096): 15,
    (4, 8, 8, 8192): 15,
    (2, 16, 4, 16384): 29,
    (1, 64, 8, 16384): 29,
}


def test_causal_walk_rule_light_rows_on_26_causal_shapes():
    """The rule's light-row count on 26 causal shapes, b1 to b66 at S = 512 to 16384 (_RULE_L), at 132 CTAs: the paired
    walk on every one, with its layout's constants consistent (pairs, the middle units, every item in a unit)."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2

    mod = _walk_module()
    for (b, h_q, h_kv, s), light in _RULE_L.items():
        walk = mod.causal_walk_plan((b, h_q, h_kv, s, s), True, SCHED_LPT_L2, h_q // h_kv, 132, 2)
        assert walk is not None and walk.light_rows == light, ((b, h_q, h_kv, s), walk)
        q_tiles = s // 128
        assert (walk.q_tiles, walk.groups, walk.ctas) == (q_tiles, h_q * b, min(walk.units, 132)), walk
        assert sum(map(len, _walk_units(q_tiles, h_q * b, light))) == q_tiles * h_q * b == 2 * walk.heavy_units - walk.lone_middle + h_q * b * light


@pytest.mark.parametrize(
    ("shape", "light"),
    [
        pytest.param((4, 24, 8, 512, 4096), None, id="b4-h24-sq512-skv4096"),
        pytest.param((8, 24, 8, 256, 2048), None, id="b8-h24-sq256-skv2048"),
        pytest.param((8, 32, 8, 2048, 16384), 0, id="b8-h32-sq2048-skv16384"),
        pytest.param((4, 32, 8, 2048, 16384), 0, id="b4-h32-sq2048-skv16384"),
    ],
)
def test_causal_walk_rule_keeps_the_stride_where_it_balances_better(shape, light):
    """Chunked prefill (bottom-right, S_q well below S_kv): every Q tile walks nearly the same KV tiles, so pairs are a
    coarser grain than single items. On the first two shapes the stride's largest CTA load is the smaller and the rule
    keeps the stride (None); on the last two the paired walk's is no larger and the rule takes it."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2

    walk = _walk_module().causal_walk_plan(shape, True, SCHED_LPT_L2, shape[1] // shape[2], 132, 2)
    assert (None if walk is None else walk.light_rows) == light, walk


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
    ],
)
def test_causal_walk_rule_is_an_exhaustive_search(shape, bottom_right, sched):
    """The rule against a brute force that visits every item, at 1, 3, 7, 64 and 132 CTAs: for each light_rows from 0 to
    the L2 cap, every CTA's load summed over the units _walk_units builds and _deal deals; for the stride, every CTA's
    load over the one-tile grid's order decoded item by item (NATURAL, LPT, lpt_l2_tile_coords on plain ints); each item
    its KV tiles plus CAUSAL_WALK_PHI. The plan is the light_rows with the smallest largest load (the smaller on a tie),
    and None exactly where the stride's is smaller; a plan with at most one item per CTA takes light_rows = T. The cap
    is recomputed from its definition: a round of single items reads up to min(ceil(CTAs / qh_per_kh) + 1, h_kv * b) KV
    heads, each to its row's last KV tile of 128 rows of K and V, within the LPT_L2 order's L2 budget (one KV group's
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
    for ctas in (1, 3, 7, 64, 132):
        walk = mod.causal_walk_plan(shape, bottom_right, sched, h_q // h_kv, ctas, 2)
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
        stride = [0] * min(ctas, q_tiles * groups)
        for i, tile in enumerate(_stride_tiles(shape, sched, h_q // h_kv, q_tiles)):
            stride[i % len(stride)] += cost[tile]
        want = best if loads[best] <= max(stride) else None
        assert (None if walk is None else walk.light_rows) == want, (ctas, cap, loads, max(stride), walk)


def test_causal_walk_rule_honours_its_constants():
    """The rule's constants as compile-time arms: a forced light-row count is taken as is (0: pairs only), clamped to T,
    and never by a plan with at most one item per CTA (light_rows = T); the L2 cap off lets light_rows past it (b23 h6/6
    s1024: 1 capped, 8 uncapped; b2 h32/32 s4096: 8 and 32); the snake off scores the walk dealt every round from CTA 0,
    which favours pairs alone (b1 h16/4 s8192: 31 with the snake, 0 without)."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2

    mod = _walk_module()

    def light(shape, **kw):
        walk = mod.causal_walk_plan(shape, True, SCHED_LPT_L2, shape[1] // shape[2], 132, 2, **kw)
        return None if walk is None else walk.light_rows

    assert [light((1, 16, 4, 8192, 8192), light_rows=n) for n in (0, 5, 64, 1000)] == [0, 5, 64, 64]
    assert [light((1, 16, 4, 1024, 1024), light_rows=n) for n in (0, 3)] == [8, 8]
    assert [light((23, 6, 6, 1024, 1024), l2_cap=cap) for cap in (True, False)] == [1, 8]
    assert [light((2, 32, 32, 4096, 4096), l2_cap=cap) for cap in (True, False)] == [8, 32]
    assert [light((1, 16, 4, 8192, 8192), snake=on) for on in (True, False)] == [31, 0]
    assert mod.causal_walk_plan((1, 16, 4, 8192, 8192), True, SCHED_LPT_L2, 4, 132, 2, snake=False).snake is False
    assert (mod.CAUSAL_WALK_LIGHT_ROWS, mod.CAUSAL_WALK_L2_CAP, mod.CAUSAL_WALK_PHI, mod.CAUSAL_WALK_SNAKE) == (-1, True, 2, True)


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
        assert make(shape, causal_walk=1, **kw).walk == mod.causal_walk_plan(shape, bottom_right, sched, shape[1] // shape[2], ctas, 2)
        assert make(shape, **kw).walk is None
    with pytest.raises(ValueError, match="orders a persistent causal record's walk"):
        make((1, 2, 2, 256, 256), is_causal=True, causal_walk=1)
    with pytest.raises(ValueError, match="orders a persistent causal record's walk"):
        make((1, 2, 2, 256, 256), persistent=True, persistent_ctas=2, causal_walk=1)


# --- The route: causal plans of the paired walk go one tile per CTA by L (host-only) -------
#
# api_dsl._sm90_d128_causal_route, run on the paired walk's own template: check_support gives a causal plan of the
# paired walk the one-tile record where it returns True (D128_CAUSAL_WALK_ROUTE); the cc 9.0 suite's
# test_d128_adapter_routes_causal_plans_one_tile_per_cta runs it through check_support.


def _route(shape, ctas=132):
    """Whether the route sends a causal bottom-right bf16 plan of ``shape``, under the adapter's causal order, one tile
    per CTA at ``ctas`` CTAs (132: an H100 SXM's SM count)."""
    from cudnn.sdpa.fwd import api_dsl

    _, h_q, h_kv, _, s_kv = shape
    sched = api_dsl._causal_sched_policy(s_kv, 128, 128, 2)
    return api_dsl._sm90_d128_causal_route(_walk_module(), shape, True, sched, h_q // h_kv, ctas, 2)


def _plan(shape, ctas=132):
    """``(kind, L)`` of the same plan, from the template's planner: "one item per CTA" (at most ``ctas`` items), "pairs"
    (the walk's rule deals pairs alone, at most one unit per CTA), "walk" (any other paired walk) or "stride" (the
    walk's rule keeps the stride), and L, the mean KV tiles per Q tile."""
    from cudnn.sdpa.fwd import api_dsl

    mod = _walk_module()
    b, h_q, h_kv, _, s_kv = shape
    kv_tiles = mod.causal_kv_tiles(shape, True)
    mean = sum(kv_tiles) / len(kv_tiles)
    if len(kv_tiles) * h_q * b <= ctas:
        return "one item per CTA", mean
    walk = mod.causal_walk_plan(shape, True, api_dsl._causal_sched_policy(s_kv, 128, 128, 2), h_q // h_kv, ctas, 2)
    return ("stride" if walk is None else "pairs" if walk.units <= ctas else "walk"), mean


# The 161 timings of D128_CAUSAL_WALK_ROUTE's comment (132 causal shapes on H100 SXM GPUs, #1159's harness, bf16, ten
# sessions from 2026-10-21 to 2026-11-04, each on its own machine), by the kind of plan and L: (session, (b, h_q, h_kv,
# S_q, S_kv) with the diagonal bottom-right (the top-left mask where S_q equals S_kv), L, t(one tile) / t(the walk's
# plan) as a median of per-round ratios, its band, the larger of 1% and the A/A twin gap (1% alone on the 26 timings of
# 2026-10-21, which ran no twin), and whether the route sends the plan one tile per CTA at 132 CTAs). Below 1 minus the
# band the one-tile kernel won, above 1 plus the band the walk's plan did. A shape timed in more than one session has a
# row per session. Paired walks stay below L 48 (_D128_ROUTE_WALK_MAX_L) whatever the head layout; strides stay up to L
# 15 inclusive with more than one Q head per KV head (_D128_ROUTE_STRIDE_MAX_L_GQA) and below L 10 on MHA
# (_D128_ROUTE_STRIDE_MAX_L_MHA); pairs alone stay at any L, and a plan of at most one item per CTA is routed at any L.
_ROUTE_MEASURED = {
    "walk": [
        ("s13", (8, 24, 8, 256, 256), 1.5, 1.2785, 0.0196, False),
        ("s11", (16, 16, 4, 512, 512), 2.5, 1.3000, 0.0100, False),
        ("s11", (33, 8, 8, 512, 512), 2.5, 1.2605, 0.0100, False),
        ("s11", (66, 4, 1, 512, 512), 2.5, 1.3806, 0.0100, False),
        ("s13", (8, 24, 8, 256, 384), 2.5, 1.2440, 0.0136, False),
        ("s14", (16, 16, 4, 512, 512), 2.5, 1.2983, 0.0100, False),
        ("s14", (29, 8, 1, 512, 512), 2.5, 1.3053, 0.0146, False),
        ("s14", (33, 8, 8, 512, 512), 2.5, 1.2662, 0.0100, False),
        ("s14", (66, 4, 1, 512, 512), 2.5, 1.3771, 0.0100, False),
        ("s19", (66, 4, 1, 512, 512), 2.5, 1.3895, 0.0198, False),
        ("s12", (4, 8, 8, 640, 640), 3.0, 1.0406, 0.0345, False),
        ("s11", (8, 16, 4, 1024, 1024), 4.5, 1.1955, 0.0100, False),
        ("s11", (17, 2, 2, 1024, 1024), 4.5, 1.1669, 0.0100, False),
        ("s11", (23, 6, 6, 1024, 1024), 4.5, 1.1275, 0.0100, False),
        ("s11", (32, 16, 4, 1024, 1024), 4.5, 1.2785, 0.0100, False),
        ("s11", (33, 4, 1, 1024, 1024), 4.5, 1.2854, 0.0100, False),
        ("s11", (33, 4, 4, 1024, 1024), 4.5, 1.1976, 0.0100, False),
        ("s11", (33, 16, 4, 1024, 1024), 4.5, 1.3316, 0.0100, False),
        ("s12", (17, 2, 2, 1024, 1024), 4.5, 1.1587, 0.0118, False),
        ("s13", (17, 2, 2, 1024, 1024), 4.5, 1.1706, 0.0224, False),
        ("s14", (8, 16, 4, 1024, 1024), 4.5, 1.1999, 0.0100, False),
        ("s14", (24, 128, 8, 1024, 1024), 4.5, 1.2941, 0.0100, False),
        ("s14", (32, 128, 8, 512, 768), 4.5, 1.7587, 0.0100, False),
        ("s14", (33, 4, 1, 1024, 1024), 4.5, 1.2986, 0.0100, False),
        ("s14", (33, 4, 4, 1024, 1024), 4.5, 1.2017, 0.0100, False),
        ("s12", (4, 16, 4, 1152, 1152), 5.0, 1.2390, 0.0100, False),
        ("s13", (4, 16, 4, 1152, 1152), 5.0, 1.2428, 0.0100, False),
        ("s12", (4, 16, 4, 1000, 1100), 5.5, 1.0780, 0.0193, False),
        ("s11", (2, 32, 8, 2048, 2048), 8.5, 1.1091, 0.0100, False),
        ("s11", (33, 2, 2, 2048, 2048), 8.5, 1.1221, 0.0100, False),
        ("s17", (12, 64, 8, 2048, 2048), 8.5, 1.1517, 0.0100, False),
        ("s13", (8, 24, 8, 256, 1280), 9.5, 1.2237, 0.0104, False),
        ("s12", (1, 6, 6, 3000, 3000), 12.5, 1.0437, 0.0219, False),
        ("s11", (1, 32, 8, 4096, 4096), 16.5, 1.0563, 0.0100, False),
        ("s11", (2, 32, 8, 4096, 4096), 16.5, 1.0708, 0.0100, False),
        ("s11", (2, 32, 32, 4096, 4096), 16.5, 1.0412, 0.0100, False),
        ("s11", (8, 32, 8, 4096, 4096), 16.5, 1.0607, 0.0100, False),
        ("s11", (33, 1, 1, 4096, 4096), 16.5, 1.0705, 0.0100, False),
        ("s19", (2, 32, 32, 4096, 4096), 16.5, 1.0438, 0.0100, False),
        ("s11", (1, 16, 4, 8192, 8192), 32.5, 1.0144, 0.0100, False),
        ("s11", (2, 32, 8, 8192, 8192), 32.5, 1.0269, 0.0100, False),
        ("s11", (4, 8, 8, 8192, 8192), 32.5, 1.0211, 0.0100, False),
        ("s11", (33, 1, 1, 8192, 8192), 32.5, 1.0307, 0.0100, False),
        ("s14", (1, 16, 4, 8192, 8192), 32.5, 1.0194, 0.0100, False),
        ("s14", (4, 32, 32, 4096, 6144), 32.5, 1.0251, 0.0100, False),
        ("s14", (4, 32, 32, 8192, 8192), 32.5, 1.0103, 0.0100, False),
        ("s14", (6, 32, 32, 4096, 6144), 32.5, 1.0247, 0.0100, False),
        ("s19", (2, 32, 8, 8192, 8192), 32.5, 1.0257, 0.0100, False),
        ("s14", (8, 32, 8, 2048, 6144), 40.5, 1.0680, 0.0100, False),
        ("s15", (8, 32, 8, 2048, 6144), 40.5, 1.0689, 0.0100, False),
        ("s16", (8, 32, 8, 2048, 6144), 40.5, 1.0440, 0.0100, False),
        ("s17", (8, 32, 8, 2048, 6144), 40.5, 1.0681, 0.0100, False),
        ("s18", (8, 32, 8, 2048, 6144), 40.5, 1.0661, 0.0100, False),
        ("s19", (8, 32, 8, 2048, 6144), 40.5, 1.0638, 0.0100, False),
        ("s17", (8, 8, 8, 4096, 7680), 44.5, 0.9976, 0.0100, False),
        ("s17", (1, 64, 8, 11776, 11776), 46.5, 1.0327, 0.0100, False),
        ("s17", (6, 16, 16, 4096, 8064), 47.5, 0.9980, 0.0100, False),
        ("s17", (16, 32, 8, 1024, 6528), 47.5, 1.0099, 0.0100, False),
        ("s14", (4, 32, 32, 4096, 8192), 48.5, 1.0011, 0.0100, True),
        ("s15", (6, 32, 32, 4096, 8192), 48.5, 1.0000, 0.0100, True),
        ("s17", (16, 32, 8, 1024, 6656), 48.5, 1.0098, 0.0100, True),
        ("s17", (2, 64, 64, 4096, 8448), 50.5, 1.0027, 0.0100, True),
        ("s17", (2, 16, 4, 13312, 13312), 52.5, 1.0118, 0.0100, True),
        ("s17", (4, 32, 4, 2048, 8192), 56.5, 1.0258, 0.0100, True),
        ("s11", (1, 64, 8, 16384, 16384), 64.5, 1.0118, 0.0100, True),
        ("s11", (2, 16, 4, 16384, 16384), 64.5, 1.0082, 0.0100, True),
        ("s11", (2, 32, 8, 16384, 16384), 64.5, 1.0088, 0.0100, True),
        ("s15", (3, 32, 32, 4096, 10240), 64.5, 1.0081, 0.0100, True),
        ("s15", (4, 32, 32, 4096, 10240), 64.5, 1.0062, 0.0100, True),
        ("s14", (8, 32, 8, 2048, 10240), 72.5, 1.0225, 0.0100, True),
        ("s15", (3, 32, 32, 4096, 12288), 80.5, 1.0157, 0.0100, True),
        ("s15", (8, 32, 8, 2048, 12288), 88.5, 1.0200, 0.0100, True),
        ("s16", (3, 32, 32, 4096, 13312), 88.5, 0.9805, 0.0100, True),
        ("s15", (8, 32, 8, 2048, 13312), 96.5, 0.9840, 0.0100, True),
        ("s16", (16, 32, 8, 1024, 13312), 100.5, 0.9770, 0.0100, True),
        ("s16", (2, 32, 32, 4096, 15360), 104.5, 0.9856, 0.0100, True),
        ("s16", (2, 32, 32, 4096, 16384), 112.5, 0.9872, 0.0100, True),
        ("s12", (4, 32, 8, 2048, 16384), 120.5, 0.9932, 0.0100, True),
        ("s12", (8, 32, 8, 2048, 16384), 120.5, 0.9857, 0.0100, True),
        ("s13", (4, 32, 8, 2048, 16384), 120.5, 0.9909, 0.0100, True),
        ("s13", (8, 32, 8, 2048, 16384), 120.5, 0.9834, 0.0100, True),
        ("s14", (8, 32, 8, 2048, 16384), 120.5, 0.9899, 0.0100, True),
        ("s15", (8, 32, 8, 2048, 16384), 120.5, 0.9889, 0.0100, True),
        ("s16", (8, 32, 8, 2048, 16384), 120.5, 0.9773, 0.0100, True),
        ("s17", (8, 32, 8, 2048, 16384), 120.5, 0.9911, 0.0100, True),
        ("s18", (8, 32, 8, 2048, 16384), 120.5, 0.9900, 0.0100, True),
        ("s19", (8, 32, 8, 2048, 16384), 120.5, 0.9884, 0.0100, True),
        ("s14", (3, 32, 32, 4096, 20480), 144.5, 0.9964, 0.0100, True),
        ("s17", (4, 64, 8, 1024, 20480), 156.5, 1.0021, 0.0100, True),
        ("s16", (16, 16, 4, 2048, 24576), 184.5, 0.9902, 0.0100, True),
        ("s16", (5, 32, 8, 2048, 26624), 200.5, 0.9818, 0.0100, True),
        ("s16", (4, 64, 8, 2048, 32768), 248.5, 0.9731, 0.0409, True),
        ("s14", (8, 32, 8, 2048, 34816), 264.5, 0.9778, 0.0443, True),
        ("s14", (8, 32, 8, 2048, 67584), 520.5, 0.9890, 0.0724, True),
    ],
    "stride": [
        ("s18", (24, 32, 32, 768, 896), 4.5, 1.2083, 0.0100, False),
        ("s17", (16, 32, 32, 2048, 2048), 8.5, 1.0238, 0.0100, False),
        ("s18", (41, 64, 8, 1024, 1536), 8.5, 1.1346, 0.0100, False),
        ("s19", (5, 64, 64, 768, 1536), 9.5, 1.0598, 0.0137, False),
        ("s18", (8, 32, 8, 2176, 2304), 10.0, 1.0979, 0.0100, False),
        ("s19", (4, 64, 32, 2176, 2304), 10.0, 1.1137, 0.0100, False),
        ("s20", (4, 64, 32, 2176, 2304), 10.0, 1.1235, 0.0100, False),
        ("s20", (9, 16, 16, 1664, 2048), 10.0, 1.0179, 0.0123, True),
        ("s20", (13, 16, 16, 1152, 1792), 10.0, 0.9610, 0.0100, True),
        ("s20", (15, 16, 8, 1152, 1792), 10.0, 1.0970, 0.0119, False),
        ("s19", (41, 64, 8, 1024, 1792), 10.5, 1.0703, 0.0100, False),
        ("s19", (8, 32, 8, 2176, 2432), 11.0, 1.0909, 0.0100, False),
        ("s20", (8, 32, 8, 2176, 2432), 11.0, 1.0976, 0.0100, False),
        ("s20", (13, 16, 16, 1152, 1920), 11.0, 0.9549, 0.0100, True),
        ("s20", (15, 16, 8, 1152, 1920), 11.0, 1.0821, 0.0100, False),
        ("s20", (2, 64, 64, 1664, 2304), 12.0, 1.0481, 0.0188, True),
        ("s20", (25, 64, 8, 384, 1664), 12.0, 1.0848, 0.0100, False),
        ("s20", (4, 32, 32, 1408, 2304), 13.0, 1.0376, 0.0191, True),
        ("s20", (17, 32, 8, 896, 2048), 13.0, 1.0718, 0.0100, False),
        ("s20", (3, 64, 64, 896, 2176), 14.0, 1.0280, 0.0189, True),
        ("s20", (7, 32, 16, 1408, 2432), 14.0, 1.0682, 0.0100, False),
        ("s20", (4, 32, 32, 2176, 2944), 15.0, 0.9811, 0.0156, True),
        ("s20", (9, 16, 16, 1664, 2688), 15.0, 0.9576, 0.0100, True),
        ("s20", (17, 32, 4, 384, 2048), 15.0, 1.0779, 0.0100, False),
        ("s20", (30, 24, 8, 384, 2048), 15.0, 1.0623, 0.0100, False),
        ("s13", (8, 24, 8, 256, 1956), 15.5, 1.0264, 0.0131, True),
        ("s13", (8, 24, 8, 256, 2048), 15.5, 1.0108, 0.0100, True),
        ("s17", (10, 16, 16, 1536, 2688), 15.5, 0.9322, 0.0114, True),
        ("s17", (16, 32, 32, 2048, 2944), 15.5, 0.9851, 0.0100, True),
        ("s18", (12, 16, 8, 1280, 2560), 15.5, 1.0277, 0.0100, True),
        ("s13", (8, 24, 8, 384, 2176), 16.0, 1.0287, 0.0100, True),
        ("s17", (8, 32, 32, 4096, 4096), 16.5, 0.9669, 0.0108, True),
        ("s17", (12, 16, 16, 3072, 4352), 22.5, 0.9573, 0.0100, True),
        ("s17", (16, 32, 8, 2048, 3840), 22.5, 1.0207, 0.0100, True),
        ("s18", (8, 64, 64, 2048, 3840), 22.5, 0.9172, 0.0100, True),
        ("s18", (12, 16, 8, 3072, 4352), 22.5, 1.0146, 0.0100, True),
        ("s17", (16, 24, 8, 2048, 4096), 24.5, 1.0308, 0.0100, True),
        ("s12", (4, 24, 8, 512, 4096), 30.5, 1.0273, 0.0100, True),
        ("s13", (4, 24, 8, 512, 4096), 30.5, 1.0231, 0.0100, True),
        ("s16", (8, 32, 32, 4096, 6144), 32.5, 0.9586, 0.0100, True),
        ("s18", (3, 16, 8, 3072, 6528), 39.5, 0.9438, 0.0100, True),
        ("s15", (3, 8, 8, 2048, 6144), 40.5, 1.0110, 0.0100, True),
        ("s15", (3, 8, 8, 2048, 7168), 48.5, 0.9202, 0.0100, True),
        ("s16", (6, 32, 8, 4096, 8192), 48.5, 1.0048, 0.0100, True),
        ("s12", (8, 64, 8, 2048, 8192), 56.5, 0.9959, 0.0100, True),
        ("s16", (16, 32, 8, 2048, 8192), 56.5, 1.0095, 0.0100, True),
        ("s15", (6, 32, 32, 4096, 10240), 64.5, 0.9148, 0.0100, True),
        ("s16", (5, 32, 32, 4096, 10240), 64.5, 0.9488, 0.0100, True),
        ("s18", (6, 32, 16, 4096, 10240), 64.5, 0.9645, 0.0100, True),
        ("s14", (3, 8, 8, 2048, 10240), 72.5, 0.9335, 0.0100, True),
        ("s16", (12, 32, 8, 2048, 11264), 80.5, 1.0117, 0.0100, True),
        ("s12", (8, 64, 8, 2048, 16384), 120.5, 1.0084, 0.0358, True),
        ("s14", (6, 32, 32, 4096, 20480), 144.5, 0.9092, 0.0474, True),
        ("s18", (3, 32, 8, 2560, 20480), 150.5, 0.9977, 0.0100, True),
        ("s18", (1, 64, 8, 2176, 28672), 216.0, 0.9902, 0.0100, True),
    ],
    "pairs": [
        ("s11", (11, 3, 3, 1024, 1024), 4.5, 1.1238, 0.0100, False),
        ("s13", (8, 24, 8, 128, 1920), 15.0, 1.0084, 0.0113, False),
        ("s13", (4, 24, 8, 256, 2048), 15.5, 1.0453, 0.0105, False),
        ("s15", (3, 8, 8, 768, 3840), 27.5, 0.9993, 0.0111, False),
        ("s14", (3, 8, 8, 768, 4864), 35.5, 1.0128, 0.0100, False),
        ("s14", (3, 8, 8, 1024, 5120), 36.5, 1.0161, 0.0100, False),
        ("s15", (3, 8, 8, 768, 5888), 43.5, 1.0133, 0.0114, False),
        ("s15", (2, 8, 8, 1152, 7296), 53.0, 0.9946, 0.0107, False),
        ("s16", (3, 4, 1, 1536, 34304), 262.5, 1.0177, 0.0109, False),
    ],
    "one item per CTA": [
        ("s11", (1, 16, 4, 1024, 1024), 4.5, 0.9947, 0.0100, True),
        ("s12", (1, 4, 1, 2048, 16384), 120.5, 0.9960, 0.0100, True),
        ("s14", (1, 4, 1, 4096, 135168), 1040.5, 0.9934, 0.0100, True),
    ],
}
# The wins the rule gives up, (session, shape): paired walks at L 52.5 to 88.5 that won by 1.2% to 2.6%, MHA strides at
# L 10 to 14 that won by 1.8% to 4.8%, and strides at L 15.5 to 80.5 that won by 1.1% to 3.1%.
_ROUTE_WINS_GIVEN_UP = {
    ("s17", (2, 16, 4, 13312, 13312)),
    ("s17", (4, 32, 4, 2048, 8192)),
    ("s11", (1, 64, 8, 16384, 16384)),
    ("s14", (8, 32, 8, 2048, 10240)),
    ("s15", (3, 32, 32, 4096, 12288)),
    ("s15", (8, 32, 8, 2048, 12288)),
    ("s20", (9, 16, 16, 1664, 2048)),
    ("s20", (2, 64, 64, 1664, 2304)),
    ("s20", (4, 32, 32, 1408, 2304)),
    ("s20", (3, 64, 64, 896, 2176)),
    ("s13", (8, 24, 8, 256, 1956)),
    ("s13", (8, 24, 8, 256, 2048)),
    ("s18", (12, 16, 8, 1280, 2560)),
    ("s13", (8, 24, 8, 384, 2176)),
    ("s17", (16, 32, 8, 2048, 3840)),
    ("s18", (12, 16, 8, 3072, 4352)),
    ("s17", (16, 24, 8, 2048, 4096)),
    ("s12", (4, 24, 8, 512, 4096)),
    ("s13", (4, 24, 8, 512, 4096)),
    ("s15", (3, 8, 8, 2048, 6144)),
    ("s16", (12, 32, 8, 2048, 11264)),
}


def test_d128_route_on_the_measured_timings():
    """The route's decision at 132 CTAs on each of the 161 timings (_ROUTE_MEASURED), with its plan's kind and L: it
    routes all 29 on which the one-tile kernel won beyond the band and 27 of the 33 ties, keeps 78 of the 99 the walk's
    plan won, and gives up the other 21 (_ROUTE_WINS_GIVEN_UP), none by more than 4.81%. The 55 strides split by head
    layout: of the 32 with more than one Q head per KV head (GQA ratios 2, 3, 4 and 8), the 14 at L 15 or less all won
    and are kept, and the 18 above (ten won, six tied, two lost) are routed; of the 23 MHA strides, the three below L 10
    won and are kept, and the 20 from L 10 up (five won, 15 lost) are routed."""
    rows = [(kind, *row) for kind, kind_rows in _ROUTE_MEASURED.items() for row in kind_rows]
    assert len(rows) == 161 and len({shape for _, _, shape, *_ in rows}) == 132
    for kind, session, shape, mean, _, _, routed in rows:
        assert _plan(shape) == (kind, mean), (session, shape, _plan(shape))
        assert _route(shape) is routed, (session, shape, routed)
    lost = [row for row in rows if row[4] < 1 - row[5]]
    won = [row for row in rows if row[4] > 1 + row[5]]
    tied = [row for row in rows if row not in lost and row not in won]
    assert len(lost) == 29 and all(row[6] for row in lost), [row[1:3] for row in lost if not row[6]]
    given_up = [row for row in won if row[6]]
    assert len(won) == 99 and {row[1:3] for row in given_up} == _ROUTE_WINS_GIVEN_UP, sorted(row[1:3] for row in given_up)
    assert max(row[4] for row in given_up) == 1.0481
    assert len(tied) == 33 and sum(row[6] for row in tied) == 27
    gqa = [row for row in rows if row[0] == "stride" and row[2][1] > row[2][2]]
    mha = [row for row in rows if row[0] == "stride" and row[2][1] == row[2][2]]
    assert (len(gqa), len(mha)) == (32, 23)
    short_gqa, long_gqa = [row for row in gqa if row[3] <= 15], [row for row in gqa if row[3] > 15]
    assert len(short_gqa) == 14 and sorted({row[2][1] // row[2][2] for row in short_gqa}) == [2, 3, 4, 8]
    assert all(row in won and not row[6] for row in short_gqa)
    assert [sum(row in calls for row in long_gqa) for calls in (won, tied, lost)] == [10, 6, 2] and all(row[6] for row in long_gqa)
    short_mha, long_mha = [row for row in mha if row[3] < 10], [row for row in mha if row[3] >= 10]
    assert len(short_mha) == 3 and all(row in won and not row[6] for row in short_mha)
    assert [sum(row in calls for row in long_mha) for calls in (won, tied, lost)] == [5, 0, 15] and all(row[6] for row in long_mha)


def test_d128_route_walk_cutoff_at_its_boundary():
    """The walk cutoff on real paired walks of b8 h32/8 at 132 CTAs, at it and half a KV tile on either side (L moves in
    half KV tiles here): S_q 2048 S_kv 7040 (L 47.5) stays, S_q 1920 S_kv 7040 (L 48.0, the cutoff) and S_q 2048 S_kv
    7168 (L 48.5) go one tile per CTA."""
    for shape, mean, routed in [((8, 32, 8, 2048, 7040), 47.5, False), ((8, 32, 8, 1920, 7040), 48.0, True), ((8, 32, 8, 2048, 7168), 48.5, True)]:
        assert _plan(shape) == ("walk", mean) and _route(shape) is routed, (shape, _plan(shape), routed)


def test_d128_route_stride_cutoffs_at_their_boundaries():
    """Each stride cutoff on real strides at 132 CTAs, at its edge (L moves in half KV tiles here). MHA, below L 10: the
    strides of b5 h64/64 S_q 768 S_kv 1536 (which the stride won by 6.0%) and b17 h8/8 S_q 256 S_kv 1280 at L 9.5 stay,
    and those of b8 h32/32 S_q 2176 S_kv 2304 and b13 h16/16 S_q 1152 S_kv 1792 (which the one-tile kernel won by 3.9%)
    at L 10.0 go one tile per CTA. More than one Q head per KV head, up to L 15 inclusive: the strides at L 15.0 of GQA
    ratio 2 (b17 h16/8 S_q 128 S_kv 1920), 3 (b30 h24/8 S_q 384 S_kv 2048, won by 6.2%), 4 (b17 h16/4 S_q 128 S_kv
    1920), 8 (b17 h32/4 S_q 384 S_kv 2048, won by 7.8%) and 32, MQA (b41 h32/1 S_q 1920 S_kv 2816), stay, and those at
    L 15.5 of ratio 2 (b12 h16/8 S_q 1280 S_kv 2560, won by 2.8%), 3 (b8 h24/8 S_q 256 S_kv 2048, won by 1.1%), 4 (b6
    h32/8 S_q 256 S_kv 2048), 8 (b41 h64/8 S_q 768 S_kv 2304) and 32 (b41 h32/1 S_q 1792 S_kv 2816) go. Each cutoff
    reads its own layout only: the GQA strides of b8 h32/16 and h32/8 S_q 2176 S_kv 2304 at L 10.0 stay (h32/8 won by
    9.8%), the MHA stride b9 h16/16 S_q 1664 S_kv 2688 at L 15.0 goes (the one-tile kernel won it by 4.2%), and the
    paired walks of b8 h32/8 S_q 2304 S_kv 2432 (L 10.5) and b9 h16/8 S_q 1664 S_kv 2688 (L 15.0) stay."""
    for shape, kind, mean, routed in [
        ((5, 64, 64, 768, 1536), "stride", 9.5, False),
        ((17, 8, 8, 256, 1280), "stride", 9.5, False),
        ((8, 32, 32, 2176, 2304), "stride", 10.0, True),
        ((13, 16, 16, 1152, 1792), "stride", 10.0, True),
        ((17, 16, 8, 128, 1920), "stride", 15.0, False),
        ((30, 24, 8, 384, 2048), "stride", 15.0, False),
        ((17, 16, 4, 128, 1920), "stride", 15.0, False),
        ((17, 32, 4, 384, 2048), "stride", 15.0, False),
        ((41, 32, 1, 1920, 2816), "stride", 15.0, False),
        ((12, 16, 8, 1280, 2560), "stride", 15.5, True),
        ((8, 24, 8, 256, 2048), "stride", 15.5, True),
        ((6, 32, 8, 256, 2048), "stride", 15.5, True),
        ((41, 64, 8, 768, 2304), "stride", 15.5, True),
        ((41, 32, 1, 1792, 2816), "stride", 15.5, True),
        ((8, 32, 16, 2176, 2304), "stride", 10.0, False),
        ((8, 32, 8, 2176, 2304), "stride", 10.0, False),
        ((9, 16, 16, 1664, 2688), "stride", 15.0, True),
        ((8, 32, 8, 2304, 2432), "walk", 10.5, False),
        ((9, 16, 8, 1664, 2688), "walk", 15.0, False),
    ]:
        assert _plan(shape) == (kind, mean) and _route(shape) is routed, (shape, _plan(shape), routed)


def test_d128_route_splits_strides_by_head_layout():
    """A stride the walk's rule keeps goes by L and by whether its plan has more than one Q head per KV head, and by
    nothing else of the layout: at 132 CTAs, the strides of b8 S_q 2176 S_kv 2560 (L 12.0, between the two stride
    cutoffs) go one tile per CTA at h32/32 (MHA) and stay at h32/16 and h32/8 (GQA ratios 2 and 4), and the MQA stride
    b41 h32/1 S_q 1920 S_kv 2432 (L 12.0) stays; the strides of h32/32, h32/16, h24/8, h32/8 and h32/4 at b8 S_q 2176
    S_kv 3328 (L 18.0, past both stride cutoffs) and b6 S_q 4096 S_kv 20480 (L 144.5, past every cutoff; h32/32 there is
    a measured stride loss of 9.1%) all go; and of the strides of the cc 9.0 suite's
    test_d128_adapter_routes_causal_plans_one_tile_per_cta away from the cutoffs, b6 h32/32 S_q 256 S_kv 512 (MHA, L
    3.5) stays, and b8 h24/8 S_q 256 S_kv 2176 (GQA ratio 3, L 16.5) and b3 h32/16 S_q 384 S_kv 6272 (GQA ratio 2, L
    48.0) go (its fourth, b3 h8/8 S_q 2048 S_kv 10240, is among the measured timings)."""
    for shape, routed in [((8, 32, 32, 2176, 2560), True), ((8, 32, 16, 2176, 2560), False), ((8, 32, 8, 2176, 2560), False), ((41, 32, 1, 1920, 2432), False)]:
        assert _plan(shape) == ("stride", 12.0) and _route(shape) is routed, (shape, _plan(shape), routed)
    for (b, s_q, s_kv), mean in [((8, 2176, 3328), 18.0), ((6, 4096, 20480), 144.5)]:
        for h_q, h_kv in ((32, 32), (32, 16), (24, 8), (32, 8), (32, 4)):
            shape = (b, h_q, h_kv, s_q, s_kv)
            assert _plan(shape) == ("stride", mean) and _route(shape), (shape, _plan(shape))
    for shape, mean, routed in [((6, 32, 32, 256, 512), 3.5, False), ((8, 24, 8, 256, 2176), 16.5, True), ((3, 32, 16, 384, 6272), 48.0, True)]:
        assert _plan(shape) == ("stride", mean) and _route(shape) is routed, (shape, _plan(shape))


def test_d128_route_reads_each_cutoff_on_its_own_kind():
    """_D128_ROUTE_WALK_MAX_L decides paired walks only, _D128_ROUTE_STRIDE_MAX_L_GQA strides with more than one Q head
    per KV head only and _D128_ROUTE_STRIDE_MAX_L_MHA MHA strides only: at 0 every plan a cutoff decides goes and past
    every L every such plan stays, and every other plan keeps its decision. The paired walks b8 h32/8 S_q 2048 at S_kv
    16384 (L 120.5, routed) and S_kv 6144 (L 40.5, kept); the GQA strides b3 h16/8 S_q 3072 S_kv 6528 (L 39.5, routed)
    and b8 h32/8 S_q 2176 S_kv 2304 (L 10.0, kept); the MHA strides b3 h8/8 S_q 2048 S_kv 10240 (L 72.5, routed) and
    b16 h32/32 S 2048 (L 8.5, kept); the plan of at most one item per CTA b1 h4/1 S_q 4096 S_kv 135168 (routed); pairs
    alone b3 h4/1 S_q 1536 S_kv 34304 (L 262.5) and b11 h3/3 S_q 1024 S_kv 6656 (L 48.5), kept."""
    from cudnn.sdpa.fwd import api_dsl

    walks = {(8, 32, 8, 2048, 16384): True, (8, 32, 8, 2048, 6144): False}
    gqa = {(3, 16, 8, 3072, 6528): True, (8, 32, 8, 2176, 2304): False}
    mha = {(3, 8, 8, 2048, 10240): True, (16, 32, 32, 2048, 2048): False}
    others = {(1, 4, 1, 4096, 135168): True, (3, 4, 1, 1536, 34304): False, (11, 3, 3, 1024, 6656): False}
    shipped = {**walks, **gqa, **mha, **others}
    assert {shape: _route(shape) for shape in shipped} == shipped
    for name, own in [("_D128_ROUTE_WALK_MAX_L", walks), ("_D128_ROUTE_STRIDE_MAX_L_GQA", gqa), ("_D128_ROUTE_STRIDE_MAX_L_MHA", mha)]:
        for cutoff, own_go in [(0, True), (10**6, False)]:
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(api_dsl, name, cutoff)
                want = {**shipped, **{shape: own_go for shape in own}}
                assert {shape: _route(shape) for shape in want} == want, (name, cutoff)


def test_d128_route_keeps_pairs_alone_and_routes_one_item_per_cta():
    """Two kinds of plan skip L. Pairs alone (at most one unit per CTA over more items than CTAs) stay at any L: at 132
    CTAs b3 h4/1 S_q 1536 S_kv 34304 (72 pairs on 72 CTAs, L 262.5, a measured win), b2 h8/8 S_q 1152 S_kv 7296 (L 53.0,
    a measured tie) and b11 h3/3 S_q 1024 S_kv 6656 (132 pairs on 132 CTAs, L 48.5); one batch more (b12, 288 items) is
    no longer pairs alone, the walk's rule keeps the stride, at L 48.5, and the route sends it. A plan of at most one item
    per CTA goes at any L: b1 h16/4 S 1024 (128 items, L 4.5), b1 h4/1 S_q 4096 S_kv 135168 (128 items, L 1040.5) and b1
    h4/1 S 4224 (132 items, L 17.0), and at 7 CTAs b1 h7/7 S_q 128 (7 items); one Q tile more than the last at 132 (b1
    h4/1 S 4352, 136 items) is a paired walk at L 17.5 and stays."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2

    for shape, mean in [((3, 4, 1, 1536, 34304), 262.5), ((2, 8, 8, 1152, 7296), 53.0), ((11, 3, 3, 1024, 6656), 48.5)]:
        assert _plan(shape) == ("pairs", mean) and not _route(shape), (shape, _plan(shape))
    assert _walk_module().causal_walk_plan((11, 3, 3, 1024, 6656), True, SCHED_LPT_L2, 1, 132, 2).units == 132
    assert _plan((12, 3, 3, 1024, 6656)) == ("stride", 48.5) and _route((12, 3, 3, 1024, 6656))
    for shape, mean in [((1, 16, 4, 1024, 1024), 4.5), ((1, 4, 1, 4096, 135168), 1040.5), ((1, 4, 1, 4224, 4224), 17.0)]:
        assert _plan(shape) == ("one item per CTA", mean) and _route(shape), (shape, _plan(shape))
    assert _plan((1, 7, 7, 128, 1024), ctas=7) == ("one item per CTA", 8.0) and _route((1, 7, 7, 128, 1024), ctas=7)
    assert _plan((1, 4, 1, 4352, 4352)) == ("walk", 17.5) and not _route((1, 4, 1, 4352, 4352))


def test_d128_route_stays_cheap_on_large_plans():
    """The route runs once per causal plan of the paired walk, in check_support, and runs the walk's rule only where the
    plan's kind can change its answer: never where the plan's stride cutoff keeps a stride (top-left b8 h32/8 S_q 2048
    S_kv 16384 at L 8.5, and top-left b2 h16/4 S_q 4096 S_kv 2048 at L 12.25, a GQA plan) or at or past the walk cutoff
    on a plan of more than 2 * ctas items (b1 h128/1 S 262144 at L 1024.5, where the rule alone takes about 7.6 s on a
    laptop, b1 h128/128 S 262144, and the walk rule's other large plans past L 48), always between the plan's stride
    cutoff and the walk cutoff (top-left b2 h16/16 S_q 4096 S_kv 2048 at L 12.25, an MHA plan, and b32 h128/8 S 8192 and
    b32 h128/128 S 8192 at L 32.5). On these plans it stays far below a compile's seconds: at most 0.3 s each on a
    laptop (Python 3.14); the bound leaves room for a slow host. Not timed here: top-left b1 h128/1 S_q 262144 S_kv 4096,
    a paired walk at L 31.8 between the cutoffs, where the rule runs and takes about 5.6 s on a laptop."""
    import time
    import types

    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2
    from cudnn.sdpa.fwd import api_dsl

    mod, calls = _walk_module(), []

    def walk_plan(*args, **kwargs):
        calls.append(args[0])
        return mod.causal_walk_plan(*args, **kwargs)

    spy = types.SimpleNamespace(causal_kv_tiles=mod.causal_kv_tiles, causal_walk_plan=walk_plan)
    for shape, bottom_right, rule_runs in [
        ((32, 32, 8, 32000, 32020), True, False),
        ((8, 64, 8, 65536, 32768), False, False),
        ((16, 32, 8, 16000, 16010), True, False),
        ((2, 16, 4, 4096, 2048), False, False),
        ((2, 16, 16, 4096, 2048), False, True),
        ((32, 128, 8, 16384, 16384), True, False),
        ((1, 128, 1, 262144, 262144), True, False),
        ((1, 128, 128, 262144, 262144), True, False),
        ((8, 32, 8, 2048, 16384), False, False),
        ((32, 128, 8, 8192, 8192), True, True),
        ((32, 128, 128, 8192, 8192), True, True),
    ]:
        calls.clear()
        start = time.perf_counter()
        api_dsl._sm90_d128_causal_route(spy, shape, bottom_right, SCHED_LPT_L2, shape[1] // shape[2], 132, 2)
        assert time.perf_counter() - start < 3.0, shape
        assert calls == ([shape] if rule_runs else []), (shape, calls)


def test_d128_route_reads_the_mask_alignment():
    """The route reads L under the alignment it is given: b8 h32/8 S_q 2048 S_kv 16384 is a paired walk at L 120.5 under
    the bottom-right mask (routed, a measured loss of the walk's plan) and at L 8.5 under the top-left one (kept), where
    each Q tile walks its own tile count from KV tile 0. check_support passes the record's alignment (the cc 9.0 suite's
    test_d128_adapter_routes_causal_plans_one_tile_per_cta runs both through it)."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT_L2
    from cudnn.sdpa.fwd import api_dsl

    mod, shape = _walk_module(), (8, 32, 8, 2048, 16384)
    for bottom_right, mean, routed in [(True, 120.5, True), (False, 8.5, False)]:
        kv_tiles = mod.causal_kv_tiles(shape, bottom_right)
        assert sum(kv_tiles) / len(kv_tiles) == mean and mod.causal_walk_plan(shape, bottom_right, SCHED_LPT_L2, 4, 132, 2).units > 132
        assert api_dsl._sm90_d128_causal_route(mod, shape, bottom_right, SCHED_LPT_L2, 4, 132, 2) is routed, bottom_right


def test_d128_route_runs_only_under_its_switch():
    """check_support calls the route at one place, under ``if causal_walk and D128_CAUSAL_WALK_ROUTE:``, so with the
    switch False no plan is routed and every causal plan of the paired walk keeps its record; where the route returns
    True the record becomes D128_PERSISTENT_CAUSAL_WALK 0's (persistent and causal_walk cleared) and nothing else
    changes. The route is given what compile() gives causal_walk_plan: the record's template, alignment, scheduler and
    head ratio, the device's SM count and the storage width, so it plans the walk the kernel would run. The head ratio is
    the expression the record's qh_per_kh is built from, so the route's MHA (a ratio of 1) is the kernel's (Q head h
    reads KV head h // qh_per_kh); a PackGQA declaration never reaches the D128 template (the per-flavor tests above).
    The cc 9.0 suite's test_d128_adapter_routes_causal_plans_one_tile_per_cta runs both settings through check_support,
    at 132 CTAs and under LPT_L2 only, so these arguments are pinned here."""
    import ast

    from cudnn.sdpa.fwd import api_dsl

    with open(api_dsl.__file__, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_sm90_d128_causal_route"]
    assert len(calls) == 1, [call.lineno for call in calls]
    inner = parents[calls[0]]
    assert isinstance(inner, ast.If) and inner.test is calls[0] and not inner.orelse, ast.unparse(inner)
    assert [ast.unparse(node) for node in inner.body] == ["self.params = replace(self.params, persistent=False, causal_walk=0)"]
    outer = parents[inner]
    assert isinstance(outer, ast.If) and outer.body == [inner] and not outer.orelse, ast.unparse(outer)
    assert ast.unparse(outer.test) == "causal_walk and D128_CAUSAL_WALK_ROUTE"
    assert not calls[0].keywords and [ast.unparse(arg) for arg in calls[0].args] == [
        "_load_sm90_kernel_module(self.flavor, self.params)",
        "(b, h_q, h_kv, s_q, s_kv)",
        "self.params.causal if self.params.bottom_right is None else self.params.bottom_right",
        "sched",
        "h_q // h_kv",
        "self._persistent_ctas(self.q_desc.device)",
        "self.q_desc.dtype.itemsize",
    ], ast.unparse(calls[0])
    records = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Sm90TemplateParams"]
    assert [ast.unparse(kw.value) for record in records for kw in record.keywords if kw.arg == "qh_per_kh"] == ["h_q // h_kv"]


def test_d128_route_ships_on_with_its_cutoffs():
    """The shipped route and its three cutoffs as committed (D128_CAUSAL_WALK_ROUTE's and the cutoffs' comments give the
    measurements): a paired walk stays below L 48, a stride up to L 15 inclusive with more than one Q head per KV head
    and below L 10 on MHA. Changing any value or adding a cutoff changes this test and the decisions above."""
    from cudnn.sdpa.fwd import api_dsl

    assert api_dsl.D128_CAUSAL_WALK_ROUTE is True
    assert (api_dsl._D128_ROUTE_WALK_MAX_L, api_dsl._D128_ROUTE_STRIDE_MAX_L_GQA, api_dsl._D128_ROUTE_STRIDE_MAX_L_MHA) == (48, 15, 10)
    assert sorted(name for name in vars(api_dsl) if name.startswith("_D128_ROUTE_")) == [
        "_D128_ROUTE_STRIDE_MAX_L_GQA",
        "_D128_ROUTE_STRIDE_MAX_L_MHA",
        "_D128_ROUTE_WALK_MAX_L",
    ]


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
    has a unit; where the rule keeps the stride it launches min(items, persistent_ctas), as a causal_walk=0 record does.
    Host-only: no trace."""
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

    paired = make((2, 8, 2, 1024, 1024), 64, 1)
    assert paired.paired_walk and paired.n_ctas == paired.walk.ctas == 64 and paired.walk.units == 64, paired.walk
    stride = make((8, 24, 8, 256, 2048), 132, 1)
    assert stride.walk is None and not stride.paired_walk and stride.n_ctas == 132
    plain = make((2, 8, 2, 1024, 1024), 64, 0)
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
