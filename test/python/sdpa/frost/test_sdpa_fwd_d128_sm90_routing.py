# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the SM90 per-flavor plumbing (issue #378): no SM90 device needed.

The SM90 row is to lower two templates for exact (128, 128) graphs, chosen by the tile knob: the
D512 body (#2426) on its 64 x 64 tile, and a dedicated D128 template on a 128 x 128 tile that
serves a dense subset. The row's and config_sm90's per-flavor seams are checked here on a
SYNTHETIC row (the shipped row plus the tile domains #378 will set on it), before any D128 kernel
exists, and on every FROST lane: no CI lane runs Hopper. The adapter's seams (its flavor pick, that
flavor's tile, the template it loads) ask the real device and are pinned in the cc 9.0 suite,
test_sdpa_fwd_dsl_sm90.py. The shipped row is not changed by this plumbing; its behavior is the
SM90 suite's to pin.
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
    """The shipped SM90 row plus the D128 template's 128 x 128 tile, spelled as #378 will spell it:
    the (128, 128) d-shape's Q and KV tile domains hold 64 (the D512 body) and 128 (D128)."""
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


# --- Tile domains per flavor (tile_ms_by_d_shape / tile_ns_by_d_shape) -------


def test_effective_tiles_read_the_selected_flavor():
    """An override entry REPLACES the row-wide domain for the d-shape the row selects (through its
    d_envelope_floors, so (120, 72) reads the (128, 128) entry too). A d-shape without an entry reads
    the row-wide domain; the default () reads it everywhere."""
    from cudnn.sdpa.fwd import engines

    row, shipped = _two_flavor_row(), _caps()
    assert shipped.tile_ms_by_d_shape == () and shipped.tile_ns_by_d_shape == ()
    for read in (engines.effective_tile_ms, engines.effective_tile_ns):
        assert read(row, _facts(128)) == read(row, _facts((120, 72))) == frozenset({64, 128})
        assert read(row, _facts(512)) == read(row, _facts((264, 504))) == read(row, _facts(256)) == read(row, _facts(64)) == frozenset({64})
        assert read(shipped, _facts(512)) == read(shipped, _facts(128)) == frozenset({64})


@pytest.mark.parametrize(
    ("d", "tile", "why"),
    [
        (128, (128, 128), None),
        (128, (64, 64), None),
        (512, (64, 64), None),
        (512, (128, 128), "requested tile_m=128 is outside this engine's domain [64]"),
        (256, (128, 128), "requested tile_m=128 is outside this engine's domain [64]"),
        ((264, 504), (128, 128), "requested tile_m=128 is outside this engine's domain [64]"),
    ],
    ids=["d128-128", "d128-64", "d512-own", "d512-pins-128", "d256-pins-128", "envelope-pins-128"],
)
def test_tile_knobs_are_judged_per_flavor(d, tile, why):
    """mismatch() judges a pinned tile against the SELECTED d-shape's domain: (128, 128) admits both
    tiles (each names its template), every other d-shape only the D512 body's 64, so no plan is
    ranked at a tile its kernel would refuse at build."""
    from cudnn.sdpa.fwd import engines

    reason = engines.mismatch(_two_flavor_row(), _facts(d), engines.SdpaFwdKnobs(tile_m=tile[0], tile_n=tile[1]))
    assert reason == why, reason


def test_every_flavor_spelled_out_equals_the_defaults_on_every_shipped_row(monkeypatch):
    """() means "the row-wide domain at every d-shape". Spelling that out (each tile override
    restating the row-wide domain for every d-shape the row has no entry for; the SM100 and SM107
    rows' own tile_ns_by_d_shape entries stay) changes no verdict and no message on any shipped half
    row: a reader that matched raw dims instead of the selected d-shape would decline the envelope
    graphs below, and one that skipped the override would differ at a pinned tile."""
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


# --- config_sm90: flavors, tiles and the template backstop -------


def test_pick_flavor_answers_d512_until_d128_has_a_template():
    """Over the default pool (D512 alone, which the adapter passes while _SM90_KERNEL_FILES has no
    D128 entry) every declaration selects D512. Over the full pool D128 is picked at exactly
    (128, 128), its floor of 127 as the row will declare it, and D512 everywhere else."""
    from cudnn.sdpa.fwd import config_sm90 as c

    pairs = [(a, b) for a in range(8, 513, 8) for b in range(8, 513, 8)]
    assert {c.pick_flavor(*pair) for pair in pairs} == {c.D512_FLAVOR}
    assert {pair for pair in pairs if c.pick_flavor(*pair, c.FLAVORS) == c.D128_FLAVOR} == {(128, 128)}
    assert c.pick_flavor(520, 512, c.FLAVORS) == c.D512_FLAVOR, "total: head_dims_mismatch, not the picker, declines it"
    assert c.FLAVORS == (c.D128_FLAVOR, c.D512_FLAVOR)
    assert c.FLAVOR_TILES == {c.D512_FLAVOR: (c.TILE_M, c.TILE_N), c.D128_FLAVOR: (128, 128)}


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
