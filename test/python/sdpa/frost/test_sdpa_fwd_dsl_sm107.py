# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM107 (Rubin) routing of the f16/bf16 forward SDPA kernels.

The adapter routes cc10.7 graphs to the SM107 sibling modules, which bake the
Rubin geometry the Blackwell modules cannot express — notably the **version-1
tcgen05 SMEM descriptor** every operand tile needs once the per-CTA SMEM budget
passes 256 KiB (a version-0 descriptor's ``start_address`` is 14 bits, so a
buffer at 256 KiB wraps to offset 0 and the MMA silently multiplies whatever
lives at the bottom of SMEM).

These tests are device-independent: everything here happens before any compile,
so they run on any host.  End-to-end numerics ride the shared SM10x suites,
which exercise whichever part is present.
"""

import dataclasses

import pytest

from frost_test_utils import requires_dsl

from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
from cudnn.sdpa.fwd.config_sm100 import TemplateParams

pytestmark = [pytest.mark.L0, requires_dsl]

_E4M3, _BF16_OUT = 0, 2
# (192, 128) is in the sweep for all three dtype families as of 2026-09-09 --
# it is the flavor whose wider K moves SMEM offsets, so it is exactly the one
# a DESC_VERSION check must cover, not skip.
_FLAVORS = [(128, 128), (192, 128), (256, 256), (512, 512)]


def _load(flavor, rubin, *, fp8=False, pertensor=False, **params):
    return _load_sm100_kernel_module(flavor, TemplateParams(**params), fp8=fp8, pertensor=pertensor, rubin=rubin)


def _code_lines(src):
    """Source with whole-line comments dropped -- the prose in these modules
    quotes ``desc_version=1`` when explaining the hazard."""
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


@pytest.mark.parametrize("flavor", _FLAVORS)
def test_f16_routes_to_the_sm107_sibling(flavor):
    """Every f16 flavor has a Rubin sibling, and Blackwell keeps the SM100 one."""
    assert "sm107" in _load(flavor, rubin=True).__name__
    assert "sm100" in _load(flavor, rubin=False).__name__


# Which flavors' MMA-operand SMEM crosses 256 KiB -- the boundary above which a
# version-0 tcgen05 descriptor cannot address the buffer.  d512 alone does: its
# Q(u)O and K(u)V slabs are 128 KiB each, putting the P transfer ring at exactly
# 262144 (and, on the MXFP8 sibling, the scale-factor tiles near 292 KiB).  Keep
# d128/d256 on version 0 so they stay byte-identical to the shipped
# sm107/prefill_d128_fp8.py sibling.
_NEEDS_DESC_V1 = {(512, 512)}

# (kind, loader kwargs) for every SM107 dtype family, so the check below covers
# the QUANTIZED kernels too -- the d512 MXFP8 sibling is where a missing
# version-1 descriptor actually shipped (LSE = +inf, O = NaN on 100% of cells).
# dtype_qkv=_E4M3 is load-bearing on the quantized arms: the FP8/MXFP8 SM107
# bodies pin idesc k_dim=1 and raise unless TILE_K_HW=64, which only the FP8
# dtypes select (config_sm107.tile_k_hw).
_DTYPE_FAMILIES = [
    ("f16", {}),
    ("fp8", {"fp8": True, "pertensor": True, "dtype_qkv": _E4M3, "dtype_o": _BF16_OUT}),
    ("mxfp8", {"fp8": True, "pertensor": False, "dtype_qkv": _E4M3, "dtype_o": _BF16_OUT}),
]


@pytest.mark.parametrize("kind,load_kw", _DTYPE_FAMILIES, ids=[k for k, _ in _DTYPE_FAMILIES])
@pytest.mark.parametrize("flavor", _FLAVORS)
def test_sm107_descriptor_version_matches_the_smem_budget(flavor, kind, load_kw):
    """A version-0 descriptor's ``start_address`` is 14 bits = a 256 KiB
    window; Rubin raises the per-CTA cap to 327 KiB.  An operand buffer at or
    above 256 KiB therefore wraps to offset 0 and the MMA multiplies whatever
    is at the bottom of SMEM -- O comes out EXACTLY zero (or, when the wrapped
    tile is a scale factor, LSE = +inf and O = NaN), no crash, both operands
    provably correct in SMEM, every race probe clean.

    Asserted against the module's own ``DESC_VERSION`` constant, which every
    ``SmemTile`` in that module is constructed with -- so this pins the value
    the kernel actually uses, not the presence of a substring that a comment
    could also supply.

    This is the counter-assertion for the flavors below the boundary: if a tile
    config ever pushes one of them over 256 KiB, move it into _NEEDS_DESC_V1
    and flip its DESC_VERSION -- do not delete the check."""
    mod = _load(flavor, rubin=True, **load_kw)
    want = 1 if flavor in _NEEDS_DESC_V1 else 0
    assert mod.DESC_VERSION == want, f"{mod.__name__}: DESC_VERSION={mod.DESC_VERSION}, expected {want}"


@pytest.mark.parametrize("kind,load_kw", _DTYPE_FAMILIES, ids=[k for k, _ in _DTYPE_FAMILIES])
@pytest.mark.parametrize("flavor", _FLAVORS)
def test_sm107_every_smem_tile_takes_the_module_desc_version(flavor, kind, load_kw):
    """DESC_VERSION is only meaningful if EVERY ``SmemTile`` is built with it.
    The d512 MXFP8 kernel shipped NaN on 100% of cells because its four
    scale-factor tiles were declared under a comment claiming "every operand
    tile here carries desc_version=1" -- without the kwarg.  One re-literalled
    call site is all it takes, so count them."""
    import re

    mod = _load(flavor, rubin=True, **load_kw)
    with open(mod.__file__, encoding="utf-8") as fh:
        code = _code_lines(fh.read())
    n_tiles = len(re.findall(r"\bSmemTile\($", code, re.M))
    assert n_tiles > 0, mod.__name__
    n_wired = code.count("desc_version=DESC_VERSION")
    assert n_wired == n_tiles, f"{mod.__name__}: {n_tiles} SmemTile(s) but {n_wired} wired to DESC_VERSION"
    assert not re.search(r"desc_version=[01]\b", code), f"{mod.__name__}: a re-literalled desc_version bypasses DESC_VERSION"


@pytest.mark.parametrize("flavor", _FLAVORS)
def test_sm107_f16_thd_specialization_matches_the_ported_flavors(flavor):
    """A THD config must TRACE for a ported f16 flavor and be REFUSED for the
    rest -- the config guard is what stops an unported body pairing its 3B+2
    metadata with the shared 4B+4 decode (a wrong-sequence read, not a raise).

    Also pins that the dense specialization stays THD-free everywhere, which is
    what keeps `get_workspace_size() == 0` for a dense stats-less graph."""
    from cudnn.sdpa.fwd.config_sm107 import SM107_F16_THD_SHAPES

    assert _load(flavor, rubin=True, seq_kv_lens_present=True).CFG.THD_VARLEN == 0

    if flavor in SM107_F16_THD_SHAPES:
        assert _load(flavor, rubin=True, seq_kv_lens_present=True, thd_varlen=1).CFG.THD_VARLEN == 1
    else:
        with pytest.raises(ValueError, match="THD/varlen"):
            _load(flavor, rubin=True, seq_kv_lens_present=True, thd_varlen=1)


def test_rubin_f16_never_picks_a_flavor_it_has_no_kernel_for():
    """The adapter must narrow the f16 candidate pool BY ARCH LINE, or a graph
    picks a flavor with no Rubin module and dies with a KeyError inside module
    loading instead of riding the next covering envelope.

    PARTIALLY INVERTED 2026-09-04: d192xd128 now HAS a Rubin sibling
    (``sm107/prefill_d192_d128_f16.py`` -- the same body as d128 with
    ``make_cfg_d192``), so a d=192 f16 graph lowers onto the NATIVE kernel
    instead of riding the d256 envelope.

    FULLY INVERTED 2026-09-09: the quantized lines gained their d192 siblings
    too, so every Rubin pool now covers all four flavors.  The pool-NARROWING
    invariant is what this test really pins and it is unchanged -- it is what
    keeps a graph off a flavor with no Rubin module, and it must survive every
    future arch line that ships a partial flavor set.
    """
    from cudnn.sdpa.fwd.api_dsl import (
        _SM100_FLAVORS,
        _SM107_FP8_KERNEL_FILES,
        _SM107_KERNEL_FILES,
        _pick_flavor,
    )

    rubin_pool = tuple(f for f in _SM100_FLAVORS if f in _SM107_KERNEL_FILES)
    assert _pick_flavor(192, 128, rubin_pool) == (192, 128)
    # Blackwell keeps its native d192xd128 kernel.
    assert _pick_flavor(192, 128, None) == (192, 128)
    # INVERTED: the quantized Rubin lines gained their d192 siblings, so a
    # d=192 FP8 graph lands on the NATIVE kernel instead of an envelope.
    rubin_fp8_pool = tuple(f for f in _SM100_FLAVORS if f in _SM107_FP8_KERNEL_FILES)
    assert (192, 128) in rubin_fp8_pool
    assert _pick_flavor(192, 128, rubin_fp8_pool) == (192, 128)
    # The narrowing itself still has teeth: every Rubin pool is a SUBSET of the
    # Blackwell flavor list, and _pick_flavor is only ever handed the pool whose
    # kernel files exist.  A pool built from a map that lacks a flavor must not
    # offer it -- this is the guard that survives the next partial arch line.
    assert set(rubin_pool) <= set(_SM100_FLAVORS) and set(rubin_fp8_pool) <= set(_SM100_FLAVORS)
    assert _pick_flavor(192, 128, tuple(f for f in rubin_fp8_pool if f != (192, 128))) != (192, 128)
    # Every flavor Rubin can pick has a module ON DISK -- the map is the only
    # thing standing between _pick_flavor and a KeyError/ImportError deep in
    # module loading, so check the file, not the map key it came from.
    import pathlib

    import cudnn.sdpa.fwd.kernels as _k

    kdir = pathlib.Path(_k.__file__).parent
    for flavor in rubin_pool:
        assert (kdir / _SM107_KERNEL_FILES[flavor]).is_file(), (flavor, _SM107_KERNEL_FILES[flavor])


# --------------------------------------------------------------------------
# Capability row: what sdpa_fwd_prefill_sm107 accepts, and what it declines.
# Every accept below corresponds to a case MEASURED on Rubin against an
# explicit fp32 reference; every decline corresponds to machinery the kernels
# genuinely lack.  Declines are asserted, never skipped -- and the ones marked
# INVERTS-WHEN are counter-assertions to flip, not delete, when the feature
# lands.
# --------------------------------------------------------------------------


def _f16_facts(**kw):
    import cudnn
    from cudnn.sdpa import graph_analyzer as ga

    base = dict(
        b=2,
        h_q=8,
        h_kv=8,
        s_q=512,
        s_kv=512,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.HALF,
        dtype_o=cudnn.data_type.HALF,
        device_cc=(10, 7),
    )
    base.update(kw)
    return ga.SdpaGraphFacts(**base)


def _caps(name):
    from cudnn.sdpa.fwd import engines

    return {s.name: s.capabilities for s in engines.ENGINE_SPECS}[name]


def test_sm107_f16_row_owns_the_rubin_lane():
    """One row per ARCH LINE: sm100 stops at cc 10.6, sm107 starts at 10.7."""
    from cudnn.sdpa.fwd import engines

    sm107, sm100 = _caps("sdpa_fwd_prefill_sm107"), _caps("sdpa_fwd_prefill_sm100")
    assert engines.mismatch(sm107, _f16_facts()) is None
    assert "SM107-119" in engines.mismatch(sm107, _f16_facts(device_cc=(10, 0)))
    assert "SM100-106" in engines.mismatch(sm100, _f16_facts(device_cc=(10, 7)))


@pytest.mark.parametrize("d", [(128, 128), (256, 256), (512, 512)])
def test_sm107_f16_accepts_every_native_shape(d):
    from cudnn.sdpa.fwd import engines

    assert engines.mismatch(_caps("sdpa_fwd_prefill_sm107"), _f16_facts(d_qk=d[0], d_v=d[1])) is None


def test_sm107_f16_accepts_both_dtypes():
    """A frozenset field is one claim PER MEMBER -- exercise bf16, not just the
    fp16 the suite happens to default to."""
    import cudnn
    from cudnn.sdpa.fwd import engines

    caps = _caps("sdpa_fwd_prefill_sm107")
    for dt in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16):
        assert engines.mismatch(caps, _f16_facts(dtype=dt, dtype_o=dt)) is None


@pytest.mark.parametrize(
    "feature",
    [
        dict(causal=True),
        dict(causal=True, bottom_right=True),
        dict(right_band_widening=True),
        dict(window_left=128, causal=True),
        dict(padded=True),
        dict(has_sink=True),
        dict(h_kv=2),  # GQA 4:1
    ],
)
def test_sm107_f16_accepts_the_measured_feature_set(feature):
    """Each of these was validated on Rubin (cos = 1.000000)."""
    from cudnn.sdpa.fwd import engines

    assert engines.mismatch(_caps("sdpa_fwd_prefill_sm107"), _f16_facts(**feature)) is None


def test_sm107_f16_serves_d192_on_its_native_kernel():
    """d192xd128 is a NATIVE f16 Rubin flavor (``sm107/prefill_d192_d128_f16.py``
    -- the d128 body with ``make_cfg_d192``), not an envelope ride.

    Both halves of that claim are pinned, because they can drift apart
    silently: the row must LIST the shape (an envelope acceptance would pass
    this assertion too, which is why the d_shapes membership is asserted
    separately), and the adapter must PICK the native kernel for it
    (test_rubin_f16_never_picks_a_flavor_it_has_no_kernel_for).  The FP8 and
    MXFP8 Rubin lines genuinely lack the sibling -- see
    test_sdpa_fp8_sm107.py, where d192 is DECLINED rather than padded."""
    from cudnn.sdpa.fwd import engines

    caps = _caps("sdpa_fwd_prefill_sm107")
    assert (192, 128) in caps.d_shapes
    assert engines.mismatch(caps, _f16_facts(d_qk=192, d_v=128)) is None


def test_sm107_f16_thd_is_served_on_every_flavor():
    """FULLY INVERTED 2026-09-09: every f16 flavor now serves THD.

    The row must cover exactly `d_shapes` -- no more (a shape with no kernel
    would KeyError inside module loading) and no less (a served shape left out
    is a capability the engine silently declines).  Asserted on real facts
    objects, not the dataclass field, so it walks the path mismatch() takes.

    This keeps `thd_d_shapes` as a named SET rather than collapsing to `True`,
    because that is what makes the next partial arch line expressible without
    reintroducing a boolean that cannot describe it."""
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.config_sm107 import SM107_F16_THD_SHAPES

    caps = _caps("sdpa_fwd_prefill_sm107")
    assert caps.thd is True
    assert caps.thd_d_shapes is SM107_F16_THD_SHAPES
    assert SM107_F16_THD_SHAPES == caps.d_shapes, "every served f16 shape must serve THD, and no unserved one may"
    for d_qk, d_v in sorted(caps.d_shapes):
        assert engines.mismatch(caps, _f16_facts(thd=True, padded=True, d_qk=d_qk, d_v=d_v)) is None, (d_qk, d_v)


def test_sm107_f16_declines_split_kv_and_pack_gqa():
    """Neither is wired in these kernels (no SplitHelpers, no PackGQA path)."""
    from cudnn.sdpa.fwd import engines

    caps = _caps("sdpa_fwd_prefill_sm107")
    assert caps.split_kv_supported is False
    assert caps.pack_gqas == frozenset({False})
    assert engines.mismatch(caps, _f16_facts(), engines.SdpaFwdKnobs(split_kv=2)) is not None
    assert engines.mismatch(caps, _f16_facts(), engines.SdpaFwdKnobs(pack_gqa=True)) is not None


def test_sm107_rows_carry_the_padded_stats_trim():
    """Every Rubin template carries the per-batch seq_len_q trim (padded q
    rows write LSE=-inf, O=0), so the rows serve dense padded Stats and the
    Capabilities record has no dense_seq_q_trim field left to declare."""
    from cudnn.sdpa.fwd import engines

    for row in ("sdpa_fwd_prefill_sm107", "sdpa_fwd_prefill_sm107_fp8", "sdpa_fwd_prefill_sm107_mxfp8"):
        assert _caps(row).padded_stats is True, row
    assert not hasattr(engines.Capabilities, "dense_seq_q_trim")


def test_sm107_rows_serve_natural_scheduling_only():
    """The ROW-WIDE floor of every Rubin row stays NATURAL-only: every LPT
    variant is claimed per flavor (`sched_policies_by_d_shape`) where it is
    validated.  SCHED_LPT_L2 needs `qh_per_kh` / `seqlen_kv` at every decode
    call site, which only the d128 / d192x128 FP8 and MXFP8 kernels pass, so it
    may appear ONLY on those flavors -- INVERTED 2026-09-14 (it used to appear
    on none, and the MXFP8 row claimed nothing beyond NATURAL).

    (This test used to assert LPT was "not honored" by the ported kernels; the
    cause was the dropped `lpt_q_tiles_in_cga_units` argument, fixed in #1001.
    The per-flavor claims are pinned by the *_advertises_lpt_* tests.)"""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL

    lpt_l2_flavors = {
        "sdpa_fwd_prefill_sm107": set(),
        "sdpa_fwd_prefill_sm107_fp8": {(128, 128), (192, 128)},
        "sdpa_fwd_prefill_sm107_mxfp8": {(128, 128), (192, 128)},
    }
    for row, l2_shapes in lpt_l2_flavors.items():
        caps = _caps(row)
        assert caps.sched_policies == frozenset({SCHED_NATURAL}), row
        assert SCHED_LPT not in caps.sched_policies, row
        assert SCHED_LPT_L2 not in caps.sched_policies, row
        claimed_l2 = {shape for shape, dom in caps.sched_policies_by_d_shape if SCHED_LPT_L2 in dom}
        assert claimed_l2 == l2_shapes, (row, claimed_l2)
    assert dict(_caps("sdpa_fwd_prefill_sm107_mxfp8").sched_policies_by_d_shape) == {
        (128, 128): frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}),
        (192, 128): frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}),
    }, "MXFP8 claims LPT and LPT_L2 exactly where its kernels thread the decode inputs"


def test_sched_points_falls_back_along_the_preference_order():
    """A row whose heuristic primary is outside its DOMAIN must fall back along
    the same preference order, not drop to NATURAL -- and must not list a policy
    twice.  The old form returned [NATURAL, LPT, NATURAL] for a causal Rubin FP8
    graph: NATURAL first (losing the LPT balancing) and a duplicate autotune
    slot.  Rows whose primary IS in domain are unaffected."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
    from cudnn.sdpa.fwd import engines, heuristics

    caps = {s.name: s.capabilities for s in engines.ENGINE_SPECS}
    rubin_fp8 = caps["sdpa_fwd_prefill_sm107_fp8"]
    facts = _f16_facts()
    facts = facts.__class__(**{**facts.__dict__, "causal": True, "is_fp8": True, "device_cc": (10, 7)})
    points = heuristics._sched_points(rubin_fp8, facts)
    assert len(points) == len(set(points)), points
    assert SCHED_LPT_L2 not in rubin_fp8.sched_policies
    # d128 FP8 claims {NATURAL, LPT, LPT_L2} (2026-09-14).  These facts have
    # h_q == h_kv (no GQA) and a tiny grid, so the Rubin rule leads with LPT
    # and keeps both other policies as autotune runners.
    assert points == [SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL], points

    # The per-flavor claim must REACH the ranking: at (256, 256) the FP8 row's
    # effective domain is {NATURAL, LPT}, the causal primary (LPT_L2) is out of
    # domain, so the fallback walks the preference order -- LPT first, NATURAL
    # as the autotune alternative.  Before the heuristics read the flavor's
    # domain this returned [NATURAL] and LPT was never proposed.
    facts256 = facts.__class__(**{**facts.__dict__, "d_qk": 256, "d_v": 256})
    assert heuristics._sched_points(rubin_fp8, facts256) == [SCHED_LPT, SCHED_NATURAL]

    # (192, 128) FP8 claims LPT_L2 too.  Without GQA the Rubin rule still leads
    # with LPT on this few-wave grid; give the KV heads Q heads to share and
    # the causal primary becomes LPT_L2, which IS in domain now, so the ranking
    # leads with it and keeps both fallbacks as autotune runners.  A mask-free
    # graph gains nothing from either remap: NATURAL alone.
    facts192 = facts.__class__(**{**facts.__dict__, "d_qk": 192, "d_v": 128})
    assert heuristics._sched_points(rubin_fp8, facts192) == [SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL]
    gqa192 = facts.__class__(**{**facts192.__dict__, "h_kv": 2})
    assert heuristics._sched_points(rubin_fp8, gqa192) == [SCHED_LPT_L2, SCHED_LPT, SCHED_NATURAL]
    dense192 = facts.__class__(**{**facts192.__dict__, "causal": False})
    assert heuristics._sched_points(rubin_fp8, dense192) == [SCHED_NATURAL]

    # ...and the multi-element case, which is the branch the single-element
    # shortcut above skips: with a causal primary (LPT_L2, a GQA graph) OUT of
    # a {NATURAL, LPT} domain, the fallback must walk the preference order
    # rather than dropping straight to NATURAL -- the bug this function was
    # fixed for.  The per-shape claims are cleared so the row-wide domain is
    # the one in force.
    widened = dataclasses.replace(rubin_fp8, sched_policies=frozenset({SCHED_NATURAL, SCHED_LPT}), sched_policies_by_d_shape=())
    widened_points = heuristics._sched_points(widened, facts.__class__(**{**facts.__dict__, "h_kv": 2}))
    assert widened_points[0] == SCHED_LPT, widened_points
    assert set(widened_points) == {SCHED_NATURAL, SCHED_LPT}, widened_points
    assert len(widened_points) == len(set(widened_points)), widened_points


def test_sm107_rows_claim_optional_stats():
    """INVERTED 2026-09-08: every Rubin kernel now const_expr's its LSE store
    out on a None ``lse_tensor`` and ``compile(has_lse=False)`` binds no dummy
    buffer, so a stats-less graph reports ``get_workspace_size() == 0``.  The
    FP8 row already claimed this from the shared spec -- which was a LIE for the
    d256/d512 FP8 kernels it was widened to in the session-5 work, and is now
    true."""
    for row in ("sdpa_fwd_prefill_sm107", "sdpa_fwd_prefill_sm107_fp8", "sdpa_fwd_prefill_sm107_mxfp8"):
        assert _caps(row).lse_optional is True, row


def test_rubin_mxfp8_forward_row_exists():
    """The Rubin MXFP8 row (slot 16) and Blackwell both have a d512 kernel.
    INVERTED 2026-09-09: d192xd128 gained a
    Rubin MXFP8 sibling, at cga2 ONLY -- at cga1 that flavor's scale-factor
    tiles start past the 256 KiB version-0 tcgen05 descriptor window.  SM100's
    row stays capped at cc 10.6."""
    from cudnn.sdpa.fwd import engines

    caps = _caps("sdpa_fwd_prefill_sm107_mxfp8")
    assert caps.is_mxfp8 is True
    assert caps.d_shapes == frozenset({(128, 128), (192, 128), (256, 256), (512, 512)})
    assert (512, 512) in _caps("sdpa_fwd_prefill_sm100_mxfp8").d_shapes
    assert (192, 128) in caps.d_shapes
    # (192, 128) takes NO cgas_by_d_shape entry, so it inherits the row default
    # cgas={2}.  That is a DESCRIPTOR constraint, not a tuning choice: at cga1
    # the four SF tiles land at 256-278 KiB, past the version-0 tcgen05
    # descriptor window, and the UTCCP would read Q data as scale factors
    # (LSE=+inf, O=NaN on every cell).  SM100 serves the same shape at {1, 2}.
    assert caps.cgas == frozenset({2})
    assert all(shape != (192, 128) for shape, _ in caps.cgas_by_d_shape)
    assert _caps("sdpa_fwd_prefill_sm100_mxfp8").cgas_by_d_shape != caps.cgas_by_d_shape
    # Exact-native only: the SF tensors are not zero-padded.
    assert caps.d_pad_multiple == 0
    # The machinery the ported kernels lack stays declined.
    assert caps.thd is False and caps.split_kv_supported is False
    assert caps.pack_gqas == frozenset({False})
    assert _caps("sdpa_fwd_prefill_sm100_mxfp8").sm_hi == 106


def test_sm107_quantized_rows_serve_d192_on_their_native_kernels():
    """ACCEPT side of the 2026-09-09 d192 addition, one case PER DTYPE MEMBER of
    each row's frozenset -- a row listing two dtypes is making two claims, and a
    suite that exercises only E4M3 leaves the other as an untested assertion
    (contract rule 12).

    The shape must land on the NATIVE (192, 128) kernel, not an envelope: the
    d256 flavor's floor (255) would otherwise swallow it onto a padded path
    nobody validated."""
    import cudnn
    from cudnn.sdpa.fwd import engines

    for row, is_mx in (("sdpa_fwd_prefill_sm107_fp8", False), ("sdpa_fwd_prefill_sm107_mxfp8", True)):
        caps = _caps(row)
        assert (192, 128) in caps.d_shapes, row
        for dt in sorted(caps.dtypes, key=lambda d: int(d)):
            facts = _quant_facts(d_qk=192, d_v=128, dtype=dt, is_mx=is_mx)
            assert engines.mismatch(caps, facts) is None, (row, dt)
            assert engines._selected_d_shape(caps, facts) == (192, 128), (row, dt)


def test_sm107_d192_mxfp8_is_cga2_only_because_of_the_descriptor_window():
    """REJECT side, and the reason is a DESCRIPTOR constraint rather than taste.

    At cga1 the K/V rings are not halved, so this flavor's four scale-factor
    tiles start at 256-278 KiB -- past the 256 KiB version-0 tcgen05 descriptor
    window, where ``start_address`` wraps to 0 and the UTCCP copies Q DATA bytes
    into the SF TMEM columns (LSE=+inf, O=NaN on 100% of cells, at every shape).

    Two independent guards, and this test pins BOTH: the row keeps (192, 128) on
    the default ``cgas={2}`` so a graph cannot request cga1, and the kernel body
    raises at import if it is ever handed one anyway.  INVERTS-WHEN DESC_VERSION
    is derived from the layout and the version-1 SF path is validated on Rubin
    -- which is not free: desc_version=1 on the d128/d256 MXFP8 tiles turned 21
    green tests red (2026-09-08)."""
    import pytest as _pytest
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps = _caps("sdpa_fwd_prefill_sm107_mxfp8")
    facts = _quant_facts(d_qk=192, d_v=128, is_mx=True)
    # Guard 1 -- knob domain. cga2 is honored, cga1 makes the plan ineligible.
    assert engines.mismatch(caps, facts, SdpaFwdKnobs(cga=2)) is None
    assert engines.mismatch(caps, facts, SdpaFwdKnobs(cga=1)) is not None
    # SM100 serves the same shape at BOTH widths -- so this is Rubin-specific,
    # which is what makes it worth pinning rather than assuming.
    assert 1 in engines.effective_cgas(_caps("sdpa_fwd_prefill_sm100_mxfp8"), facts, 1)

    # Guard 2 -- the kernel body itself, so the constraint survives a row edit.
    with _pytest.raises(ValueError, match="CTA_MMA must be 2"):
        _load((192, 128), rubin=True, fp8=True, pertensor=False, dtype_qkv=_E4M3, dtype_o=_BF16_OUT, cta_mma=1)


def _quant_facts(**kw):
    import cudnn
    from cudnn.sdpa import graph_analyzer as ga

    is_mx = kw.pop("is_mx", False)
    base = dict(
        b=2,
        h_q=4,
        h_kv=4,
        s_q=384,
        s_kv=384,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.BFLOAT16,
        device_cc=(10, 7),
    )
    base.update(kw)
    base["is_mxfp8" if is_mx else "is_fp8"] = True
    return ga.SdpaGraphFacts(**base)


def test_sm107_ones_tile_stays_inside_the_v0_descriptor_window():
    """The row-sum "ones" tile is allocated LAST and read as an MMA operand, so
    its OFFSET -- not its size -- has to stay under 256 KiB while DESC_VERSION
    is 0.  At cga1 the K/V rings are not halved, and with a HALF-PRECISION O
    the d192 tile lands at 272 KiB: its version-0 descriptor wraps to offset 0
    and the Sigma MMA multiplies sQ instead of all-ones, so `total_sum` becomes
    a function of Q -- wrong LSE and wrong O on every row, no crash.

    Caught in review, not by a test, because the kernel docstring ARGUED cga1
    was safe from a hand-summed figure that omitted two buffers.  The guard is
    therefore derived from the allocator's own constants; this pins that it
    fires exactly on the unsafe config and on nothing else, so the safe ones
    cannot be walled off by a future over-broad tightening."""
    import pytest as _pytest

    _E4M3_OUT, _BF16 = 0, 2
    # (id, flavor, cta_mma, dtype_o, per-tensor?).  The MXFP8 rows matter on
    # their own: those kernels carry FOUR scale-factor slabs the per-tensor
    # sibling does not, so their ones tile sits ~7 KiB higher and a guard that
    # forgot the SF slabs would under-report the offset rather than fire.
    safe = [
        ("d128", (128, 128), 1, _E4M3_OUT, True),
        ("d128", (128, 128), 1, _BF16, True),
        ("d128", (128, 128), 2, _BF16, True),
        ("d192", (192, 128), 1, _E4M3_OUT, True),
        ("d192", (192, 128), 2, _BF16, True),
        ("d128-mx", (128, 128), 1, _E4M3_OUT, False),
        ("d128-mx", (128, 128), 1, _BF16, False),
        ("d128-mx", (128, 128), 2, _E4M3_OUT, False),
        ("d128-mx", (128, 128), 2, _BF16, False),
        ("d192-mx", (192, 128), 2, _E4M3_OUT, False),
        ("d192-mx", (192, 128), 2, _BF16, False),
    ]
    for _id, flavor, cta, dto, pertensor in safe:
        mod = _load(flavor, rubin=True, fp8=True, pertensor=pertensor, dtype_qkv=_E4M3, dtype_o=dto, cta_mma=cta)
        assert mod._ONES_SMEM_OFFSET < 256 * 1024, (_id, cta, dto, mod._ONES_SMEM_OFFSET)

    # d192 MXFP8 x cga1 never reaches the ones guard: its own CTA_MMA raise
    # fires first (the four SF tiles cross the window before the ones tile
    # does).  Pin WHICH guard speaks, so a future reordering cannot silently
    # swap a precise diagnostic for a vaguer one.
    with _pytest.raises(ValueError, match="CTA_MMA must be 2"):
        _load((192, 128), rubin=True, fp8=True, pertensor=False, dtype_qkv=_E4M3, dtype_o=_BF16, cta_mma=1)

    # The one unsafe combination: d192 x cga1 x half-precision O.
    with _pytest.raises(ValueError, match="version-0 tcgen05 descriptor window"):
        _load((192, 128), rubin=True, fp8=True, pertensor=True, dtype_qkv=_E4M3, dtype_o=_BF16, cta_mma=1)


def test_sm107_mxfp8_tmem_map_fills_the_rubin_allocation_exactly():
    """The row-sum (Sigma) columns took the 32 TMEM columns the scale-factor
    tiles used to leave free: at d192 the map ends EXACTLY at the 576-column
    Rubin allocation, at d128 at 564.  The import-time raise catches an
    overflow; this pins the layout so a re-literalled offset that still fits
    cannot drift unnoticed."""
    for flavor, end in (((192, 128), 576), ((128, 128), 564)):
        mod = _load(flavor, rubin=True, fp8=True, pertensor=False, dtype_qkv=_E4M3, dtype_o=_BF16_OUT, cta_mma=2)
        assert mod.LAYOUT.TOTAL_COLS == 576, flavor
        assert mod.LAYOUT.SF_V_OFF + mod.SF_TMEM_COLS_V == end, (flavor, mod.LAYOUT.SF_V_OFF, mod.SF_TMEM_COLS_V)

    # ...and the STANDALONE wrapper must decline it too, rather than letting a
    # bare ValueError escape from compile() (contract rule 8b' -- the kernel
    # raise and the wrapper gate are two enforcement points for one fact).
    # Assert the wrapper's OWN decision function, so this holds from any host:
    # the Rubin arm is unreachable on a non-cc-10.7 box, which is exactly the
    # kind of branch that rots untested.
    from cudnn.sdpa.fwd.api_dsl import supported_cgas_for

    assert supported_cgas_for((192, 128), fp8=True, device_cc=(10, 7)) == (2,)
    # ...and check_support must ACT on that, not merely compute it.  The helper
    # was extracted in review and the `_value_error_if` that consumed it got
    # dropped in the same edit, so `supported_cgas` was computed and discarded --
    # an explicit cga=1 then cleared support validation and died inside
    # compile(), which is exactly the rule-8b' failure the helper exists to
    # prevent.  Pin the CONSUMPTION, since the value alone was already correct.
    import inspect

    from cudnn.sdpa.fwd import api_dsl as _api_dsl

    _src = inspect.getsource(_api_dsl.SdpaFwdDslSm100.check_support)
    assert "supported_cgas_for(" in _src, "check_support no longer derives the CGA domain"
    assert "self.cga not in supported_cgas" in _src, "check_support computes supported_cgas but never validates self.cga against it"

    # ...and only there: Blackwell keeps both widths, and the f16 Rubin path is
    # not narrowed (it has no quantized SF or ones tile to push over the line).
    assert supported_cgas_for((192, 128), fp8=True, device_cc=(10, 0)) == (1, 2)
    assert supported_cgas_for((192, 128), fp8=False, device_cc=(10, 7)) == (1, 2)
    assert supported_cgas_for((128, 128), fp8=True, device_cc=(10, 7)) == (2,)
    assert supported_cgas_for((256, 256), fp8=True, device_cc=(10, 7)) == (1,)


# --- LPT on Rubin, advertised PER D-SHAPE -----------------------------------
#
# `sched_policies` is row-wide but LPT support is per-KERNEL: the SM107 port
# dropped `lpt_q_tiles_in_cga_units=True` on 9 of 11 flavors, and without it the
# LPT decode claims no tile and the kernel writes nothing. The argument is
# restored everywhere; only the flavors validated under LPT are ADVERTISED.


@pytest.mark.L0
def test_sm107_advertises_lpt_only_for_the_validated_d_shape():
    """(256, 256) serves LPT; the row-wide default stays NATURAL-only.

    An accept AND a reject, because a capability that is only ever exercised on
    its accepting side is an untested assertion (engine contract, Rule 9).
    """
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
    from cudnn.sdpa.fwd import engines

    caps = _caps("sdpa_fwd_prefill_sm107")
    assert caps.sched_policies == frozenset({SCHED_NATURAL}), "the row-wide floor must stay NATURAL"

    d256 = engines.effective_sched_policies(caps, _f16_facts(d_qk=256, d_v=256))
    assert SCHED_LPT in d256, "the d256 kernel honours LPT and must advertise it"

    d128 = engines.effective_sched_policies(caps, _f16_facts(d_qk=128, d_v=128))
    assert SCHED_LPT not in d128, "d128 is unvalidated under LPT; advertising it would be dishonest"

    # LPT_L2 is a separate, still-open gap: its decode needs qh_per_kh and
    # seqlen_kv, which the SM107 call sites do not pass. No flavor claims it.
    for shape in ((128, 128), (256, 256), (512, 512)):
        assert SCHED_LPT_L2 not in engines.effective_sched_policies(caps, _f16_facts(d_qk=shape[0], d_v=shape[1]))


def test_sm107_fp8_advertises_lpt_only_for_the_validated_d_shape():
    """The FP8 twin of the f16 test above: (256, 256), (192, 128) and -- since
    2026-09-14 -- (128, 128) serve LPT (validated through the standalone
    adapter, bit-identical to NATURAL -- see the row's comment); d512 does
    not; the row-wide floor stays NATURAL.  LPT_L2 needs qh_per_kh / seqlen_kv
    at every decode call site: the d128 and d192x128 kernels thread them and
    both claim it; d256 / d512 pass neither argument."""
    import cudnn
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
    from cudnn.sdpa.fwd import engines

    caps = _caps("sdpa_fwd_prefill_sm107_fp8")
    assert caps.sched_policies == frozenset({SCHED_NATURAL}), "the row-wide floor must stay NATURAL"
    fp8 = dict(dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16, is_fp8=True)
    for shape in ((256, 256), (192, 128), (128, 128)):
        assert SCHED_LPT in engines.effective_sched_policies(caps, _f16_facts(d_qk=shape[0], d_v=shape[1], **fp8)), shape
    assert SCHED_LPT not in engines.effective_sched_policies(caps, _f16_facts(d_qk=512, d_v=512, **fp8))
    for shape in ((192, 128), (128, 128)):
        assert SCHED_LPT_L2 in engines.effective_sched_policies(caps, _f16_facts(d_qk=shape[0], d_v=shape[1], **fp8)), shape
    for shape in ((256, 256), (512, 512)):
        assert SCHED_LPT_L2 not in engines.effective_sched_policies(caps, _f16_facts(d_qk=shape[0], d_v=shape[1], **fp8)), shape


@pytest.mark.L0
def test_sm107_fp8_lpt_knob_is_honored_or_ineligible_per_d_shape():
    """Requesting LPT on the FP8 row: ACCEPTED at d256, d192xd128 and d128,
    DECLINED (typed, naming the knob) at d512; LPT_L2: ACCEPTED at d192xd128
    and d128 only -- never silently downgraded (engine contract, Rule 4)."""
    import cudnn
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps = _caps("sdpa_fwd_prefill_sm107_fp8")
    fp8 = dict(dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16, is_fp8=True)
    for shape in ((256, 256), (192, 128), (128, 128)):
        assert engines.mismatch(caps, _f16_facts(d_qk=shape[0], d_v=shape[1], **fp8), SdpaFwdKnobs(sched_policy=SCHED_LPT)) is None, shape
    why = engines.mismatch(caps, _f16_facts(d_qk=512, d_v=512, **fp8), SdpaFwdKnobs(sched_policy=SCHED_LPT))
    assert why is not None and "sched_policy" in why
    for shape in ((192, 128), (128, 128)):
        assert engines.mismatch(caps, _f16_facts(d_qk=shape[0], d_v=shape[1], **fp8), SdpaFwdKnobs(sched_policy=SCHED_LPT_L2)) is None, shape
    for shape in ((256, 256), (512, 512)):
        why = engines.mismatch(caps, _f16_facts(d_qk=shape[0], d_v=shape[1], **fp8), SdpaFwdKnobs(sched_policy=SCHED_LPT_L2))
        assert why is not None and "sched_policy" in why, shape


@pytest.mark.L0
def test_sm107_lpt_knob_is_honored_or_ineligible_per_d_shape():
    """Requesting LPT must be ACCEPTED at d256 and DECLINED at d128 -- never
    silently downgraded to NATURAL (engine contract, Rule 4)."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps = _caps("sdpa_fwd_prefill_sm107")
    assert engines.mismatch(caps, _f16_facts(d_qk=256, d_v=256), SdpaFwdKnobs(sched_policy=SCHED_LPT)) is None
    why = engines.mismatch(caps, _f16_facts(d_qk=128, d_v=128), SdpaFwdKnobs(sched_policy=SCHED_LPT))
    assert why is not None and "sched_policy" in why


@pytest.mark.L0
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_sm107_d256_desc_version_follows_the_stages_kv_layout(depth):
    """STAGES_KV moves the last V stage past the 14-bit tcgen05 address window
    at depth 4, so DESC_VERSION must be DERIVED, not a literal.

    This is the guard for a silent wrong answer: at depth 4 with a version-0
    descriptor the last two V stages alias the bottom of SMEM and the result is
    wrong from the KV iteration that first touches one (measured cos 0.68 / 0.50
    / 0.46 at n_kv 3/4/8).
    """
    from dataclasses import replace

    import cudnn.sdpa.fwd.kernels.sm107.prefill_d256_f16 as kern

    cfg = replace(kern.CFG, STAGES_KV=depth)
    want = 1 if depth >= 4 else 0
    assert int(kern._needs_desc_v1(cfg)) == want, f"STAGES_KV={depth} must select desc_version {want}"


@pytest.mark.L0
@pytest.mark.parametrize("bad_depth", [0, 1, 5])
def test_sm107_d256_rejects_an_out_of_domain_stages_kv(bad_depth):
    """An out-of-domain STAGES_KV must RAISE, never be silently defaulted.

    `make_cfg_d256` reads `stages_kv` as an OPTIONAL attribute, because the
    shared forward `TemplateParams` has no such field yet -- nothing on the
    shipped path sets it, so this is a latent path, not a live one. It stops
    being latent the day the knob is declared, and the failure it would have
    then is the quiet kind: a truthiness test maps an explicit 0 onto the
    default 2, so the engine runs a depth the caller did not ask for and the
    2..4 check below never sees it. That is knob SUBSTITUTION, which the
    engine contract forbids -- a knob is honored or the engine is ineligible.

    Pinning 0 specifically: 1 and 5 fail under any spelling, but 0 is the only
    value a truth test swallows, so it is the one that regresses silently.
    """
    from dataclasses import dataclass

    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd.config_sm107 import make_cfg_d256

    @dataclass(frozen=True)
    class _ParamsWithStagesKv(TemplateParams):
        stages_kv: int = None

    with pytest.raises(ValueError, match="STAGES_KV"):
        make_cfg_d256(_ParamsWithStagesKv(stages_kv=bad_depth))


@pytest.mark.L0
def test_sm107_d256_stages_kv_defaults_when_the_attribute_is_absent():
    """The accept side of the test above: a plain TemplateParams (no
    `stages_kv` attribute at all) still gets the depth-2 default, so the fix
    for the explicit-zero case did not break the only path that ships."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams
    from cudnn.sdpa.fwd.config_sm107 import make_cfg_d256

    cfg, _ = make_cfg_d256(TemplateParams())
    assert cfg.STAGES_KV == 2


@pytest.mark.L0
def test_thd_stats_padded_is_appended_to_the_public_signature():
    """The adapters' constructors are public and not keyword-only; an old
    positional call (..., thd, max_total_seq_len_q, max_total_seq_len_kv) must
    keep binding the same way, so every new parameter is APPENDED after the
    legacy prefix, in the order it landed.

    Tail history: ``thd_stats_padded`` closed the legacy prefix; #931 / #983
    appended ``sample_amax_o``, ``pv_bf16`` and ``stats_log2``; the fused
    epilogue gate then appended ``sample_gate`` and ``has_amax_o`` after those,
    and the SM100 adapter's ``execute`` gained a trailing ``gate`` after
    ``block_table_v``.  The pin moves with the tail: every addition must sit
    after the prefix in landing order, so a positional caller of any earlier
    signature still binds where it always did."""
    import inspect

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDsl, SdpaFwdDslSm100

    params = list(inspect.signature(SdpaFwdDsl.__init__).parameters)
    legacy_tail = [
        "paged_table_stride",
        "paged_table_v_stride",
        "thd_stats_padded",
    ]
    assert params[params.index("paged_table_stride") : params.index("thd_stats_padded") + 1] == legacy_tail
    extension_start = params.index("sample_amax_o")
    assert extension_start == params.index("thd_stats_padded") + 1, params[extension_start - 1 : extension_start + 1]
    assert params[extension_start:] == ["sample_amax_o", "pv_bf16", "stats_log2", "sample_gate", "has_amax_o"], params[extension_start:]
    assert inspect.signature(SdpaFwdDsl.__init__).parameters["stats_log2"].default is False
    assert params.index("thd") + 1 == params.index("max_total_seq_len_q")
    # Both gate parameters default OFF, so every pre-gate call site is untouched.
    sig = inspect.signature(SdpaFwdDsl.__init__).parameters
    assert sig["sample_gate"].default is None and sig["has_amax_o"].default is True
    exec_params = list(inspect.signature(SdpaFwdDslSm100.execute).parameters)
    assert exec_params[-2:] == ["block_table_v", "gate"], exec_params[-3:]
    assert inspect.signature(SdpaFwdDslSm100.execute).parameters["gate"].default is None


# --- MXFP8 scheduler-policy claims (2026-09-14) -------------------------------
# The d128 and d192x128 MXFP8 kernels thread qh_per_kh / seqlen_kv into every
# decode call site (the LPT_L2 cost-model inputs the shared decode raises
# without), so they honour LPT AND LPT_L2; d256 / d512 pass neither and stay
# NATURAL.  Pure tests first (row + ranking), then the Rubin e2e that is the
# evidence for the claim.


def _mxfp8_facts(**kw):
    import cudnn

    return _f16_facts(dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16, is_mxfp8=True, **kw)


@pytest.mark.L0
def test_sm107_mxfp8_advertises_lpt_and_lpt_l2_per_d_shape():
    """INVERTED 2026-09-14 (the row was NATURAL-only): d128 and d192x128 claim
    LPT and LPT_L2, d256 and d512 do not, the row-wide floor stays NATURAL --
    and the claim REACHES the ranking: a causal GQA graph leads with LPT_L2, a
    causal graph without GQA on a few-wave grid leads with LPT, dense stays
    NATURAL.  An accept AND a reject per shape (engine contract, Rule 9)."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
    from cudnn.sdpa.fwd import engines, heuristics

    caps = _caps("sdpa_fwd_prefill_sm107_mxfp8")
    assert caps.sched_policies == frozenset({SCHED_NATURAL}), "the row-wide floor must stay NATURAL"
    for shape in ((128, 128), (192, 128)):
        dom = engines.effective_sched_policies(caps, _mxfp8_facts(d_qk=shape[0], d_v=shape[1]))
        assert dom == frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}), (shape, dom)
        assert heuristics._sched_points(caps, _mxfp8_facts(d_qk=shape[0], d_v=shape[1], causal=True, h_kv=2)) == [SCHED_LPT_L2, SCHED_LPT, SCHED_NATURAL], shape
        assert heuristics._sched_points(caps, _mxfp8_facts(d_qk=shape[0], d_v=shape[1], causal=True)) == [SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL], shape
        assert heuristics._sched_points(caps, _mxfp8_facts(d_qk=shape[0], d_v=shape[1])) == [SCHED_NATURAL], shape
    for shape in ((256, 256), (512, 512)):
        dom = engines.effective_sched_policies(caps, _mxfp8_facts(d_qk=shape[0], d_v=shape[1]))
        assert dom == frozenset({SCHED_NATURAL}), (shape, dom)
        assert heuristics._sched_points(caps, _mxfp8_facts(d_qk=shape[0], d_v=shape[1], causal=True)) == [SCHED_NATURAL], shape


@pytest.mark.L0
def test_sm107_mxfp8_sched_knob_is_honored_or_ineligible_per_d_shape():
    """Requesting LPT or LPT_L2 on the MXFP8 row: ACCEPTED at d128 and
    d192x128, DECLINED (typed, naming the knob) at d256 and d512 -- never
    silently downgraded (engine contract, Rule 4)."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    caps = _caps("sdpa_fwd_prefill_sm107_mxfp8")
    for pol in (SCHED_LPT, SCHED_LPT_L2):
        for shape in ((128, 128), (192, 128)):
            assert engines.mismatch(caps, _mxfp8_facts(d_qk=shape[0], d_v=shape[1]), SdpaFwdKnobs(sched_policy=pol)) is None, (pol, shape)
        for shape in ((256, 256), (512, 512)):
            why = engines.mismatch(caps, _mxfp8_facts(d_qk=shape[0], d_v=shape[1]), SdpaFwdKnobs(sched_policy=pol))
            assert why is not None and "sched_policy" in why, (pol, shape)


@pytest.mark.parametrize("d_qk, d_v", [(128, 128), (192, 128)])
@pytest.mark.parametrize("causal, b, hq, hkv, s", [(True, 2, 16, 4, 2048), (False, 1, 8, 2, 1024)])
def test_mxfp8_sched_policies_are_bit_identical_to_natural(d_qk, d_v, causal, b, hq, hkv, s):
    """Rubin e2e behind the MXFP8 (128, 128) and (192, 128) LPT / LPT_L2
    claims: the same block-scaled problem under EVERY policy the row claims
    for the flavor.  The scheduler only reorders whole (batch, head, q-tile)
    work items -- each tile's KV loop is unchanged -- so O and LSE must be
    BIT-IDENTICAL to NATURAL, every cell written (sentinel-filled O, NaN-filled
    LSE), and all within the fp32 oracle bound.  The causal case is a
    multi-wave grid (256 work items over ~106 clusters) so the persistent
    loop walks the remapped order for several rounds."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the sm107 MXFP8 kernels serve cc10.7 only")
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    from cudnn.frost.tile_dsl.constants import SCHED_NATURAL
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    caps = _caps("sdpa_fwd_prefill_sm107_mxfp8")
    policies = sorted(engines.effective_sched_policies(caps, _mxfp8_facts(d_qk=d_qk, d_v=d_v, causal=causal)))
    assert SCHED_NATURAL in policies and len(policies) == 3, policies

    torch.manual_seed(0)
    dev = "cuda"
    qf = torch.randn(b, hq, s, d_qk, device=dev) * 0.5
    kf = torch.randn(b, hkv, s, d_qk, device=dev) * 0.5
    vf = torch.randn(b, hkv, s, d_v, device=dev) * 0.5

    def mx(x, h, d, columnwise):
        data_d, _, swz_d, data_s, _, swz_s = quantize_to_mxfp8(x.contiguous(), b, h, s, d, 32, torch.float8_e4m3fn, with_ref=False)
        data, swz = (data_s, swz_s) if columnwise else (data_d, swz_d)
        return data.permute(0, 2, 1, 3).contiguous().transpose(1, 2), swz.contiguous()  # BHSD view over BSHD storage

    q8, sfq = mx(qf, hq, d_qk, False)
    k8, sfk = mx(kf, hkv, d_qk, False)
    v8, sfv = mx(vf, hkv, d_v, True)
    outs, lses = {}, {}
    for pol in policies:
        out = torch.full((b, s, hq, d_v), 1.5e30, device=dev, dtype=torch.bfloat16).transpose(1, 2)  # sentinel: an unclaimed tile stays visible
        lse = torch.full((b, hq, s), float("nan"), device=dev, dtype=torch.float32)
        api = SdpaFwdDslSm100(
            q8, k8, v8, out, lse, scale_softmax=d_qk**-0.5, is_causal=causal, pertensor_fp8=False, dtype_o=torch.bfloat16, sched_policy=pol, cga=2
        )
        assert api.check_support()
        api.compile()
        api.execute(q8, k8, v8, out, lse_tensor=lse, sf_q=sfq, sf_k=sfk, sf_v=sfv)
        torch.cuda.synchronize()
        outs[pol], lses[pol] = out.clone(), lse.clone()
    rep = hq // hkv
    logits = qf @ kf.repeat_interleave(rep, 1).transpose(-1, -2) * d_qk**-0.5
    if causal:
        logits = logits.masked_fill(~torch.tril(torch.ones(s, s, dtype=torch.bool, device=dev)), float("-inf"))
    ref = torch.softmax(logits, dim=-1) @ vf.repeat_interleave(rep, 1)
    scale = ref.abs().max().item()
    for pol in policies:
        assert torch.isfinite(outs[pol]).all(), f"policy {pol}: non-finite / unwritten O cells"
        assert torch.isfinite(lses[pol]).all(), f"policy {pol}: unwritten LSE rows"
        err = (outs[pol].float() - ref).abs().max().item()
        assert err <= 0.1 * scale, f"policy {pol}: max err {err} vs fp32 reference (scale {scale})"
    for pol in policies:
        if pol != SCHED_NATURAL:
            assert torch.equal(
                outs[pol], outs[SCHED_NATURAL]
            ), f"policy {pol}: O must be bit-identical to NATURAL -- a different tile walk, the same per-tile math"
            assert torch.equal(lses[pol], lses[SCHED_NATURAL]), f"policy {pol}: LSE must be bit-identical to NATURAL"


@pytest.mark.L0
def test_sm107_causal_ranking_picks_the_policy_by_gqa_and_wave_count():
    """The Rubin causal rule (heuristics._sched_points, 2026-09-14), pinned on
    the two charted layouts.  Perf node, kernel-level, vs NATURAL:
    DSv3 d192x128 H128 (no GQA) -- LPT_L2 -9.7 % at S=2K, LPT +16 % at S=4K,
    LPT -6.5 / -13 / -7.9 % at S=8K/16K/32K, LPT_L2 within 1 % there; Llama
    d128 H64/8 (GQA) -- LPT_L2 +4.2..+7.6 % at every S.  So: without GQA
    LPT_L2 has nothing to group and is never proposed first; LPT leads while
    the grid is at most 24 waves of 2-CTA clusters and NATURAL beyond; with
    GQA the L2-budget rule (LPT_L2) stands.  SM100 keeps its own rule."""
    import cudnn
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
    from cudnn.sdpa.fwd import heuristics

    fp8 = dict(dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16, is_fp8=True, causal=True, b=1)
    rubin = _caps("sdpa_fwd_prefill_sm107_fp8")
    dsv3 = dict(h_q=128, h_kv=128, d_qk=192, d_v=128, **fp8)
    # 1 x 128 heads x 8 q-clusters / 106 clusters = 9.7 waves -> LPT; 19 waves at S=4K -> LPT; 39 waves at S=8K -> NATURAL.
    assert heuristics._sched_points(rubin, _f16_facts(s_q=2048, s_kv=2048, **dsv3)) == [SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL]
    assert heuristics._sched_points(rubin, _f16_facts(s_q=4096, s_kv=4096, **dsv3)) == [SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL]
    for s in (8192, 16384, 32768):
        assert heuristics._sched_points(rubin, _f16_facts(s_q=s, s_kv=s, **dsv3)) == [SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2], s
    llama = dict(h_q=64, h_kv=8, d_qk=128, d_v=128, **fp8)
    for s in (2048, 8192, 32768):
        assert heuristics._sched_points(rubin, _f16_facts(s_q=s, s_kv=s, **llama)) == [SCHED_LPT_L2, SCHED_LPT, SCHED_NATURAL], s
    # The SM100 row is untouched by the Rubin rule: the L2-budget primary stays.
    sm100 = _caps("sdpa_fwd_prefill_sm100_fp8")
    assert heuristics._sched_points(sm100, _f16_facts(s_q=2048, s_kv=2048, device_cc=(10, 0), **dsv3)) == [SCHED_LPT_L2, SCHED_LPT, SCHED_NATURAL]
    # SM120 (sm_lo=120) is NOT the Rubin line even though 120 >= 107: a causal no-GQA
    # GeForce-Blackwell graph keeps the SM100/SM120 L2-budget primary too (PR #1059 review).
    sm120 = _caps("sdpa_fwd_prefill_sm120")
    f16 = dict(dtype=cudnn.data_type.HALF, dtype_o=cudnn.data_type.HALF, causal=True, b=1, h_q=32, h_kv=32, d_qk=128, d_v=128)
    assert heuristics._sched_points(sm120, _f16_facts(s_q=2048, s_kv=2048, device_cc=(12, 0), **f16)) == [SCHED_LPT_L2, SCHED_LPT, SCHED_NATURAL]


@pytest.mark.L0
@pytest.mark.parametrize("d_qk, d_v", [(128, 128), (192, 128)])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_mxfp8_stats_is_the_exact_softmax_lse(d_qk, d_v, causal):
    """Rubin e2e for the MXFP8 d128 / d192x128 kernels (row-sum-in-MMA since
    #1059): the PUBLISHED Stats is the fp32 log-sum-exp of the block-scaled
    problem the kernel saw, not the log of the quantized-P sum that
    normalizes O -- cuDNN's mxfp8 backward recomputes P = exp(S - Stats), and
    the fp8 twin lost a dK row to exactly that (test_mhas_v2 fp8_bwd_ragged
    test31).  (1) LSE within 1e-4 of the exact value from the DEQUANTIZED
    inputs; (2) O bit-identical with and without Stats."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the sm107 MXFP8 kernels serve cc10.7 only")
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    torch.manual_seed(0)
    b, hq, hkv, s = 2, 8, 2, 1024
    dev = "cuda"
    qf = torch.randn(b, hq, s, d_qk, device=dev) * 0.5
    kf = torch.randn(b, hkv, s, d_qk, device=dev) * 0.5
    vf = torch.randn(b, hkv, s, d_v, device=dev) * 0.5

    def mx(x, h, d, columnwise):
        data_d, sf_d, swz_d, data_s, sf_s, swz_s = quantize_to_mxfp8(x.contiguous(), b, h, s, d, 32, torch.float8_e4m3fn, with_ref=True)
        data, sf, swz = (data_s, sf_s, swz_s) if columnwise else (data_d, sf_d, swz_d)
        # sf_*_ref are the per-element fp32 DEQUANT SCALES [b, h, s, d]; the value the kernel sees is data * scale.
        # fp64 so the reference logits below are exact: the DLFW containers run fp32 matmul in TF32, which a
        # causal row with one valid column turns into a 1e-4-class LSE error (test_fp8_stats_is_the_exact_softmax_lse).
        deq = data.double().reshape(b, h, s, d) * sf.double().reshape(b, h, s, d)
        return data.permute(0, 2, 1, 3).contiguous().transpose(1, 2), deq, swz.contiguous()

    q8, q_deq, sfq = mx(qf, hq, d_qk, False)
    k8, k_deq, sfk = mx(kf, hkv, d_qk, False)
    v8, _, sfv = mx(vf, hkv, d_v, True)
    outs = {}
    lse = torch.full((b, hq, s), float("nan"), device=dev, dtype=torch.float32)
    for with_stats in (True, False):
        out = torch.full((b, s, hq, d_v), 1.5e30, device=dev, dtype=torch.bfloat16).transpose(1, 2)
        api = SdpaFwdDslSm100(
            q8, k8, v8, out, lse if with_stats else None, scale_softmax=d_qk**-0.5, is_causal=causal, pertensor_fp8=False, dtype_o=torch.bfloat16, cga=2
        )
        assert api.check_support()
        api.compile()
        api.execute(q8, k8, v8, out, lse_tensor=lse if with_stats else None, sf_q=sfq, sf_k=sfk, sf_v=sfv)
        torch.cuda.synchronize()
        outs[with_stats] = out.clone()
    assert torch.equal(outs[True], outs[False]), "O must not depend on whether Stats is requested (the MMA row-sum normalizes O in both specializations)"
    rep = hq // hkv
    logits = q_deq @ k_deq.repeat_interleave(rep, 1).transpose(-1, -2) * d_qk**-0.5
    if causal:
        logits = logits.masked_fill(~torch.tril(torch.ones(s, s, dtype=torch.bool, device=dev)), float("-inf"))
    lse_ref = torch.logsumexp(logits, dim=-1).float()
    assert torch.isfinite(lse).all(), "unwritten LSE rows"
    err = (lse - lse_ref).abs()
    assert (
        err.max().item() <= 1e-4
    ), f"Stats is not the exact log-sum-exp: max |dLSE| {err.max().item():.3e}, rms {err.pow(2).mean().sqrt().item():.3e} (quantized-sum LSE reads ~1e-3..1e-2)"


@pytest.mark.parametrize("kind,load_kw", _DTYPE_FAMILIES, ids=[k for k, _ in _DTYPE_FAMILIES])
@pytest.mark.parametrize("flavor", _FLAVORS)
def test_sm107_stats_log2_specializes_every_dtype_and_flavor(flavor, kind, load_kw):
    natural = _load(flavor, rubin=True, stats_log2=False, **load_kw)
    log2 = _load(flavor, rubin=True, stats_log2=True, **load_kw)
    assert natural is not log2
    assert natural.CFG.STATS_LOG2 == 0
    assert log2.CFG.STATS_LOG2 == 1


# ============================================================================
# Fused epilogue gate -- O := O * sigmoid(G) (PR-A, 2026-09-15)
#
# The Rubin d256 f16/bf16 and per-tensor FP8 kernels carry the gate behind
# ``TemplateParams.epilogue_gate`` (a module-cache key) / ``CFG.EPILOGUE_GATE``
# (const_expr seams).  Four enforcement points share ONE constant,
# ``config_sm107.SM107_EPILOGUE_GATE_SHAPES``: the two engine rows
# (``epilogue_gate_d_shapes``), the standalone adapter's rule-8b twin in
# ``SdpaFwdDslSm100.check_support``, the config validator
# (``_EPILOGUE_GATE_FLAVORS``) and the gated-attention block's geometry pin.
# The tests below walk them in that order -- rows, config/template, adapter --
# and end with the Rubin e2e that is the evidence behind every claim.
# ============================================================================

_GATE_ROWS = ("sdpa_fwd_prefill_sm107", "sdpa_fwd_prefill_sm107_fp8")
_D256 = (256, 256)
_GATE_SEAMS = (
    "cfg",
    "kernel_params",
    "smem",
    "bars",
    "init",
    "dispatch_corr",
    "dispatch_tmaldg",
    "tmaldg_state",
    "tmaldg_issue",
    "corr_state",
    "corr_prologue",
    "corr_chunk",
    "corr_release",
    "host_desc",
    "host_launch",
    "compile",
)
# The fork-era module knobs the production kernels must NOT carry (engine
# contract rule 5: no module-level knobs; the block drives the adapter).
_RETIRED_GATE_KNOBS = ("GATE_SOURCE", "GATE_ISSUE", "GATE_MATH", "AMAX_O", "KEEP_SASS")
# cta_mma=1 is what the adapter pins for FP8 d256 (supported_cgas_for((256, 256), fp8=True) == (1,)).
_FP8_LOAD_KW = dict(fp8=True, pertensor=True, dtype_qkv=_E4M3, dtype_o=_BF16_OUT, cta_mma=1)


def _gate_facts(**kw):
    """f16 facts for a GATED (256, 256) graph; the gate dtype defaults to Q's."""
    import cudnn

    kw.setdefault("d_qk", 256)
    kw.setdefault("d_v", 256)
    kw.setdefault("has_epilogue_gate", True)
    kw.setdefault("epilogue_gate_dtype", kw.get("dtype", cudnn.data_type.HALF))
    return _f16_facts(**kw)


def _fp8_gate_facts(**kw):
    """Per-tensor FP8 facts for a GATED (256, 256) graph with a bf16 G."""
    import cudnn

    kw.setdefault("dtype", cudnn.data_type.FP8_E4M3)
    kw.setdefault("dtype_o", cudnn.data_type.BFLOAT16)
    kw.setdefault("epilogue_gate_dtype", cudnn.data_type.BFLOAT16)
    return _gate_facts(is_fp8=True, **kw)


def _fp8_ungated_kw(**kw):
    """The UNGATED per-tensor FP8 (256, 256) facts kwargs -- the control next to ``_fp8_gate_facts``."""
    import cudnn

    base = dict(is_fp8=True, d_qk=256, d_v=256, dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16)
    base.update(kw)
    return base


def _gate_kernel_modules():
    """(f16, fp8) d256 Rubin modules loaded with the gate ON."""
    return _load(_D256, rubin=True, epilogue_gate=True), _load(_D256, rubin=True, epilogue_gate=True, **_FP8_LOAD_KW)


# --- Rows / mismatch -----------------------------------------------------------


def test_sm107_gate_rows_claim_exactly_d256():
    """Both Rubin rows claim the gate at EXACTLY (256, 256) -- and nothing else
    does.  The exact-dims rule is deliberate (plan S7 Q6): a d=200 graph rides the
    (256, 256) envelope for plain attention, but the gate tile would multiply the
    zero-padded columns too and nobody has validated a padded G, so the envelope
    flavor is declined with the gate's own reason rather than silently served."""
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.config_sm107 import SM107_EPILOGUE_GATE_SHAPES

    assert SM107_EPILOGUE_GATE_SHAPES == frozenset({_D256})
    for row in _GATE_ROWS:
        caps = _caps(row)
        assert caps.epilogue_gate is True, row
        assert caps.epilogue_gate_d_shapes == frozenset({_D256}), row
        assert caps.epilogue_gate_d_shapes is SM107_EPILOGUE_GATE_SHAPES, f"{row}: the row must consume the shared constant, not a copy"
    for spec in engines.ENGINE_SPECS:
        if spec.name not in _GATE_ROWS:
            assert spec.capabilities.epilogue_gate is False, spec.name
            assert spec.capabilities.epilogue_gate_d_shapes is None, spec.name
    # No new engine row: the gate is a FEATURE of the two existing Rubin rows.
    assert len(engines.ENGINE_SPECS) == 9

    f16, fp8 = _caps(_GATE_ROWS[0]), _caps(_GATE_ROWS[1])
    assert engines.mismatch(f16, _gate_facts()) is None
    assert engines.mismatch(fp8, _fp8_gate_facts()) is None
    # Every other flavor -- and the d=200 envelope ride -- declines by the gate's reason.
    for d_qk, d_v in ((128, 128), (192, 128), (512, 512), (200, 200)):
        why = engines.mismatch(f16, _gate_facts(d_qk=d_qk, d_v=d_v))
        assert why is not None and "epilogue gate" in why, (d_qk, d_v, why)
    for d_qk, d_v in ((128, 128), (192, 128), (512, 512)):
        why = engines.mismatch(fp8, _fp8_gate_facts(d_qk=d_qk, d_v=d_v))
        assert why is not None and "epilogue gate" in why, (d_qk, d_v, why)
    # An UNGATED graph is untouched by the new fields on every row.
    assert engines.mismatch(f16, _f16_facts(d_qk=256, d_v=256)) is None
    assert engines.mismatch(f16, _f16_facts(d_qk=128, d_v=128)) is None


def test_sm107_gate_accepts_every_dtype_member():
    """A frozenset field is one claim PER MEMBER (contract rule 9).  f16 row:
    HALF and BFLOAT16, gate dtype == Q's.  FP8 row: every input dtype x every O
    dtype the row lists, with the bf16 gate the kernel stages
    (``GATE_STORAGE_DTYPE = cutlass.BFloat16``); a HALF gate on that row is an
    asserted decline naming the axis."""
    import cudnn
    from cudnn.sdpa.fwd import engines

    f16 = _caps(_GATE_ROWS[0])
    assert f16.epilogue_gate_dtypes is None, "None = G must equal Q's dtype"
    for dt in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16):
        assert engines.mismatch(f16, _gate_facts(dtype=dt, dtype_o=dt, epilogue_gate_dtype=dt)) is None, dt
    # ...and a gate of the OTHER half dtype declines, naming the gate dtype.
    why = engines.mismatch(f16, _gate_facts(dtype=cudnn.data_type.HALF, dtype_o=cudnn.data_type.HALF, epilogue_gate_dtype=cudnn.data_type.BFLOAT16))
    assert why is not None and "gate dtype" in why, why

    fp8 = _caps(_GATE_ROWS[1])
    assert fp8.epilogue_gate_dtypes == frozenset({cudnn.data_type.BFLOAT16})
    assert fp8.dtypes == frozenset({cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2})
    assert fp8.out_dtypes == frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2})
    for dt in sorted(fp8.dtypes, key=int):
        for dto in sorted(fp8.out_dtypes, key=int):
            assert engines.mismatch(fp8, _fp8_gate_facts(dtype=dt, dtype_o=dto)) is None, (dt, dto)
    why = engines.mismatch(fp8, _fp8_gate_facts(epilogue_gate_dtype=cudnn.data_type.HALF))
    assert why is not None and "gate dtype" in why, why
    # The MXFP8 row is PR-B: it must decline the tail outright.
    why = engines.mismatch(
        _caps("sdpa_fwd_prefill_sm107_mxfp8"), _mxfp8_facts(d_qk=256, d_v=256, has_epilogue_gate=True, epilogue_gate_dtype=cudnn.data_type.BFLOAT16)
    )
    assert why is not None and "epilogue" in why, why


def test_sm107_gate_declines_the_interactions():
    """Gate x THD is REACHABLE (the f16 row serves THD at d256), gate x paged
    is REACHABLE (paged is wired on d256), so both must be declined by the gate
    block itself -- and the knob interactions (split, PackGQA) by their knob
    blocks.  Structural: the split flavors and the gate flavors are disjoint on
    every row, so no lowering can ever be asked to gate a partial."""
    import cudnn
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    f16, fp8 = _caps(_GATE_ROWS[0]), _caps(_GATE_ROWS[1])
    assert engines.mismatch(f16, _f16_facts(d_qk=256, d_v=256, thd=True, padded=True)) is None, "THD at d256 is served -- the interaction is live"
    why = engines.mismatch(f16, _gate_facts(thd=True, padded=True))
    assert why is not None and "dense-only" in why, why
    paged = dict(has_paged_kv=True, padded=True, page_size=128)
    # Neither Rubin row claims paged KV today, so gate x paged is declined by the feature table first;
    # the gate block carries its own paged decline for the day a gate row gains paged, so pin THAT on
    # a synthetic row that does (the gate block runs before the paged block in mismatch()).
    assert not f16.paged_kv and not fp8.paged_kv
    why = engines.mismatch(f16, _gate_facts(**paged))
    assert why is not None and "paged" in why, why
    why = engines.mismatch(dataclasses.replace(f16, paged_kv=True), _gate_facts(**paged))
    assert why is not None and "paged" in why and "gate" in why, why
    # Knob interactions: a split or packed plan can never carry the gate.
    why = engines.mismatch(fp8, _fp8_gate_facts(), SdpaFwdKnobs(split_kv=2))
    assert why is not None and "gate" in why, why
    # PackGQA on the REAL fp8 row is declined by its flavor table first (pack_gqa_d_shapes = {(128, 128)}); the
    # gate clause is what returns on a row that WOULD pack a d256 graph, so pin it on that synthetic row.
    why = engines.mismatch(fp8, _fp8_gate_facts(h_kv=2), SdpaFwdKnobs(pack_gqa=True))
    assert why is not None, why
    packs_d256 = dataclasses.replace(fp8, pack_gqas=frozenset({False, True}), pack_gqa_d_shapes=None)
    assert engines.mismatch(packs_d256, _f16_facts(**_fp8_ungated_kw(h_kv=2)), SdpaFwdKnobs(pack_gqa=True)) is None, "the control must pack"
    why = engines.mismatch(packs_d256, _fp8_gate_facts(h_kv=2), SdpaFwdKnobs(pack_gqa=True))
    assert why is not None and "gate" in why, why
    for row in _GATE_ROWS:
        caps = _caps(row)
        assert not ((caps.split_d_shapes or frozenset()) & caps.epilogue_gate_d_shapes), row
        assert not ((caps.pack_gqa_d_shapes or frozenset()) & caps.epilogue_gate_d_shapes), row
    # A broadcast G / an undeclared O_v are DECLINES (legal graphs for the backend), never facts.invalid.
    why = engines.mismatch(f16, _gate_facts(epilogue_gate_shape_ok=False))
    assert why is not None and "shape" in why, why
    # A G the kernel cannot TMA-load zero-copy (head-major / unaligned strides) is the row's twin of the
    # adapter's "TMA-expressible" ValueError (rule 8b: both halves, or the row admits what the adapter rejects).
    why = engines.mismatch(f16, _gate_facts(epilogue_gate_layout_ok=False))
    assert why is not None and "zero-copy" in why, why
    why = engines.mismatch(f16, _gate_facts(sdpa_o_virtual_declared=False))
    assert why is not None and "set_dim" in why, why
    for ok in (None, cudnn.data_type.FLOAT, cudnn.data_type.HALF):
        assert engines.mismatch(f16, _gate_facts(sdpa_o_virtual_dtype=ok)) is None, ok
    why = engines.mismatch(f16, _gate_facts(sdpa_o_virtual_dtype=cudnn.data_type.BFLOAT16))
    assert why is not None and "virtual O" in why, why


def test_epilogue_gate_is_not_a_knob():
    """The gate is a graph FACT (the tail is there or it is not), never a tuning
    axis: the knob vocabulary has no such field and the heuristics cannot
    propose it (contract rule 4)."""
    from cudnn.sdpa.fwd import engines, heuristics

    assert "epilogue_gate" not in engines.SdpaFwdKnobs.__dataclass_fields__
    assert not hasattr(heuristics, "_epilogue_gate_points")
    # ...but it IS a template-params field (the module-cache key) with the gate OFF by default.
    assert TemplateParams.__dataclass_fields__["epilogue_gate"].default is False


def test_sm107_gate_row_matches_the_adapter_twin():
    """Rule 8b': a row decline has a STANDALONE-WRAPPER twin, and the two must
    read ONE constant.  Iterate ROWS (not one named row -- the d_envelope_floors
    lesson), check identity with the shared object, and pin that
    ``check_support`` CONSUMES it rather than a private literal."""
    import inspect

    from cudnn.sdpa.fwd import api_dsl, engines
    from cudnn.sdpa.fwd.config_sm107 import SM107_EPILOGUE_GATE_SHAPES

    gate_rows = [s for s in engines.ENGINE_SPECS if s.capabilities.epilogue_gate]
    assert {s.name for s in gate_rows} == set(_GATE_ROWS)
    for spec in gate_rows:
        assert spec.capabilities.epilogue_gate_d_shapes is SM107_EPILOGUE_GATE_SHAPES, spec.name
        assert spec.capabilities.epilogue_gate_d_shapes <= spec.capabilities.d_shapes, spec.name
    assert api_dsl._SM107_EPILOGUE_GATE_SHAPES is SM107_EPILOGUE_GATE_SHAPES
    src = inspect.getsource(api_dsl.SdpaFwdDslSm100.check_support)
    assert "_SM107_EPILOGUE_GATE_SHAPES" in src, "check_support no longer consumes the shared gate-shape constant"
    assert "gate_desc" in src, "check_support does not look at the gate descriptor at all"
    # The config validator is the third consumer, keyed by flavor string.
    from cudnn.sdpa.fwd import config_sm107

    assert config_sm107._EPILOGUE_GATE_FLAVORS == frozenset({"sm107 d256"})
    assert "SM107_EPILOGUE_GATE_SHAPES" in config_sm107.__all__


# --- Config / template -----------------------------------------------------------


def test_sm107_gate_template_loads_for_d256_only():
    """The config backstop behind the rows: a gated (256, 256) module loads on
    BOTH dtype families with ``CFG.EPILOGUE_GATE == 1``; every other flavor
    raises at config build (a body without the seams would trace an ungated
    epilogue and silently return O instead of O * sigmoid(G))."""
    f16, fp8 = _gate_kernel_modules()
    assert f16.CFG.EPILOGUE_GATE == 1 and fp8.CFG.EPILOGUE_GATE == 1
    assert f16.CFG.GATE_BPE == 2 and fp8.CFG.GATE_BPE == 2, "the gate is 2 B/elem on both kernels (Q dtype / bf16)"
    assert "sm107" in f16.__name__ and "sm107" in fp8.__name__
    for flavor in ((128, 128), (192, 128), (512, 512)):
        with pytest.raises(ValueError, match="epilogue_gate"):
            _load(flavor, rubin=True, epilogue_gate=True)
        with pytest.raises(ValueError, match="epilogue_gate"):
            _load(flavor, rubin=True, epilogue_gate=True, **_FP8_LOAD_KW)
    # MXFP8 d256 is PR-B: same flavor family, different flavor string, refused.
    with pytest.raises(ValueError, match="epilogue_gate"):
        _load(_D256, rubin=True, epilogue_gate=True, fp8=True, pertensor=False, dtype_qkv=_E4M3, dtype_o=_BF16_OUT)
    # The SM100 line has no gated body at all.
    with pytest.raises(ValueError, match="epilogue_gate"):
        _load(_D256, rubin=False, epilogue_gate=True)
    # The feature interactions are refused by the config too (backstop of the rows / adapter twin).
    for bad in (
        dict(thd_varlen=True, seq_kv_lens_present=True),
        dict(split_kv=2),
        dict(pack_gqa=True, qh_per_kh=4),
        dict(paged_kv=True, page_size=128, seq_kv_lens_present=True),
    ):
        # split_kv > 1 is refused by the (earlier) SplitHelpers guard on this flavor; the rest by the gate rule.
        with pytest.raises(ValueError, match="epilogue_gate|split_kv > 1 is not wired"):
            _load(_D256, rubin=True, epilogue_gate=True, **bad)


def test_sm107_gate_off_is_the_default_module():
    """Gate on/off are TWO coexisting specializations of one template, and
    gate-off is the module every pre-gate caller already loads: same
    ``TemplateParams``, same module object, same source digest."""
    assert TemplateParams() == TemplateParams(epilogue_gate=False)
    for kw in ({}, _FP8_LOAD_KW):
        off = _load(_D256, rubin=True, **kw)
        assert off is _load(_D256, rubin=True, epilogue_gate=False, **kw)
        assert off.CFG.EPILOGUE_GATE == 0
        on = _load(_D256, rubin=True, epilogue_gate=True, **kw)
        assert on is not off
        assert (
            on.FROST_SOURCE_DIGEST != off.FROST_SOURCE_DIGEST
        ), "the digest must key the gate so the compiled cache cannot hand a gated artifact to an ungated caller"
        assert on.__file__ == off.__file__, "one template file, two specializations"


def test_sm107_gate_smem_tally():
    """The gate adds ONE 64 KiB staging tile (TILES_Q x TILE_M x TILE_O x 2 B,
    NOT divided by CTA_MMA -- each CTA of a cga2 pair gates its own full-width
    128 q rows).  Pure arithmetic on the shared tally, then the validator: depth
    2 fits at every configuration the block and engine path run, depth 3 does
    not, and the raise NAMES the gate term so a reader sees why a depth that
    fits ungated no longer does."""
    from dataclasses import dataclass

    from cudnn.sdpa.fwd.config_sm107 import SMEM_USABLE_BYTES, make_cfg_d256, sdpa_smem_bytes

    KIB = 1024
    fixed = 2 * KIB  # _SMEM_FIXED_OVERHEAD: barriers + scheduler ring + tmem ptr

    def tally(*, cta_mma, stages_kv, bpe_qkv, bpe_o, gate_bpe):
        return sdpa_smem_bytes(128, 128, 256, 256, 1, stages_kv, cta_mma, bpe_qkv, bpe_o, qo_alias=True, gate_bpe=gate_bpe) + fixed

    # Gate-off numbers are today's (194 / 162 KiB); the gate term is +64 KiB.
    assert tally(cta_mma=2, stages_kv=2, bpe_qkv=2, bpe_o=2, gate_bpe=0) == 194 * KIB
    assert tally(cta_mma=1, stages_kv=2, bpe_qkv=1, bpe_o=1, gate_bpe=0) == 162 * KIB
    assert tally(cta_mma=2, stages_kv=2, bpe_qkv=2, bpe_o=2, gate_bpe=2) == 258 * KIB  # d256 f16 cga2 (block + engine path)
    assert tally(cta_mma=1, stages_kv=2, bpe_qkv=1, bpe_o=1, gate_bpe=2) == 226 * KIB  # d256 fp8 e4m3-O (block's fully-fused mode)
    assert tally(cta_mma=1, stages_kv=2, bpe_qkv=1, bpe_o=2, gate_bpe=2) == 258 * KIB  # d256 fp8 bf16-O (unfused fp8)
    assert tally(cta_mma=2, stages_kv=3, bpe_qkv=2, bpe_o=2, gate_bpe=2) == 322 * KIB > SMEM_USABLE_BYTES
    assert tally(cta_mma=1, stages_kv=3, bpe_qkv=1, bpe_o=2, gate_bpe=2) == 322 * KIB > SMEM_USABLE_BYTES
    # ...and the same shapes carry the 4-CTA-wide gate term a d128 cga2 flavor would need (TILES_Q=2): not claimed.
    assert sdpa_smem_bytes(128, 128, 128, 128, 2, 4, 2, 2, 2, qo_alias=False, gate_bpe=2) + fixed == 322 * KIB

    @dataclass(frozen=True)
    class _ParamsWithStagesKv(TemplateParams):
        stages_kv: int = None

    cfg, _ = make_cfg_d256(_ParamsWithStagesKv(stages_kv=2, epilogue_gate=True))
    assert cfg.EPILOGUE_GATE == 1 and cfg.STAGES_KV == 2
    cfg, _ = make_cfg_d256(_ParamsWithStagesKv(stages_kv=3, epilogue_gate=False))
    assert cfg.STAGES_KV == 3, "depth 3 still fits UNGATED (290 KiB)"
    with pytest.raises(ValueError, match="GATE") as ei:
        make_cfg_d256(_ParamsWithStagesKv(stages_kv=3, epilogue_gate=True))
    assert "322 KiB" in str(ei.value) and "EPILOGUE_GATE=1" in str(ei.value), str(ei.value)
    # FP8 twins: e4m3-O and bf16-O at depth 2 fit; bf16-O at depth 3 does not.
    for dto in (_E4M3, _BF16_OUT):
        cfg, _ = make_cfg_d256(_ParamsWithStagesKv(stages_kv=2, epilogue_gate=True, dtype_qkv=_E4M3, dtype_o=dto, cta_mma=1))
        assert cfg.EPILOGUE_GATE == 1
    with pytest.raises(ValueError, match="GATE"):
        make_cfg_d256(_ParamsWithStagesKv(stages_kv=3, epilogue_gate=True, dtype_qkv=_E4M3, dtype_o=_BF16_OUT, cta_mma=1))


def test_sm107_gate_desc_version_unchanged():
    """The gate tile is allocated AFTER every MMA operand (sQO, sK, sV) and is
    never an MMA / UTCCP operand itself, so it cannot move an operand across
    the 256 KiB version-0 tcgen05 descriptor window: DESC_VERSION and its
    STAGES_KV derivation are identical with the gate on and off, on both
    kernels.  (The FP8 sibling now derives it too -- its literal 0 was one
    depth away from the same wrap.)"""
    from dataclasses import replace

    for kw in ({}, _FP8_LOAD_KW):
        off = _load(_D256, rubin=True, **kw)
        on = _load(_D256, rubin=True, epilogue_gate=True, **kw)
        assert on.DESC_VERSION == off.DESC_VERSION == 0, (kw, on.DESC_VERSION, off.DESC_VERSION)
        assert callable(getattr(on, "_needs_desc_v1", None)), f"{on.__name__}: DESC_VERSION must be DERIVED, not a literal"
        for depth in (2, 3, 4):
            want = int(off._needs_desc_v1(replace(off.CFG, STAGES_KV=depth)))
            assert int(on._needs_desc_v1(replace(on.CFG, STAGES_KV=depth))) == want, (kw, depth)
            # ...and both agree with the layout: the LAST V stage's START (Q(u)O byte-max + K ring + the
            # preceding V stages) crossing the 14-bit window.  The gate tile is not in this sum.
            c = on.CFG
            qo = max(c.TILE_M * c.TILE_K * c.BPE, c.TILE_M * c.TILE_O * c.BPE_O)
            k_stage = (c.TILE_N * c.TILE_K // c.CTA_MMA) * c.BPE
            v_stage = (c.TILE_O * c.TILE_N // c.CTA_MMA) * c.BPE
            assert want == int(qo + depth * k_stage + (depth - 1) * v_stage >= 256 * 1024), (kw, depth, want)
        # The f16 cga2 layout crosses at depth 4 exactly as the existing STAGES_KV test pins; fp8 bf16-O cga1 too.
        assert int(on._needs_desc_v1(replace(on.CFG, STAGES_KV=4))) == 1, kw


def test_sm107_gate_kernel_signatures_are_append_only():
    """Public-API signatures evolve append-only (AGENTS.md): both kernels'
    ``compile`` gain their gate parameters at the END (fp8 also ``has_amax``),
    both ``_host`` gain ``gate_tensor`` LAST -- after ``stream``, which every
    caller passes by keyword.  A ``gate_stride`` handed to an UNGATED module is
    a ValueError, not a silently ignored kwarg."""
    import inspect

    f16, fp8 = _gate_kernel_modules()
    f16_c = list(inspect.signature(f16.compile).parameters)
    fp8_c = list(inspect.signature(fp8.compile).parameters)
    assert f16_c[-1] == "gate_stride", f16_c[-3:]
    assert fp8_c[-2:] == ["gate_stride", "has_amax"], fp8_c[-3:]
    for params in (f16_c, fp8_c):
        assert params.index("lse_stride") < params.index("gate_stride")
        assert params[: params.index("gate_stride")] == [p for p in params if p not in ("gate_stride", "has_amax")]
    for mod in (f16, fp8):
        sig = inspect.signature(mod.compile).parameters
        assert sig["gate_stride"].default is None
        host = list(inspect.signature(mod._host).parameters)
        assert host[-2:] == ["stream", "gate_tensor"], (mod.__name__, host[-3:])
        assert inspect.signature(mod._host).parameters["gate_tensor"].default is None
    assert inspect.signature(fp8.compile).parameters["has_amax"].default is True
    assert "gate_tensor" in inspect.signature(fp8._kernel).parameters and "tma_gate_desc" in inspect.signature(fp8._kernel).parameters
    assert "gate_tensor" in inspect.signature(f16._kernel).parameters and "tma_gate_desc" in inspect.signature(f16._kernel).parameters

    # Ungated modules refuse a gate stride before any trace.
    for kw in ({}, _FP8_LOAD_KW):
        off = _load(_D256, rubin=True, **kw)
        with pytest.raises(ValueError, match="epilogue_gate"):
            off.compile(b=1, qh=1, kh=1, sq=256, skv=256, gate_stride=(256 * 256, 256, 256, 1))


def test_sm107_gate_seams_are_named_once():
    """Every splice in both kernels is marked ``# == EPILOGUE_FUSION_SEAM(<name>) ==``
    -- the 16 names exactly once each, so a reviewer can walk the fusion in
    pipeline order with one grep and a dropped / duplicated seam is visible.
    And none of the fork's module knobs survived the port (contract rule 5)."""
    import re

    f16, fp8 = _gate_kernel_modules()
    for mod in (f16, fp8):
        with open(mod.__file__, encoding="utf-8") as fh:
            src = fh.read()
        seams = re.findall(r"# == EPILOGUE_FUSION_SEAM\((\w+)\) ==", src)
        counts = {name: seams.count(name) for name in set(seams)}
        assert set(seams) == set(_GATE_SEAMS), f"{mod.__name__}: seams {sorted(set(seams) ^ set(_GATE_SEAMS))} missing or unknown"
        assert all(n == 1 for n in counts.values()), f"{mod.__name__}: duplicated seams {[k for k, n in counts.items() if n != 1]}"
        code = _code_lines(src)
        for knob in _RETIRED_GATE_KNOBS:
            assert not re.search(rf"^{knob}\b\s*[:=]", code, re.M), f"{mod.__name__}: fork-era module knob {knob} survived"
            assert not re.search(rf"\b{knob}\b", code), f"{mod.__name__}: fork-era knob {knob} is still referenced"
        assert 'options="--enable-tvm-ffi"' in code, f"{mod.__name__}: compile options must stay the production ones"


# --- Adapter accept / reject (CPU; the device capability is monkeypatched) ------


def _desc(shape, dtype, name, stride=None):
    """A BSHD-physical (BHSD-logical) TensorDesc with no storage behind it."""
    from cudnn.api_base import TensorDesc

    b, h, s, d = shape
    stride = tuple(stride) if stride is not None else (s * h * d, d, h * d, 1)
    return TensorDesc(
        dtype=dtype, shape=tuple(shape), stride=stride, stride_order=TensorDesc._compute_stride_order(tuple(shape), stride), device="cuda", name=name
    )


def _fake_cc(monkeypatch, cc):
    """check_support reads the LIVE device; pin it so the Rubin arm runs on any host."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: cc)


def _gate_api(
    *,
    b=1,
    h=8,
    h_kv=None,
    s=512,
    d=256,
    d_v=None,
    dtype=None,
    fp8=False,
    pertensor=True,
    gate_dtype=None,
    gate_shape=None,
    gate_stride=None,
    with_gate=True,
    **kw,
):
    """A d256-shaped SdpaFwdDslSm100 built from descriptors only."""
    import torch

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    h_kv = h if h_kv is None else h_kv
    d_v = d if d_v is None else d_v
    dtype = dtype or torch.float16
    qkv_dtype = torch.float8_e4m3fn if fp8 else dtype
    o_dtype = torch.bfloat16 if fp8 else dtype
    gate_dtype = gate_dtype or (torch.bfloat16 if fp8 else dtype)
    q = _desc((b, h, s, d), qkv_dtype, "q")
    k = _desc((b, h_kv, s, d), qkv_dtype, "k")
    v = _desc((b, h_kv, s, d_v), qkv_dtype, "v")
    o = _desc((b, h, s, d_v), o_dtype, "o")
    if with_gate:
        kw["sample_gate"] = _desc(gate_shape or (b, h, s, d_v), gate_dtype, "gate", stride=gate_stride)
    if fp8:
        kw.update(pertensor_fp8=pertensor, dtype_o=o_dtype)
    return SdpaFwdDslSm100(q, k, v, o, None, **kw)


def _stub_compiled(monkeypatch, api):
    """Let execute() reach its presence contracts without a kernel or a stream."""
    api._compiled_kernel = object()
    monkeypatch.setattr(api, "_get_default_stream", lambda stream: stream)


def test_gate_check_support_declines_typed(monkeypatch):
    """Rule 8b twins of the rows' gate claims, on the STANDALONE adapter -- each
    a typed decline (ValueError for a malformed request, NotImplementedError for
    "not mine"), never a bare escape from compile().  The accept side first, so
    a decline cannot pass by the whole path being dead."""
    import torch

    _fake_cc(monkeypatch, (10, 7))
    for dt in (torch.float16, torch.bfloat16):
        api = _gate_api(dtype=dt)
        assert api.check_support()
        assert api.gate_desc is not None and api.gate_desc.dtype == dt
        assert api._gate_declared is None, "a compact G declares no stride"
        assert api.template_params().epilogue_gate is True
    assert _gate_api(with_gate=False).template_params().epilogue_gate is False
    assert _gate_api(fp8=True).check_support(), "bf16 G on the per-tensor FP8 path"
    assert _gate_api(fp8=True, has_amax_o=False).check_support(), "the Amax_O fold-out is a quantized-path option"

    # Malformed requests -> ValueError.
    with pytest.raises(ValueError, match="GATE"):
        _gate_api(gate_shape=(1, 8, 256, 256)).check_support()
    with pytest.raises(ValueError, match="GATE"):
        _gate_api(gate_dtype=torch.bfloat16).check_support()
    with pytest.raises(ValueError, match="GATE"):
        _gate_api(fp8=True, gate_dtype=torch.float16).check_support()
    # A G whose head stride is not a 16-byte multiple is not TMA-expressible; the adapter never hides that behind a copy.
    with pytest.raises(ValueError, match="TMA-expressible"):
        _gate_api(gate_stride=(512 * 8 * 260, 260, 8 * 260, 1)).check_support()
    with pytest.raises(ValueError, match="has_amax_o"):
        _gate_api(with_gate=False, has_amax_o=False).check_support()

    # "Not mine" -> NotImplementedError, naming the axis.
    for d, d_v in ((128, 128), (192, 128), (512, 512)):
        with pytest.raises(NotImplementedError, match="256"):
            _gate_api(d=d, d_v=d_v).check_support()
    with pytest.raises(NotImplementedError, match="dense-only"):
        _gate_api(thd=True).check_support()
    # split_kv=2: the PRE-EXISTING SM107 split guard ("split_kv > 1 on cc10.7 is wired only for per-tensor FP8
    # d128") fires first, gate on or off -- the gate's own split clause is unreachable on Rubin, because the
    # only split-wired flavor (fp8 d128) is declined by the gate's head-dim guard before it.  Pinned on the
    # shared "split_kv" token so either ordering is a typed decline.
    with pytest.raises(NotImplementedError, match="split_kv"):
        _gate_api(split_kv=2).check_support()
    with pytest.raises(NotImplementedError, match="PackGQA"):
        _gate_api(h_kv=2, pack_gqa=True).check_support()
    with pytest.raises(NotImplementedError, match="MXFP8"):
        _gate_api(fp8=True, pertensor=False).check_support()
    # Paged KV: K/V are page pools; the d256 f16 flavor serves paged, the gate does not ride it.
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    pool = _desc((16, 8, 128, 256), torch.float16, "k")
    with pytest.raises(NotImplementedError, match="paged"):
        SdpaFwdDslSm100(
            _desc((1, 8, 512, 256), torch.float16, "q"),
            pool,
            _desc((16, 8, 128, 256), torch.float16, "v"),
            _desc((1, 8, 512, 256), torch.float16, "o"),
            None,
            seq_kv_lens_present=True,
            paged_page_size=128,
            paged_max_seq_len_kv=512,
            sample_gate=_desc((1, 8, 512, 256), torch.float16, "gate"),
        ).check_support()

    # Execute presence contracts, both directions, plus the Amax_O fold-out.
    t = torch.empty(1)
    api = _gate_api()
    assert api.check_support()
    _stub_compiled(monkeypatch, api)
    with pytest.raises(ValueError, match="gate"):
        api.execute(t, t, t, t)
    plain = _gate_api(with_gate=False)
    assert plain.check_support()
    _stub_compiled(monkeypatch, plain)
    with pytest.raises(ValueError, match="sample_gate"):
        plain.execute(t, t, t, t, gate=t)
    no_amax = _gate_api(fp8=True, with_gate=False, has_amax_o=False)
    assert no_amax.check_support()
    _stub_compiled(monkeypatch, no_amax)
    with pytest.raises(ValueError, match="has_amax_o"):
        no_amax.execute(t, t, t, t, amax_o=t)


def test_gate_check_support_declines_other_arch_lines(monkeypatch):
    """The gate is served by the Rubin d256 kernels ONLY: the SM100 arm of the
    same adapter and the SM120 / SM80 adapters all decline it typed."""
    import torch

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120, SdpaFwdDslSm80

    _fake_cc(monkeypatch, (10, 0))
    with pytest.raises(NotImplementedError, match="Rubin"):
        _gate_api().check_support()
    assert _gate_api(with_gate=False).check_support(), "the ungated d256 graph is served on SM100"
    for cls in (SdpaFwdDslSm120, SdpaFwdDslSm80):
        q = _desc((1, 8, 512, 128), torch.float16, "q")
        api = cls(q, q, q, _desc((1, 8, 512, 128), torch.float16, "o"), None, sample_gate=_desc((1, 8, 512, 128), torch.float16, "gate"))
        with pytest.raises(NotImplementedError, match="SM107"):
            api.check_support()


# --- Rubin e2e, standalone adapter ----------------------------------------------
#
# Sentinel-filled O (an unwritten tile stays visible), NaN-filled LSE, and the
# two-launch trick (a first-launch race shows as a second-launch delta) on every
# run.  Tolerances are the shared SM10x suites' (O atol 5e-2 / rtol 3e-2, LSE
# atol 2e-2); bitwise claims are ``torch.equal``.

_GATE_O_TOL = dict(atol=5e-2, rtol=3e-2)
_GATE_LSE_TOL = dict(atol=2e-2, rtol=2e-2)


def _rubin_only():
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the fused epilogue gate is served by the sm107 d256 kernels (cc10.7) only")


def _gate_problem(b, h, h_kv, s, d, dtype, *, seed=0):
    """BSHD-physical / BHSD-logical Q, K, V and a gate G of O's shape."""
    import torch

    torch.manual_seed(seed)
    dev = "cuda"
    q = (torch.randn(b, s, h, d, device=dev) * 0.5).to(dtype).transpose(1, 2)
    k = (torch.randn(b, s, h_kv, d, device=dev) * 0.5).to(dtype).transpose(1, 2)
    v = (torch.randn(b, s, h_kv, d, device=dev) * 0.5).to(dtype).transpose(1, 2)
    gate = (torch.randn(b, s, h, d, device=dev) * 2.0).to(dtype).transpose(1, 2)
    return q, k, v, gate


def _gate_reference(q, k, v, gate, *, causal, scale, seq_kv_lens=None):
    """fp32 softmax(QK^T) V * sigmoid(G) and the fp64 natural-log LSE (-inf on an empty row)."""
    import torch

    rep = q.shape[1] // k.shape[1]
    kf, vf = k.float().repeat_interleave(rep, 1), v.float().repeat_interleave(rep, 1)
    logits = q.float() @ kf.transpose(-1, -2) * scale
    s_q, s_kv = q.shape[2], k.shape[2]
    if causal:
        i = torch.arange(s_q, device=q.device).view(s_q, 1)
        j = torch.arange(s_kv, device=q.device).view(1, s_kv)
        logits = logits.masked_fill(j > i + (s_kv - s_q), float("-inf"))
    if seq_kv_lens is not None:
        cols = torch.arange(s_kv, device=q.device).view(1, 1, 1, s_kv)
        logits = logits.masked_fill(cols >= seq_kv_lens.view(-1, 1, 1, 1), float("-inf"))
    lse = torch.logsumexp(logits.double(), dim=-1).float()
    p = torch.softmax(logits, dim=-1).nan_to_num(0.0)  # an empty row is all -inf: O = 0
    return (p @ vf) * torch.sigmoid(gate.float()), lse


def _run_gated(q, k, v, gate, *, causal, cga=None, sched_policy=None, seq_kv_lens=None, with_lse=True, gate_on=True):
    """Build, compile and launch TWICE; return (api, O, LSE) with the sentinel / two-launch checks done."""
    import torch

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, h, s, d_v = q.shape[0], q.shape[1], q.shape[2], v.shape[3]
    dtype = q.dtype
    out = torch.full((b, s, h, d_v), 1.5e30, device=q.device, dtype=torch.float32).to(dtype).transpose(1, 2)  # sentinel (inf in fp16)
    sentinel = out[0, 0, 0, 0].item()
    lse = torch.full((b, h, s), float("nan"), device=q.device, dtype=torch.float32) if with_lse else None
    api = SdpaFwdDslSm100(
        q,
        k,
        v,
        out,
        lse,
        is_causal=causal,
        scale_softmax=q.shape[3] ** -0.5,
        seq_kv_lens_present=seq_kv_lens is not None,
        cga=cga,
        sched_policy=sched_policy,
        **({"sample_gate": gate} if gate_on else {}),
    )
    assert api.check_support()
    api.compile()
    kw = dict(lse_tensor=lse, seq_kv_lens=seq_kv_lens)
    if gate_on:
        kw["gate"] = gate
    api.execute(q, k, v, out, **kw)
    torch.cuda.synchronize()
    first_o, first_lse = out.clone(), (lse.clone() if lse is not None else None)
    api.execute(q, k, v, out, **kw)
    torch.cuda.synchronize()
    assert torch.equal(out, first_o), "two-launch delta on O: a first-launch race (missing fence_proxy?)"
    assert lse is None or torch.equal(lse, first_lse), "two-launch delta on LSE"
    assert torch.isfinite(out.float()).all() and not (out == sentinel).any(), "unwritten (sentinel) or non-finite O cells"
    return api, out, lse


@pytest.mark.parametrize(
    "dtype, causal, s, cga",
    [
        ("fp16", False, 512, None),
        ("fp16", True, 1000, None),
        ("bf16", False, 512, None),
        ("bf16", True, 1000, None),
    ],
    ids=["fp16-dense-512", "fp16-causal-1000", "bf16-dense-512", "bf16-causal-1000"],
)
def test_sm107_gate_matches_the_fp32_oracle(dtype, causal, s, cga):
    """Rubin e2e for the f16/bf16 row's gate claim: O == softmax(QK^T)V *
    sigmoid(G) against the fp32 oracle at the shared tolerance; LSE is the
    plain (ungated) fp64 log-sum-exp.  S=1000 causal covers a KV tail.  Always
    the default cga (2): the f16 d256 flavor serves cga2 ONLY on Rubin -- see
    test_sm107_gate_f16_d256_declines_an_explicit_cga1."""
    import torch

    _rubin_only()
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    b, h, h_kv, d = 2, 8, 2, 256
    q, k, v, gate = _gate_problem(b, h, h_kv, s, d, dt)
    _, out, lse = _run_gated(q, k, v, gate, causal=causal, cga=cga)
    ref_o, ref_lse = _gate_reference(q, k, v, gate, causal=causal, scale=d**-0.5)
    torch.testing.assert_close(out.float(), ref_o, **_GATE_O_TOL)
    torch.testing.assert_close(lse, ref_lse, **_GATE_LSE_TOL)


def test_sm107_gate_f16_d256_declines_an_explicit_cga1():
    """The f16 d256 flavor is cga2-only on Rubin (``supported_cgas_for((256, 256),
    fp8=False, (10, 7)) == (2,)``: at cga1 the Q-union-O alias plus the K/V rings
    at STAGES_KV=2 exceed the 320 KiB carveout), with or without the gate.  The
    PR-A plan's Q18 assumed (1, 2) and asked for a cga=1 oracle case; the adapter
    declines it before any kernel loads (measured on the dev node 2026-09-15:
    "only supports cga in (2,)").  Pin the decline with the gate ON so a widening
    of the flavor's cga domain has to come back here and add the oracle case."""
    import torch

    _rubin_only()
    b, h, h_kv, s, d = 2, 8, 2, 1000, 256
    q, k, v, gate = _gate_problem(b, h, h_kv, s, d, torch.bfloat16)
    with pytest.raises(ValueError, match="cga"):
        _run_gated(q, k, v, gate, causal=True, cga=1)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_sm107_gate_dead_padded_entry_is_exactly_zero(dtype):
    """sdpa-invariants S1/S2: a batch entry with seq_kv_len == 0 runs ZERO KV
    iterations, so its epilogue must SELECT zero, never multiply the (possibly
    NaN) accumulator residue -- and the gate fma must sit BEFORE that select,
    so a +-1e4 gate on the dead entry cannot leak through: O exactly 0, LSE
    exactly -inf.  The live entry stays on the oracle."""
    import torch

    _rubin_only()
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    b, h, h_kv, s, d = 2, 8, 2, 512, 256
    q, k, v, gate = _gate_problem(b, h, h_kv, s, d, dt)
    sign = torch.where(torch.arange(s * h * d, device="cuda") % 2 == 0, 1.0, -1.0).view(s, h, d)
    gate.transpose(1, 2)[1] = (sign * 1e4).to(dt)
    lens = torch.tensor([s, 0], dtype=torch.int32, device="cuda")
    _, out, lse = _run_gated(q, k, v, gate, causal=False, seq_kv_lens=lens)
    assert (out[1] == 0).all(), "the dead entry must be EXACTLY zero (select, not residue * 0)"
    assert torch.isneginf(lse[1]).all(), "an empty row's LSE is -inf (no floor may leak: -69.08 = log 1e-30)"
    ref_o, ref_lse = _gate_reference(q, k, v, gate, causal=False, scale=d**-0.5, seq_kv_lens=lens)
    torch.testing.assert_close(out[0].float(), ref_o[0], **_GATE_O_TOL)
    torch.testing.assert_close(lse[0], ref_lse[0], **_GATE_LSE_TOL)


def test_sm107_gate_reads_a_strided_slab_bitwise():
    """G sliced out of a fused projection slab (token stride N, not h*d) is
    read ZERO-COPY through the declared BSHD stride and yields the BITWISE
    same O as a compact G with the same values -- first G alone, then Q/K/V/G
    all strided, which is what the gated attention block runs."""
    import torch

    _rubin_only()
    b, h, s, d = 2, 8, 512, 256
    dt = torch.bfloat16
    q, k, v, gate = _gate_problem(b, h, h, s, d, dt)
    api_c, out_c, lse_c = _run_gated(q, k, v, gate, causal=True)
    assert api_c._gate_declared is None

    n = 4 * h * d  # a [B, S, N] slab: q | k | v | gate column blocks
    slab = torch.zeros(b, s, n, device="cuda", dtype=dt)
    for i, t in enumerate((q, k, v, gate)):
        slab[:, :, i * h * d : (i + 1) * h * d] = t.transpose(1, 2).reshape(b, s, h * d)
    sliced = [slab[:, :, i * h * d : (i + 1) * h * d].view(b, s, h, d).transpose(1, 2) for i in range(4)]
    assert torch.equal(sliced[3], gate) and not sliced[3].is_contiguous()

    # G strided alone.
    api_g, out_g, lse_g = _run_gated(q, k, v, sliced[3], causal=True)
    assert api_g._gate_declared == (s * n, n, d, 1), api_g._gate_declared
    assert api_g._bshd_zero_copy_stride(api_g.gate_desc, 2) == (s * n, n, d, 1)
    assert torch.equal(out_g, out_c) and torch.equal(lse_g, lse_c), "a strided G must read bitwise as the compact one"
    # Everything strided (the block's layout).
    api_s, out_s, lse_s = _run_gated(*sliced, causal=True)
    assert api_s._gate_declared == (s * n, n, d, 1)
    assert all(st == (s * n, n, d, 1) for st in api_s._bshd_declared[:3]), api_s._bshd_declared
    assert torch.equal(out_s, out_c) and torch.equal(lse_s, lse_c), "strided Q/K/V/G must read bitwise as compact"


def test_sm107_gate_lse_is_bitwise_independent_of_the_gate():
    """The gate multiplies the fp32 pre-cast accumulator in the epilogue and
    nothing upstream of it: the published LSE is BITWISE the ungated kernel's,
    and O is the ungated O times sigmoid(G) to rounding."""
    import torch

    _rubin_only()
    b, h, h_kv, s, d = 2, 8, 2, 1000, 256
    q, k, v, gate = _gate_problem(b, h, h_kv, s, d, torch.bfloat16)
    _, out_on, lse_on = _run_gated(q, k, v, gate, causal=True, gate_on=True)
    _, out_off, lse_off = _run_gated(q, k, v, gate, causal=True, gate_on=False)
    assert torch.equal(lse_on, lse_off), "LSE must not depend on the gate"
    torch.testing.assert_close(out_on.float(), out_off.float() * torch.sigmoid(gate.float()), **_GATE_O_TOL)
    assert not torch.equal(out_on, out_off), "the gate must actually apply (a +-2 sigma G is far from sigmoid == 1)"


def test_sm107_gate_lpt_is_bitwise_natural():
    """The f16 row claims LPT at (256, 256); the scheduler only reorders whole
    work items, so O and LSE under SCHED_LPT with the gate ON are BITWISE the
    NATURAL run's (a multi-wave causal grid so the persistent loop walks the
    remapped order for several rounds)."""
    import torch

    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_NATURAL
    from cudnn.sdpa.fwd import engines

    _rubin_only()
    caps = _caps(_GATE_ROWS[0])
    assert SCHED_LPT in engines.effective_sched_policies(caps, _gate_facts(causal=True))
    b, h, h_kv, s, d = 2, 16, 4, 1024, 256
    q, k, v, gate = _gate_problem(b, h, h_kv, s, d, torch.bfloat16)
    _, out_nat, lse_nat = _run_gated(q, k, v, gate, causal=True, sched_policy=SCHED_NATURAL)
    _, out_lpt, lse_lpt = _run_gated(q, k, v, gate, causal=True, sched_policy=SCHED_LPT)
    assert torch.equal(out_lpt, out_nat), "LPT must be bit-identical to NATURAL -- a different tile walk, the same per-tile math"
    assert torch.equal(lse_lpt, lse_nat)
    ref_o, _ = _gate_reference(q, k, v, gate, causal=True, scale=d**-0.5)
    torch.testing.assert_close(out_nat.float(), ref_o, **_GATE_O_TOL)
