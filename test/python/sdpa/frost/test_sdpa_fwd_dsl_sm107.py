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
    keep binding the same way, so the new parameter is the LAST one."""
    import inspect

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDsl

    params = list(inspect.signature(SdpaFwdDsl.__init__).parameters)
    assert params[-1] == "thd_stats_padded"
    assert params.index("thd") + 1 == params.index("max_total_seq_len_q")


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
