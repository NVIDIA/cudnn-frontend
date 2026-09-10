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


def test_sm107_f16_declines_the_stats_trim_it_lacks():
    """padded_stats / dense_seq_q_trim both need the per-batch seq_len_q LSE
    trim (padded q rows write LSE=-inf, O=0), which these kernels do not
    carry.  INVERTS-WHEN that epilogue lands."""
    caps = _caps("sdpa_fwd_prefill_sm107")
    assert caps.padded_stats is False
    assert caps.dense_seq_q_trim is False


def test_sm107_rows_serve_natural_scheduling_only():
    """SCHED_LPT is NOT honored by the ported Rubin kernels: with the row
    claiming it, a causal graph picks LPT and comes back with max|O-ref| ~ 1.9
    (dense stays correct).  Session 6 had already dropped SCHED_LPT_L2 for the
    same reason -- the ported decode sites never thread qh_per_kh / seqlen_kv --
    and LPT only stayed hidden because heuristics could not REACH it (its causal
    primary is LPT_L2, and the old out-of-domain fallback dropped to NATURAL).

    A knob is honored or the engine is ineligible.  INVERTS-WHEN the LPT decode
    is threaded and validated on the ported Rubin kernels."""
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL

    for row in ("sdpa_fwd_prefill_sm107", "sdpa_fwd_prefill_sm107_fp8", "sdpa_fwd_prefill_sm107_mxfp8"):
        caps = _caps(row)
        assert caps.sched_policies == frozenset({SCHED_NATURAL}), row
        assert SCHED_LPT not in caps.sched_policies, row
        assert SCHED_LPT_L2 not in caps.sched_policies, row


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
    # Single-element domain today (see the row): the ranking must still be that
    # one element, never a NATURAL fallback bolted on beside it.
    assert points == [SCHED_NATURAL], points

    # ...and the multi-element case, which is the branch the single-element
    # shortcut above skips: with a causal primary (LPT_L2) OUT of domain, the
    # fallback must walk the preference order rather than dropping straight to
    # NATURAL -- the bug this function was fixed for.
    widened = dataclasses.replace(rubin_fp8, sched_policies=frozenset({SCHED_NATURAL, SCHED_LPT}))
    widened_points = heuristics._sched_points(widened, facts)
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
    safe = [
        ("d128", (128, 128), 1, _E4M3_OUT),
        ("d128", (128, 128), 1, _BF16),
        ("d128", (128, 128), 2, _BF16),
        ("d192", (192, 128), 1, _E4M3_OUT),
        ("d192", (192, 128), 2, _BF16),
    ]
    for _id, flavor, cta, dto in safe:
        mod = _load(flavor, rubin=True, fp8=True, pertensor=True, dtype_qkv=_E4M3, dtype_o=dto, cta_mma=cta)
        assert mod._ONES_SMEM_OFFSET < 256 * 1024, (_id, cta, dto, mod._ONES_SMEM_OFFSET)

    # The one unsafe combination: d192 x cga1 x half-precision O.
    with _pytest.raises(ValueError, match="version-0 tcgen05 descriptor window"):
        _load((192, 128), rubin=True, fp8=True, pertensor=True, dtype_qkv=_E4M3, dtype_o=_BF16, cta_mma=1)

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
