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

import pytest

from frost_test_utils import requires_dsl

from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module
from cudnn.sdpa.fwd.config_sm100 import TemplateParams

pytestmark = [pytest.mark.L0, requires_dsl]

_E4M3, _BF16_OUT = 0, 2
_FLAVORS = [(128, 128), (256, 256), (512, 512)]


def _load(flavor, rubin, **params):
    return _load_sm100_kernel_module(flavor, TemplateParams(**params), rubin=rubin)


@pytest.mark.parametrize("flavor", _FLAVORS)
def test_f16_routes_to_the_sm107_sibling(flavor):
    """Every f16 flavor has a Rubin sibling, and Blackwell keeps the SM100 one."""
    assert "sm107" in _load(flavor, rubin=True).__name__
    assert "sm100" in _load(flavor, rubin=False).__name__


# Which flavors' MMA-operand SMEM crosses 256 KiB -- the boundary above which a
# version-0 tcgen05 descriptor cannot address the buffer.  d512 alone does: its
# Q(u)O and K(u)V slabs are 128 KiB each, putting the P transfer ring at exactly
# 262144.  Keep d128/d256 on version 0 so they stay byte-identical to the
# shipped prefill_d128_fp8_sm107.py sibling.
_NEEDS_DESC_V1 = {(512, 512)}


@pytest.mark.parametrize("flavor", _FLAVORS)
def test_sm107_descriptor_version_matches_the_smem_budget(flavor):
    """A version-0 descriptor's ``start_address`` is 14 bits = a 256 KiB
    window; Rubin raises the per-CTA cap to 327 KiB.  An operand buffer at or
    above 256 KiB therefore wraps to offset 0 and the MMA multiplies whatever
    is at the bottom of SMEM -- O comes out EXACTLY zero, no crash, both
    operands provably correct in SMEM, every race probe clean.

    This is the counter-assertion for the flavors below the boundary: if a tile
    config ever pushes one of them over 256 KiB, move it into _NEEDS_DESC_V1
    and set desc_version=1 on its operand tiles -- do not delete the check."""
    mod = _load(flavor, rubin=True)
    src = open(mod.__file__).read() if getattr(mod, "__file__", None) else ""
    has_v1 = "desc_version=1" in src
    assert has_v1 is (flavor in _NEEDS_DESC_V1), f"{mod.__name__}: desc_version=1 present={has_v1}, expected={flavor in _NEEDS_DESC_V1}"


def test_sm107_f16_kernels_do_not_claim_thd():
    """THD is NOT ported: the setup-kernel call site still speaks the
    pre-upstream 7-arg contract against a 14-arg helper, and the metadata
    layout differs (3B+2 vs 4B+4).  compile() must refuse loudly rather than
    fail as an arity error deep in a trace -- and no engine row may advertise
    it."""
    mod = _load((128, 128), rubin=True, seq_kv_lens_present=True)
    assert mod.CFG.THD_VARLEN == 0


def test_rubin_f16_never_picks_a_flavor_it_has_no_kernel_for():
    """The adapter must narrow the f16 candidate pool BY ARCH LINE, or a graph
    picks a flavor with no Rubin module and dies with a KeyError inside module
    loading instead of riding the next covering envelope.

    PARTIALLY INVERTED 2026-09-04: d192xd128 now HAS a Rubin sibling
    (``prefill_d192_d128_f16_sm107.py`` -- the same body as d128 with
    ``make_cfg_d192``), so a d=192 f16 graph lowers onto the NATIVE kernel
    instead of riding the d256 envelope.  The pool-narrowing invariant is
    unchanged and is what this test really pins; the FP8/MXFP8 lines still have
    no d192 sibling, which is why the narrowing cannot be deleted.
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
    # The quantized Rubin lines have NO d192 sibling -- a d=192 FP8 graph must
    # still ride the next covering envelope rather than KeyError.
    rubin_fp8_pool = tuple(f for f in _SM100_FLAVORS if f in _SM107_FP8_KERNEL_FILES)
    assert (192, 128) not in rubin_fp8_pool
    assert _pick_flavor(192, 128, rubin_fp8_pool) == (256, 256)
    # Every flavor Rubin can pick has a module on disk.
    for flavor in rubin_pool:
        assert flavor in _SM107_KERNEL_FILES


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


def test_sm107_f16_serves_d192_through_the_d256_envelope():
    """No native d192xd128 Rubin kernel, but the graph is still served -- it
    rides the next covering envelope at that flavor's MMA cost."""
    from cudnn.sdpa.fwd import engines

    assert engines.mismatch(_caps("sdpa_fwd_prefill_sm107"), _f16_facts(d_qk=192, d_v=128)) is None


def test_sm107_f16_declines_thd():
    """INVERTS-WHEN the THD setup-kernel contract is ported: the call site
    still passes 7 args to a 14-arg helper and the metadata layout differs
    (3B+2 vs 4B+4).  Asserted on a real facts object, not just the dataclass
    field, so it exercises the path mismatch() actually walks."""
    from cudnn.sdpa.fwd import engines

    caps = _caps("sdpa_fwd_prefill_sm107")
    assert caps.thd is False
    assert engines.mismatch(caps, _f16_facts(thd=True, padded=True)) is not None


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
    """INVERTED 2026-09-04: the Rubin MXFP8 row landed (slot 16).  It is WIDER
    than its Blackwell counterpart at the top end -- Rubin has a d512 MXFP8
    kernel and SM100 does not -- and narrower at d192xd128, which has no Rubin
    MXFP8 sibling.  SM100's row stays capped at cc 10.6."""
    from cudnn.sdpa.fwd import engines

    caps = _caps("sdpa_fwd_prefill_sm107_mxfp8")
    assert caps.is_mxfp8 is True
    assert caps.d_shapes == frozenset({(128, 128), (256, 256), (512, 512)})
    assert (512, 512) not in _caps("sdpa_fwd_prefill_sm100_mxfp8").d_shapes
    assert (192, 128) not in caps.d_shapes
    # Exact-native only: the SF tensors are not zero-padded.
    assert caps.d_pad_multiple == 0
    # The machinery the ported kernels lack stays declined.
    assert caps.thd is False and caps.split_kv_supported is False
    assert caps.pack_gqas == frozenset({False})
    assert _caps("sdpa_fwd_prefill_sm100_mxfp8").sm_hi == 106
