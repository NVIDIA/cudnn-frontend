# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM107 (Rubin) routing of the per-tensor FP8 d128 SDPA kernel.

The adapter routes cc10.7 per-tensor-FP8 graphs to the SM107 sibling module
(``sm107/prefill_d128_fp8.py``), which bakes the Rubin dense-FP8 K=64 MMA
geometry; Blackwell keeps the untouched SM100 module. These tests pin the
routing and both modules' derived constants — device-independent (everything
here happens before any compile). End-to-end coverage rides the existing
``test_sdpa_fwd_fp8_sm100.py`` suite, which exercises whichever SM10x part
is present (Rubin included) through the same adapter.
"""

import pytest

from frost_test_utils import requires_dsl

from cudnn.sdpa.fwd.api_dsl import _sm100_fp8_shapes
from cudnn.sdpa.fwd.config_sm100 import TemplateParams

pytestmark = [pytest.mark.L0, requires_dsl]

_E4M3, _BF16_OUT = 0, 2


def _load(rubin, **params):
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module

    return _load_sm100_kernel_module(
        (128, 128),
        TemplateParams(dtype_qkv=_E4M3, dtype_o=_BF16_OUT, **params),
        fp8=True,
        pertensor=True,
        rubin=rubin,
    )


def test_sm107_module_bakes_rubin_geometry():
    mod = _load(rubin=True)
    assert "sm107" in mod.__name__
    # Dense-FP8 K=64 steps + the 9-stage KV ring (GR100 SMEM).
    assert (mod.CFG.TILE_K_HW_BMM1, mod.CFG.TILE_K_HW_BMM2) == (64, 64)
    assert mod.CFG.STAGES_KV == 9
    assert mod.NUM_KPHASES_PV == 2  # TILE_N / 64 — in lockstep with the idesc


def test_sm100_module_unchanged():
    mod = _load(rubin=False)
    assert "sm100" in mod.__name__
    assert (mod.CFG.TILE_K_HW_BMM1, mod.CFG.TILE_K_HW_BMM2) == (32, 32)
    assert mod.CFG.STAGES_KV == 4
    assert mod.NUM_KPHASES_PV == 4


def test_sm107_per_tensor_fp8_native_shapes():
    """INVERTED 2026-09-04: the SM107 port added d256 and d512 per-tensor FP8.
    INVERTED again 2026-09-09: d192xd128 gained its Rubin sibling
    (sm107/prefill_d192_d128_fp8.py), so both arch lines now carry all four
    native flavors and the two shape sets are identical."""
    assert _sm100_fp8_shapes(pertensor=True, device_cc=(10, 7)) == frozenset({(128, 128), (192, 128), (256, 256), (512, 512)})
    assert (192, 128) in _sm100_fp8_shapes(pertensor=True, device_cc=(10, 7))
    assert _sm100_fp8_shapes(pertensor=True, device_cc=(10, 7)) == _sm100_fp8_shapes(pertensor=True, device_cc=(10, 0))
    assert (192, 128) in _sm100_fp8_shapes(pertensor=True, device_cc=(10, 0))
    assert (256, 256) in _sm100_fp8_shapes(pertensor=True, device_cc=(10, 0))


def test_per_tensor_fp8_rows_split_per_arch_line():
    """The per-tensor FP8 rows are split at the Rubin boundary — each row
    declares exactly what its own lowering carries, with no knob x arch
    notches: kernel flavors, HALF softmax, and split/LPT capabilities are row DATA."""
    import cudnn as _c
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
    from cudnn.sdpa.fwd import engines

    caps = {s.name: s.capabilities for s in engines.ENGINE_SPECS}
    sm100 = caps[engines.engine_name(fp8=True)]
    sm107 = caps[engines.engine_name(arch="sm107", fp8=True)]

    # Arch ranges tile the SM100 family at the Rubin boundary, no overlap.
    assert (sm100.sm_lo, sm100.sm_hi) == (100, 106)
    assert (sm107.sm_lo, sm107.sm_hi) == (107, 119)
    # Kernel flavors are row DATA.  FULLY INVERTED 2026-09-09: the Rubin line
    # now carries all four per-tensor FP8 flavors, d192xd128 included, so the
    # two rows agree on d_shapes.  They still differ on THD, split-KV, PackGQA
    # and the scheduler domain -- which is the point of splitting the rows.
    assert sm100.d_shapes == frozenset({(128, 128), (192, 128), (256, 256), (512, 512)})
    assert sm107.d_shapes == frozenset({(128, 128), (192, 128), (256, 256), (512, 512)})
    assert (192, 128) in sm107.d_shapes
    # The envelope floors are arch-INDEPENDENT (api_dsl._SM100_FP8_ENVELOPE_FLOORS),
    # so a row that gains a flavor must gain its floor in the same commit --
    # otherwise mismatch() admits a graph check_support kills (contract 8b').
    assert dict(sm107.d_envelope_floors) == dict(sm100.d_envelope_floors)
    # Envelope FLOORS keep an inexact graph off a flavor whose padded path is
    # not validated.  BOTH rows carry them, and the Rubin one differs only by
    # the (192, 128) entry it has no flavor for -- the floors' rationale is the
    # kernel geometry (a cga4x1 d512 role-split; an unvalidated d256 padded
    # path), which the Rubin ports inherit unchanged.  They must also match
    # api_dsl._SM100_FP8_ENVELOPE_FLOORS, which the adapter enforces on BOTH
    # arch lines -- a row that admits what the adapter rejects is a plan that
    # enters the ranked list only to die in check_support.
    assert sm100.d_envelope_floors == (((192, 128), 128), ((256, 256), 255), ((512, 512), 256))
    # INVERTED 2026-09-09: Rubin gained the d192 flavor, so it gained that
    # flavor's floor too -- the two rows' floor tables are now identical.
    assert sm107.d_envelope_floors == (((192, 128), 128), ((256, 256), 255), ((512, 512), 256))

    # The f16x2 exponent arm is Rubin-row data, not a notch.
    assert sm100.softmax_precisions == frozenset({_c.data_type.FLOAT})
    assert sm107.softmax_precisions == frozenset({_c.data_type.FLOAT, _c.data_type.HALF})

    # Both fp8 rows now wire the split path (the SM107 sibling carries the same
    # make_split_helpers plumbing as its SM100 twin) and the LPT/LPT_L2 remap
    # (issue #653) — every SM107 decode call site threads qh_per_kh/seqlen_kv,
    # so the sched domain is the same on both rows.
    assert sm107.split_kv_supported is True
    assert sm100.split_kv_supported is True
    # The Rubin ROW-WIDE floor is NATURAL: LPT_L2 is honoured by no SM107
    # kernel (the ported decode sites never thread qh_per_kh / seqlen_kv), and
    # plain LPT is claimed PER FLAVOR through `sched_policies_by_d_shape` --
    # (256, 256) and (192, 128) since 2026-09-11, validated bit-identical to
    # NATURAL; d512 stays out (its role-split kernel still lacks the #1001
    # `lpt_q_tiles_in_cga_units` argument and writes nothing under LPT, which
    # is what the old "causal d512 FP8 under LPT returns NaN" report was).
    assert sm107.sched_policies == frozenset({SCHED_NATURAL})
    assert dict(sm107.sched_policies_by_d_shape) == {(256, 256): frozenset({SCHED_NATURAL, SCHED_LPT}), (192, 128): frozenset({SCHED_NATURAL, SCHED_LPT})}
    assert sm100.sched_policies_by_d_shape == ()
    assert sm100.sched_policies == frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2})

    # Both d128 cells carry the write_thd_meta THD leg.
    assert sm100.thd and sm107.thd and sm100.cu_seq_len and sm107.cu_seq_len


def test_sm107_row_ranks_natural_only_for_causal():
    """The LPT/LPT_L2 remap (issue #653) is an SM100-row property.  A causal
    per-tensor FP8 graph ranks [LPT_L2, LPT, NATURAL] there.  On the Rubin row
    the d128 FLAVOR's effective domain is the row-wide floor, {NATURAL} (LPT is
    claimed per flavor, at (256, 256) and (192, 128) only), so ranking must
    offer exactly [NATURAL] there and never bolt a fallback on beside it.  The
    d256 ranking is pinned in test_sdpa_fwd_dsl_sm107.py.  Pure -- facts pin
    the device, and nothing here compiles."""
    import cudnn as _c
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.fwd import engines, heuristics

    def facts(cc):
        return ga.SdpaGraphFacts(
            b=1,
            h_q=8,
            h_kv=8,
            s_q=4096,
            s_kv=4096,
            d_qk=128,
            d_v=128,
            dtype=_c.data_type.FP8_E4M3,
            dtype_o=_c.data_type.BFLOAT16,
            is_fp8=True,
            causal=True,
            device_cc=cc,
        )

    caps = {s.name: s.capabilities for s in engines.ENGINE_SPECS}
    sm100 = caps[engines.engine_name(fp8=True)]
    sm107 = caps[engines.engine_name(arch="sm107", fp8=True)]

    # One head's K+V here is 4096 * 256 * 1 B = 1 MiB, far inside the L2 budget
    # the remap groups against, so the L2 variant leads on BOTH rows. Before
    # the port the Rubin row had a one-element domain and took _sched_points'
    # sole-element shortcut, which is what pinned it to NATURAL.
    # INVERTED: the Rubin row dropped SCHED_LPT_L2 (its ported kernels raise on
    # the L2 decode), so heuristics ranks LPT first there while the SM100 twin
    # still leads with LPT_L2.  A proposal outside the row's domain would be a
    # heuristics bug, so the ranking must not offer it at all.
    # The Rubin row's domain is now a single element, so ranking has one point.
    assert heuristics._sched_points(sm107, facts((10, 7))) == [SCHED_NATURAL]
    assert heuristics._sched_points(sm100, facts((10, 0))) == [SCHED_LPT_L2, SCHED_LPT, SCHED_NATURAL]

    # The d128 LPT specialization template-LOADS (the decode is correct since
    # #1001 and bit-identical to NATURAL); the ROW withholds the claim until the
    # d128 causal path clears the suite tolerance under NATURAL too.
    assert _load(rubin=True, sched_policy=SCHED_LPT).CFG.SCHEDULER_POLICY == SCHED_LPT


def test_softmax_half_declines_by_row_domain():
    """The sdpa(softmax_precision=HALF) op attribute is a graph FACT: the sm100
    row declines it through its capability domain; the sm107 row admits it.
    Pure — facts pin the device."""
    import cudnn as _c
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.fwd import engines

    def facts(cc, softmax_precision=None):
        return ga.SdpaGraphFacts(
            b=1,
            h_q=4,
            h_kv=4,
            s_q=256,
            s_kv=256,
            d_qk=128,
            d_v=128,
            dtype=_c.data_type.FP8_E4M3,
            dtype_o=_c.data_type.BFLOAT16,
            is_fp8=True,
            device_cc=cc,
            softmax_precision=softmax_precision,
        )

    caps = {s.name: s.capabilities for s in engines.ENGINE_SPECS}
    sm100 = caps[engines.engine_name(fp8=True)]
    sm107 = caps[engines.engine_name(arch="sm107", fp8=True)]
    half = _c.data_type.HALF

    assert "softmax_precision" in engines.mismatch(sm100, facts((10, 0), half), None)
    assert engines.mismatch(sm107, facts((10, 7), half), None) is None
    # It is not a tuning knob: the knob vocabulary has no such axis.
    assert "softmax_precision" not in engines.SdpaFwdKnobs.__dataclass_fields__
    # And the rows keep their arch lanes regardless of the attribute.
    assert "SM107-119" in engines.mismatch(sm107, facts((10, 0)), None)
    assert "SM100-106" in engines.mismatch(sm100, facts((10, 7)), None)


@pytest.mark.parametrize("rubin", [True, False], ids=["sm107", "sm100"])
def test_fp8_thd_leg_loads(rubin):
    """The write_thd_meta THD leg (issue #552) is baked into BOTH fp8 d128
    siblings: a THD specialization template-loads with the flag folded in and
    exports the envelope tile constant the adapter's plan-time grid derives
    from. End-to-end THD numerics ride test_sdpa_fwd_fp8_sm100.py on whichever
    SM10x part is present (Rubin included) through the same adapter."""
    mod = _load(rubin=rubin, thd_varlen=True, seq_kv_lens_present=True)
    assert mod.CFG.THD_VARLEN == 1
    assert mod.CFG.SEQ_KV_LENS_PRESENT == 1  # THD overloads the metadata buffer
    assert mod.CGA_TILE_M == mod.CFG.TILES_Q * mod.CFG.TILE_M * mod.CFG.CTA_MMA


def test_softmax_f16_module_derivation():
    """softmax_precision=HALF folds to the SM107 sibling's f16x2 exponent
    path (SOFTMAX_F16=1); the SM100 module refuses the flag outright."""
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module

    mod = _load_sm100_kernel_module(
        (128, 128),
        TemplateParams(dtype_qkv=_E4M3, dtype_o=_BF16_OUT, softmax_f16=True),
        fp8=True,
        pertensor=True,
        rubin=True,
    )
    assert "sm107" in mod.__name__
    assert mod.SOFTMAX_F16 == 1
    # Default stays the f32 exponent chain.
    assert _load(rubin=True).SOFTMAX_F16 == 0
    with pytest.raises(ValueError, match="softmax_f16"):
        _load_sm100_kernel_module(
            (128, 128),
            TemplateParams(dtype_qkv=_E4M3, dtype_o=_BF16_OUT, softmax_f16=True),
            fp8=True,
            pertensor=True,
            rubin=False,
        )


def test_softmax_f16_rejects_half_inputs():
    """Config-validator backstop: f16/bf16 inputs never run the f16x2
    softmax (their softmax is the f32 pipeline already)."""
    from cudnn.sdpa.fwd.config_sm100 import make_cfg_d128

    _FP16 = 3
    with pytest.raises(ValueError, match="softmax_f16"):
        make_cfg_d128(TemplateParams(dtype_qkv=_FP16, softmax_f16=True))


def test_softmax_precision_is_never_a_heuristic_axis():
    """HALF is numerics-changing, so it is not a tuning axis at all: the
    heuristics' knob vocabulary has no softmax field, and the only way to get
    the f16x2 arm is the sdpa(softmax_precision=HALF) op attribute, which a
    row admits or declines through its capability domain (never degraded)."""
    import cudnn as _c
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.engines import Capabilities, SdpaFwdKnobs, mismatch

    assert "softmax_precision" not in SdpaFwdKnobs.__dataclass_fields__
    assert not hasattr(heuristics, "_softmax_points")

    base = dict(
        b=1, h_q=4, h_kv=4, s_q=256, s_kv=256, d_qk=128, d_v=128, dtype=_c.data_type.FP8_E4M3, dtype_o=_c.data_type.BFLOAT16, is_fp8=True, device_cc=(10, 7)
    )
    lit = Capabilities(
        sm_lo=100, sm_hi=119, phase="prefill", d_shapes=frozenset({(128, 128)}), softmax_precisions=frozenset({_c.data_type.FLOAT, _c.data_type.HALF})
    )
    dark = Capabilities(sm_lo=100, sm_hi=119, phase="prefill", d_shapes=frozenset({(128, 128)}))
    # No request: every row runs its f32 pipeline, nothing to gate.
    for caps in (lit, dark):
        assert "softmax_precision" not in (mismatch(caps, ga.SdpaGraphFacts(**base), None) or "")
    # An explicit HALF request: honored by the row that carries the arm, declined by the one that does not.
    half = ga.SdpaGraphFacts(**base, softmax_precision=_c.data_type.HALF)
    # The row that carries the arm reaches the same verdict as for no request,
    # i.e. the softmax gate (which runs before every other gate) passed. Compared
    # with the no-request verdict rather than asserted None so the unit stays
    # device- and DSL-independent (the later gates need both).
    assert mismatch(lit, half, None) == mismatch(lit, ga.SdpaGraphFacts(**base), None)
    assert "softmax_precision" in mismatch(dark, half, None)


@requires_dsl
def test_softmax_precision_knob_gate():
    """Vocabulary and arch gate: unknown dtypes raise; HALF is declined
    everywhere except per-tensor FP8 on cc10.7; FLOAT is accepted on the
    per-tensor FP8 path (it is the pipeline that path already runs)."""
    import torch
    import cudnn as _c

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("needs an fp8-admitted SM10x part")

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    B, H, S, D = 1, 1, 256, 128
    dev = "cuda"
    q8 = torch.zeros(B, H, S, D, device=dev, dtype=torch.float8_e4m3fn)
    o = torch.empty(B, H, S, D, device=dev, dtype=torch.bfloat16)
    lse = torch.empty(B, H, S, device=dev, dtype=torch.float32)

    def mk(precision):
        return SdpaFwdDslSm100(
            sample_q=q8, sample_k=q8, sample_v=q8, sample_o=o, sample_lse=lse, scale_softmax=0.1, pertensor_fp8=True, softmax_precision=precision
        )

    with pytest.raises(ValueError, match="softmax_precision"):
        mk(_c.data_type.BFLOAT16).check_support()

    assert mk(_c.data_type.FLOAT).check_support()  # the pipeline this path runs

    if torch.cuda.get_device_capability() == (10, 7):
        assert mk(_c.data_type.HALF).check_support()
    else:
        with pytest.raises(ValueError, match="HALF"):
            mk(_c.data_type.HALF).check_support()


@requires_dsl
def test_fp8_softmax_f16_e2e():
    """Rubin e2e: the f16x2 exponent path against the FLOAT run of the same
    problem — same kernel family, same quantized-P contract, so the two
    agree to fp8-quantization noise."""
    import torch
    import cudnn as _c

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("softmax_precision=HALF serves cc10.7 only")

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    torch.manual_seed(0)
    B, H, S, D = 2, 4, 512, 128
    dev = "cuda"
    Q8, K8, V8 = ((torch.randn(B, H, S, D, device=dev) * 0.5).to(torch.float8_e4m3fn) for _ in range(3))
    lse = torch.empty(B, H, S, device=dev, dtype=torch.float32)
    outs = {}
    for precision in (_c.data_type.FLOAT, _c.data_type.HALF):
        out = torch.empty(B, H, S, D, device=dev, dtype=torch.bfloat16)
        api = SdpaFwdDslSm100(
            sample_q=Q8, sample_k=K8, sample_v=V8, sample_o=out, sample_lse=lse, scale_softmax=D**-0.5, pertensor_fp8=True, softmax_precision=precision
        )
        assert api.check_support()
        api.compile()
        api.execute(q_tensor=Q8, k_tensor=K8, v_tensor=V8, o_tensor=out, lse_tensor=lse)
        torch.cuda.synchronize()
        outs[precision] = out.float()

    ref = torch.softmax(Q8.float() @ K8.float().transpose(-1, -2) * D**-0.5, dim=-1) @ V8.float()
    for precision, out in outs.items():
        err = (out - ref).abs().max().item()
        assert err <= 0.1 * ref.abs().max().item(), f"{precision}: max err {err} vs fp32 reference"
    xerr = (outs[_c.data_type.HALF] - outs[_c.data_type.FLOAT]).abs().max().item()
    assert xerr <= 0.05 * ref.abs().max().item(), f"HALF-vs-FLOAT softmax divergence {xerr}"


def test_fp8_rows_serve_dense_envelope():
    """BOTH per-tensor FP8 rows (sm100 and sm107) serve the dense head-dim
    ENVELOPE of their kernel flavors (TMA zero-padding — exact in FP8;
    d % 16 at 1 byte/elem, and arch-independent since the descales are
    scalars). THD stays native-tile (the packed THD compile key carries no
    head-dim entries) and MXFP8 stays exact-native (d_pad_multiple=0)."""
    from cudnn.sdpa.fwd import engines

    caps = {s.name: s.capabilities for s in engines.ENGINE_SPECS}
    for arch in ("sm100", "sm107"):
        row = caps[engines.engine_name(arch=arch, fp8=True)]
        assert row.d_pad_multiple == 16, arch
    # FULLY INVERTED 2026-09-09: the Rubin per-tensor FP8 THD leg now covers every
    # native flavor -- d192xd128 came free with the DSv3 port (same body as d128),
    # and d256/d512 were moved onto the FROST THD contract.  Both lines now serve
    # THD at every shape they serve dense.
    rubin_fp8 = caps[engines.engine_name(arch="sm107", fp8=True)]
    assert rubin_fp8.thd_d_shapes == rubin_fp8.d_shapes
    assert caps[engines.engine_name(fp8=True)].thd_d_shapes == frozenset({(128, 128), (192, 128), (256, 256), (512, 512)})

    # The row and the STANDALONE wrapper enforce the same fact at two places
    # (rule 8b'), so they now share ONE constant instead of two copies kept in
    # step by hand -- which is what failed when the row was widened first and a
    # d192 THD graph died with a bare NotImplementedError inside check_support.
    # Assert IDENTITY with the shared object, not equality with a literal: a
    # literal here would just be a third copy to drift.
    from cudnn.sdpa.fwd.api_dsl import _SM107_FP8_THD_SHAPES
    from cudnn.sdpa.fwd.config_sm107 import SM107_FP8_THD_SHAPES

    assert rubin_fp8.thd_d_shapes is SM107_FP8_THD_SHAPES
    assert _SM107_FP8_THD_SHAPES is SM107_FP8_THD_SHAPES
    # Every THD shape must also be a shape the row SERVES at all.
    assert SM107_FP8_THD_SHAPES <= rubin_fp8.d_shapes
    assert caps[engines.engine_name(arch="sm100", fp8=True)].thd_d_shapes == frozenset({(128, 128), (192, 128), (256, 256), (512, 512)})
    assert caps[engines.engine_name(mxfp8=True)].d_pad_multiple == 0


def _fp8_facts(**kw):
    import cudnn
    from cudnn.sdpa import graph_analyzer as ga

    base = dict(
        b=2,
        h_q=4,
        h_kv=4,
        s_q=384,
        s_kv=384,
        d_qk=80,
        d_v=80,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.BFLOAT16,
        is_fp8=True,
        device_cc=(10, 0),
    )
    base.update(kw)
    return ga.SdpaGraphFacts(**base)


def test_fp8_envelope_mismatch_rules():
    """Honest eligibility for the fp8 envelope: mismatch() admits dense d80 on
    the d128 rows of both arch lines, enforces d % 16, and keeps THD
    native-tile — no plan may enter the ranked list only to die at build in
    the adapter."""
    from cudnn.sdpa.fwd import engines

    caps = {s.name: s.capabilities for s in engines.ENGINE_SPECS}
    sm100 = caps[engines.engine_name(fp8=True)]
    assert engines.mismatch(sm100, _fp8_facts()) is None
    assert engines.mismatch(sm100, _fp8_facts(d_qk=96, d_v=64)) is None
    # The d192xd128 and D256 flavors serve ONLY their exact shapes (floors 128 / 255):
    # d_qk zero-padded into d192 is numerically wrong and the D256 padded envelope is
    # nondeterministic in test_mhas_v2, so the row declines the whole (128, 256) inexact
    # region and the classic backend verdict applies there.
    assert engines.mismatch(sm100, _fp8_facts(d_qk=192, d_v=128)) is None
    assert engines.mismatch(sm100, _fp8_facts(d_qk=256, d_v=256)) is None
    for dq, dv in ((160, 96), (176, 128), (192, 96), (160, 160), (224, 208)):
        assert "no kernel-flavor envelope" in engines.mismatch(sm100, _fp8_facts(d_qk=dq, d_v=dv)), (dq, dv)
    assert engines._selected_d_shape(sm100, _fp8_facts(d_qk=192, d_v=128)) == (192, 128)
    assert engines._selected_d_shape(sm100, _fp8_facts(d_qk=256, d_v=256)) == (256, 256)
    assert "no kernel-flavor envelope" in engines.mismatch(sm100, _fp8_facts(d_qk=272, d_v=256))
    assert "multiples of 16" in engines.mismatch(sm100, _fp8_facts(d_qk=88, d_v=88))
    assert "dense-only" in engines.mismatch(sm100, _fp8_facts(thd=True, padded=True))
    assert engines.mismatch(sm100, _fp8_facts(d_qk=128, d_v=128, thd=True, padded=True)) is None
    # SM100 carries THD at each native per-tensor FP8 shape.
    assert engines.mismatch(sm100, _fp8_facts(d_qk=192, d_v=128, thd=True, padded=True)) is None
    assert engines.mismatch(sm100, _fp8_facts(d_qk=256, d_v=256)) is None
    assert engines.mismatch(sm100, _fp8_facts(d_qk=256, d_v=256, thd=True, padded=True)) is None
    # d512 is a NATIVE shape (accepted exactly, dense and THD) and serves the
    # (256, 512] envelope band on BOTH head dims -- at most 2x zero-padding.
    assert engines.mismatch(sm100, _fp8_facts(d_qk=512, d_v=512)) is None
    assert engines.mismatch(sm100, _fp8_facts(d_qk=512, d_v=512, thd=True, padded=True)) is None
    assert engines.mismatch(sm100, _fp8_facts(d_qk=384, d_v=448)) is None
    assert engines.mismatch(sm100, _fp8_facts(d_qk=464, d_v=368)) is None
    assert engines.mismatch(sm100, _fp8_facts(d_qk=272, d_v=272)) is None
    # Straddling the d512 floor declines rather than routing onto that kernel.
    assert "no kernel-flavor envelope" in engines.mismatch(sm100, _fp8_facts(d_qk=512, d_v=256))
    assert "no kernel-flavor envelope" in engines.mismatch(sm100, _fp8_facts(d_qk=384, d_v=128))
    # THD stays native-tile: the (256, 512] envelope band is dense-only.
    assert "dense-only" in engines.mismatch(sm100, _fp8_facts(d_qk=384, d_v=448, thd=True, padded=True))
    # d % 16 still applies inside the band (TMA 16-byte global-stride rule).
    assert "multiples of 16" in engines.mismatch(sm100, _fp8_facts(d_qk=392, d_v=392))
    # The Rubin row serves the same dense envelope (the ViT d=72-in-80 case).
    sm107 = caps[engines.engine_name(arch="sm107", fp8=True)]
    assert engines.mismatch(sm107, _fp8_facts(device_cc=(10, 7))) is None
    # INVERTED 2026-09-09: the Rubin d192 FP8 sibling landed, so the EXACT
    # shape is now served on both rows.  Its floor (128) came with it, so the
    # inexact region below stays declined -- d_qk zero-padded into d192 is
    # numerically wrong, and that is a kernel property, not an arch one.
    assert engines.mismatch(sm107, _fp8_facts(device_cc=(10, 7), d_qk=192, d_v=128)) is None
    assert engines._selected_d_shape(sm107, _fp8_facts(device_cc=(10, 7), d_qk=192, d_v=128)) == (192, 128)
    for dq, dv in ((160, 96), (176, 128), (192, 96)):
        assert "no kernel-flavor envelope" in engines.mismatch(sm107, _fp8_facts(device_cc=(10, 7), d_qk=dq, d_v=dv)), (dq, dv)
    assert "dense-only" in engines.mismatch(sm107, _fp8_facts(device_cc=(10, 7), thd=True, padded=True))
    # INVERTED: the Rubin line gained a d512 per-tensor FP8 kernel, so the
    # native d512 shape is now SERVED rather than declined, and the (256, 512]
    # floor applies here exactly as on the SM100 row.
    assert engines.mismatch(sm107, _fp8_facts(device_cc=(10, 7), d_qk=512, d_v=512)) is None
    assert engines.mismatch(sm107, _fp8_facts(device_cc=(10, 7), d_qk=384, d_v=448)) is None
    # ...and straddling the floor declines on BOTH rows, identically.
    assert "no kernel-flavor envelope" in engines.mismatch(sm107, _fp8_facts(device_cc=(10, 7), d_qk=512, d_v=256))
    assert "no kernel-flavor envelope" in engines.mismatch(sm100, _fp8_facts(d_qk=512, d_v=256))


# --- KV split on Rubin -------------------------------------------------------


def test_sm107_split_is_wired_only_for_per_tensor_fp8_d128():
    """config_sm107 permits the split for the ONE Rubin kernel that wires
    make_split_helpers, and refuses it for every other flavor.

    A blanket refusal contradicted the engine row, which advertises
    ``split_d_shapes={(128, 128)}`` on the Rubin FP8 row: a long-KV Rubin graph
    could be handed an automatically proposed split plan and then fail at
    compile. The gate follows the ROW, not merely whether a body wires
    SplitHelpers -- d192xd128 FP8 does, but is neither advertised nor carries an
    o_partial_f32 slot, so it must still decline."""
    from cudnn.sdpa.fwd import config_sm107 as cfg

    def tp(**kw):
        return cfg.TemplateParams(split_kv=4, **kw)

    # The wired cell builds.
    cfg.make_cfg_d128(tp(dtype_qkv=_E4M3, dtype_o=_BF16_OUT))

    # Every other Rubin cell still refuses -- same entry point for the half
    # d128 and d192 kernels, so the gate cannot key on the flavor string alone.
    unwired = [
        ("d128 half", cfg.make_cfg_d128, dict(dtype_qkv=_BF16_OUT, dtype_o=_BF16_OUT)),
        ("d128 mxfp8", cfg.make_cfg_d128_mxfp8, dict(dtype_qkv=_E4M3, dtype_o=_BF16_OUT)),
        ("d192", cfg.make_cfg_d192, dict(dtype_qkv=_BF16_OUT, dtype_o=_BF16_OUT)),
        ("d256", cfg.make_cfg_d256, dict(dtype_qkv=_E4M3, dtype_o=_BF16_OUT)),
        ("d512", cfg.make_cfg_d512, dict(dtype_qkv=_E4M3, dtype_o=_BF16_OUT)),
    ]
    for name, make, params in unwired:
        with pytest.raises(ValueError, match="split_kv > 1 is not wired"):
            make(tp(**params))
        assert make(cfg.TemplateParams(**params)) is not None, f"{name}: unsplit must still build"


def test_sm107_split_matches_unsplit_on_rubin():
    """End-to-end split numerics on cc10.7 silicon.

    The one executing test in this otherwise device-independent module: the
    SM107 kernel's split path -- and the fp32 partial store it now takes
    unconditionally -- has no other hardware coverage, since the split-KV suite
    is marked pre-Rubin."""
    import math

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("cc10.7 (Rubin) part required")

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    b, h_q, s_q, s_kv, d, dev = 1, 8, 512, 8192, 128, "cuda"
    torch.manual_seed(0)

    def run(split):
        def mk(*sh):
            return (torch.randn(*sh, device=dev) * 0.5).to(torch.float8_e4m3fn)

        torch.manual_seed(0)
        q, k, v = mk(b, h_q, s_q, d), mk(b, 1, s_kv, d), mk(b, 1, s_kv, d)
        o = torch.zeros(b, h_q, s_q, d, device=dev, dtype=torch.float16)
        one = torch.ones(1, dtype=torch.float32, device=dev)
        api = SdpaFwdDslSm100(
            sample_q=q, sample_k=k, sample_v=v, sample_o=o, dtype_o=torch.float16, split_kv=split, pertensor_fp8=True, scale_softmax=1.0 / math.sqrt(d)
        )
        assert api.check_support()
        api.compile()
        wsb = api.scratch_workspace_bytes()
        ws = torch.empty(wsb, dtype=torch.uint8, device=dev) if wsb else None
        api.execute(q_tensor=q, k_tensor=k, v_tensor=v, o_tensor=o, workspace=ws, descale_q=one, descale_k=one, descale_v=one, scale_o=one)
        torch.cuda.synchronize()
        return api, o.float().clone(), (q, k, v)

    _, base, _ = run(1)
    for split in (2, 4, 8):
        api, got, (q, k, v) = run(split)
        assert api._fp32_partial_split(), "the Rubin split must take the fp32-partial path"
        assert not torch.isnan(got).any(), f"split={split}: NaN"
        qf = q.double()
        kf, vf = (t.double().repeat_interleave(h_q, dim=1) for t in (k, v))
        ref = torch.softmax(qf @ kf.transpose(-1, -2) / math.sqrt(d), dim=-1) @ vf
        assert (got.double() - ref).abs().max().item() <= 5e-2, f"split={split}: off the oracle"
        assert (got - base).abs().max().item() <= 5e-2, f"split={split}: diverges from unsplit"


@requires_dsl
@pytest.mark.parametrize("d_qk, d_v", [(256, 256), (192, 128)])
@pytest.mark.parametrize("causal, b, s", [(True, 2, 1000), (False, 1, 1024)])
def test_fp8_lpt_is_bit_identical_to_natural_on_the_claimed_flavors(d_qk, d_v, causal, b, s):
    """Rubin e2e for the FP8 (256, 256) and (192, 128) LPT claims: the same
    quantized problem under SCHED_LPT and SCHED_NATURAL.  The scheduler only
    reorders whole (batch, head, q-tile) work items -- each tile's KV loop is
    unchanged -- so O must be BIT-IDENTICAL across the two policies (measured
    0.0 on every case, 2026-09-11), and both stay within the module's fp32
    oracle bound.  S=1000 causal covers a KV tail; dense needs S % 128 == 0."""
    import torch
    import cudnn as _c

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the sm107 FP8 kernels serve cc10.7 only")
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_NATURAL
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    torch.manual_seed(0)
    hq, hkv = 8, 2
    dev = "cuda"
    fmax = 448.0

    def quant(x):
        dsc = (x.abs().amax().clamp_min(1e-8) / fmax).item()
        return (x / dsc).clamp(-fmax, fmax).to(torch.float8_e4m3fn), torch.full((1,), dsc, device=dev, dtype=torch.float32)

    q8, dq = quant(torch.randn(b, hq, s, d_qk, device=dev) * 0.5)
    k8, dk = quant(torch.randn(b, hkv, s, d_qk, device=dev) * 0.5)
    v8, dv = quant(torch.randn(b, hkv, s, d_v, device=dev) * 0.5)
    outs = {}
    for pol in (SCHED_NATURAL, SCHED_LPT):
        out = torch.full((b, hq, s, d_v), 1.5e30, device=dev, dtype=torch.bfloat16)  # sentinel: an unclaimed tile stays visible
        api = SdpaFwdDslSm100(
            sample_q=q8, sample_k=k8, sample_v=v8, sample_o=out, scale_softmax=d_qk**-0.5, is_causal=causal, pertensor_fp8=True, sched_policy=pol
        )
        assert api.check_support()
        api.compile()
        api.execute(q_tensor=q8, k_tensor=k8, v_tensor=v8, o_tensor=out, descale_q=dq, descale_k=dk, descale_v=dv)
        torch.cuda.synchronize()
        outs[pol] = out.clone()
    rep = hq // hkv
    qf, kf, vf = q8.float() * dq, (k8.float() * dk).repeat_interleave(rep, 1), (v8.float() * dv).repeat_interleave(rep, 1)
    logits = qf @ kf.transpose(-1, -2) * d_qk**-0.5
    if causal:
        logits = logits.masked_fill(~torch.tril(torch.ones(s, s, dtype=torch.bool, device=dev)), float("-inf"))
    ref = torch.softmax(logits, dim=-1) @ vf
    scale = ref.abs().max().item()
    for pol, out in outs.items():
        assert torch.isfinite(out).all(), f"policy {pol}: non-finite / unwritten cells"
        err = (out.float() - ref).abs().max().item()
        assert err <= 0.1 * scale, f"policy {pol}: max err {err} vs fp32 reference (scale {scale})"
    assert torch.equal(outs[SCHED_LPT], outs[SCHED_NATURAL]), "LPT must be bit-identical to NATURAL -- a different tile walk, the same per-tile math"
