# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM107 (Rubin) routing of the per-tensor FP8 d128 SDPA kernel.

The adapter routes cc10.7 per-tensor-FP8 graphs to the SM107 sibling module
(``prefill_d128_fp8_sm107.py``), which bakes the Rubin dense-FP8 K=64 MMA
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


def test_sm107_per_tensor_fp8_advertises_only_d128():
    assert _sm100_fp8_shapes(pertensor=True, device_cc=(10, 7)) == frozenset({(128, 128)})
    assert (192, 128) in _sm100_fp8_shapes(pertensor=True, device_cc=(10, 0))


def test_per_tensor_fp8_rows_split_per_arch_line():
    """The per-tensor FP8 rows are split at the Rubin boundary — each row
    declares exactly what its own lowering carries, with no knob x arch
    notches: HALF softmax and the missing split/LPT paths are row DATA."""
    import cudnn as _c
    from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
    from cudnn.sdpa.fwd import engines

    caps = {s.name: s.capabilities for s in engines.ENGINE_SPECS}
    sm100 = caps[engines.engine_name(fp8=True)]
    sm107 = caps[engines.engine_name(arch="sm107", fp8=True)]

    # Arch ranges tile the SM100 family at the Rubin boundary, no overlap.
    assert (sm100.sm_lo, sm100.sm_hi) == (100, 106)
    assert (sm107.sm_lo, sm107.sm_hi) == (107, 119)
    # Kernel flavors are row DATA: Rubin has no d192 sibling.
    assert sm100.d_shapes == frozenset({(128, 128), (192, 128)})
    assert sm107.d_shapes == frozenset({(128, 128)})

    # The f16x2 exponent arm is Rubin-row data, not a notch.
    assert sm100.softmax_precisions == frozenset({_c.data_type.FLOAT})
    assert sm107.softmax_precisions == frozenset({_c.data_type.FLOAT, _c.data_type.HALF})

    # The SM107 sibling has no split path and no LPT remap yet (issue #653).
    assert sm107.split_kvs == frozenset({1})
    assert sm100.split_kvs == frozenset({1, 2, 4})
    assert sm107.sched_policies == frozenset({SCHED_NATURAL})
    assert sm100.sched_policies == frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2})

    # Both d128 cells carry the write_thd_meta THD leg.
    assert sm100.thd and sm107.thd and sm100.cu_seq_len and sm107.cu_seq_len


def test_softmax_half_declines_by_row_domain():
    """A HALF request on the sm100 row declines through the generic knob
    domain gate; the sm107 row admits it. Pure — facts pin the device."""
    import cudnn as _c
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.fwd import engines

    def facts(cc):
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
        )

    caps = {s.name: s.capabilities for s in engines.ENGINE_SPECS}
    sm100 = caps[engines.engine_name(fp8=True)]
    sm107 = caps[engines.engine_name(arch="sm107", fp8=True)]
    half = engines.SdpaFwdKnobs(softmax_precision=_c.data_type.HALF)

    assert "domain" in engines.mismatch(sm100, facts((10, 0)), half)
    assert engines.mismatch(sm107, facts((10, 7)), half) is None
    # And the rows keep their arch lanes regardless of the knob.
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


def test_softmax_points_never_propose_half():
    """propose() fills the axis with FLOAT where the row serves it — HALF is
    numerics-changing and reachable by explicit request only."""
    import cudnn as _c
    from cudnn.sdpa.fwd.engines import Capabilities
    from cudnn.sdpa.fwd.heuristics import _softmax_points

    lit = Capabilities(
        sm_lo=100,
        sm_hi=119,
        phase="prefill",
        d_shapes=frozenset({(128, 128)}),
        softmax_precisions=frozenset({_c.data_type.FLOAT, _c.data_type.HALF}),
    )
    assert _softmax_points(lit) == [_c.data_type.FLOAT]
    dark = Capabilities(sm_lo=100, sm_hi=119, phase="prefill", d_shapes=frozenset({(128, 128)}))
    assert _softmax_points(dark) == [None]
    # A HALF-only row must still not get HALF auto-proposed (numerics-changing).
    half_only = Capabilities(sm_lo=100, sm_hi=119, phase="prefill", d_shapes=frozenset({(128, 128)}), softmax_precisions=frozenset({_c.data_type.HALF}))
    assert _softmax_points(half_only) == [None]


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
        assert row.thd_d_shapes == frozenset({(128, 128)}), arch
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
    # The d192xd128 flavor serves its envelope too (kernel takes d_qk/d_v).
    assert engines.mismatch(sm100, _fp8_facts(d_qk=160, d_v=96)) is None
    assert "no kernel-flavor envelope" in engines.mismatch(sm100, _fp8_facts(d_qk=160, d_v=160))
    assert "multiples of 16" in engines.mismatch(sm100, _fp8_facts(d_qk=88, d_v=88))
    assert "dense-only" in engines.mismatch(sm100, _fp8_facts(thd=True, padded=True))
    assert engines.mismatch(sm100, _fp8_facts(d_qk=128, d_v=128, thd=True, padded=True)) is None
    # THD at the d192 native shape is dense-only (thd_d_shapes = {(128, 128)}).
    assert "dense-only" in engines.mismatch(sm100, _fp8_facts(d_qk=192, d_v=128, thd=True, padded=True))
    # The Rubin row serves the same dense envelope (the ViT d=72-in-80 case)
    # but has no d192 flavor at all.
    sm107 = caps[engines.engine_name(arch="sm107", fp8=True)]
    assert engines.mismatch(sm107, _fp8_facts(device_cc=(10, 7))) is None
    assert "no kernel-flavor envelope" in engines.mismatch(sm107, _fp8_facts(device_cc=(10, 7), d_qk=192, d_v=128))
    assert "dense-only" in engines.mismatch(sm107, _fp8_facts(device_cc=(10, 7), thd=True, padded=True))
