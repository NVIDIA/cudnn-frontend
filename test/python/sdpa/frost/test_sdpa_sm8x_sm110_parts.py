# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SM80 rows on cc 8.6 / 8.9 (the shared-memory budget picks the tiles and levers) and Thor (cc 11.0) on the SM100-line rows."""

import pytest

import cudnn
from cudnn.frost import buffers
from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa import native_sm80
from cudnn.sdpa.bwd import config_sm80 as bwd_config
from cudnn.sdpa.bwd import engines as bwd_engines
from cudnn.sdpa.fwd import config_sm80 as fwd_config
from cudnn.sdpa.fwd import engines as fwd_engines

pytestmark = [pytest.mark.L0]

A100, GA10X = native_sm80.SMEM_OPTIN_BYTES[(8, 0)], native_sm80.SMEM_OPTIN_BYTES[(8, 9)]


def _fwd_facts(**over):
    base = dict(b=2, h_q=4, h_kv=4, s_q=512, s_kv=512, d_qk=128, d_v=128, dtype=cudnn.data_type.BFLOAT16, dtype_o=cudnn.data_type.BFLOAT16, device_cc=(8, 9))
    base.update(over)
    return ga.SdpaGraphFacts(**base)


def _bwd_facts(**over):
    return _fwd_facts(is_backward=True, **over)


def _row(specs, name):
    return next(s for s in specs if s.name == name).capabilities


def _family(specs, lo):
    """The half-precision rows of one arch line: the policy row and the explicit implementations it delegates to."""
    return [s.capabilities for s in specs if s.capabilities.sm_lo == lo and not (s.capabilities.is_fp8 or s.capabilities.is_mxfp8)]


def _served(rows, mismatch, facts):
    """Some row of the family admits the graph (a policy row may delegate it to one of its explicit implementations)."""
    return any(mismatch(c, facts) is None for c in rows)


def test_part_table():
    assert native_sm80.part_mismatch((8, 0), "x") is None and native_sm80.part_mismatch((8, 6), "x") is None and native_sm80.part_mismatch((8, 9), "x") is None
    assert "found SM87" in native_sm80.part_mismatch((8, 7), "x") and "found SM90" in native_sm80.part_mismatch((9, 0), "x")
    assert native_sm80.part_mismatch((8, 6), "x", native_sm80.SM80_BACKWARD_PARTS) is None
    assert "requires SM80 or SM86; found SM89" in native_sm80.part_mismatch((8, 9), "x", native_sm80.SM80_BACKWARD_PARTS)
    assert native_sm80.smem_budget_bytes((8, 0)) == A100 and native_sm80.smem_budget_bytes((8, 6)) == GA10X and native_sm80.smem_budget_bytes((9, 0)) is None


@pytest.mark.parametrize(
    "dims,tile,fits_ga10x",
    [
        ((64, 64), (128, 64), True),
        ((128, 128), (128, 64), True),
        ((192, 128), (128, 64), True),
        ((256, 256), (128, 64), False),
        ((256, 256), (128, 32), True),
        ((256, 256), (64, 64), False),
        ((256, 256), (64, 32), True),
        ((512, 512), (32, 16), True),
        ((576, 512), (32, 16), True),
    ],
)
def test_prefill_smem_model(dims, tile, fits_ga10x):
    used = fwd_config.prefill_smem_bytes(*dims, *tile)
    assert used <= A100 and (used <= GA10X) == fits_ga10x, used


def test_backward_smem_model_and_levers():
    assert bwd_config.d64_smem_bytes() <= GA10X
    assert bwd_config.envelope_dims(100, 100) == (128, 128) and bwd_config.envelope_dims(160, 128) == (192, 128) and bwd_config.envelope_dims(600, 600) is None
    for dims in ((64, 64), (128, 128), (192, 128), (256, 256)):
        assert bwd_config.levers_within_budget(*dims, A100) == bwd_config.default_levers(dims[0]), dims
    assert bwd_config.levers_within_budget(64, 64, GA10X) == bwd_config.default_levers(64)
    assert bwd_config.levers_within_budget(128, 128, GA10X) == (1, False)
    assert bwd_config.levers_within_budget(128, 128, GA10X, has_rope=True) is None
    assert bwd_config.levers_within_budget(192, 128, GA10X) is None and bwd_config.levers_within_budget(256, 256, GA10X) is None
    assert bwd_config.backward_smem_mismatch(128, 128, GA10X) is None
    assert "D_QK=192/D_V=128" in bwd_config.backward_smem_mismatch(160, 128, GA10X)
    assert bwd_config.backward_smem_mismatch(256, 256, A100) is None
    params = bwd_config.bwd_params_for_flavor("llama", d_qk=128, d_v=128)
    assert bwd_config.resolved_levers(params) == (2, True)
    assert bwd_config.resolved_levers(bwd_config.bwd_params_for_flavor("llama", d_qk=128, d_v=128, qo_stages=1, dq_smem_coalesce=False)) == (1, False)


def test_sm80_rows_admit_the_named_parts_only():
    fwd, bwd = _family(fwd_engines.ENGINE_SPECS, 80), _family(bwd_engines.ENGINE_SPECS, 80)
    assert fwd and bwd and all(c.sm_extra == frozenset({86, 89}) for c in fwd) and all(c.sm_extra == frozenset({86}) for c in bwd)
    for cc in ((8, 0), (8, 6), (8, 9)):
        assert _served(fwd, fwd_engines.mismatch, _fwd_facts(device_cc=cc)), cc
    for cc in ((8, 0), (8, 6)):
        assert _served(bwd, bwd_engines.mismatch, _bwd_facts(device_cc=cc)), cc
    assert "SM80-80 or SM86 or SM89" in fwd_engines.mismatch(_row(fwd_engines.ENGINE_SPECS, "sdpa_fwd_prefill_sm80"), _fwd_facts(device_cc=(8, 7)))
    assert "SM80-80 or SM86" in bwd_engines.mismatch(_row(bwd_engines.ENGINE_SPECS, "sdpa_bwd_sm80"), _bwd_facts(device_cc=(9, 0)))
    assert all("SM8" in fwd_engines.mismatch(c, _fwd_facts(device_cc=(8, 7))) for c in fwd)
    assert all("SM8" in bwd_engines.mismatch(c, _bwd_facts(device_cc=(9, 0))) for c in bwd)
    assert all("SM8" in bwd_engines.mismatch(c, _bwd_facts(device_cc=(8, 9))) for c in bwd)
    ada = [_row(bwd_engines.ENGINE_SPECS, n) for n in ("sdpa_bwd_sm89", "sdpa_bwd_sm89_f16_d64")]
    assert all(c.sm_extra == frozenset() for c in ada)
    assert _served(ada, bwd_engines.mismatch, _bwd_facts(device_cc=(8, 9)))
    assert all("SM89" in bwd_engines.mismatch(c, _bwd_facts(device_cc=(8, 6))) for c in ada)


def test_sm80_backward_rows_decline_the_envelopes_that_cannot_fit():
    bwd = _family(bwd_engines.ENGINE_SPECS, 80)
    cc = (8, 6)
    assert _served(bwd, bwd_engines.mismatch, _bwd_facts(device_cc=cc, d_qk=64, d_v=64))
    assert _served(bwd, bwd_engines.mismatch, _bwd_facts(device_cc=cc))
    for d_qk, d_v in ((192, 128), (256, 256)):
        reasons = [bwd_engines.mismatch(c, _bwd_facts(device_cc=cc, d_qk=d_qk, d_v=d_v)) for c in bwd]
        assert all(reasons) and any("shared memory" in r for r in reasons), reasons
    assert _served(bwd, bwd_engines.mismatch, _bwd_facts(device_cc=(8, 0), d_qk=256, d_v=256))
    ada = _row(bwd_engines.ENGINE_SPECS, "sdpa_bwd_sm89")
    for d_qk, d_v in ((128, 128), (192, 128), (256, 256)):
        assert bwd_engines.mismatch(ada, _bwd_facts(device_cc=(8, 9), d_qk=d_qk, d_v=d_v)) is None, (d_qk, d_v)


def test_thor_rides_the_sm100_rows():
    thor = (11, 0)
    assert thor in fwd_engines._THD_ORIGIN_CCS
    assert _served(_family(fwd_engines.ENGINE_SPECS, 100), fwd_engines.mismatch, _fwd_facts(device_cc=thor))
    rubin_fwd = _family(fwd_engines.ENGINE_SPECS, 107) + [
        _row(fwd_engines.ENGINE_SPECS, n) for n in ("sdpa_fwd_prefill_sm107_mxfp8", "sdpa_fwd_prefill_sm107_fp8")
    ]
    assert all("SM107" in fwd_engines.mismatch(c, _fwd_facts(device_cc=thor)) for c in rubin_fwd)
    assert fwd_engines.mismatch(_row(fwd_engines.ENGINE_SPECS, "sdpa_fwd_prefill_sm100"), _fwd_facts(device_cc=(11, 5))).startswith(
        "requires SM100-106 or SM110"
    )
    sm100_bwd = _family(bwd_engines.ENGINE_SPECS, 100)
    for over in (dict(d_qk=512, d_v=512), {}, dict(d_qk=192, d_v=128), dict(d_qk=256, d_v=256)):
        assert _served(sm100_bwd, bwd_engines.mismatch, _bwd_facts(device_cc=thor, **over)), over
    assert all("SM107" in bwd_engines.mismatch(c, _bwd_facts(device_cc=thor, d_qk=256, d_v=256)) for c in _family(bwd_engines.ENGINE_SPECS, 107))
    for name in ("sdpa_bwd_sm100_d128_pair", "sdpa_bwd_sm100_d128_accumulation_2cta"):
        assert "requires SM" in bwd_engines.mismatch(_row(bwd_engines.ENGINE_SPECS, name), _bwd_facts(device_cc=thor)), name


def test_dsl_target_requirement_names_the_part(monkeypatch):
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    monkeypatch.setattr(buffers, "_cutedsl_has_sm110", lambda: False)
    assert buffers.cutedsl_arch_requirement_error((10, 0)) is None and buffers.cutedsl_arch_requirement_error((10, 7)) is None
    assert "SM110 requires a CuTe DSL build supporting sm_110a" in buffers.cutedsl_arch_requirement_error((11, 0))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm110", lambda: True)
    assert buffers.cutedsl_arch_requirement_error((11, 0)) is None
    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: False)
    assert "SM107 requires a CuTe DSL build supporting sm_107a" in buffers.cutedsl_arch_requirement_error((10, 7))


def test_sm80_policy_row_delegates_on_every_admitted_part():
    policy = _row(fwd_engines.ENGINE_SPECS, "sdpa_fwd_prefill_sm80")
    for cc in native_sm80.SM80_ROW_PARTS:
        for d_qk, d_v in ((64, 64), (128, 128), (192, 128), (256, 256), (512, 512), (576, 512)):
            reason = fwd_engines.mismatch(policy, _fwd_facts(device_cc=cc, d_qk=d_qk, d_v=d_v))
            assert reason is not None and "owned by the explicit implementation" in reason, (cc, d_qk, d_v, reason)


def _thd_api(case_kwargs, api_type):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from d128_thd_test_utils import make_api, make_case

    return make_api(make_case((65, 0, 3), (17, 129, 0), **case_kwargs), api_type=api_type)


@pytest.mark.parametrize(
    "module,name,case_kwargs",
    [
        ("cudnn.sdpa.bwd.api_dsl_sm120_thd", "SdpaBwdDslSm120Thd", {}),
        ("cudnn.sdpa.bwd.api_dsl_sm120_wide", "SdpaBwdDslSm120Wide", dict(heads=6, kv_heads=2, d_qk=512, d_v=512, causal=True, bottom_right=True)),
    ],
)
def test_sm120_backward_check_support_records_the_part(monkeypatch, module, name, case_kwargs):
    import importlib

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (12, 0))
    api = _thd_api(case_kwargs, getattr(importlib.import_module(module), name))
    assert api.check_support() is True
    assert api._device_cc == (12, 0) and api._smem_levers is None


def test_d192_thd_backward_requires_the_dsl_target(monkeypatch):
    from cudnn.frost import device as frost_device
    from cudnn.sdpa.bwd.api_dsl_sm100_d192 import SdpaBwdDslSm100D192

    api = _thd_api(dict(d_qk=192, d_v=128), SdpaBwdDslSm100D192)
    monkeypatch.setattr(frost_device, "compute_capability", lambda device: (11, 0))
    monkeypatch.setattr(buffers, "_cutedsl_has_sm110", lambda: False)
    with pytest.raises(NotImplementedError, match="sm_110a"):
        api.check_support()
    monkeypatch.setattr(buffers, "_cutedsl_has_sm110", lambda: True)
    assert api.check_support() is True
