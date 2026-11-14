# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for independently replayable wide half main implementations."""

from dataclasses import replace

import pytest

import cudnn
from cudnn.sdpa.fwd import engines, heuristics, placement
from cudnn.sdpa.fwd.config_common import TemplateParams
from cudnn.sdpa.fwd.kernels.sm100.d512_f16_contract import ROLE_SPLIT, TWO_BY_TWO, explicit_geometry
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0
SPECS = {s.name: s for s in engines.ENGINE_SPECS}
OFFERED = {"sdpa_fwd_prefill_sm100": 20511, ROLE_SPLIT: 20521, TWO_BY_TWO: 20522}


class _Tensor:
    def __init__(self, b, h, s, d):
        self.dim = (b, h, s, d)
        self.stride = (s * h * d, d, h * d, 1)

    def get_dim(self):
        return self.dim

    def get_stride(self):
        return self.stride

    def get_data_type(self):
        return cudnn.data_type.BFLOAT16


def facts(**overrides):
    values = dict(b=1, h_q=32, h_kv=8, s_q=128, s_kv=8192, d_qk=512, d_v=512, dtype=cudnn.data_type.BFLOAT16, device_cc=(10, 0), device_sm_count=148)
    values.update(overrides)
    b, h, kh, sq, sk, dq, dv = (values[k] for k in ("b", "h_q", "h_kv", "s_q", "s_kv", "d_qk", "d_v"))
    values.update(q_t=_Tensor(b, h, sq, dq), k_t=_Tensor(b, kh, sk, dq), v_t=_Tensor(b, kh, sk, dv), o_t=_Tensor(b, h, sq, dv))
    return SdpaGraphFacts(**values)


@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
@pytest.mark.parametrize("dq", [512, 576])
def test_ownership_and_complete_cluster_geometry(cc, dq):
    f = facts(device_cc=cc, d_qk=dq)
    assert "owned by" in engines.mismatch(SPECS["sdpa_fwd_prefill_sm100"].capabilities, f)
    choices = (ROLE_SPLIT, TWO_BY_TWO) if dq == 512 else (TWO_BY_TWO,)
    for name in choices:
        geometry = explicit_geometry(name, f)
        knobs = geometry.canonical_knobs(engines.SdpaFwdKnobs())
        assert engines.mismatch(SPECS[name].capabilities, f, knobs) is None
        assert engines.SdpaFwdKnobs.from_public(knobs.to_public()) == knobs
        assert geometry.cga == geometry.cluster_m
        assert geometry.cta_group == 2
        assert geometry.cluster_m == (2 if dq == 576 else 4)
        assert geometry.query_tile == (128 if name == ROLE_SPLIT else 64)
    if dq == 576:
        assert engines.mismatch(SPECS[ROLE_SPLIT].capabilities, f) is not None


@pytest.mark.parametrize("cc", [(10, 7), (12, 0)])
@pytest.mark.parametrize("name", [ROLE_SPLIT, TWO_BY_TWO])
def test_other_architectures_do_not_enter_the_new_rows(cc, name):
    assert "requires SM100/SM103" in engines.mismatch(SPECS[name].capabilities, facts(device_cc=cc))


def test_envelope_keeps_its_existing_owner():
    f = facts(d_qk=504, d_v=504)
    assert engines.mismatch(SPECS["sdpa_fwd_prefill_sm100"].capabilities, f) is None
    assert all(engines.mismatch(SPECS[name].capabilities, f) is not None for name in (ROLE_SPLIT, TWO_BY_TWO))


@pytest.mark.parametrize("name", [ROLE_SPLIT, TWO_BY_TWO])
@pytest.mark.parametrize("sq", [1, 129, 8192])
def test_explicit_lowering_never_reselects_the_main_body(monkeypatch, name, sq):
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit replay entered performance selection")

    monkeypatch.setattr(heuristics, "select_sm100_d512_geometry_for_shape", forbidden)
    monkeypatch.setattr(heuristics, "select_sm100_d512_geometry", forbidden)
    calls = []
    monkeypatch.setattr(engines, "lower_dsl_prefill", lambda spec, f, k, *, selected_kernel: calls.append((spec, f, k, selected_kernel)))
    f = facts(s_q=sq)
    SPECS[name].lower(SPECS[name], f, None)
    assert len(calls) == 1
    _, _, knobs, geometry = calls[0]
    assert geometry.implementation == name
    for opposite in (False, True):
        params = geometry.configure_params(TemplateParams(mma_2x2=opposite))
        assert params.mma_2x2 == (name == TWO_BY_TWO)
    assert knobs.tile_m == geometry.query_tile


def test_paging_and_packing_use_the_selected_bodys_limits():
    f = facts(has_paged_kv=True, page_size=16, padded=True)
    assert "does not serve paged" in engines.mismatch(SPECS[TWO_BY_TWO].capabilities, f)
    f = facts(h_q=128, h_kv=1)
    role = explicit_geometry(ROLE_SPLIT, f).canonical_knobs(engines.SdpaFwdKnobs(pack_gqa=True))
    pair = explicit_geometry(TWO_BY_TWO, f).canonical_knobs(role)
    assert engines.mismatch(SPECS[ROLE_SPLIT].capabilities, f, role) is None
    assert "whole GQA groups" in engines.mismatch(SPECS[TWO_BY_TWO].capabilities, f, pair)
    assert "split-KV is unqualified" in engines.mismatch(SPECS[TWO_BY_TWO].capabilities, f, replace(pair, split_kv=2))
    assert "physical launch cluster" in engines.mismatch(SPECS[ROLE_SPLIT].capabilities, f, replace(role, cga=2))


@pytest.mark.parametrize("name", [ROLE_SPLIT, TWO_BY_TWO])
@pytest.mark.parametrize("verdict", [False, True])
def test_current_name_measurement_precedes_legacy_inheritance(monkeypatch, name, verdict):
    from cudnn.sdpa import placement_table

    f = facts()
    knobs = explicit_geometry(name, f).canonical_knobs(engines.SdpaFwdKnobs())
    monkeypatch.setattr(placement_table, "lookup", lambda row, facts: verdict if row == name else not verdict)
    assert placement.place(SPECS[name], f, knobs) == (placement.LEAD if verdict else placement.TRAIL)


@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
@pytest.mark.parametrize("dq", [512, 576])
def test_default_budget_and_body_match_the_historical_policy(kind, dq):
    f = facts(d_qk=dq)
    policy = heuristics.sm100_legacy_policy_spec()
    old_knobs = heuristics._knob_sets(policy, f) if kind == "A" else [heuristics._fallback_knobs(policy, f)]
    expected = []
    for knobs in old_knobs:
        if engines.mismatch(policy.capabilities, f, knobs) is None:
            geometry = heuristics.select_sm100_d512_geometry(f, knobs)
            expected.append((OFFERED[geometry.implementation], geometry.canonical_knobs(knobs)))
    plans = heuristics.recommend(kind, f, OFFERED)
    assert [(p.engine_id, p.knobs) for p in plans] == list(dict.fromkeys(expected))
    assert all(p.engine_id != 20511 for p in plans)


def test_half_configuration_is_isolated_from_the_quantized_record():
    from cudnn.sdpa.fwd.config_sm100 import CfgD512 as QuantizedRecord, make_cfg_d512
    from cudnn.sdpa.fwd.kernels.sm100.d512_f16_role_split.config import CfgD512 as HalfRecord

    half, _ = make_cfg_d512(TemplateParams(dtype_qkv=3))
    fp8, _ = make_cfg_d512(TemplateParams(dtype_qkv=0, dtype_o=3))
    assert type(half) is HalfRecord
    assert type(fp8) is QuantizedRecord
    assert HalfRecord is not QuantizedRecord


@pytest.mark.parametrize("name,dq,cluster", [(ROLE_SPLIT, 512, 4), (TWO_BY_TWO, 512, 4), (TWO_BY_TWO, 576, 2)])
@pytest.mark.parametrize("requested", [1, 2, 4, 8])
def test_public_cluster_width_is_honored_or_rejected(name, dq, cluster, requested):
    f = facts(d_qk=dq)
    knobs = engines.SdpaFwdKnobs(cga=requested)
    why = engines.mismatch(SPECS[name].capabilities, f, engines.SdpaFwdKnobs.from_public(knobs.to_public()))
    assert (why is None) == (requested == cluster)
    assert engines.effective_cgas(SPECS[name].capabilities, f) == frozenset({cluster})
