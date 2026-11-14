# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for the exact per-tensor FP8 D512 implementation owner."""

from dataclasses import replace

import pytest

import cudnn
from cudnn.sdpa.fwd import engines, heuristics, placement
from cudnn.sdpa.fwd.kernels.sm100.d512_fp8_role_split.contract import Geometry, LEGACY_ROW, NAME
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0
SPECS = {s.name: s for s in engines.ENGINE_SPECS}
OFFERED = {LEGACY_ROW: 20513, NAME: 20523}


def facts(**overrides):
    values = dict(
        b=1,
        h_q=8,
        h_kv=2,
        s_q=128,
        s_kv=4096,
        d_qk=512,
        d_v=512,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.BFLOAT16,
        is_fp8=True,
        uniform_dtype=True,
        device_cc=(10, 0),
        device_sm_count=148,
    )
    values.update(overrides)
    return SdpaGraphFacts(**values)


@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
@pytest.mark.parametrize("dtype", [cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2])
def test_exact_owner_and_public_cluster_geometry(cc, dtype):
    f = facts(device_cc=cc, dtype=dtype)
    assert "owned by" in engines.mismatch(SPECS[LEGACY_ROW].capabilities, f)
    knobs = Geometry().canonical_knobs(engines.SdpaFwdKnobs())
    assert engines.mismatch(SPECS[NAME].capabilities, f, knobs) is None
    assert engines.SdpaFwdKnobs.from_public(knobs.to_public()) == knobs
    assert (knobs.tile_m, knobs.cga, Geometry().cta_group) == (128, 4, 2)
    assert engines.effective_cgas(SPECS[NAME].capabilities, f) == frozenset({4})
    assert "physical launch cluster" in engines.mismatch(SPECS[NAME].capabilities, f, replace(knobs, cga=2))


@pytest.mark.parametrize(
    "overrides",
    [
        {"device_cc": (10, 7)},
        {"device_cc": (12, 0)},
        {"d_qk": 384, "d_v": 320},
        {"dtype": cudnn.data_type.BFLOAT16, "is_fp8": False},
        {"is_mxfp8": True},
        {"o_block_scale": 32},
        {"dtype_o": cudnn.data_type.FP4_E2M1},
    ],
)
def test_other_implementations_do_not_enter_the_owner(overrides):
    assert engines.mismatch(SPECS[NAME].capabilities, facts(**overrides)) is not None


def test_dense_envelope_keeps_legacy_owner():
    assert engines.mismatch(SPECS[LEGACY_ROW].capabilities, facts(d_qk=384, d_v=320)) is None


@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
def test_candidate_order_and_budget_preserve_the_legacy_policy(kind):
    f = facts()
    policy = heuristics.sm100_legacy_policy_spec(LEGACY_ROW)
    old = heuristics._knob_sets(policy, f) if kind == "A" else [heuristics._fallback_knobs(policy, f)]
    expected = [Geometry().canonical_knobs(k) for k in old if engines.mismatch(policy.capabilities, f, k) is None]
    plans = heuristics.recommend(kind, f, OFFERED)
    assert plans
    assert all(p.engine_id == 20523 for p in plans)
    assert [p.knobs for p in plans] == list(dict.fromkeys(expected))


def test_explicit_lowering_does_not_consult_candidate_selection(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit record entered performance selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    calls = []
    monkeypatch.setattr(engines, "lower_dsl_prefill", lambda spec, f, k, *, selected_kernel: calls.append((spec, f, k, selected_kernel)))
    SPECS[NAME].lower(SPECS[NAME], facts(), None)
    assert len(calls) == 1
    assert calls[0][3].implementation == NAME
    assert calls[0][3].lowering_row_name == LEGACY_ROW


@pytest.mark.parametrize("verdict", [False, True])
def test_current_name_study_precedes_inherited_placement(monkeypatch, verdict):
    from cudnn.sdpa import placement_table

    monkeypatch.setattr(placement_table, "lookup", lambda row, f: verdict if row == NAME else not verdict)
    assert placement.place(SPECS[NAME], facts(), Geometry().canonical_knobs(engines.SdpaFwdKnobs())) == (placement.LEAD if verdict else placement.TRAIL)
