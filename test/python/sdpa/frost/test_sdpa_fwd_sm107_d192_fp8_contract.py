# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for the exact per-tensor FP8 D192 implementation owner."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

import cudnn
from cudnn.sdpa.fwd import engines, heuristics, placement
from cudnn.sdpa.fwd.kernels.sm107.d192_fp8_multi_q.contract import Geometry, LEGACY_ROW, NAME
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0
SPECS = {s.name: s for s in engines.ENGINE_SPECS}
OFFERED = {LEGACY_ROW: 20514, NAME: 20561}


def facts(**overrides):
    values = dict(
        b=1,
        h_q=8,
        h_kv=2,
        s_q=128,
        s_kv=4096,
        d_qk=192,
        d_v=128,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.BFLOAT16,
        is_fp8=True,
        uniform_dtype=True,
        device_cc=(10, 7),
        device_sm_count=212,
    )
    values.update(overrides)
    return SdpaGraphFacts(**values)


@pytest.mark.parametrize("cc", [(10, 7)])
@pytest.mark.parametrize("dtype", [cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2])
def test_exact_owner_and_public_cluster_geometry(cc, dtype):
    f = facts(device_cc=cc, dtype=dtype)
    assert "owned by" in engines.mismatch(SPECS[LEGACY_ROW].capabilities, f)
    knobs = Geometry().canonical_knobs(engines.SdpaFwdKnobs())
    assert engines.mismatch(SPECS[NAME].capabilities, f, knobs) is None
    assert engines.SdpaFwdKnobs.from_public(knobs.to_public()) == knobs
    assert (knobs.tile_m, knobs.cga, Geometry().cta_group) == (128, 2, 2)
    assert engines.effective_cgas(SPECS[NAME].capabilities, f) == frozenset({2})
    assert "physical launch cluster" in engines.mismatch(SPECS[NAME].capabilities, f, replace(knobs, cga=4))


@pytest.mark.parametrize(
    "overrides",
    [
        {"device_cc": (10, 0)},
        {"device_cc": (12, 0)},
        {"d_qk": 176, "d_v": 112},
        {"dtype": cudnn.data_type.BFLOAT16, "is_fp8": False},
        {"is_mxfp8": True},
        {"o_block_scale": 32},
        {"has_epilogue_gate": True},
        {"dtype_o": cudnn.data_type.FP4_E2M1},
    ],
)
def test_other_implementations_do_not_enter_the_owner(overrides):
    assert engines.mismatch(SPECS[NAME].capabilities, facts(**overrides)) is not None


def test_dense_envelope_keeps_legacy_owner():
    assert engines.mismatch(SPECS[LEGACY_ROW].capabilities, facts(d_qk=176, d_v=112)) is None


@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
def test_candidate_order_and_budget_preserve_the_legacy_policy(kind):
    f = facts()
    policy = heuristics.sm100_legacy_policy_spec(LEGACY_ROW)
    old = heuristics._knob_sets(policy, f) if kind == "A" else [heuristics._fallback_knobs(policy, f)]
    expected = [Geometry(cga=k.cga).canonical_knobs(k) for k in old if engines.mismatch(policy.capabilities, f, k) is None]
    plans = heuristics.recommend(kind, f, OFFERED)
    assert plans
    assert all(p.engine_id == 20561 for p in plans)
    assert [p.knobs for p in plans] == list(dict.fromkeys(expected))


def test_explicit_lowering_does_not_consult_candidate_selection(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit record entered performance selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    calls = []
    monkeypatch.setattr(engines, "lower_dsl_prefill", lambda spec, f, k, *, selected_kernel: calls.append((spec, f, k, selected_kernel)))
    spec = next(s for s in engines._sm107_mid_fp8_specs() if s.name == NAME)
    spec.lower(spec, facts(), None)
    assert len(calls) == 1
    assert calls[0][3].implementation == NAME
    assert calls[0][3].lowering_row_name == LEGACY_ROW


def test_cold_host_import_does_not_import_registry():
    # CI can run source tests against a wheel without an in-tree native extension.
    package_root = str(Path(cudnn.__file__).resolve().parent.parent)
    code = "\n".join(
        (
            "import sys",
            "sys.meta_path[:] = [f for f in sys.meta_path if 'editable' not in str(f).lower()]",
            f"sys.path.insert(0, {package_root!r})",
            "import cudnn.sdpa.fwd.kernels.sm107.d192_fp8_multi_q.host",
            "assert 'cudnn.sdpa.fwd.engines' not in sys.modules",
        )
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("verdict", [False, True])
def test_current_name_study_precedes_inherited_placement(monkeypatch, verdict):
    from cudnn.sdpa import placement_table

    monkeypatch.setattr(placement_table, "lookup", lambda row, f: verdict if row == NAME else not verdict)
    assert placement.place(SPECS[NAME], facts(), Geometry().canonical_knobs(engines.SdpaFwdKnobs())) == (placement.LEAD if verdict else placement.TRAIL)


def test_owned_config_does_not_delegate_to_legacy_factories(monkeypatch):
    from cudnn.sdpa.fwd import config_sm107 as legacy
    from cudnn.sdpa.fwd.kernels.sm107.d192_fp8_multi_q import config

    def forbidden(*args, **kwargs):
        raise AssertionError("owned config consulted legacy implementation state")

    for name in ("make_cfg_d192", "make_cfg_d256", "_make_cfg_d128_family", "_make_cfg_d256_family", "CfgD192", "CfgD256"):
        monkeypatch.setattr(legacy, name, forbidden)
    cfg, _ = config.make_cfg_d192(config.TemplateParams(dtype_qkv=0, dtype_o=2, cta_mma=2))
    assert type(cfg) is config.CfgD192
    assert (cfg.CGA_M, cfg.CTA_MMA, cfg.TILE_M) == (2, 2, 128)
    with pytest.raises(ValueError, match="E4M3/E5M2"):
        config.make_cfg_d192(config.TemplateParams(dtype_qkv=2))
    with pytest.raises(ValueError, match="2x2"):
        config.make_cfg_d192(config.TemplateParams(dtype_qkv=0, mma_2x2=True))


def test_cga1_retains_the_bounded_dense_causal_fp8_mha_domain():
    caps = SPECS[NAME].capabilities
    f = facts(h_q=2, h_kv=2, causal=True, dtype_o=cudnn.data_type.FP8_E4M3)
    knobs = Geometry(cga=1).canonical_knobs(engines.SdpaFwdKnobs())
    assert engines.mismatch(caps, f, knobs) is None
    assert engines.effective_cgas(caps, f) == frozenset({1, 2})
    for changes in ({"thd": True, "padded": True}, {"causal": False}, {"dtype_o": cudnn.data_type.BFLOAT16}, {"h_q": 8}, {"s_kv": 1 << 30}):
        assert engines.mismatch(caps, replace(f, **changes), knobs) is not None
    assert engines.mismatch(caps, f, replace(knobs, split_kv=4)) is not None
