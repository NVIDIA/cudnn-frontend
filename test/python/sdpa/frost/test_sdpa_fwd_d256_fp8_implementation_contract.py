# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for the per-tensor FP8 D256 implementation owner."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

import cudnn
from cudnn.sdpa.fwd import engines, heuristics, placement
from cudnn.sdpa.fwd.kernels.sm100.d256_fp8_single_q.contract import Geometry, LEGACY_ROW, NAME
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

from cudnn.sdpa.fwd import config_sm100 as legacy
from cudnn.sdpa.fwd.config_common import TemplateParams
from cudnn.sdpa.fwd.kernels.sm100.d256_fp8_single_q import config
from cudnn.sdpa.fwd.kernels.sm100.d256_f16_lookahead import config as half_config

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("dtype", [0, 1])
def test_per_tensor_configuration_is_independent_of_half_and_mxfp8(monkeypatch, dtype):
    def forbidden(*args, **kwargs):
        raise AssertionError("per-tensor configuration consulted a sibling factory or validator")

    monkeypatch.setattr(legacy, "_make_cfg_d256", forbidden)
    monkeypatch.setattr(legacy, "_validate_cfg_d256", forbidden)
    monkeypatch.setattr(half_config, "make_cfg_d256", forbidden)
    params = TemplateParams(dtype_qkv=dtype, dtype_o=2, cta_mma=1)
    cfg, _ = legacy.make_cfg_d256(params)
    assert type(cfg) is config.CfgD256
    assert config.CfgD256 is not legacy.CfgD256
    assert config.CfgD256 is not half_config.CfgD256
    assert (cfg.CGA_M, cfg.CTA_MMA, cfg.TILE_M, cfg.STAGES_KV) == (1, 1, 128, 2)


def test_per_tensor_factory_is_not_on_the_half_or_block_scaled_path(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("sibling configuration consulted the per-tensor factory")

    monkeypatch.setattr(config, "make_cfg_d256", forbidden)
    half, _ = legacy.make_cfg_d256(TemplateParams(dtype_qkv=2, cta_mma=2))
    block, _ = legacy.make_cfg_d256_mxfp8(TemplateParams(dtype_qkv=0, dtype_o=2, cta_mma=1))
    assert type(half) is half_config.CfgD256
    assert type(block) is legacy.CfgD256


@pytest.mark.parametrize("dtype", [2, 3])
def test_per_tensor_owner_rejects_half_inputs(dtype):
    with pytest.raises(ValueError, match="requires E4M3/E5M2"):
        config.make_cfg_d256(TemplateParams(dtype_qkv=dtype, cta_mma=2))


SPECS = {s.name: s for s in engines.ENGINE_SPECS}
OFFERED = {LEGACY_ROW: 20513, NAME: 20524}


def facts(**overrides):
    values = dict(
        b=1,
        h_q=8,
        h_kv=2,
        s_q=128,
        s_kv=4096,
        d_qk=256,
        d_v=256,
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
    assert (knobs.tile_m, knobs.cga, Geometry().cta_group) == (128, 1, 1)
    assert engines.effective_cgas(SPECS[NAME].capabilities, f) == frozenset({1})
    assert "physical launch cluster" in engines.mismatch(SPECS[NAME].capabilities, f, replace(knobs, cga=2))


@pytest.mark.parametrize(
    "overrides",
    [
        {"device_cc": (10, 7)},
        {"device_cc": (12, 0)},
        {"d_qk": 224, "d_v": 208},
        {"dtype": cudnn.data_type.BFLOAT16, "is_fp8": False},
        {"is_mxfp8": True},
        {"o_block_scale": 32},
        {"dtype_o": cudnn.data_type.FP4_E2M1},
    ],
)
def test_other_implementations_do_not_enter_the_owner(overrides):
    assert engines.mismatch(SPECS[NAME].capabilities, facts(**overrides)) is not None


def test_dense_envelope_keeps_legacy_owner():
    assert engines.mismatch(SPECS[LEGACY_ROW].capabilities, facts(d_qk=224, d_v=208)) is None


@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
def test_candidate_order_and_budget_preserve_the_legacy_policy(kind):
    f = facts()
    policy = heuristics.sm100_legacy_policy_spec(LEGACY_ROW)
    old = heuristics._knob_sets(policy, f) if kind == "A" else [heuristics._fallback_knobs(policy, f)]
    expected = [Geometry().canonical_knobs(k) for k in old if engines.mismatch(policy.capabilities, f, k) is None]
    plans = heuristics.recommend(kind, f, OFFERED)
    assert plans
    assert all(p.engine_id == 20524 for p in plans)
    assert [p.knobs for p in plans] == list(dict.fromkeys(expected))


def test_explicit_lowering_does_not_consult_candidate_selection(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit record entered performance selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    calls = []
    monkeypatch.setattr(engines, "lower_dsl_prefill", lambda spec, f, k, *, selected_kernel: calls.append((spec, f, k, selected_kernel)))
    spec = engines._sm100_d256_fp8_spec()
    spec.lower(spec, facts(), None)
    assert len(calls) == 1
    assert calls[0][3].implementation == NAME
    assert calls[0][3].lowering_row_name == LEGACY_ROW


@pytest.mark.parametrize("verdict", [False, True])
def test_current_name_study_precedes_inherited_placement(monkeypatch, verdict):
    from cudnn.sdpa import placement_table

    monkeypatch.setattr(placement_table, "lookup", lambda row, f: verdict if row == NAME else not verdict)
    assert placement.place(SPECS[NAME], facts(), Geometry().canonical_knobs(engines.SdpaFwdKnobs())) == (placement.LEAD if verdict else placement.TRAIL)


def test_cold_host_import_does_not_import_registry():
    # CI can run source tests against a wheel without an in-tree native extension.
    package_root = str(Path(cudnn.__file__).resolve().parent.parent)
    code = "\n".join(
        (
            "import sys",
            "sys.meta_path[:] = [f for f in sys.meta_path if 'editable' not in str(f).lower()]",
            f"sys.path.insert(0, {package_root!r})",
            "import cudnn.sdpa.fwd.kernels.sm100.d256_fp8_single_q.host",
            "assert 'cudnn.sdpa.fwd.engines' not in sys.modules",
        )
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
