# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for the exact SM120 wide-head half implementation owner."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

import cudnn
from cudnn.sdpa.fwd import config_sm120, engines, heuristics, placement
from cudnn.sdpa.fwd.kernels.sm120.d512_f16_paired.contract import DEVICE_CCS, Geometry, LEGACY_ROW, NAME, SHAPES
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0
SPECS = {s.name: s for s in engines.ENGINE_SPECS}
OFFERED = {LEGACY_ROW: 20505, NAME: 20526}


def facts(**overrides):
    values = dict(
        b=2,
        h_q=16,
        h_kv=2,
        s_q=256,
        s_kv=4096,
        d_qk=512,
        d_v=512,
        dtype=cudnn.data_type.BFLOAT16,
        dtype_o=cudnn.data_type.BFLOAT16,
        device_cc=(12, 0),
        device_sm_count=188,
    )
    values.update(overrides)
    return SdpaGraphFacts(**values)


def test_registry_slot_and_native_geometry_match_the_shared_config():
    from cudnn.engines.manifest import MANIFEST

    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    assert (family.offered_ids()[NAME], family.offered_ids()[LEGACY_ROW]) == (20526, 20505)
    assert set(SHAPES) == config_sm120.WIDE_F16_FLAVORS
    assert (Geometry().query_tile, Geometry().kv_tile) == config_sm120.D512_TILE
    caps = SPECS[NAME].capabilities
    assert (caps.sm_lo // 10, caps.sm_hi // 10) == (12, 12) and len(DEVICE_CCS) == caps.sm_hi - caps.sm_lo + 1


@pytest.mark.parametrize("cc", [(12, 0), (12, 1)])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16])
def test_exact_owner_and_public_tile_geometry(cc, shape, dtype):
    f = facts(device_cc=cc, d_qk=shape[0], d_v=shape[1], dtype=dtype, dtype_o=dtype)
    assert "owned by the explicit implementation engines" in engines.mismatch(SPECS[LEGACY_ROW].capabilities, f)
    knobs = Geometry().canonical_knobs(engines.SdpaFwdKnobs())
    assert engines.mismatch(SPECS[NAME].capabilities, f, knobs) is None
    assert engines.SdpaFwdKnobs.from_public(knobs.to_public()) == knobs
    assert (knobs.tile_m, knobs.tile_n, knobs.cga) == (64, 32, 1)
    for bad in (dict(tile_m=128), dict(tile_n=64), dict(cga=2), dict(tile_m=16, tile_n=16)):
        assert engines.mismatch(SPECS[NAME].capabilities, f, replace(knobs, **bad)) is not None, bad


def test_paged_fp8_kv_storage_follows_the_policy_domain():
    # Mixed FP8-KV pages run this body; whether they are admitted (native binder support) is the legacy row's answer.
    f = facts(has_paged_kv=True, padded=True, page_size=16, uniform_dtype=False, kv_dtype=cudnn.data_type.FP8_E4M3, s_q=1)
    assert "owned by" in engines.mismatch(SPECS[LEGACY_ROW].capabilities, f)
    policy = heuristics.sm100_legacy_policy_spec(LEGACY_ROW)
    assert engines.mismatch(SPECS[NAME].capabilities, f) == engines.mismatch(policy.capabilities, f)


@pytest.mark.parametrize(
    "overrides",
    [
        {"device_cc": (10, 0), "device_sm_count": 148},
        {"device_cc": (10, 7), "device_sm_count": 216},
        {"d_qk": 384, "d_v": 384},
        {"d_qk": 576, "d_v": 384},
        {"d_qk": 256, "d_v": 256},
        {"dtype": cudnn.data_type.FP8_E4M3, "dtype_o": cudnn.data_type.HALF, "is_fp8": True},
        {"dtype": cudnn.data_type.FP8_E4M3, "is_mxfp8": True},
    ],
)
def test_other_implementations_do_not_enter_the_owner(overrides):
    assert engines.mismatch(SPECS[NAME].capabilities, facts(**overrides)) is not None


@pytest.mark.parametrize("dims", [(264, 264), (384, 448), (512, 384), (520, 512), (576, 384)])
def test_padded_envelope_keeps_the_legacy_owner(dims):
    f = facts(d_qk=dims[0], d_v=dims[1])
    assert engines.mismatch(SPECS[LEGACY_ROW].capabilities, f) is None
    assert engines.mismatch(SPECS[NAME].capabilities, f) is not None
    assert all(p.engine_id == 20505 for p in heuristics.recommend("A", f, OFFERED))


@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
@pytest.mark.parametrize(
    "overrides",
    [{}, dict(d_qk=576), dict(s_q=1, h_q=128, h_kv=1, causal=True, bottom_right=True), dict(window_left=128, causal=True, bottom_right=True)],
)
def test_candidate_order_and_budget_preserve_the_legacy_policy(kind, overrides):
    f = facts(**overrides)
    policy = heuristics.sm100_legacy_policy_spec(LEGACY_ROW)
    old = heuristics._knob_sets(policy, f) if kind == "A" else [heuristics._fallback_knobs(policy, f)]
    expected = [Geometry().canonical_knobs(k) for k in old if engines.mismatch(policy.capabilities, f, k) is None]
    plans = heuristics.recommend(kind, f, OFFERED)
    assert plans
    assert all(p.engine_id == 20526 for p in plans)
    # The parent's records: the canonical knobs equal the policy's sets field for field.
    assert [p.knobs for p in plans] == list(dict.fromkeys(expected)) == list(dict.fromkeys(k for k in old if k in expected))


def test_placement_inherits_the_legacy_study_keys():
    f = facts(s_q=1, has_paged_kv=True, padded=True, page_size=16)
    knobs = Geometry().canonical_knobs(engines.SdpaFwdKnobs())
    assert placement.place(SPECS[NAME], f, knobs) == placement.place(heuristics.sm100_legacy_policy_spec(LEGACY_ROW), f)


@pytest.mark.parametrize("verdict", [False, True])
def test_current_name_study_precedes_inherited_placement(monkeypatch, verdict):
    from cudnn.sdpa import placement_table

    monkeypatch.setattr(placement_table, "lookup", lambda row, f: verdict if row == NAME else not verdict)
    assert placement.place(SPECS[NAME], facts(), Geometry().canonical_knobs(engines.SdpaFwdKnobs())) == (placement.LEAD if verdict else placement.TRAIL)


def test_explicit_lowering_does_not_consult_candidate_selection(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit record entered performance selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    calls = []
    monkeypatch.setattr(engines, "lower_dsl_prefill", lambda spec, f, k, *, api_type, selected_kernel: calls.append((api_type, k, selected_kernel)))
    spec = engines._sm120_d512_paired_spec()
    spec.lower(spec, facts(), None)
    assert len(calls) == 1
    api_type, knobs, geometry = calls[0]
    assert api_type == "SdpaFwdDslSm120"
    assert (knobs.tile_m, knobs.tile_n, knobs.cga) == (64, 32, 1)
    assert (geometry.implementation, geometry.lowering_row_name, geometry.kernel_file) == (NAME, LEGACY_ROW, "sm120/d512_f16_paired/kernel.py")


def test_compiled_identity_check_names_the_disagreement():
    from types import SimpleNamespace

    knobs = Geometry().canonical_knobs(engines.SdpaFwdKnobs(pack_gqa=True, split_kv=3))
    params = SimpleNamespace(q_tile=64, kv_tile=32, sched_policy=0, pack_gqa=True, split_kv=3)
    config, why = Geometry().compiled_mismatch("prefill_d512_f16", params, knobs)
    assert why is None and config["query_tile"] == 64
    for change in (dict(split_kv=1), dict(kv_tile=64), dict(pack_gqa=False)):
        assert Geometry().compiled_mismatch("prefill_d512_f16", SimpleNamespace(**{**vars(params), **change}), knobs)[1] is not None
    assert Geometry().compiled_mismatch("prefill_f16", params, knobs)[1] is not None


def test_cold_host_import_does_not_import_registry_or_runtimes():
    source_root = str(Path(cudnn.__file__).resolve().parents[1])
    code = "\n".join(
        (
            "import json, sys",
            "sys.meta_path[:] = [f for f in sys.meta_path if 'editable' not in str(f).lower()]",
            f"sys.path.insert(0, {source_root!r})",
            "import cudnn.sdpa.fwd.kernels.sm120.d512_f16_paired.host",
            "print(json.dumps(sorted(sys.modules)))",
        )
    )
    loaded = set(json.loads(subprocess.check_output([sys.executable, "-c", code], text=True)))
    forbidden = {
        "torch",
        "cutlass",
        "cudnn.sdpa.fwd.engines",
        "cudnn.sdpa.fwd.api_dsl",
        "cudnn.sdpa.fwd.config_sm120",
        "cudnn.sdpa.fwd.kernels.sm120.d512_f16_paired.kernel",
    }
    assert not forbidden.intersection(loaded)
