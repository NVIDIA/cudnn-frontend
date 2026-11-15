# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for the exact SM120 wide-head per-tensor FP8 and MXFP8 implementation owners."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

import cudnn
from cudnn.sdpa.fwd import config_sm120, engines, heuristics, placement
from cudnn.sdpa.fwd.kernels.sm120.d512_fp8_paired import contract as fp8_wide
from cudnn.sdpa.fwd.kernels.sm120.d512_mxfp8_paired import contract as mxfp8_wide
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0
SPECS = {s.name: s for s in engines.ENGINE_SPECS}
# contract, quantization flags, legacy id, explicit id
OWNERS = {
    "fp8": (fp8_wide, dict(is_fp8=True), 20507, 20533),
    "mxfp8": (mxfp8_wide, dict(is_mxfp8=True), 20518, 20534),
}


def facts(quant, **overrides):
    values = dict(
        b=2,
        h_q=16,
        h_kv=2,
        s_q=256,
        s_kv=4096,
        d_qk=512,
        d_v=512,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.BFLOAT16,
        device_cc=(12, 0),
        device_sm_count=188,
        **OWNERS[quant][1],
    )
    values.update(overrides)
    return SdpaGraphFacts(**values)


def offered(quant):
    contract, _, legacy_id, explicit_id = OWNERS[quant]
    return {contract.LEGACY_ROW: legacy_id, contract.NAME: explicit_id}


@pytest.mark.parametrize("quant", sorted(OWNERS))
def test_registry_slot_and_native_geometry_match_the_shared_config(quant):
    from cudnn.engines.manifest import MANIFEST

    contract = OWNERS[quant][0]
    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    assert {k: family.offered_ids()[k] for k in offered(quant)} == offered(quant)
    assert contract.SHAPE == config_sm120.D512_FLAVOR
    assert (contract.Geometry().query_tile, contract.Geometry().kv_tile) == config_sm120.FP8_D512_TILE


@pytest.mark.parametrize("quant", sorted(OWNERS))
@pytest.mark.parametrize("cc", [(12, 0), (12, 1)])
@pytest.mark.parametrize("dtype", [cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2])
def test_exact_owner_and_public_tile_geometry(quant, cc, dtype):
    contract = OWNERS[quant][0]
    f = facts(quant, device_cc=cc, dtype=dtype)
    assert "owned by the explicit implementation engines" in engines.mismatch(SPECS[contract.LEGACY_ROW].capabilities, f)
    knobs = contract.Geometry().canonical_knobs(engines.SdpaFwdKnobs())
    assert engines.mismatch(SPECS[contract.NAME].capabilities, f, knobs) is None
    assert engines.SdpaFwdKnobs.from_public(knobs.to_public()) == knobs
    assert (knobs.tile_m, knobs.tile_n, knobs.cga) == (64, 64, 1)
    for bad in (dict(tile_m=128), dict(tile_n=128), dict(cga=2)):
        assert engines.mismatch(SPECS[contract.NAME].capabilities, f, replace(knobs, **bad)) is not None, bad


@pytest.mark.parametrize("quant", sorted(OWNERS))
@pytest.mark.parametrize(
    "overrides",
    [
        {"device_cc": (10, 0), "device_sm_count": 148},
        {"d_qk": 384, "d_v": 384},
        {"d_qk": 256, "d_v": 256},
        {"dtype": cudnn.data_type.BFLOAT16, "is_fp8": False, "is_mxfp8": False},
    ],
)
def test_other_implementations_do_not_enter_the_owner(quant, overrides):
    assert engines.mismatch(SPECS[OWNERS[quant][0].NAME].capabilities, facts(quant, **overrides)) is not None


def test_the_two_quantizations_do_not_cross():
    assert engines.mismatch(SPECS[fp8_wide.NAME].capabilities, facts("mxfp8")) is not None
    assert engines.mismatch(SPECS[mxfp8_wide.NAME].capabilities, facts("fp8")) is not None


@pytest.mark.parametrize("quant", sorted(OWNERS))
@pytest.mark.parametrize("dims", [(272, 272), (384, 448), (512, 384), (496, 512)])
def test_padded_envelope_keeps_the_legacy_owner(quant, dims):
    contract, _, legacy_id, _ = OWNERS[quant]
    f = facts(quant, d_qk=dims[0], d_v=dims[1])
    assert engines.mismatch(SPECS[contract.LEGACY_ROW].capabilities, f) is None
    assert engines.mismatch(SPECS[contract.NAME].capabilities, f) is not None
    assert all(p.engine_id == legacy_id for p in heuristics.recommend("A", f, offered(quant)))


@pytest.mark.parametrize("quant", sorted(OWNERS))
@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
@pytest.mark.parametrize("overrides", [{}, dict(s_q=1, h_q=128, h_kv=1, causal=True, bottom_right=True), dict(window_left=128, causal=True, bottom_right=True)])
def test_candidate_order_and_budget_preserve_the_legacy_policy(quant, kind, overrides):
    contract, _, _, explicit_id = OWNERS[quant]
    f = facts(quant, **overrides)
    policy = heuristics.sm100_legacy_policy_spec(contract.LEGACY_ROW)
    old = heuristics._knob_sets(policy, f) if kind == "A" else [heuristics._fallback_knobs(policy, f)]
    expected = [k for k in old if engines.mismatch(policy.capabilities, f, k) is None]
    plans = heuristics.recommend(kind, f, offered(quant))
    assert plans
    assert all(p.engine_id == explicit_id for p in plans)
    assert [p.knobs for p in plans] == list(dict.fromkeys(contract.Geometry().canonical_knobs(k) for k in expected)) == list(dict.fromkeys(expected))


@pytest.mark.parametrize("quant", sorted(OWNERS))
def test_placement_inherits_the_legacy_study_keys(quant):
    contract = OWNERS[quant][0]
    f = facts(quant, s_q=1)
    knobs = contract.Geometry().canonical_knobs(engines.SdpaFwdKnobs())
    assert placement.place(SPECS[contract.NAME], f, knobs) == placement.place(heuristics.sm100_legacy_policy_spec(contract.LEGACY_ROW), f)


@pytest.mark.parametrize("quant", sorted(OWNERS))
@pytest.mark.parametrize("verdict", [False, True])
def test_current_name_study_precedes_inherited_placement(monkeypatch, quant, verdict):
    from cudnn.sdpa import placement_table

    contract = OWNERS[quant][0]
    monkeypatch.setattr(placement_table, "lookup", lambda row, f: verdict if row == contract.NAME else not verdict)
    knobs = contract.Geometry().canonical_knobs(engines.SdpaFwdKnobs())
    assert placement.place(SPECS[contract.NAME], facts(quant), knobs) == (placement.LEAD if verdict else placement.TRAIL)


@pytest.mark.parametrize("quant", sorted(OWNERS))
def test_explicit_lowering_does_not_consult_candidate_selection(monkeypatch, quant):
    contract = OWNERS[quant][0]

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit record entered performance selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    calls = []
    monkeypatch.setattr(engines, "lower_dsl_prefill", lambda spec, f, k, *, api_type, selected_kernel: calls.append((api_type, k, selected_kernel)))
    spec = {"fp8": engines._sm120_d512_fp8_paired_spec, "mxfp8": engines._sm120_d512_mxfp8_paired_spec}[quant]()
    spec.lower(spec, facts(quant), None)
    ((api_type, knobs, geometry),) = calls
    assert api_type == "SdpaFwdDslSm120"
    assert (knobs.tile_m, knobs.tile_n, knobs.cga) == (64, 64, 1)
    assert (geometry.implementation, geometry.lowering_row_name, geometry.kernel_file) == (contract.NAME, contract.LEGACY_ROW, contract.KERNEL_FILE)


@pytest.mark.parametrize("module", ["d512_fp8_paired", "d512_mxfp8_paired"])
def test_cold_host_import_does_not_import_registry_or_runtimes(module):
    source_root = str(Path(cudnn.__file__).resolve().parents[1])
    code = "\n".join(
        (
            "import json, sys",
            "sys.meta_path[:] = [f for f in sys.meta_path if 'editable' not in str(f).lower()]",
            f"sys.path.insert(0, {source_root!r})",
            f"import cudnn.sdpa.fwd.kernels.sm120.{module}.host",
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
        f"cudnn.sdpa.fwd.kernels.sm120.{module}.kernel",
    }
    assert not forbidden.intersection(loaded)
