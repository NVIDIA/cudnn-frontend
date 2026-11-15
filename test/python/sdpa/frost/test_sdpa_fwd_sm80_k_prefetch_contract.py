# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for the SM80 K-prefetch half owner (gptoss / llama / dsv3 flavor envelopes)."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import cudnn
from cudnn.sdpa.fwd import config_sm80, engines, heuristics, placement
from cudnn.sdpa.fwd.kernels.sm80.f16_k_prefetch import contract
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0
SPECS = {s.name: s for s in engines.ENGINE_SPECS}
OFFERED = {contract.LEGACY_ROW: 20508, contract.NAME: 20546}


def facts(**overrides):
    values = dict(
        b=2,
        h_q=16,
        h_kv=4,
        s_q=256,
        s_kv=4096,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        dtype_o=cudnn.data_type.BFLOAT16,
        device_cc=(8, 0),
        device_sm_count=108,
    )
    values.update(overrides)
    return SdpaGraphFacts(**values)


def test_domain_is_the_flavor_envelopes():
    flavors = {name: (cfg.D_QK, cfg.D_V) for name, cfg in config_sm80.FLAVOR_CFGS.items()}
    assert {config_sm80.pick_flavor(q, v) for q, v in contract.SHAPES} == set(contract.FLAVORS)
    assert max(flavors[f][0] for f in contract.FLAVORS) == contract.D_QK_MAX
    assert max(flavors[f][1] for f in contract.FLAVORS) == contract.D_V_MAX
    assert len(contract.SHAPES) == (contract.D_QK_MAX // 8) * (contract.D_V_MAX // 8)


def test_registry_slot():
    from cudnn.engines.manifest import MANIFEST

    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    assert {k: family.offered_ids()[k] for k in OFFERED} == OFFERED


@pytest.mark.parametrize("dims", [(8, 8), (64, 64), (128, 128), (192, 128), (72, 40), (192, 8)])
def test_owned_dims_have_one_owner(dims):
    f = facts(d_qk=dims[0], d_v=dims[1])
    assert "owned by the explicit implementation engines" in engines.mismatch(SPECS[contract.LEGACY_ROW].capabilities, f)
    assert engines.mismatch(SPECS[contract.NAME].capabilities, f) is None


@pytest.mark.parametrize("dims", [(200, 128), (128, 136), (256, 256), (512, 512), (576, 512), (100, 100)])
def test_other_dims_do_not_enter_the_owner(dims):
    f = facts(d_qk=dims[0], d_v=dims[1])
    assert engines.mismatch(SPECS[contract.NAME].capabilities, f) is not None
    if dims[0] % 8 == 0 and dims[1] % 8 == 0:
        assert engines.mismatch(SPECS[contract.LEGACY_ROW].capabilities, f) is None


@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
@pytest.mark.parametrize("dims", [(64, 64), (128, 128), (192, 128)])
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        dict(s_q=1, h_q=32, h_kv=8, s_kv=8192),
        dict(causal=True, bottom_right=True, s_kv=256),
        dict(window_left=128, causal=True, bottom_right=True),
        dict(s_q=4, has_paged_kv=True, padded=True, page_size=16),
    ],
)
def test_candidate_order_and_budget_preserve_the_legacy_policy(kind, dims, overrides):
    f = facts(d_qk=dims[0], d_v=dims[1], **overrides)
    policy = heuristics.sm100_legacy_policy_spec(contract.LEGACY_ROW)
    old = heuristics._knob_sets(policy, f) if kind == "A" else [heuristics._fallback_knobs(policy, f)]
    expected = list(dict.fromkeys(k for k in old if engines.mismatch(policy.capabilities, f, k) is None))
    plans = heuristics.recommend(kind, f, OFFERED)
    assert [p.knobs for p in plans] == expected
    assert all(p.engine_id == 20546 for p in plans)
    assert all(p.knobs.tile_m is None and p.knobs.tile_n is None for p in plans)


def test_placement_and_trailing_inherit_the_legacy_row():
    f = facts(s_q=8, h_q=32, h_kv=4)
    assert placement.place(SPECS[contract.NAME], f) == placement.place(heuristics.sm100_legacy_policy_spec(contract.LEGACY_ROW), f)
    assert contract.NAME in placement.TRAIL_ONLY_ROWS and contract.LEGACY_ROW in placement.TRAIL_ONLY_ROWS


def test_explicit_lowering_does_not_consult_candidate_selection(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit record entered performance selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    calls = []
    monkeypatch.setattr(engines, "lower_dsl_prefill", lambda spec, f, k, *, api_type, selected_kernel: calls.append((api_type, k, selected_kernel)))
    spec = engines._sm80_f16_k_prefetch_spec()
    spec.lower(spec, facts(), None)
    ((api_type, knobs, geometry),) = calls
    assert api_type == "SdpaFwdDslSm80"
    assert (knobs.tile_m, knobs.tile_n, geometry.implementation, geometry.lowering_row_name) == (None, None, contract.NAME, contract.LEGACY_ROW)


def test_compiled_identity_check_names_the_disagreement():
    geometry = contract.Geometry()
    params = SimpleNamespace(d_qk=128, d_v=128, tile_m=64, tile_n=64, sched_policy=0, split_kv=3)
    knobs = engines.SdpaFwdKnobs(split_kv=3)
    assert geometry.compiled_mismatch("prefill_f16", params, knobs)[1] is None
    assert geometry.compiled_mismatch("prefill_d256_f16", params, knobs)[1] is not None
    assert geometry.compiled_mismatch("prefill_f16", params, engines.SdpaFwdKnobs(split_kv=2))[1] is not None
    assert geometry.compiled_mismatch("prefill_f16", params, engines.SdpaFwdKnobs(tile_m=128, split_kv=3))[1] is not None


def test_cold_host_import_does_not_import_registry_or_runtimes():
    source_root = str(Path(cudnn.__file__).resolve().parents[1])
    code = "\n".join(
        (
            "import json, sys",
            "sys.meta_path[:] = [f for f in sys.meta_path if 'editable' not in str(f).lower()]",
            f"sys.path.insert(0, {source_root!r})",
            "import cudnn.sdpa.fwd.kernels.sm80.f16_k_prefetch.host",
            "print(json.dumps(sorted(sys.modules)))",
        )
    )
    loaded = set(json.loads(subprocess.check_output([sys.executable, "-c", code], text=True)))
    forbidden = {
        "torch",
        "cutlass",
        "cudnn.sdpa.fwd.engines",
        "cudnn.sdpa.fwd.api_dsl",
        "cudnn.sdpa.fwd.config_sm80",
        "cudnn.sdpa.fwd.kernels.sm80.f16_k_prefetch.kernel",
    }
    assert not forbidden.intersection(loaded)
