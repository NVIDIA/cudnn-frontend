# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for the two exact-D256 SM120 half owners (all-compute and key-group decode)."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import cudnn
from cudnn.sdpa.fwd import engines, heuristics, placement
from cudnn.sdpa.fwd.kernels.sm120 import d256_f16_contract as d256
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0
SPECS = {s.name: s for s in engines.ENGINE_SPECS}
OFFERED = {d256.LEGACY_ROW: 20505, d256.ALL_COMPUTE: 20537, d256.KEY_GROUP: 20538}
IDS = {20537: d256.ALL_COMPUTE, 20538: d256.KEY_GROUP}


def facts(**overrides):
    values = dict(
        b=2,
        h_q=16,
        h_kv=2,
        s_q=256,
        s_kv=4096,
        d_qk=256,
        d_v=256,
        dtype=cudnn.data_type.BFLOAT16,
        dtype_o=cudnn.data_type.BFLOAT16,
        device_cc=(12, 0),
        device_sm_count=188,
    )
    values.update(overrides)
    return SdpaGraphFacts(**values)


def test_registry_slots():
    from cudnn.engines.manifest import MANIFEST

    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    assert {k: family.offered_ids()[k] for k in OFFERED} == OFFERED


@pytest.mark.parametrize("cc", [(12, 0), (12, 1)])
@pytest.mark.parametrize("owner,tile_m", [(d256.ALL_COMPUTE, 64), (d256.ALL_COMPUTE, 128), (d256.KEY_GROUP, 16), (d256.KEY_GROUP, 32)])
def test_tile_m_names_the_owner(cc, owner, tile_m):
    f = facts(device_cc=cc, s_q=1 if owner == d256.KEY_GROUP else 256)
    assert "owned by the explicit implementation engines" in engines.mismatch(SPECS[d256.LEGACY_ROW].capabilities, f)
    knobs = engines.SdpaFwdKnobs(sched_policy=0, tile_m=tile_m, cga=1, pack_gqa=False, split_kv=1)
    assert d256.select_geometry(f, knobs).implementation == owner
    other = d256.KEY_GROUP if owner == d256.ALL_COMPUTE else d256.ALL_COMPUTE
    assert engines.mismatch(SPECS[other].capabilities, f, knobs) is not None
    assert engines.SdpaFwdKnobs.from_public(knobs.to_public()) == knobs


@pytest.mark.parametrize("dims", [(248, 248), (248, 256), (256, 248), (128, 128), (512, 512)])
def test_other_head_dims_do_not_enter_the_owners(dims):
    f = facts(d_qk=dims[0], d_v=dims[1])
    for owner in d256.IMPLEMENTATIONS:
        assert engines.mismatch(SPECS[owner].capabilities, f) is not None, (owner, dims)


def test_envelope_keeps_the_legacy_owner():
    f = facts(d_qk=248, d_v=256)
    assert engines.mismatch(SPECS[d256.LEGACY_ROW].capabilities, f) is None
    assert all(p.engine_id == 20505 for p in heuristics.recommend("A", f, OFFERED))


@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
@pytest.mark.parametrize(
    "overrides",
    [{}, dict(s_q=1, h_q=32, h_kv=8, s_kv=8192), dict(thd=True, padded=True, max_total_seq_len_q=512), dict(window_left=128, causal=True, bottom_right=True)],
)
def test_candidate_order_and_budget_preserve_the_legacy_policy(kind, overrides):
    f = facts(**overrides)
    policy = heuristics.sm100_legacy_policy_spec(d256.LEGACY_ROW)
    old = heuristics._knob_sets(policy, f) if kind == "A" else [heuristics._fallback_knobs(policy, f)]
    expected = list(dict.fromkeys(k for k in old if engines.mismatch(policy.capabilities, f, k) is None))
    plans = heuristics.recommend(kind, f, OFFERED)
    assert [p.knobs for p in plans] == expected
    for plan in plans:
        assert IDS[plan.engine_id] == d256.select_geometry(f, plan.knobs).implementation


def test_placement_inherits_the_legacy_study_keys():
    f = facts(s_q=1, has_paged_kv=True, padded=True, page_size=16)
    knobs = engines.SdpaFwdKnobs(sched_policy=0, tile_m=16, tile_n=16, cga=1, pack_gqa=True, split_kv=1)
    assert placement.place(SPECS[d256.KEY_GROUP], f, knobs) == placement.place(heuristics.sm100_legacy_policy_spec(d256.LEGACY_ROW), f)


@pytest.mark.parametrize("owner,default", [(d256.ALL_COMPUTE, 128), (d256.KEY_GROUP, 16)])
def test_explicit_lowering_does_not_consult_candidate_selection(monkeypatch, owner, default):
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit record entered performance selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    calls = []
    monkeypatch.setattr(engines, "lower_dsl_prefill", lambda spec, f, k, *, api_type, selected_kernel: calls.append((api_type, k, selected_kernel)))
    spec = {d256.ALL_COMPUTE: engines._sm120_d256_all_compute_spec, d256.KEY_GROUP: engines._sm120_decode_key_group_spec}[owner]()
    spec.lower(spec, facts(), None)
    ((api_type, knobs, geometry),) = calls
    assert api_type == "SdpaFwdDslSm120"
    assert (knobs.tile_m, knobs.cga, geometry.implementation, geometry.lowering_row_name) == (default, 1, owner, d256.LEGACY_ROW)


def test_compiled_identity_check_names_the_disagreement():
    geometry = d256.D256Geometry(d256.KEY_GROUP, 16)
    knobs = geometry.canonical_knobs(engines.SdpaFwdKnobs(tile_n=16, split_kv=3, pack_gqa=True))
    params = SimpleNamespace(q_tile=16, kv_tile=16, sched_policy=0, pack_gqa=True, split_kv=3)
    assert geometry.compiled_mismatch("decode_f16", params, knobs)[1] is None
    assert geometry.compiled_mismatch("prefill_d256_f16", params, knobs)[1] is not None
    assert geometry.compiled_mismatch("decode_f16", SimpleNamespace(**{**vars(params), "q_tile": 32}), knobs)[1] is not None


@pytest.mark.parametrize("module", ["d256_f16_all_compute", "decode_f16_key_group"])
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
