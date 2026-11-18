# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for the SM120 half owners on exact 16-wide head tiles (all-compute D256, general, key-group decode)."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import cudnn
from cudnn.sdpa.fwd import config_sm120, engines, heuristics, placement
from cudnn.sdpa.fwd.kernels.sm120 import half_exact_contract as exact
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0
SPECS = {s.name: s for s in engines.ENGINE_SPECS}
OFFERED = {exact.LEGACY_ROW: 20505, exact.ALL_COMPUTE: 20537, exact.KEY_GROUP: 20538, exact.GENERAL: 20544}
IDS = {20537: exact.ALL_COMPUTE, 20538: exact.KEY_GROUP, 20544: exact.GENERAL}


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


def test_shared_tiles_match_the_config():
    assert exact.HEAD_TILES == config_sm120.GENERAL_HEAD_TILES


def test_registry_slots():
    from cudnn.engines.manifest import MANIFEST

    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    assert {k: family.offered_ids()[k] for k in OFFERED} == OFFERED


@pytest.mark.parametrize("cc", [(12, 0), (12, 1)])
@pytest.mark.parametrize("dims", [(256, 256), (128, 128), (16, 16), (48, 80), (256, 128), (128, 256)])
@pytest.mark.parametrize("tile_m", [16, 32, 64, 128])
def test_tile_m_names_the_owner(cc, dims, tile_m):
    f = facts(device_cc=cc, d_qk=dims[0], d_v=dims[1], s_q=1 if tile_m in (16, 32) else 256)
    assert "owned by the explicit implementation engines" in engines.mismatch(SPECS[exact.LEGACY_ROW].capabilities, f)
    knobs = engines.SdpaFwdKnobs(sched_policy=0, tile_m=tile_m, cga=1, pack_gqa=False, split_kv=1)
    prefill = exact.ALL_COMPUTE if dims == exact.D256 else exact.GENERAL
    owner = exact.KEY_GROUP if tile_m in (16, 32) else prefill
    assert exact.select_geometry(f, knobs).implementation == owner
    for other in exact.IMPLEMENTATIONS - {owner}:
        assert engines.mismatch(SPECS[other].capabilities, f, knobs) is not None, other
    assert engines.SdpaFwdKnobs.from_public(knobs.to_public()) == knobs


@pytest.mark.parametrize("dims", [(512, 512), (576, 512), (264, 264), (132, 132)])
def test_wide_and_unaligned_heads_do_not_enter_the_owners(dims):
    f = facts(d_qk=dims[0], d_v=dims[1])
    for owner in exact.IMPLEMENTATIONS:
        assert engines.mismatch(SPECS[owner].capabilities, f) is not None, (owner, dims)


@pytest.mark.parametrize(
    "dims,owner", [((248, 248), exact.ALL_COMPUTE), ((248, 256), exact.ALL_COMPUTE), ((120, 120), exact.GENERAL), ((72, 200), exact.GENERAL)]
)
def test_padded_heads_enter_the_owner_of_their_tile(dims, owner):
    f = facts(d_qk=dims[0], d_v=dims[1])
    other = exact.GENERAL if owner == exact.ALL_COMPUTE else exact.ALL_COMPUTE
    assert engines.mismatch(SPECS[owner].capabilities, f) is None
    assert engines.mismatch(SPECS[exact.KEY_GROUP].capabilities, f) is None
    assert engines.mismatch(SPECS[other].capabilities, f) is not None


@pytest.mark.parametrize("dims", [(248, 256), (120, 120)])
def test_padded_heads_leave_the_legacy_row(dims):
    f = facts(d_qk=dims[0], d_v=dims[1])
    assert "padded to" in engines.mismatch(SPECS[exact.LEGACY_ROW].capabilities, f)
    plans = heuristics.recommend("A", f, OFFERED)
    assert plans and 20505 not in {p.engine_id for p in plans}


@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
@pytest.mark.parametrize("dims", [(256, 256), (128, 128), (64, 64)])
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        dict(s_q=1, h_q=32, h_kv=8, s_kv=8192),
        dict(thd=True, padded=True, max_total_seq_len_q=512),
        dict(window_left=128, causal=True, bottom_right=True),
        dict(s_q=4, has_paged_kv=True, padded=True, page_size=16),
    ],
)
def test_candidate_order_and_budget_preserve_the_legacy_policy(kind, dims, overrides):
    f = facts(d_qk=dims[0], d_v=dims[1], **overrides)
    policy = heuristics.sm100_legacy_policy_spec(exact.LEGACY_ROW)
    old = heuristics._knob_sets(policy, f) if kind == "A" else [heuristics._fallback_knobs(policy, f)]
    expected = list(dict.fromkeys(k for k in old if engines.mismatch(policy.capabilities, f, k) is None))
    plans = heuristics.recommend(kind, f, OFFERED)
    assert [p.knobs for p in plans] == expected
    for plan in plans:
        assert IDS[plan.engine_id] == exact.select_geometry(f, plan.knobs).implementation


class _Bias:
    dim = (1, 16, 256, 256)

    def get_dim(self):
        return self.dim

    def get_stride(self):
        return (16 * 256 * 256, 256 * 256, 256, 1)

    def get_data_type(self):
        return cudnn.data_type.BFLOAT16


def test_dense_d128_bias_stays_on_the_general_owner():
    bias = _Bias()
    f = facts(d_qk=128, d_v=128, s_kv=256, has_bias=True, bias_t=bias, bias_dim=bias.dim, bias_stride=bias.get_stride(), bias_dtype=bias.get_data_type())
    plans = heuristics.recommend("A", f, OFFERED)
    assert plans and all(p.engine_id == 20544 for p in plans)
    assert all(engines.mismatch(SPECS[exact.GENERAL].capabilities, f, p.knobs) is None for p in plans)
    decode = engines.SdpaFwdKnobs(sched_policy=0, tile_m=16, cga=1, pack_gqa=False, split_kv=1)
    assert engines.mismatch(SPECS[exact.KEY_GROUP].capabilities, f, decode) is not None


def test_partial_public_record_takes_the_owner_default():
    """A record naming only TILE_N runs the owner's fixed Q tile, as an empty record does; a wrong stage still declines."""
    decode = facts(s_q=3)
    key_group = SPECS[exact.KEY_GROUP].capabilities
    assert engines.mismatch(key_group, decode, engines.SdpaFwdKnobs()) is None
    assert engines.mismatch(key_group, decode, engines.SdpaFwdKnobs(tile_n=16)) is None
    assert "16-key K/V stages" in engines.mismatch(key_group, decode, engines.SdpaFwdKnobs(tile_n=64))
    assert engines.mismatch(SPECS[exact.ALL_COMPUTE].capabilities, facts(), engines.SdpaFwdKnobs(tile_n=64)) is None
    assert engines.mismatch(key_group, decode, engines.SdpaFwdKnobs(tile_m=64, tile_n=16)) is not None
    d128 = facts(s_q=3, d_qk=128, d_v=128)
    assert engines.mismatch(key_group, d128, engines.SdpaFwdKnobs(tile_n=16)) is None
    assert engines.mismatch(SPECS[exact.GENERAL].capabilities, facts(d_qk=128, d_v=128), engines.SdpaFwdKnobs(tile_n=64)) is None


def test_placement_inherits_the_legacy_study_keys():
    f = facts(s_q=1, has_paged_kv=True, padded=True, page_size=16)
    knobs = engines.SdpaFwdKnobs(sched_policy=0, tile_m=16, tile_n=16, cga=1, pack_gqa=True, split_kv=1)
    assert placement.place(SPECS[exact.KEY_GROUP], f, knobs) == placement.place(heuristics.sm100_legacy_policy_spec(exact.LEGACY_ROW), f)
    g = facts(d_qk=128, d_v=128, s_q=4, has_paged_kv=True, padded=True, page_size=16)
    knobs = engines.SdpaFwdKnobs(sched_policy=0, tile_m=64, tile_n=64, cga=1, pack_gqa=False, split_kv=1)
    assert placement.place(SPECS[exact.GENERAL], g, knobs) == placement.place(heuristics.sm100_legacy_policy_spec(exact.LEGACY_ROW), g)


@pytest.mark.parametrize("owner,default", [(exact.ALL_COMPUTE, 128), (exact.KEY_GROUP, 16), (exact.GENERAL, 128)])
def test_explicit_lowering_does_not_consult_candidate_selection(monkeypatch, owner, default):
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit record entered performance selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    calls = []
    monkeypatch.setattr(engines, "lower_dsl_prefill", lambda spec, f, k, *, api_type, selected_kernel: calls.append((api_type, k, selected_kernel)))
    builders = {
        exact.ALL_COMPUTE: engines._sm120_d256_all_compute_spec,
        exact.KEY_GROUP: engines._sm120_decode_key_group_spec,
        exact.GENERAL: engines._sm120_f16_general_spec,
    }
    spec = builders[owner]()
    spec.lower(spec, facts(d_qk=128 if owner == exact.GENERAL else 256, d_v=128 if owner == exact.GENERAL else 256), None)
    ((api_type, knobs, geometry),) = calls
    assert api_type == "SdpaFwdDslSm120"
    assert (knobs.tile_m, knobs.cga, geometry.implementation, geometry.lowering_row_name) == (default, 1, owner, exact.LEGACY_ROW)


def test_compiled_identity_check_names_the_disagreement():
    geometry = exact.HalfGeometry(exact.KEY_GROUP, 16)
    knobs = geometry.canonical_knobs(engines.SdpaFwdKnobs(tile_n=16, split_kv=3, pack_gqa=True))
    params = SimpleNamespace(q_tile=16, kv_tile=16, sched_policy=0, pack_gqa=True, split_kv=3)
    assert geometry.compiled_mismatch("decode_f16", params, knobs)[1] is None
    assert geometry.compiled_mismatch("prefill_d256_f16", params, knobs)[1] is not None
    assert geometry.compiled_mismatch("prefill_f16", params, knobs)[1] is not None
    assert geometry.compiled_mismatch("decode_f16", SimpleNamespace(**{**vars(params), "q_tile": 32}), knobs)[1] is not None
    general = exact.HalfGeometry(exact.GENERAL, 64)
    assert general.compiled_mismatch("prefill_f16", SimpleNamespace(**{**vars(params), "q_tile": 64}), general.canonical_knobs(knobs))[1] is None


@pytest.mark.parametrize("module", ["d256_f16_all_compute", "decode_f16_key_group", "f16_general"])
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
