# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU admission tests for plan-specific Stats stride override declarations."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import cudnn
from cudnn.engines.base import PlanConfig
from cudnn.sdpa.fwd import engines
from cudnn.sdpa.fwd.engine import FrostSdpaFwdEngine
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0
SPECS = {s.name: s for s in engines.ENGINE_SPECS}
NOTE = cudnn.behavior_note.SUPPORTS_SDPA_STATS_STRIDE_OVERRIDE


def _facts(*, packing="HN", **changes):
    stats = SimpleNamespace(
        get_data_type=lambda: cudnn.data_type.FLOAT,
        get_dim=lambda: (2, 8, 16, 1),
        ragged_offset=object() if packing != "padded" else None,
        get_stride=lambda: (256, 32, 1, 1) if packing != "NH" else (256, 1, 8, 1),
    )
    return SdpaGraphFacts(
        **dict(
            dict(
                b=2,
                h_q=8,
                h_kv=2,
                s_q=16,
                s_kv=64,
                d_qk=128,
                d_v=128,
                dtype=cudnn.data_type.BFLOAT16,
                device_cc=(10, 0),
                device_sm_count=148,
                thd=True,
                padded=True,
                shape_overrides=True,
                wants_stats=True,
                stats_t=stats,
            ),
            **changes,
        )
    )


def _notes(name, facts, knobs=None):
    engine = FrostSdpaFwdEngine(SPECS[name], 20519)
    graph = SimpleNamespace(_facts_for=lambda analyzer: facts)
    return engine.get_behavior_notes(graph, PlanConfig(engine.engine_id, knobs))


@pytest.mark.parametrize(
    "name,d,cc",
    [
        ("sdpa_fwd_sm100_d256_lookahead", 256, (10, 0)),
        ("sdpa_fwd_sm100_d256_lookahead", 256, (10, 3)),
        ("sdpa_fwd_sm100_d128_multi_q", 128, (10, 0)),
        ("sdpa_fwd_sm100_d128_multi_q", 128, (10, 3)),
    ],
)
def test_eligible_half_plans_declare_note(name, d, cc, monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    f = _facts(d_qk=d, d_v=d, device_cc=cc)
    assert _notes(name, f) == (cudnn.behavior_note.RUNTIME_COMPILATION, NOTE)


@pytest.mark.parametrize(
    "changes",
    [dict(packing="NH"), dict(packing="padded"), dict(shape_overrides=False), dict(wants_stats=False, stats_t=None), dict(thd=False), dict(is_fp8=True)],
)
def test_note_excludes_ineligible_graphs(changes):
    assert NOTE not in _notes("sdpa_fwd_sm100_d128_multi_q", _facts(**changes))


@pytest.mark.parametrize("cc", [(12, 0), (12, 1), (12, 2)])
def test_stats_override_follows_the_implementation_on_every_admitted_cc(cc):
    # 12.2 is synthetic: changing only the minor CC must not require a second opt-in.
    name = "sdpa_fwd_sm120_f16_general"
    facts = _facts(device_cc=cc)
    assert engines.mismatch(SPECS[name].capabilities, facts) is None
    assert NOTE in _notes(name, facts)


def test_template_opt_in_is_required_even_when_the_graph_and_device_match():
    owner = SPECS["sdpa_fwd_sm120_f16_general"]
    facts = _facts(device_cc=(12, 1))
    assert engines.mismatch(owner.capabilities, facts) is None
    sibling = engines.EngineSpec("new_template", owner.capabilities, owner.lower)
    graph = SimpleNamespace(_facts_for=lambda analyzer: facts)
    engine = FrostSdpaFwdEngine(sibling, 20519)
    assert NOTE not in engine.get_behavior_notes(graph, PlanConfig(engine.engine_id, None))


@pytest.mark.parametrize(
    "name,dq,dv,tile,cluster,splits,warp",
    [
        ("sdpa_fwd_sm107_d128_multi_q", 128, 128, 128, 2, 1, 0),
        ("sdpa_fwd_sm107_d192_multi_q", 192, 128, 128, 2, 1, 0),
        ("sdpa_fwd_sm107_d256_lookahead", 256, 256, 128, 2, 1, 0),
        ("sdpa_fwd_sm107_d256_lookahead", 128, 128, 128, 1, 1, 1),
        ("sdpa_fwd_sm107_shared_single_q", 128, 128, 128, 1, 2, 0),
        ("sdpa_fwd_sm107_shared_single_q", 192, 128, 128, 1, 1, 0),
        ("sdpa_fwd_sm107_shared_d256_lookahead", 256, 256, 128, 2, 2, 0),
        ("sdpa_fwd_sm107_d512_role_split", 512, 512, 128, 4, 1, 0),
        ("sdpa_fwd_sm107_d512_2x2", 512, 512, 64, 4, 1, 0),
    ],
)
def test_admitted_sm107_half_owners_declare_qualified_notes(name, dq, dv, tile, cluster, splits, warp, monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    f = _facts(d_qk=dq, d_v=dv, device_cc=(10, 7), device_sm_count=208, max_total_seq_len_q=32)
    knobs = engines.SdpaFwdKnobs(tile_m=tile, cga=cluster, split_kv=splits, pack_gqa=False, warp_spec_cfg=warp)
    # A retired/ineligible row would also return RTC alone. Require a legal
    # selected record so this tests the public note's qualification boundary.
    assert engines.mismatch(SPECS[name].capabilities, f, knobs) is None
    assert _notes(name, f, knobs.to_public()) == (cudnn.behavior_note.RUNTIME_COMPILATION, NOTE)


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 7)])
@pytest.mark.parametrize("kv_dtype", [cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2])
def test_mixed_kv_half_template_keeps_its_stats_contract(cc, kv_dtype, monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    name = "sdpa_fwd_sm107_d128_multi_q" if cc == (10, 7) else "sdpa_fwd_sm100_d128_multi_q"
    knobs = engines.SdpaFwdKnobs(tile_m=128, tile_n=128, cga=2, split_kv=1, pack_gqa=False)
    facts = _facts(device_cc=cc, dtype_o=cudnn.data_type.BFLOAT16, kv_dtype=kv_dtype, uniform_dtype=False, max_total_seq_len_q=32)
    assert engines.mismatch(SPECS[name].capabilities, facts, knobs) is None
    assert NOTE in _notes(name, facts, knobs)


def test_new_swap_ab_does_not_inherit_lookahead_notes():
    f = _facts(d_qk=256, d_v=256)
    assert NOTE not in _notes("sdpa_fwd_sm100_d256_swap_ab", f)


def test_paged_q1_note_follows_selected_binder():
    offset = SimpleNamespace(get_data_type=lambda: cudnn.data_type.INT32)
    tensor = SimpleNamespace(ragged_offset=offset, get_stride=lambda: (1024, 128, 1024, 1), get_ragged_offset_multiplier=lambda: 1)
    facts = _facts(s_q=1, q_t=tensor, o_t=tensor, has_paged_kv=True)
    facts.stats_t.ragged_offset = offset
    facts.stats_t.get_ragged_offset_multiplier = lambda: 1
    prefill = SPECS["sdpa_fwd_sm100_d128_multi_q"]
    decode = SPECS["sdpa_fwd_sm100_d128_single_q"]
    assert engines._thd_decode_leg(prefill.capabilities, facts)
    assert prefill.supports_stats_stride_override(facts, engines.SdpaFwdKnobs(cga=2, split_kv=1))
    assert not decode.supports_stats_stride_override(facts, engines.SdpaFwdKnobs(cga=1, split_kv=2))


def test_ineligible_knobs_and_sibling_plans_do_not_leak_notes():
    f = _facts(d_qk=256, d_v=256)
    good = engines.SdpaFwdKnobs(tile_m=128, cga=2)
    bad = engines.SdpaFwdKnobs(tile_m=16, cga=1)
    for knobs, capable in ((good, True), (bad, False), (good.to_public(), True)):
        assert (NOTE in _notes("sdpa_fwd_sm100_d256_lookahead", f, knobs)) == capable
    assert FrostSdpaFwdEngine.behavior_notes == (cudnn.behavior_note.RUNTIME_COMPILATION,)


def test_note_requires_the_native_binder_feature(monkeypatch):
    monkeypatch.setattr(cudnn._pybind_module, "_SdpaThdBinder", SimpleNamespace(supports_stats_stride_override=False))
    assert NOTE not in _notes("sdpa_fwd_sm100_d128_multi_q", _facts())


@pytest.mark.parametrize(
    "name,dq,tile,cluster",
    [
        ("sdpa_fwd_sm100_d512_role_split", 512, 128, 4),
        ("sdpa_fwd_sm100_d512_2x2", 512, 64, 4),
        ("sdpa_fwd_sm100_d512_2x2", 576, 64, 2),
    ],
)
@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
def test_wide_half_notes_follow_selected_plan(name, dq, tile, cluster, cc):
    f = _facts(d_qk=dq, d_v=512, device_cc=cc)
    good = engines.SdpaFwdKnobs(tile_m=tile, cga=cluster, split_kv=1, pack_gqa=False)
    assert NOTE in _notes(name, f, good.to_public())
    assert NOTE not in _notes(name, f, replace(good, cga=1))
    assert NOTE not in _notes(name, _facts(packing="NH", d_qk=dq, d_v=512, device_cc=cc), good)
    assert NOTE not in _notes(name, _facts(d_qk=dq, d_v=512, device_cc=cc, shape_overrides=False), good)


# The native lifecycle suite uses these same explicit public plan records.
QUALIFIED_OVERRIDE_CASES = [
    (107, "sm107_d128_multi_q", 64, 64, 128, 128, 1, 1, 0, 0, False),
    (107, "sm107_d128_multi_q", 128, 128, 128, 128, 2, 1, 0, 0, False),
    (107, "sm107_d192_multi_q", 192, 128, 128, 128, 2, 1, 0, 0, False),
    (107, "sm107_d256_lookahead", 256, 256, 128, 128, 2, 1, 0, 0, False),
    (107, "sm107_d256_lookahead", 128, 128, 128, 128, 1, 1, 0, 1, False),
    (107, "sm107_shared_single_q", 128, 128, 128, 128, 1, 2, 1, 0, True),
    (107, "sm107_shared_single_q", 192, 128, 128, 128, 1, 1, 0, 0, False),
    (107, "sm107_shared_d256_lookahead", 256, 256, 128, 128, 2, 2, 0, 0, False),
    (107, "sm107_d512_role_split", 512, 512, 128, 128, 4, 1, 0, 0, False),
    (107, "sm107_d512_2x2", 512, 512, 64, 128, 4, 1, 0, 0, False),
] + [
    (sm, *case)
    for sm in (120, 121)
    for case in (
        ("sm120_f16_general", 64, 64, 64, 64, 1, 1, 0, 0, False),
        ("sm120_f16_general", 128, 128, 64, 64, 1, 1, 0, 0, False),
        ("sm120_f16_general", 192, 128, 64, 64, 1, 1, 0, 0, False),
        ("sm120_d256_all_compute", 256, 256, 64, 64, 1, 1, 0, 0, False),
        ("sm120_d512_paired", 512, 512, 64, 32, 1, 1, 0, 0, False),
        ("sm120_d512_paired", 576, 512, 64, 32, 1, 1, 0, 0, False),
    )
]


@pytest.mark.parametrize("case", QUALIFIED_OVERRIDE_CASES, ids=lambda c: f"sm{c[0]}-{c[1]}-d{c[2]}-split{c[7]}-warp{c[9]}")
def test_qualified_override_owner_notes(case, monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    arch, owner, dq, dv, tile_m, tile_n, cga, split, pack, warp, paged = case
    knobs = engines.SdpaFwdKnobs(tile_m=tile_m, tile_n=tile_n, cga=cga, split_kv=split, pack_gqa=bool(pack), warp_spec_cfg=warp)
    changes = dict(
        d_qk=dq,
        d_v=dv,
        device_cc=(arch // 10, arch % 10),
        device_sm_count=208 if arch == 107 else 170,
        max_total_seq_len_q=32,
        has_paged_kv=paged,
        page_size=16 if paged else 0,
    )
    name = "sdpa_fwd_" + owner
    f = _facts(**changes)
    assert engines.mismatch(SPECS[name].capabilities, f, knobs) is None
    assert NOTE in _notes(name, f, knobs.to_public())
    for excluded in (dict(packing="NH"), dict(packing="padded"), dict(shape_overrides=False), dict(wants_stats=False, stats_t=None)):
        assert NOTE not in _notes(name, _facts(**changes, **excluded), knobs)
