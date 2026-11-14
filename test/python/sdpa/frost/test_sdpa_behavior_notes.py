# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU admission tests for plan-specific Stats stride override declarations."""

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
        ("sdpa_fwd_prefill_sm100", 128, (10, 0)),
        ("sdpa_fwd_prefill_sm100", 128, (10, 3)),
    ],
)
def test_eligible_new_and_legacy_half_plans_declare_note(name, d, cc, monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    f = _facts(d_qk=d, d_v=d, device_cc=cc)
    assert _notes(name, f) == (cudnn.behavior_note.RUNTIME_COMPILATION, NOTE)


@pytest.mark.parametrize(
    "changes",
    [dict(packing="NH"), dict(packing="padded"), dict(shape_overrides=False), dict(wants_stats=False, stats_t=None), dict(thd=False), dict(is_fp8=True)],
)
def test_familiar_legacy_name_is_not_a_capability(changes):
    assert NOTE not in _notes("sdpa_fwd_prefill_sm100", _facts(**changes))


@pytest.mark.parametrize("name,cc", [("sdpa_fwd_prefill_sm107", (10, 7)), ("sdpa_fwd_prefill_sm120", (12, 0))])
def test_unqualified_architectures_do_not_advertise_the_shared_binder(name, cc, monkeypatch):
    from cudnn.frost import buffers

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    assert NOTE not in _notes(name, _facts(device_cc=cc))


def test_new_swap_ab_does_not_inherit_lookahead_notes():
    f = _facts(d_qk=256, d_v=256)
    assert NOTE not in _notes("sdpa_fwd_sm100_d256_swap_ab", f)


def test_ineligible_knobs_and_sibling_plans_do_not_leak_notes():
    f = _facts(d_qk=256, d_v=256)
    good = engines.SdpaFwdKnobs(tile_m=128, cga=2)
    bad = engines.SdpaFwdKnobs(tile_m=16, cga=1)
    for knobs, capable in ((good, True), (bad, False), (good.to_public(), True)):
        assert (NOTE in _notes("sdpa_fwd_sm100_d256_lookahead", f, knobs)) == capable
    assert FrostSdpaFwdEngine.behavior_notes == (cudnn.behavior_note.RUNTIME_COMPILATION,)


def test_note_requires_the_native_binder_feature(monkeypatch):
    monkeypatch.setattr(cudnn._pybind_module, "_SdpaThdBinder", SimpleNamespace(supports_stats_stride_override=False))
    assert NOTE not in _notes("sdpa_fwd_prefill_sm100", _facts())


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
    from dataclasses import replace

    f = _facts(d_qk=dq, d_v=512, device_cc=cc)
    good = engines.SdpaFwdKnobs(tile_m=tile, cga=cluster, split_kv=1, pack_gqa=False)
    assert NOTE in _notes(name, f, good.to_public())
    assert NOTE not in _notes(name, f, replace(good, cga=1))
    assert NOTE not in _notes(name, _facts(packing="NH", d_qk=dq, d_v=512, device_cc=cc), good)
    assert NOTE not in _notes(name, _facts(d_qk=dq, d_v=512, device_cc=cc, shape_overrides=False), good)
