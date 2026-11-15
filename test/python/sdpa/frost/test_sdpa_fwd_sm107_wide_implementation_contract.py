# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Explicit Rubin wide-half records preserve geometry and isolate their factories."""

from dataclasses import asdict, replace
from functools import partial

import pytest
import cudnn
from cudnn.sdpa.fwd import engines, heuristics, placement
from cudnn.sdpa.fwd.config_common import TemplateParams
from cudnn.sdpa.fwd.kernels.sm107.d512_half_contract import LEGACY_ROW, ROLE_SPLIT, TWO_BY_TWO, explicit_geometry
from sdpa.frost.test_sdpa_fwd_d512_implementation_contract import facts as _facts

pytestmark = pytest.mark.L0
SPECS = {s.name: s for s in engines.ENGINE_SPECS}


def facts(**kwargs):
    return _facts(device_cc=(10, 7), device_sm_count=208, **kwargs)


@pytest.mark.parametrize("name,dq", [(ROLE_SPLIT, 512), (TWO_BY_TWO, 512), (TWO_BY_TWO, 576)])
def test_explicit_owner_and_physical_cluster(name, dq):
    f = facts(d_qk=dq)
    g = explicit_geometry(name, f)
    k = g.canonical_knobs(engines.SdpaFwdKnobs())
    assert "owned by" in engines.mismatch(SPECS[LEGACY_ROW].capabilities, f)
    assert engines.mismatch(SPECS[name].capabilities, f, k) is None
    assert engines.SdpaFwdKnobs.from_public(k.to_public()) == k
    assert (k.tile_m, k.cga, g.cta_group) == (128 if name == ROLE_SPLIT else 64, 2 if dq == 576 else 4, 2)
    assert engines.effective_cgas(SPECS[name].capabilities, f) == frozenset({g.cluster_m})
    assert engines.mismatch(SPECS[name].capabilities, f, replace(k, cga=1))
    assert engines.mismatch(SPECS[name].capabilities, f, replace(k, tile_m=32))


@pytest.mark.parametrize("name", [ROLE_SPLIT, TWO_BY_TWO])
@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 8), (12, 0)])
def test_other_architectures_decline(name, cc):
    assert engines.mismatch(SPECS[name].capabilities, replace(facts(), device_cc=cc))


@pytest.mark.parametrize("dims", [(504, 504), (128, 128), (256, 256)])
def test_legacy_retains_nonmigrated_domain(dims):
    f = facts(d_qk=dims[0], d_v=dims[1])
    assert engines.mismatch(SPECS[LEGACY_ROW].capabilities, f) is None
    assert all(engines.mismatch(SPECS[name].capabilities, f) for name in (ROLE_SPLIT, TWO_BY_TWO))


@pytest.mark.parametrize("sq", [1, 129, 8192])
@pytest.mark.parametrize("name", [ROLE_SPLIT, TWO_BY_TWO])
def test_fixed_host_does_not_reenter_selection(monkeypatch, sq, name):
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit record reentered body selection")

    monkeypatch.setattr(heuristics, "select_sm107_d512_geometry", forbidden)
    spec = SPECS[name]
    assert isinstance(spec.lower, partial)
    calls = []
    f = facts(s_q=sq)
    spec.lower.func(spec, f, None, lower_impl=lambda *args, **kw: calls.append((args, kw)), knob_type=engines.SdpaFwdKnobs)
    args, kw = calls[0]
    g = kw["selected_kernel"]
    assert g.implementation == name
    assert args[2].tile_m == g.query_tile
    for before in (False, True):
        assert g.configure_params(TemplateParams(mma_2x2=before)).mma_2x2 == (name == TWO_BY_TWO)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("pack,split", [(False, 1), (True, 1), (True, 3), (False, 3)])
def test_native_packed_and_split_2x2_domain(paged, pack, split):
    f = facts(has_paged_kv=paged, page_size=128 if paged else 0, padded=paged, h_q=12, h_kv=2)
    g = explicit_geometry(TWO_BY_TWO, f)
    k = g.canonical_knobs(engines.SdpaFwdKnobs(pack_gqa=pack, split_kv=split))
    assert engines.mismatch(SPECS[TWO_BY_TWO].capabilities, f, k) is None
    role = explicit_geometry(ROLE_SPLIT, f).canonical_knobs(k)
    assert bool(engines.mismatch(SPECS[ROLE_SPLIT].capabilities, f, role)) == bool(paged or pack or split > 1)


def test_mixed_kv_uses_one_pair_and_declines_other_storage():
    f = facts(uniform_dtype=False, kv_dtype=cudnn.data_type.FP8_E4M3)
    g = explicit_geometry(TWO_BY_TWO, f)
    k = g.canonical_knobs(engines.SdpaFwdKnobs(pack_gqa=True, split_kv=3))
    assert (g.cluster_m, g.cta_group) == (2, 2)
    assert engines.mismatch(SPECS[TWO_BY_TWO].capabilities, f, k) is None
    assert engines.mismatch(SPECS[TWO_BY_TWO].capabilities, replace(f, thd=True), k)
    assert engines.mismatch(SPECS[ROLE_SPLIT].capabilities, f)


@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
@pytest.mark.parametrize("dq", [512, 576])
def test_central_policy_keeps_budget_order_and_placement(kind, dq):
    f = facts(d_qk=dq)
    policy = engines.sm100_legacy_policy_spec(LEGACY_ROW)
    old = heuristics._knob_sets(policy, f) if kind == "A" else [heuristics._fallback_knobs(policy, f)]
    offered = {ROLE_SPLIT: 20554, TWO_BY_TWO: 20555, LEGACY_ROW: 20514}
    expected = []
    for k in old:
        if engines.mismatch(policy.capabilities, f, k) is not None:
            continue
        g = heuristics.select_sm107_d512_geometry(f, k)
        record = (offered[g.implementation], g.canonical_knobs(k))
        if record not in expected:
            expected.append(record)
    got = heuristics.recommend(kind, f, offered)
    assert [(p.engine_id, p.knobs) for p in got] == expected
    for p in got:
        name = next(name for name, eid in offered.items() if eid == p.engine_id)
        assert placement.place(SPECS[name], f, p.knobs) == placement.place(policy, f, replace(p.knobs, tile_m=128, cga=2))


@pytest.mark.parametrize("name", [ROLE_SPLIT, TWO_BY_TWO])
@pytest.mark.parametrize("verdict", [False, True])
def test_current_owner_study_precedes_inherited_placement(monkeypatch, name, verdict):
    from cudnn.sdpa import placement_table

    f = facts()
    k = explicit_geometry(name, f).canonical_knobs(engines.SdpaFwdKnobs())
    monkeypatch.setattr(placement_table, "lookup", lambda row, facts: verdict if row == name else not verdict)
    assert placement.place(SPECS[name], f, k) == (placement.LEAD if verdict else placement.TRAIL)


def test_config_factories_do_not_consume_sibling_types_or_factories(monkeypatch):
    from cudnn.sdpa.fwd import config_sm107 as old
    from cudnn.sdpa.fwd.kernels.sm107.d512_f16_role_split import config as role
    from cudnn.sdpa.fwd.kernels.sm107.d512_f16_2x2 import config as two

    p = TemplateParams(dtype_qkv=2, d_flavor=512, cta_mma=2)
    controls = [(role.make_cfg_d512, p), (two.make_cfg_d512_2x2, replace(p, mma_2x2=True))]
    want = [(asdict(fn(p)[0]), asdict(fn(p)[1])) for fn, p in controls]

    def forbidden(*args, **kwargs):
        raise AssertionError("sibling config was used")

    for name in ("make_cfg_d512", "make_cfg_d512_2x2", "_make_cfg_d512_family", "CfgD512", "CfgD512X2", "CfgD512Mixed"):
        monkeypatch.setattr(old, name, forbidden)
    assert [(asdict(fn(p)[0]), asdict(fn(p)[1])) for fn, p in controls] == want
