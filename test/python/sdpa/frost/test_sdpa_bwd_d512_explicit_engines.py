# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed SM100 D512 records retain the shared admission domain and body identity."""

from dataclasses import replace

import pytest

import cudnn
from cudnn.engines.manifest import MANIFEST
from cudnn.sdpa.bwd import engines, heuristics
from cudnn.sdpa.bwd.kernels.sm100 import d512_half_contract as contract
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 4), (10, 7)])
@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16])
@pytest.mark.parametrize("d", [256, 264, 384, 504, 512, 520])
@pytest.mark.parametrize("profile", ["dense", "causal", "deterministic", "gqa", "thd", "missing_totals", "padding", "decode"])
def test_d512_owners_preserve_admission_and_default_order(cc, dtype, d, profile):
    specs = {spec.name: spec for spec in engines.ENGINE_SPECS}
    row = specs[contract.LEGACY_ROW]
    legacy = replace(row.capabilities, select_implementation=None, delegated=frozenset())
    changes = {
        "dense": {},
        "causal": {"causal": True},
        "deterministic": {"deterministic": True},
        "gqa": {"h_kv": 2},
        "thd": {"thd": True, "padded": True, "max_total_seq_len_q": 512, "max_total_seq_len_kv": 512},
        "missing_totals": {"thd": True, "padded": True},
        "padding": {"padded": True},
        "decode": {"s_q": 1},
    }[profile]
    facts = SdpaGraphFacts(**(dict(is_backward=True, b=2, h_q=4, h_kv=4, s_q=256, s_kv=256, d_qk=d, d_v=d, dtype=dtype, device_cc=cc) | changes))
    baseline = engines.mismatch(legacy, facts)
    for name in (contract.ROLE_SPLIT, contract.TWO_BY_TWO):
        assert engines.mismatch(specs[name].capabilities, facts) == baseline
        assert engines.policy_name(specs[name]) == row.name
    family = next(f for f in MANIFEST if f.name == "frost_sdpa_bwd")
    offered = {name: family.engine_id + family.slots[name].slot for name in (row.name, contract.ROLE_SPLIT, contract.TWO_BY_TWO)}
    plans = heuristics.recommend("A", facts, offered)
    if baseline is None:
        assert "owned by the explicit implementation" in engines.mismatch(row.capabilities, facts)
        assert [(p.engine_id, p.knobs) for p in plans] == [(20626, None), (20627, None)]
    else:
        assert not plans


@pytest.mark.parametrize("name,slot", [(contract.ROLE_SPLIT, 26), (contract.TWO_BY_TWO, 27)])
def test_d512_explicit_identity_and_no_tile_knobs(name, slot):
    family = next(f for f in MANIFEST if f.name == "frost_sdpa_bwd")
    assert family.slots[name].slot == slot
    assert family.slots[name].opt_in == (name == contract.TWO_BY_TWO)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == name)
    assert spec.capabilities.implementation == name
    assert not spec.capabilities.tile_ms and not spec.capabilities.tile_ns
    assert not spec.capabilities.delegated


@pytest.mark.parametrize("enabled", [False, True])
def test_d512_discovery_keeps_additional_body_opt_in(monkeypatch, enabled):
    if enabled:
        monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    else:
        monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    family = next(f for f in MANIFEST if f.name == "frost_sdpa_bwd")
    offered = family.offered_ids()
    assert contract.ROLE_SPLIT in offered
    assert (contract.TWO_BY_TWO in offered) == enabled
    facts = SdpaGraphFacts(is_backward=True, b=2, h_q=4, h_kv=4, s_q=256, s_kv=256, d_qk=512, d_v=512, dtype=cudnn.data_type.BFLOAT16, device_cc=(10, 0))
    assert [p.engine_id for p in heuristics.recommend("A", facts, offered)] == ([20626, 20627] if enabled else [20626])


@requires_dsl
@requires_pre_rubin_blackwell
@pytest.mark.parametrize("twin", [False, True], ids=["role-split", "2x2"])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"], ids=["fp16", "bf16"])
def test_public_d512_record_rebuild_ignores_legacy_switch(monkeypatch, twin, dtype_name):
    import torch

    from cudnn.sdpa.bwd import api_dsl
    import test_sdpa_bwd_dsl_sm100 as helper

    dtype = getattr(torch, dtype_name)
    case = helper._prepared_case(dtype=dtype, causal=True, hkv=2)
    owner = contract.TWO_BY_TWO if twin else contract.ROLE_SPLIT
    engine_id = 20627 if twin else 20626
    template = "sm100/d512_f16_2x2/kernel.py" if twin else "sm100/d512_f16_role_split/kernel.py"
    seen, original = [], api_dsl.load_template

    def spy(path, params, tag="template"):
        if tag == owner + "_stage2":
            seen.append(path.rsplit("/kernels/", 1)[-1])
        return original(path, params, tag)

    monkeypatch.setattr(api_dsl, "load_template", spy)
    b, h, sq, d = case.tensors["q"].shape
    _, hk, sk, _ = case.tensors["k"].shape

    def build(record):
        g, refs, outputs = helper._build_graph(b, h, hk, sq, sk, d, d**-0.5, dt=dtype, record=record, use_causal_mask=True)
        g.select_plan(0)
        g.check_support()
        g.build_plans()
        refs.update(zip(("dq", "dk", "dv"), outputs))
        return g, {refs[name]: value for name, value in case.tensors.items()}, torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)

    monkeypatch.setattr(api_dsl, "STAGE2_2X2", not twin)
    g, pack, workspace = build((engine_id, {}))
    g.execute(pack, workspace)
    helper._check_prepared(case)
    expected = [case.tensors[name].clone() for name in ("dq", "dk", "dv")]
    record = g.get_engine_and_knobs_at_index(0)
    assert record == (engine_id, {})

    monkeypatch.setattr(api_dsl, "STAGE2_2X2", twin)
    rebuilt, pack, workspace = build(record)
    for name in ("dq", "dk", "dv"):
        case.tensors[name].fill_(float("nan"))
    workspace.fill_(0xBD)
    rebuilt.execute(pack, workspace)
    helper._check_prepared(case)
    for name, want in zip(("dq", "dk", "dv"), expected):
        assert torch.equal(case.tensors[name].view(torch.int16), want.view(torch.int16)), name
    assert seen == [template, template]

    with pytest.raises((ValueError, RuntimeError), match="20602|owned by|not supported|No execution plans"):
        build((20602, {}))
