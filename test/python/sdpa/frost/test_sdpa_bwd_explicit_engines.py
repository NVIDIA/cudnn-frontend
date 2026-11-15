# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backward explicit engines (no GPU): a multi-body row and the engines that own its bodies partition the row's graphs by
the body its declaration selects, and the owners inherit the row's heuristics decisions."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from cudnn.sdpa.bwd import engines as bwd_engines
from cudnn.sdpa.bwd.engines import Capabilities, EngineSpec, explicit_engine, mismatch, policy_name

pytestmark = pytest.mark.L0


def _lower(*args, **kwargs):
    raise AssertionError("not lowered")


def _row(delegated=frozenset()):
    caps = Capabilities(sm_lo=120, sm_hi=121, d=frozenset({64}), select_implementation=lambda facts: facts.body, delegated=delegated)
    return EngineSpec("row", caps, _lower)


def test_selected_body_partitions_row_and_owner(monkeypatch):
    monkeypatch.setattr(bwd_engines, "_declared_mismatch", lambda caps, facts, requested: None)
    row = _row(delegated=frozenset({"owner_a"}))
    owner = explicit_engine(row, "owner_a", _lower)
    a, b = SimpleNamespace(body="owner_a"), SimpleNamespace(body="b")
    assert mismatch(owner.capabilities, a) is None
    assert mismatch(row.capabilities, a) == "this graph is owned by the explicit implementation engine owner_a"
    assert mismatch(owner.capabilities, b) == "owner_a does not run this graph; its declaration selects b"
    assert mismatch(row.capabilities, b) is None


def test_two_legal_bodies_are_both_selectable_and_the_default_ranks_first(monkeypatch):
    """Legality is not the default choice: when a graph admits two bodies, each owner keeps its explicit record and
    the candidate list names the row's default first."""
    from cudnn.sdpa.bwd import heuristics

    monkeypatch.setattr(bwd_engines, "_declared_mismatch", lambda caps, facts, requested: None)
    caps = Capabilities(sm_lo=120, sm_hi=121, d=frozenset({64}), select_implementation=lambda facts: facts.bodies)
    row = EngineSpec("row", replace(caps, delegated=frozenset({"pair"})), _lower)
    one, pair = explicit_engine(row, "one", _lower), explicit_engine(row, "pair", _lower)
    both = SimpleNamespace(bodies=("pair", "one"))
    assert mismatch(one.capabilities, both) is None and mismatch(pair.capabilities, both) is None
    assert mismatch(row.capabilities, both) == "this graph is owned by the explicit implementation engine pair"
    assert mismatch(row.capabilities, SimpleNamespace(bodies=("one", "pair"))) is None  # its default is still the row's
    monkeypatch.setattr(heuristics, "ENGINE_SPECS", (row, one, pair))
    plans = heuristics.recommend("A", both, {"row": 1, "one": 2, "pair": 3})
    assert [p.engine_id for p in plans] == [3, 2]


def test_declared_reason_comes_before_the_selection(monkeypatch):
    monkeypatch.setattr(bwd_engines, "_declared_mismatch", lambda caps, facts, requested: "declared")
    row = _row(delegated=frozenset({"owner_a"}))
    assert mismatch(row.capabilities, SimpleNamespace(body="owner_a")) == "declared"


def test_owner_names_its_body_and_inherits_the_row_policy():
    row = _row(delegated=frozenset({"owner_a"}))
    owner = explicit_engine(row, "owner_a", _lower, d=frozenset({32, 64}))
    assert (owner.name, owner.capabilities.implementation, owner.capabilities.delegated) == ("owner_a", "owner_a", frozenset())
    assert owner.capabilities.d == frozenset({32, 64}) and owner.capabilities.select_implementation is row.capabilities.select_implementation
    assert (policy_name(owner), policy_name(row)) == ("row", "row")


def _sm120_facts(**over):
    import cudnn
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

    base = dict(is_backward=True, b=2, h_q=8, h_kv=8, s_q=256, s_kv=256, d_qk=128, d_v=128, dtype=cudnn.data_type.HALF, causal=True, device_cc=(12, 0))
    base.update(over)
    return SdpaGraphFacts(**base)


def _spec(name):
    return next(spec for spec in bwd_engines.ENGINE_SPECS if spec.name == name)


@pytest.mark.parametrize("d", [32, 40, 64, 128, 184, 192, 256])
@pytest.mark.parametrize("cc", [(12, 0), (12, 1)])
def test_sm120_fused_body_owns_dense_graphs_through_d256(d, cc):
    from cudnn.frost.buffers import cutedsl_state

    if not cutedsl_state()[0]:
        pytest.skip("mismatch declines every row without the DSL")
    row, fused = _spec("sdpa_bwd_sm120"), _spec("sdpa_bwd_sm120_f16_fused")
    for over in ({}, {"padded": True}, {"h_kv": 2, "deterministic": True}, {"causal": False, "window_left": 64}):
        facts = _sm120_facts(d_qk=d, d_v=d, device_cc=cc, **over)
        assert mismatch(fused.capabilities, facts) is None
        assert mismatch(row.capabilities, facts) == "this graph is owned by the explicit implementation engine sdpa_bwd_sm120_f16_fused"


@pytest.mark.parametrize("over", [{"d_qk": 264, "d_v": 264}, {"d_qk": 512, "d_v": 512}, {"thd": True, "max_total_seq_len_q": 512, "max_total_seq_len_kv": 512}])
def test_sm120_two_subgroup_body_owns_packed_and_wide_graphs(over):
    from cudnn.frost.buffers import cutedsl_state

    if not cutedsl_state()[0]:
        pytest.skip("mismatch declines every row without the DSL")
    facts = _sm120_facts(**over)
    assert mismatch(_spec("sdpa_bwd_sm120_f16_two_subgroup").capabilities, facts) is None
    assert mismatch(_spec("sdpa_bwd_sm120_f16_fused").capabilities, facts) is not None
    assert mismatch(_spec("sdpa_bwd_sm120").capabilities, facts) == "this graph is owned by the explicit implementation engine sdpa_bwd_sm120_f16_two_subgroup"


def test_sm120_fused_identity():
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.bwd import api_dsl
    from cudnn.sdpa.bwd.kernels.sm120.f16_fused import contract

    family = next(f for f in MANIFEST if f.name == "frost_sdpa_bwd")
    assert family.engine_id + family.slots[contract.NAME].slot == 20618 and not family.slots[contract.NAME].opt_in
    fused = _spec(contract.NAME)
    assert (fused.capabilities.implementation, policy_name(fused)) == (contract.NAME, contract.LEGACY_ROW)
    assert api_dsl._SM120_KERNEL_FILE == contract.KERNEL_FILE == "sm120/f16_fused/kernel.py"
    assert max(fused.capabilities.d) == 256 and 16 not in fused.capabilities.tile_ms | fused.capabilities.tile_ns
    names = [spec.name for spec in bwd_engines.ENGINE_SPECS]
    assert names.index(contract.NAME) == names.index(contract.LEGACY_ROW) + 1


def test_sm120_fused_inherits_the_row_default_tiles():
    from cudnn.sdpa.bwd.heuristics import default_knobs

    for d in (40, 64, 128, 256):
        facts = _sm120_facts(d_qk=d, d_v=d)
        assert default_knobs(_spec("sdpa_bwd_sm120_f16_fused"), facts) == default_knobs(_spec("sdpa_bwd_sm120"), facts)


def test_sm120_fused_host_refuses_other_bodies():
    from cudnn.sdpa.bwd.kernels.sm120.f16_fused.host import lower

    with pytest.raises(ValueError, match="dense graphs with D_QK <= 256 only"):
        lower(None, _sm120_facts(thd=True), None, lower_impl=_lower)


@pytest.mark.parametrize("fmt,slot", [("fp8", 16), ("mxfp8", 17)])
def test_sm120_quantized_rows_name_their_one_body(fmt, slot):
    import importlib
    import pathlib

    from cudnn.engines.manifest import MANIFEST

    contract = importlib.import_module(f"cudnn.sdpa.bwd.kernels.sm120.{fmt}.contract")
    family = next(f for f in MANIFEST if f.name == "frost_sdpa_bwd")
    assert family.slots[contract.NAME].slot == slot and family.slots[contract.NAME].opt_in
    caps = _spec(contract.NAME).capabilities
    assert (caps.implementation, caps.select_implementation, caps.delegated) == (contract.NAME, None, frozenset())
    assert (pathlib.Path(bwd_engines.__file__).parent / "kernels" / contract.KERNEL_FILE).is_file()


@pytest.mark.parametrize(
    "name,slot,kernel_file",
    [
        ("sdpa_bwd_sm80_f16_d64", 19, "sm80/f16_d64/kernel.py"),
        ("sdpa_bwd_sm80_f16_two_subgroup", 20, "sm80/f16_two_subgroup/kernel.py"),
        ("sdpa_bwd_sm120_f16_two_subgroup", 21, "sm80/f16_two_subgroup/kernel.py"),
    ],
)
def test_two_subgroup_and_d64_identities(name, slot, kernel_file):
    import importlib
    import pathlib

    from cudnn.engines.manifest import MANIFEST

    family = next(f for f in MANIFEST if f.name == "frost_sdpa_bwd")
    assert family.slots[name].slot == slot and not family.slots[name].opt_in
    arch, impl = ("sm120", "f16_two_subgroup") if name.startswith("sdpa_bwd_sm120") else ("sm80", name.split("sdpa_bwd_sm80_")[1])
    contract = importlib.import_module(f"cudnn.sdpa.bwd.kernels.{arch}.{impl}.contract")
    assert (contract.NAME, contract.KERNEL_FILE) == (name, kernel_file)
    assert (pathlib.Path(bwd_engines.__file__).parent / "kernels" / kernel_file).is_file()
    spec = _spec(name)
    assert (spec.capabilities.implementation, policy_name(spec)) == (name, contract.LEGACY_ROW)
    names = [s.name for s in bwd_engines.ENGINE_SPECS]
    assert names.index(contract.LEGACY_ROW) < names.index(name) <= names.index(contract.LEGACY_ROW) + 2


def test_sm80_loaders_and_d64_tile_follow_the_contracts():
    from cudnn.sdpa.bwd import api_dsl
    from cudnn.sdpa.bwd.kernels.sm80 import half_contract
    from cudnn.sdpa.bwd.kernels.sm80.f16_d64 import kernel as d64
    from cudnn.sdpa.bwd.kernels.sm80.f16_two_subgroup import contract as two_subgroup

    assert api_dsl._SM80_BWD_KERNEL_FILE == two_subgroup.KERNEL_FILE
    assert api_dsl._sm80_bwd_kernel_mod("d64") is d64
    assert half_contract.D64_TILE == (d64.M_BLOCK, d64.N_BLOCK)


def _sm80_facts(**over):
    base = dict(device_cc=(8, 0), d_qk=64, d_v=64, causal=False, s_q=256, s_kv=256)
    base.update(over)
    return _sm120_facts(**base)


@pytest.mark.parametrize(
    "over",
    [
        {},
        {"causal": True},
        {"window_left": 64},
        {"bottom_right": True, "causal": True},
        {"padded": True},
        {"has_bias": True},
        {"has_sink": True},
        {"deterministic": True},
        {"h_kv": 2},
        {"d_qk": 128, "d_v": 128},
        {"s_q": 200},
        {"s_kv": 192},
        {"thd": True, "max_total_seq_len_q": 512, "max_total_seq_len_kv": 512},
        {"right_band_widening": True, "right_bound": 64},
        {"right_band_widening": True, "right_bound": 64, "causal": True},
    ],
)
def test_sm80_selector_is_the_adapters_d64_rule(over):
    """The D64 owner's domain is the adapter's own fast-path rule (api_dsl._sm80_d64_fast_path_eligible), restated on facts."""
    from cudnn.sdpa.bwd.api_dsl import _sm80_d64_fast_path_eligible
    from cudnn.sdpa.bwd.kernels.sm80 import half_contract

    f = _sm80_facts(**over)
    is_causal = f.causal or f.right_band_widening  # what lower_dsl_bwd hands the adapter
    mask = "none" if not (is_causal or f.window_left is not None) else "masked"
    adapter = not f.thd and _sm80_d64_fast_path_eligible(
        d_qk=f.d_qk,
        d_v=f.d_v,
        h_q=f.h_q,
        h_kv=f.h_kv,
        s_q=f.s_q,
        s_kv=f.s_kv,
        mask_token=mask,
        right_bound=f.right_bound if f.right_band_widening else 0,
        causal_bottom_right=f.bottom_right,
        bw_kwargs=dict(
            seq_kv_lens=object() if f.padded else None,
            bias=object() if f.has_bias else None,
            sinks=object() if f.has_sink else None,
            deterministic=f.deterministic,
        ),
    )
    assert (half_contract.select_implementation(f) == half_contract.D64) == adapter
    assert adapter == (over == {})


def test_sm80_row_delegates_every_body():
    from cudnn.frost.buffers import cutedsl_state

    if not cutedsl_state()[0]:
        pytest.skip("mismatch declines every row without the DSL")
    row = _spec("sdpa_bwd_sm80")
    for over, owner in (({}, "sdpa_bwd_sm80_f16_d64"), ({"causal": True}, "sdpa_bwd_sm80_f16_two_subgroup")):
        f = _sm80_facts(**over)
        assert mismatch(_spec(owner).capabilities, f) is None
        assert mismatch(row.capabilities, f) == f"this graph is owned by the explicit implementation engine {owner}"
