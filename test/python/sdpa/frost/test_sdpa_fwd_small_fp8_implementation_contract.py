# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for the independent small-head per-tensor FP8 records."""

from dataclasses import asdict, replace
import importlib
from pathlib import Path
import subprocess
import sys

import pytest

import cudnn
from cudnn.sdpa.fwd import config_sm100 as legacy
from cudnn.sdpa.fwd import engines, heuristics, placement
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0


@pytest.mark.parametrize(
    "variant,record",
    [
        ("multi_q", "CfgD64"),
        ("multi_q", "CfgD128"),
        ("dual_kv", "CfgD128"),
        ("swap_ab", "CfgD128Fp8Decode"),
    ],
)
def test_configuration_defaults_are_independent(variant, record, monkeypatch):
    module = importlib.import_module(f"cudnn.sdpa.fwd.kernels.sm100.d128_fp8_{variant}.config")
    own_record = getattr(module, record)
    expected = asdict(getattr(legacy, record)())
    assert own_record.__bases__ == (object,)
    assert asdict(own_record()) == expected
    # Class-level role tuning in a legacy sibling must not change this body.
    monkeypatch.setattr(getattr(legacy, record), "STAGES_KV", 999)
    assert own_record.STAGES_KV == expected["STAGES_KV"]


@pytest.mark.parametrize("dual", [False, True])
def test_prefill_factory_cannot_switch_main_implementation(dual):
    variant = "dual_kv" if dual else "multi_q"
    module = importlib.import_module(f"cudnn.sdpa.fwd.kernels.sm100.d128_fp8_{variant}.config")
    params = legacy.TemplateParams(dtype_qkv=0, dtype_o=2, cta_mma=1, dual_kv_stream=dual)
    config, _ = module.make_cfg_d128(params)
    assert bool(config.DUAL_KV_STREAM) == dual
    with pytest.raises(ValueError, match="different FP8 main implementation"):
        module.make_cfg_d128(legacy.TemplateParams(dtype_qkv=0, dual_kv_stream=not dual))


@pytest.mark.parametrize("variant", ["multi_q", "dual_kv", "swap_ab"])
def test_factories_do_not_call_legacy_implementations(variant, monkeypatch):
    module = importlib.import_module(f"cudnn.sdpa.fwd.kernels.sm100.d128_fp8_{variant}.config")

    def forbidden(*args, **kwargs):
        raise AssertionError("owner consulted a legacy implementation factory")

    for name in ("make_cfg_d64", "make_cfg_d128", "make_cfg_d128_fp8_decode", "_validate_cfg_d128"):
        monkeypatch.setattr(legacy, name, forbidden)
    if variant == "swap_ab":
        params = legacy.TemplateParamsDecode(dtype_qkv=0, cta_mma=1, decode_q_tile=16, decode_live_columns=8)
        config, _ = module.make_cfg_d128_fp8_decode(params)
        assert config.N_Q == 16
    else:
        params = legacy.TemplateParams(dtype_qkv=0, cta_mma=1, dual_kv_stream=variant == "dual_kv")
        config, _ = module.make_cfg_d128(params)
        assert config.TILE_K == 128
        if variant == "multi_q":
            config, _ = module.make_cfg_d64(params)
            assert config.TILE_K == 64


def _facts(**overrides):
    values = dict(
        b=2,
        h_q=8,
        h_kv=2,
        s_q=1,
        s_kv=4096,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.BFLOAT16,
        is_fp8=True,
        uniform_dtype=True,
        device_cc=(10, 0),
    )
    values.update(overrides)
    return SdpaGraphFacts(**values)


@pytest.mark.parametrize(
    "overrides,knobs,expected",
    [
        ({}, dict(cga=1, pack_gqa=True), "dual_kv"),
        ({}, dict(cga=1, pack_gqa=True, swap_ab=1), "swap_ab"),
        ({}, dict(cga=2, pack_gqa=True), "multi_q"),
        ({}, dict(pack_gqa=True), "multi_q"),
        ({"s_q": 33}, dict(cga=1, pack_gqa=True), "multi_q"),
        ({"s_q": 32}, dict(cga=1, pack_gqa=True), "dual_kv"),
        ({"has_paged_kv": True}, dict(cga=1, pack_gqa=True), "swap_ab"),
        ({"has_paged_kv": True, "s_q": 3}, dict(cga=1, pack_gqa=True), "dual_kv"),
        ({"has_paged_kv": True, "device_cc": (10, 3)}, dict(cga=1, pack_gqa=True), "dual_kv"),
        ({"has_paged_kv": True, "b": 65536}, dict(cga=1, pack_gqa=True), "dual_kv"),
        ({"has_paged_kv": True, "thd": True}, dict(cga=1, pack_gqa=True), "multi_q"),
        ({"has_paged_kv": True, "o_block_scale": 32}, dict(cga=1, pack_gqa=True), "dual_kv"),
        ({"has_paged_kv": True, "d_qk": 64, "d_v": 64, "s_q": 8}, dict(cga=1, pack_gqa=True, swap_ab=1), "swap_ab"),
        ({"has_paged_kv": True, "d_qk": 64, "d_v": 64, "s_q": 8}, dict(cga=1, pack_gqa=True, swap_ab=0), "multi_q"),
    ],
)
def test_central_selection_preserves_body_boundaries(overrides, knobs, expected, monkeypatch):
    monkeypatch.setattr(legacy, "FP8_DUAL_KV_STREAM_ROUTED", True)
    selected = heuristics.select_sm100_small_fp8_geometry(_facts(**overrides), engines.SdpaFwdKnobs(**knobs))
    assert selected.implementation == f"sdpa_fwd_sm100_d128_fp8_{expected}"


@pytest.mark.parametrize("variant", ["multi_q", "dual_kv", "swap_ab"])
def test_fixed_host_never_reenters_selection(variant, monkeypatch):
    host = importlib.import_module(f"cudnn.sdpa.fwd.kernels.sm100.d128_fp8_{variant}.host")

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed host reentered implementation selection")

    monkeypatch.setattr(heuristics, "select_sm100_small_fp8_geometry", forbidden)
    monkeypatch.setattr(heuristics, "recommend", forbidden)
    calls = []
    host.lower(None, _facts(), engines.SdpaFwdKnobs(cga=1), lower_impl=lambda *args, **kwargs: calls.append((args, kwargs)), knob_type=engines.SdpaFwdKnobs)
    assert len(calls) == 1
    args, kwargs = calls[0]
    selected = kwargs["selected_kernel"]
    assert selected.implementation.endswith(variant)
    assert engines.SdpaFwdKnobs.from_public(args[2].to_public()) == args[2]
    params = selected.configure_params(legacy.TemplateParams(dtype_qkv=0, dtype_o=2))
    # Existing global routing switches cannot change a retained implementation.
    monkeypatch.setattr(legacy, "FP8_DUAL_KV_STREAM_ROUTED", False)
    assert selected.configure_params(params) == params
    cfg_mod = importlib.import_module(f"cudnn.sdpa.fwd.kernels.sm100.d128_fp8_{variant}.config")
    factory = cfg_mod.make_cfg_d128_fp8_decode if variant == "swap_ab" else cfg_mod.make_cfg_d128
    cfg, _ = factory(params)
    if variant == "swap_ab":
        assert (args[2].tile_m, cfg.N_Q, params.decode_live_columns) == (16, 16, 8)
        assert args[2].sched_policy == 0
    else:
        assert bool(cfg.DUAL_KV_STREAM) == (variant == "dual_kv")


@pytest.mark.parametrize("variant", ["multi_q", "dual_kv", "swap_ab"])
def test_cold_host_import_does_not_import_registry(variant):
    package_root = str(Path(cudnn.__file__).resolve().parent.parent)
    code = "\n".join(
        (
            "import sys",
            "sys.meta_path[:] = [f for f in sys.meta_path if 'editable' not in str(f).lower()]",
            f"sys.path.insert(0, {package_root!r})",
            f"import cudnn.sdpa.fwd.kernels.sm100.d128_fp8_{variant}.host",
            "assert 'cudnn.sdpa.fwd.engines' not in sys.modules",
        )
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


SPECS = {spec.name: spec for spec in engines.ENGINE_SPECS}
PREFIX = "sdpa_fwd_sm100_d128_fp8_"
OFFERED = {PREFIX + variant: engine for variant, engine in (("multi_q", 20527), ("dual_kv", 20528), ("swap_ab", 20529))}


@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
@pytest.mark.parametrize("d", [64, 128])
def test_exact_domain_moves_to_explicit_owners(cc, d):
    facts = _facts(device_cc=cc, d_qk=d, d_v=d)
    assert "owned by" in engines.mismatch(SPECS["sdpa_fwd_prefill_sm100_fp8"].capabilities, facts)
    for kind in ("A", "FALLBACK"):
        plans = heuristics.recommend(kind, facts, OFFERED)
        assert plans
        for plan in plans:
            assert engines.SdpaFwdKnobs.from_public(plan.knobs.to_public()) == plan.knobs
            assert plan.knobs.swap_ab is None


@pytest.mark.parametrize("variant", ["multi_q", "dual_kv", "swap_ab"])
@pytest.mark.parametrize("changes", [{"device_cc": (10, 7)}, {"d_qk": 112, "d_v": 112}, {"is_mxfp8": True}, {"is_fp8": False}])
def test_other_implementations_remain_outside_owner(variant, changes):
    assert engines.mismatch(SPECS[PREFIX + variant].capabilities, _facts(**changes)) is not None


@pytest.mark.parametrize("variant", ["multi_q", "dual_kv"])
@pytest.mark.parametrize("sq", [1, 129, 257])
def test_prefill_body_eligibility_is_not_a_query_length_policy(variant, sq):
    facts = _facts(s_q=sq)
    knobs = engines.SdpaFwdKnobs(tile_m=128, tile_n=128, cga=1, pack_gqa=False, split_kv=1, sched_policy=0)
    assert engines.mismatch(SPECS[PREFIX + variant].capabilities, facts, knobs) is None
    assert engines.mismatch(SPECS[PREFIX + variant].capabilities, facts, replace(knobs, swap_ab=1)) is not None


@pytest.mark.parametrize(
    "changes,pack",
    [
        ({"s_q": 9}, False),
        ({"s_q": 3}, True),
        ({"b": 65536}, False),
        ({"thd": True}, False),
        ({"device_cc": (10, 3), "has_paged_kv": True}, False),
        ({"d_qk": 64, "d_v": 64}, False),
        ({"d_qk": 64, "d_v": 64, "has_paged_kv": True, "s_q": 9}, True),
        ({"attn_scale_prefolded": True}, False),
        ({"o_block_scale": 32}, False),
    ],
)
def test_swap_ab_declines_outside_qualified_geometry(changes, pack):
    knobs = engines.SdpaFwdKnobs(tile_m=16, tile_n=128, cga=1, pack_gqa=pack, split_kv=1, sched_policy=0)
    assert engines.mismatch(SPECS[PREFIX + "swap_ab"].capabilities, _facts(**changes), knobs) is not None


@pytest.mark.parametrize("variant", ["multi_q", "dual_kv", "swap_ab"])
@pytest.mark.parametrize("verdict", [False, True])
def test_own_study_precedes_inherited_placement(monkeypatch, variant, verdict):
    from cudnn.sdpa import placement_table

    name = PREFIX + variant
    monkeypatch.setattr(placement_table, "lookup", lambda row, facts: verdict if row == name else not verdict)
    assert placement.place(SPECS[name], _facts()) == (placement.LEAD if verdict else placement.TRAIL)


def test_manual_alternate_has_no_inherited_lead_claim(monkeypatch):
    from cudnn.sdpa import placement_table

    monkeypatch.setattr(placement_table, "lookup", lambda row, facts: None if row in OFFERED else True)
    monkeypatch.setattr(legacy, "FP8_DUAL_KV_STREAM_ROUTED", True)
    knobs = engines.SdpaFwdKnobs(tile_m=128, tile_n=128, cga=1, pack_gqa=True, split_kv=1, sched_policy=0)
    assert placement.place(SPECS[PREFIX + "multi_q"], _facts(), knobs) == placement.TRAIL
    assert placement.place(SPECS[PREFIX + "dual_kv"], _facts(), knobs) == placement.LEAD


def test_unsplit_thd_keeps_the_existing_two_cta_requirement():
    facts = _facts(s_q=17, thd=True, padded=True)
    knobs = engines.SdpaFwdKnobs(tile_m=128, tile_n=128, cga=1, pack_gqa=False, split_kv=1, sched_policy=0)
    policy = engines.sm100_legacy_policy_spec("sdpa_fwd_prefill_sm100_fp8")
    before = engines.mismatch(policy.capabilities, facts, knobs)
    after = engines.mismatch(SPECS[PREFIX + "multi_q"].capabilities, facts, knobs)
    assert before is not None and "other THD graphs run the cga2" in before
    assert after == before
    assert engines.mismatch(SPECS[PREFIX + "multi_q"].capabilities, facts, replace(knobs, cga=2)) is None
