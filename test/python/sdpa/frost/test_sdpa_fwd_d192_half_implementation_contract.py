# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for isolated half D192 multi-Q and shared single-Q bodies."""

from dataclasses import asdict, replace
import importlib
from pathlib import Path
import subprocess
import sys

import pytest

import cudnn
from cudnn.sdpa.fwd import config_sm100 as legacy
from cudnn.sdpa.fwd import engines, heuristics, placement
from cudnn.sdpa.fwd.kernels.sm100.d192_f16_multi_q import contract as multi
from cudnn.sdpa.fwd.kernels.sm100.d128_f16_single_q import contract as single
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0
SPECS = {spec.name: spec for spec in engines.ENGINE_SPECS}
OFFERED = {multi.NAME: 20531, single.NAME: 20532, multi.LEGACY_ROW: 20511}


def facts(**overrides):
    values = dict(
        b=2,
        h_q=8,
        h_kv=8,
        s_q=257,
        s_kv=4096,
        d_qk=192,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        dtype_o=cudnn.data_type.BFLOAT16,
        uniform_dtype=True,
        device_cc=(10, 0),
        device_sm_count=148,
    )
    values.update(overrides)
    return SdpaGraphFacts(**values)


@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16])
@pytest.mark.parametrize("thd,cga,owner", [(False, 1, multi.NAME), (False, 2, multi.NAME), (True, 1, single.NAME), (True, 2, multi.NAME)])
def test_registered_owner_and_public_record(cc, dtype, thd, cga, owner):
    f = facts(device_cc=cc, dtype=dtype, dtype_o=dtype, thd=thd, padded=thd)
    assert "owned by" in engines.mismatch(SPECS[multi.LEGACY_ROW].capabilities, f)
    k = engines.SdpaFwdKnobs(tile_m=128, tile_n=128, cga=cga, pack_gqa=False, split_kv=1, sched_policy=0)
    selected = heuristics.select_sm100_d192_half_geometry(f, k)
    assert selected.implementation == owner
    assert engines.mismatch(SPECS[owner].capabilities, f, k) is None
    assert engines.SdpaFwdKnobs.from_public(k.to_public()) == k
    assert (selected.cluster_m, selected.cta_group) == (cga, cga)


@pytest.mark.parametrize("owner", [multi.NAME, single.NAME])
@pytest.mark.parametrize(
    "overrides",
    [
        {"device_cc": (10, 7)},
        {"d_qk": 177, "d_v": 112},
        {"is_fp8": True},
        {"is_mxfp8": True},
        {"uniform_dtype": False},
        {"has_bias": True},
        {"has_score_max": True},
    ],
)
def test_other_implementation_domains_are_not_admitted(owner, overrides):
    assert engines.mismatch(SPECS[owner].capabilities, facts(thd=True, padded=True, **overrides)) is not None


@pytest.mark.parametrize("thd,cga,owner", [(False, 1, multi.NAME), (False, 2, multi.NAME), (True, 1, single.NAME), (True, 2, multi.NAME)])
def test_head_envelope_has_an_explicit_owner(thd, cga, owner):
    f = facts(d_qk=176, d_v=112, thd=thd, padded=thd)
    k = engines.SdpaFwdKnobs(cga=cga, tile_m=128, tile_n=128)
    assert "owned by" in engines.mismatch(SPECS[multi.LEGACY_ROW].capabilities, f)
    assert engines.mismatch(SPECS[owner].capabilities, f, k) is None
    assert heuristics.select_sm100_d192_half_geometry(f, k).implementation == owner


@pytest.mark.parametrize("mode", ["A", "FALLBACK"])
@pytest.mark.parametrize("thd", [False, True])
def test_recommended_order_retains_existing_policy(mode, thd):
    f = facts(thd=thd, padded=thd)
    policy = engines.sm100_legacy_policy_spec()
    old = heuristics._knob_sets(policy, f) if mode == "A" else [heuristics._fallback_knobs(policy, f)]
    old = [k for k in old if engines.mismatch(policy.capabilities, f, k) is None]
    plans = heuristics.recommend(mode, f, OFFERED)
    assert plans and [p.knobs for p in plans] == old
    assert [p.engine_id for p in plans] == [OFFERED[heuristics.select_sm100_d192_half_geometry(f, k).implementation] for k in old]


def test_single_q_retains_nonpaged_thd_split():
    f = facts(thd=True, padded=True)
    k = engines.SdpaFwdKnobs(tile_m=128, tile_n=128, cga=1, pack_gqa=False, split_kv=2, sched_policy=1)
    assert engines.mismatch(SPECS[single.NAME].capabilities, f, k) is None


def test_head_envelope_retains_the_previous_packed_split_boundary():
    f = facts(d_qk=176, d_v=112, thd=True, padded=True)
    k = engines.SdpaFwdKnobs(tile_m=128, tile_n=128, cga=1, pack_gqa=False, split_kv=3, sched_policy=0)
    policy = engines.sm100_legacy_policy_spec()
    assert not engines.thd_split_domain(policy.capabilities, f)
    # The legacy row rejects its cluster domain first; the owner already fixes
    # that geometry and reports the unsupported packed split directly.
    assert engines.mismatch(policy.capabilities, f, k) is not None
    assert "split_kv" in engines.mismatch(SPECS[single.NAME].capabilities, f, k)
    assert engines.mismatch(SPECS[multi.NAME].capabilities, f, k) is not None


@pytest.mark.parametrize("owner", [multi.NAME, single.NAME])
def test_explicit_lowering_never_reselects_the_body(monkeypatch, owner):
    def forbidden(*args, **kwargs):
        raise AssertionError("fixed record consulted performance selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    monkeypatch.setattr(heuristics, "select_sm100_d192_half_geometry", forbidden)
    calls = []
    monkeypatch.setattr(engines, "lower_dsl_prefill", lambda spec, f, k, *, selected_kernel: calls.append((k, selected_kernel)))
    spec = next(spec for spec in engines._sm100_d192_half_specs() if spec.name == owner)
    spec.lower(spec, facts(thd=owner == single.NAME, padded=owner == single.NAME), None)
    assert len(calls) == 1
    assert calls[0][1].implementation == owner
    assert calls[0][1].lowering_row_name == multi.LEGACY_ROW


@pytest.mark.parametrize("folder", ["d192_f16_multi_q", "d128_f16_single_q"])
def test_cold_host_import_does_not_import_registry(folder):
    package_root = str(Path(cudnn.__file__).resolve().parent.parent)
    code = "\n".join(
        (
            "import sys",
            "sys.meta_path[:] = [f for f in sys.meta_path if 'editable' not in str(f).lower()]",
            f"sys.path.insert(0, {package_root!r})",
            f"import cudnn.sdpa.fwd.kernels.sm100.{folder}.host",
            "assert 'cudnn.sdpa.fwd.engines' not in sys.modules",
            "assert 'cutlass' not in sys.modules",
        )
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("owner", [multi.NAME, single.NAME])
@pytest.mark.parametrize("verdict", [False, True])
def test_explicit_study_precedes_inherited_placement(monkeypatch, owner, verdict):
    from cudnn.sdpa import placement_table

    monkeypatch.setattr(placement_table, "lookup", lambda row, f: verdict if row == owner else not verdict)
    f = facts(thd=owner == single.NAME, padded=owner == single.NAME)
    k = engines.SdpaFwdKnobs(cga=1, pack_gqa=False, split_kv=1)
    assert placement.place(SPECS[owner], f, k) == (placement.LEAD if verdict else placement.TRAIL)


@pytest.mark.parametrize("owner,record", [("d192_f16_multi_q", "CfgD192"), ("d128_f16_single_q", "CfgD128Decode"), ("d128_f16_single_q", "CfgD64Decode")])
def test_configuration_defaults_are_independent(owner, record, monkeypatch):
    module = importlib.import_module(f"cudnn.sdpa.fwd.kernels.sm100.{owner}.config")
    own = getattr(module, record)
    expected = asdict(getattr(legacy, record)())
    assert own.__bases__ == (object,)
    assert asdict(own()) == expected
    monkeypatch.setattr(getattr(legacy, record), "STAGES_KV", 999)
    assert own.STAGES_KV == expected["STAGES_KV"]


@pytest.mark.parametrize("dtype", [0, 1])
def test_multi_q_cannot_become_a_quantized_body(dtype):
    from cudnn.sdpa.fwd.kernels.sm100.d192_f16_multi_q.config import make_cfg_d192

    with pytest.raises(ValueError, match="FP16/BF16"):
        make_cfg_d192(legacy.TemplateParams(dtype_qkv=dtype))


@pytest.mark.parametrize("head_dim", [64, 128, 192])
def test_single_q_cannot_select_quadrant_cooperation(head_dim):
    from cudnn.sdpa.fwd.kernels.sm100.d128_f16_single_q import config

    params = legacy.TemplateParams(
        cta_mma=1,
        d_flavor=64 if head_dim == 64 else 128,
        decode_tile=head_dim == 64,
        single_q_head_dim=192 if head_dim == 192 else 128,
        thd_varlen=head_dim == 192,
        seq_kv_lens_present=head_dim == 192,
    )
    factory = config.make_cfg_d64_decode if head_dim == 64 else config.make_cfg_d128_decode
    cfg, _ = factory(params)
    assert cfg.TILES_Q == 1 and cfg.TILE_K == head_dim
    with pytest.raises(ValueError, match="quadrant"):
        factory(replace(params, decode_quad=True))


@pytest.mark.parametrize(
    "owner,function,params",
    [
        ("d192_f16_multi_q", "make_cfg_d192", legacy.TemplateParams()),
        ("d128_f16_single_q", "make_cfg_d128_decode", legacy.TemplateParams(cta_mma=1, single_q_head_dim=192, thd_varlen=True, seq_kv_lens_present=True)),
        ("d128_f16_single_q", "make_cfg_d64_decode", legacy.TemplateParams(cta_mma=1, d_flavor=64, decode_tile=True)),
    ],
)
def test_factories_do_not_call_legacy_implementations(owner, function, params, monkeypatch):
    module = importlib.import_module(f"cudnn.sdpa.fwd.kernels.sm100.{owner}.config")

    def forbidden(*args, **kwargs):
        raise AssertionError("isolated body consulted a legacy configuration factory")

    for name in ("make_cfg_d192", "make_cfg_d128_decode", "make_cfg_d64_decode", "_validate_cfg_d192", "_validate_cfg_d128_decode", "_validate_cfg_d64_decode"):
        monkeypatch.setattr(legacy, name, forbidden)
    cfg, _ = getattr(module, function)(params)
    assert cfg.DTYPE_QKV == params.dtype_qkv


@pytest.mark.parametrize("owner,cga,query_tiles", [("d192_f16_multi_q", 1, 2), ("d192_f16_multi_q", 2, 2), ("d128_f16_single_q", 1, 1)])
def test_fixed_host_owns_its_geometry(owner, cga, query_tiles):
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    module = importlib.import_module(f"cudnn.sdpa.fwd.kernels.sm100.{owner}.host")
    config = importlib.import_module(f"cudnn.sdpa.fwd.kernels.sm100.{owner}.config")
    spec, graph_facts = object(), facts(thd=query_tiles == 1)
    seen = []

    def lower_impl(actual_spec, actual_facts, knobs, *, selected_kernel):
        assert actual_spec is spec and actual_facts is graph_facts
        params = legacy.TemplateParams(dtype_qkv=2, cta_mma=cga, thd_varlen=query_tiles == 1, seq_kv_lens_present=query_tiles == 1)
        params = selected_kernel.configure_params(params)
        factory = config.make_cfg_d192 if query_tiles == 2 else config.make_cfg_d128_decode
        cfg, _ = factory(params)
        assert cfg.TILES_Q == query_tiles and cfg.TILE_K == 192
        assert cfg.CGA_M == selected_kernel.cluster_m == knobs.cga == cga
        assert not params.decode_quad and not params.early_q
        seen.append(selected_kernel)
        return selected_kernel

    result = module.lower(spec, graph_facts, SdpaFwdKnobs(cga=cga), lower_impl=lower_impl, knob_type=SdpaFwdKnobs)
    assert seen == [result]


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("cga", [1, 2])
def test_thd_body_boundaries_are_explicit(paged, cga):
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
    from cudnn.sdpa.fwd.kernels.sm100.d192_f16_multi_q import contract as multi
    from cudnn.sdpa.fwd.kernels.sm100.d128_f16_single_q import contract as single

    facts = SdpaGraphFacts(
        b=2,
        h_q=8,
        h_kv=8,
        s_q=257,
        s_kv=1024,
        d_qk=192,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        dtype_o=cudnn.data_type.BFLOAT16,
        device_cc=(10, 0),
        thd=True,
        has_paged_kv=paged,
    )
    knobs = SdpaFwdKnobs(cga=cga, pack_gqa=False)
    assert (single.mismatch(facts, knobs) is None) == (not paged and cga == 1)
    assert (multi.mismatch(facts, knobs) is None) == (paged or cga == 2)
