# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-body ownership for SM100 small and mid-width block-scaled attention."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

import cudnn
from cudnn.sdpa.fwd import config_sm100 as legacy
from cudnn.sdpa.fwd import engines, heuristics, placement
from cudnn.sdpa.fwd.config_common import TemplateParams
from cudnn.sdpa.fwd.kernels.sm100.d128_mxfp8_multi_q import config as small_config, contract as small
from cudnn.sdpa.fwd.kernels.sm100.d192_mxfp8_multi_q import config as mid_config, contract as mid
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0
SPECS = {s.name: s for s in engines.ENGINE_SPECS}
OFFERED = {small.LEGACY_ROW: 20512, small.NAME: 20550, mid.NAME: 20551}


def facts(dim=128, **overrides):
    values = dict(
        b=2,
        h_q=8,
        h_kv=2,
        s_q=129,
        s_kv=4096,
        d_qk=dim,
        d_v=min(dim, 128),
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.BFLOAT16,
        is_mxfp8=True,
        uniform_dtype=True,
        device_cc=(10, 0),
        device_sm_count=148,
    )
    values.update(overrides)
    return SdpaGraphFacts(**values)


def geometry(dim, cga):
    return small.Geometry(d_flavor=dim, cga=cga) if dim <= 128 else mid.Geometry(cga=cga)


@pytest.mark.parametrize("dim,cga", [(64, 1), (128, 1), (128, 2), (192, 1), (192, 2)])
@pytest.mark.parametrize("dtype", [0, 1])
def test_configuration_is_independent_of_per_tensor_and_legacy(dim, cga, dtype, monkeypatch):
    from cudnn.sdpa.fwd.kernels.sm100.d128_fp8_multi_q import config as small_fp8
    from cudnn.sdpa.fwd.kernels.sm100.d192_fp8_multi_q import config as mid_fp8

    owner = small_config if dim <= 128 else mid_config
    sibling = small_fp8 if dim <= 128 else mid_fp8
    record = getattr(owner, f"CfgD{dim}")
    assert record.__bases__ == (object,)
    assert record not in (getattr(legacy, f"CfgD{dim}"), getattr(sibling, f"CfgD{dim}"))

    def forbidden(*args, **kwargs):
        raise AssertionError("MXFP8 owner consulted a sibling configuration implementation")

    for module in (legacy, sibling):
        monkeypatch.setattr(module, f"make_cfg_d{dim}", forbidden)
        monkeypatch.setattr(module, f"_validate_cfg_d{dim}", forbidden)
        monkeypatch.setattr(getattr(module, f"CfgD{dim}"), "STAGES_KV", 999)
    kwargs = {"cga1_kv_stages": 3} if dim == 128 else {}
    # D192's dedicated factory historically receives generic d_flavor=128.
    p = TemplateParams(dtype_qkv=dtype, dtype_o=2, cta_mma=cga, d_flavor=dim if dim <= 128 else 128)
    cfg, _ = getattr(owner, f"make_cfg_d{dim}")(p, **kwargs)
    assert type(cfg) is record
    assert (cfg.TILE_K, cfg.TILE_O, cfg.CGA_M, cfg.CTA_MMA) == (dim, min(dim, 128), cga, cga)
    assert cfg.STAGES_KV != 999
    if dim == 128 and cga == 1:
        assert cfg.STAGES_KV == 3


@pytest.mark.parametrize("dim,cga", [(64, 1), (128, 1), (128, 2), (192, 1), (192, 2)])
@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
@pytest.mark.parametrize("dtype", [cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2])
def test_record_names_exact_owner_before_lowering(dim, cga, cc, dtype):
    f = facts(dim, device_cc=cc, dtype=dtype)
    g = geometry(dim, cga)
    knobs = g.canonical_knobs(engines.SdpaFwdKnobs())
    assert "owned by" in engines.mismatch(SPECS[small.LEGACY_ROW].capabilities, f)
    assert engines.mismatch(SPECS[g.implementation].capabilities, f, knobs) is None
    assert engines.SdpaFwdKnobs.from_public(knobs.to_public()) == knobs
    assert (knobs.tile_m, knobs.cga, g.cta_group) == (128, cga, cga)
    assert engines.mismatch(SPECS[g.implementation].capabilities, f, replace(knobs, tile_m=64)) is not None
    assert engines.mismatch(SPECS[g.implementation].capabilities, f, replace(knobs, cga=4)) is not None


@pytest.mark.parametrize("dim", [64, 128, 192])
@pytest.mark.parametrize(
    "changes", [{"device_cc": (10, 7)}, {"device_cc": (12, 0)}, {"is_fp8": True}, {"uniform_dtype": False}, {"d_v": 96}, {"is_backward": True}]
)
def test_adjacent_domains_do_not_enter_owner(dim, changes):
    owner = small if dim <= 128 else mid
    assert engines.mismatch(SPECS[owner.NAME].capabilities, facts(dim, **changes)) is not None


@pytest.mark.parametrize("dim,cga", [(64, 1), (128, 1), (128, 2), (192, 1), (192, 2)])
def test_explicit_lowering_does_not_reselect_body(dim, cga, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit owner called candidate generation")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    calls = []
    monkeypatch.setattr(engines, "lower_dsl_prefill", lambda spec, f, k, *, selected_kernel: calls.append((k, selected_kernel)))
    g = geometry(dim, cga)
    spec = next(s for s in engines._sm100_small_mxfp8_specs() if s.name == g.implementation)
    for rows in (1, 4097):
        spec.lower(spec, facts(dim, s_q=rows), engines.SdpaFwdKnobs(cga=cga))
    assert len(calls) == 2
    assert all(x[1] == g for x in calls)
    assert all(x[0].cga == cga for x in calls)


@pytest.mark.parametrize("dim", [64, 128, 192])
@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
def test_candidate_order_and_budget_follow_original_policy(dim, kind):
    f = facts(dim)
    policy = engines.sm100_legacy_policy_spec(small.LEGACY_ROW)
    old = heuristics._knob_sets(policy, f) if kind == "A" else [heuristics._fallback_knobs(policy, f)]
    expected = [geometry(dim, k.cga).canonical_knobs(k) for k in old if engines.mismatch(policy.capabilities, f, k) is None]
    plans = heuristics.recommend(kind, f, OFFERED)
    assert plans
    assert all(p.engine_id == (20550 if dim <= 128 else 20551) for p in plans)
    assert [p.knobs for p in plans] == list(dict.fromkeys(expected))


@pytest.mark.parametrize("flag", [None, "0", "1"])
def test_discovery_offers_both_owners_and_keeps_the_own_block_unlock(monkeypatch, flag):
    from cudnn.engines import MANIFEST

    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    if flag is None:
        monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    else:
        monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", flag)
    # Default since 2026-11-16, behind the backend (placement.TRAIL_ONLY_ROWS).
    assert family.offered_ids()[small.NAME] == 20550
    assert family.offered_ids()[mid.NAME] == 20551
    assert {small.NAME, mid.NAME} <= placement.TRAIL_ONLY_ROWS
    assert family.slots[small.NAME].unlocked_by == ("causal_exclude_own_block",)
    assert not family.slots[mid.NAME].unlocked_by


@pytest.mark.parametrize("owner,dim", [(small, 128), (mid, 192)])
@pytest.mark.parametrize("verdict", [False, True])
def test_current_owner_placement_study_takes_precedence(owner, dim, verdict, monkeypatch):
    from cudnn.sdpa import placement_table

    monkeypatch.setattr(placement_table, "lookup", lambda row, f: verdict if row == owner.NAME else not verdict)
    assert placement.place(SPECS[owner.NAME], facts(dim), geometry(dim, 2).canonical_knobs(engines.SdpaFwdKnobs())) == (
        placement.LEAD if verdict else placement.TRAIL
    )


@pytest.mark.parametrize("implementation", ["d128_mxfp8_multi_q", "d192_mxfp8_multi_q"])
def test_cold_host_and_configuration_import_without_registry_or_siblings(implementation):
    package_root = str(Path(cudnn.__file__).resolve().parent.parent)
    code = "\n".join(
        (
            "import sys",
            "sys.meta_path[:] = [f for f in sys.meta_path if 'editable' not in str(f).lower()]",
            f"sys.path.insert(0, {package_root!r})",
            f"import cudnn.sdpa.fwd.kernels.sm100.{implementation}.host",
            f"import cudnn.sdpa.fwd.kernels.sm100.{implementation}.config",
            "assert 'cudnn.sdpa.fwd.engines' not in sys.modules",
            "assert 'cudnn.sdpa.fwd.config_sm100' not in sys.modules",
            "assert not any(n.endswith('.kernel') for n in sys.modules)",
        )
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_d128_thd_one_cta_decline_is_preserved():
    f = facts(128, thd=True, padded=True, max_total_seq_len_q=258)
    knobs = small.Geometry(cga=1).canonical_knobs(engines.SdpaFwdKnobs())
    old = engines.sm100_legacy_policy_spec(small.LEGACY_ROW)
    before = engines.mismatch(old.capabilities, f, knobs)
    after = engines.mismatch(SPECS[small.NAME].capabilities, f, knobs)
    assert before is not None and "cga=1" in before
    assert after == before
    assert engines.mismatch(SPECS[small.NAME].capabilities, f, replace(knobs, cga=2)) is None
