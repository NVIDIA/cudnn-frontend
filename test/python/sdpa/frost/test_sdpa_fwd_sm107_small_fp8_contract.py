# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed native and shared-decode records for SM107 small-head per-tensor FP8."""

from dataclasses import replace
import pytest
import cudnn
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts
from cudnn.sdpa.fwd import engines, heuristics
from cudnn.sdpa.fwd.kernels.sm107.d128_fp8_multi_q import contract as native
from cudnn.sdpa.fwd.kernels.sm107.d128_fp8_swap_ab import contract as swap
from cudnn.sdpa.fwd.kernels.sm100.d128_fp8_swap_ab import contract as shared_swap

pytestmark = pytest.mark.L0
SPECS = {s.name: s for s in engines.ENGINE_SPECS}
OFFERED = {native.LEGACY_ROW: 20514, native.NAME: 20563, swap.NAME: 20564}


def facts(**changes):
    values = dict(
        b=2,
        h_q=4,
        h_kv=2,
        s_q=128,
        s_kv=1024,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.BFLOAT16,
        is_fp8=True,
        uniform_dtype=True,
        device_cc=(10, 7),
        device_sm_count=208,
    )
    values.update(changes)
    return SdpaGraphFacts(**values)


@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("cga", [1, 2])
@pytest.mark.parametrize("dtype", [cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2])
def test_native_exact_identity_and_cluster(width, cga, dtype):
    f = facts(d_qk=width, d_v=width, dtype=dtype)
    assert "owned by" in engines.mismatch(SPECS[native.LEGACY_ROW].capabilities, f)
    geometry = native.Geometry(d_flavor=width, cga=cga)
    knobs = geometry.canonical_knobs(engines.SdpaFwdKnobs())
    assert engines.mismatch(SPECS[native.NAME].capabilities, f, knobs) is None
    assert (geometry.cluster_m, geometry.cta_group, knobs.tile_m) == (cga, cga, 128)
    assert engines.SdpaFwdKnobs.from_public(knobs.to_public()) == knobs


@pytest.mark.parametrize("width", [64, 128])
@pytest.mark.parametrize("mode", ["A", "FALLBACK"])
def test_native_candidates_keep_legacy_order_and_resolve_d64_default(width, mode):
    f = facts(d_qk=width, d_v=width, h_q=2, h_kv=2, s_q=1)
    policy = engines.sm100_legacy_policy_spec(native.LEGACY_ROW)
    old = heuristics._knob_sets(policy, f) if mode == "A" else [heuristics._fallback_knobs(policy, f)]
    expected = []
    for k in old:
        if engines.mismatch(policy.capabilities, f, k) is None:
            cga = k.cga if k.cga is not None else 1 if width == 64 else 2
            expected.append(native.Geometry(d_flavor=width, cga=cga).canonical_knobs(k))
    plans = heuristics.recommend(mode, f, OFFERED)
    assert plans and {p.engine_id for p in plans} == {20563}
    assert [p.knobs for p in plans] == list(dict.fromkeys(expected))
    if width == 64:
        assert plans[0].knobs.cga == 1


@pytest.mark.parametrize("changes", [{"device_cc": (10, 0)}, {"d_qk": 96, "d_v": 96}, {"is_mxfp8": True}, {"has_paged_kv": True, "page_size": 16}])
def test_native_owner_declines_other_domains(changes):
    assert engines.mismatch(SPECS[native.NAME].capabilities, facts(**changes)) is not None


def test_thd_is_d128_two_cta_only():
    caps = SPECS[native.NAME].capabilities
    f = facts(thd=True, padded=True)
    assert engines.mismatch(caps, f, engines.SdpaFwdKnobs(cga=2)) is None
    assert engines.mismatch(caps, f, engines.SdpaFwdKnobs(cga=1)) is not None
    assert engines.mismatch(caps, replace(f, d_qk=64, d_v=64), engines.SdpaFwdKnobs(cga=2)) is not None


@pytest.mark.parametrize("pack,group,sq", [(False, 3, 8), (True, 2, 4), (True, 4, 2), (True, 8, 1)])
def test_swap_record_reuses_f20_with_its_own_rubin_domain(pack, group, sq):
    f = facts(h_q=2 * group, s_q=sq)
    g = swap.Geometry()
    assert g.kernel_file == shared_swap.KERNEL_FILE
    knobs = g.canonical_knobs(engines.SdpaFwdKnobs(pack_gqa=pack, split_kv=4))
    assert (g.query_tile, g.cluster_m, g.cta_group) == (16, 1, 1)
    assert engines.mismatch(SPECS[swap.NAME].capabilities, f, knobs) is None
    assert engines.effective_cgas(SPECS[swap.NAME].capabilities, f) == frozenset({1})
    assert engines.SdpaFwdKnobs.from_public(knobs.to_public()) == knobs
    assert engines.mismatch(SPECS[native.NAME].capabilities, f, knobs) is not None


@pytest.mark.parametrize(
    "changes",
    [
        {"device_cc": (10, 0)},
        {"d_qk": 64, "d_v": 64},
        {"s_q": 9},
        {"thd": True, "padded": True},
        {"has_paged_kv": True, "page_size": 16},
        {"softmax_precision": cudnn.data_type.HALF},
        {"o_block_scale": 32},
    ],
)
def test_swap_rejects_non_decode_domains(changes):
    assert engines.mismatch(SPECS[swap.NAME].capabilities, facts(s_q=1, **changes) if "s_q" not in changes else facts(**changes)) is not None


@pytest.mark.parametrize("owner", [native, swap])
def test_fixed_lowering_bypasses_candidate_policy(owner, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("fixed record entered candidate policy")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    calls = []
    monkeypatch.setattr(engines, "lower_dsl_prefill", lambda spec, f, k, *, selected_kernel: calls.append((k, selected_kernel)))
    spec = next(s for s in engines._sm107_small_fp8_specs() if s.name == owner.NAME)
    spec.lower(spec, facts(s_q=1), None)
    assert len(calls) == 1 and calls[0][1].implementation == owner.NAME
    assert calls[0][1].lowering_row_name == native.LEGACY_ROW


@pytest.mark.parametrize("width", [64, 128])
def test_native_config_is_independent_of_legacy_factory_and_records(width, monkeypatch):
    from cudnn.sdpa.fwd import config_sm107 as legacy
    from cudnn.sdpa.fwd.kernels.sm107.d128_fp8_multi_q import config

    def forbidden(*args, **kwargs):
        raise AssertionError("native owner consulted legacy implementation state")

    for name in ("make_cfg_d64", "make_cfg_d128", "_make_cfg_d128_family", "CfgD64", "CfgD128"):
        monkeypatch.setattr(legacy, name, forbidden)
    cfg, _ = getattr(config, f"make_cfg_d{width}")(config.TemplateParams(dtype_qkv=0, dtype_o=2, cta_mma=1, d_flavor=width))
    assert type(cfg) is getattr(config, f"CfgD{width}")
    assert (cfg.TILE_K, cfg.TILE_O, cfg.CTA_MMA) == (width, width, 1)
