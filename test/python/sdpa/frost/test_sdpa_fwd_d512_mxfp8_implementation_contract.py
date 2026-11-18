# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for the exact MXFP8 D512 owners: output-slice recompute and retained P."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import cudnn
from cudnn.sdpa.fwd import engines, heuristics, placement
from cudnn.sdpa.fwd.config_sm100 import TemplateParams, d512_mxfp8_retains_p, make_cfg_d512_mxfp8
from cudnn.sdpa.fwd.kernels.sm100 import d512_mxfp8_contract as mx
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0
SPECS = {s.name: s for s in engines.ENGINE_SPECS}
OFFERED = {mx.LEGACY_ROW: 20512, mx.SLICE_RECOMPUTE: 20552, mx.RETAINED_P: 20553}
SWA = dict(causal=True, window_left=127)


def facts(**overrides):
    values = dict(
        b=1,
        h_q=8,
        h_kv=2,
        s_q=2048,
        s_kv=2048,
        d_qk=512,
        d_v=512,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.BFLOAT16,
        is_mxfp8=True,
        device_cc=(10, 0),
        device_sm_count=148,
    )
    values.update(overrides)
    return SdpaGraphFacts(**values)


def test_registry_slots_are_opt_in_like_the_row_they_split():
    from cudnn.engines.manifest import MANIFEST

    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    assert {name: family.engine_id + family.slots[name].slot for name in OFFERED} == OFFERED
    assert all(family.slots[name].opt_in for name in OFFERED)


@pytest.mark.parametrize(
    "overrides,split,owner",
    [
        ({}, 1, mx.SLICE_RECOMPUTE),
        ({"causal": True}, 1, mx.SLICE_RECOMPUTE),
        (SWA, 1, mx.RETAINED_P),
        (SWA, 2, mx.SLICE_RECOMPUTE),
        (dict(SWA, thd=True, padded=True), 1, mx.RETAINED_P),
        (dict(SWA, padded=True), 1, mx.SLICE_RECOMPUTE),
        (dict(SWA, bottom_right=True), 1, mx.SLICE_RECOMPUTE),
        (dict(SWA, window_left=126), 1, mx.SLICE_RECOMPUTE),
        (dict(SWA, window_left=128), 1, mx.SLICE_RECOMPUTE),
        ({"window_left": 127}, 1, mx.SLICE_RECOMPUTE),
        ({"right_band_widening": True, "window_left": 127}, 1, mx.SLICE_RECOMPUTE),
    ],
)
@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
def test_the_band_and_split_name_one_owner(overrides, split, owner, cc):
    f = facts(device_cc=cc, **overrides)
    knobs = mx.explicit_geometry(owner).canonical_knobs(engines.SdpaFwdKnobs(split_kv=split))
    assert "owned by the explicit implementation engines" in engines.mismatch(SPECS[mx.LEGACY_ROW].capabilities, f)
    assert engines.mismatch(SPECS[owner].capabilities, f, knobs) is None
    (other,) = mx.IMPLEMENTATIONS - {owner}
    assert engines.mismatch(SPECS[other].capabilities, f, knobs) is not None


@pytest.mark.parametrize("overrides,admitted", [({}, {mx.SLICE_RECOMPUTE}), (SWA, {mx.SLICE_RECOMPUTE, mx.RETAINED_P})])
def test_knobless_admission_keeps_every_body_a_record_can_name(overrides, admitted):
    # SPLIT_KV names the body on the retained band, so an eligibility check without knobs must not decide it.
    f = facts(**overrides)
    assert {name for name in mx.IMPLEMENTATIONS if engines.mismatch(SPECS[name].capabilities, f) is None} == admitted


@pytest.mark.parametrize(
    "params,split,thd,padded",
    [
        (dict(window_left=127, window_right=0), 1, False, False),
        (dict(window_left=127, window_right=0), 2, False, False),
        (dict(window_left=127, window_right=0, seq_kv_lens_present=True), 1, False, True),
        (dict(window_left=127, window_right=0, thd_varlen=True, seq_kv_lens_present=True), 1, True, True),
        (dict(window_left=127, window_right=0, bottom_right=True), 1, False, False),
        (dict(window_left=126, window_right=0), 1, False, False),
        (dict(window_left=127), 1, False, False),
        (dict(window_right=0), 1, False, False),
        (dict(), 1, False, False),
    ],
)
def test_the_graph_rule_matches_the_kernel_rule(params, split, thd, padded):
    # The graph-side owner rule restates config_sm100.d512_mxfp8_retains_p, which each kernel asserts at load.
    record = TemplateParams(dtype_qkv=0, dtype_o=2, cta_mma=1, split_kv=split, **params)
    cfg, _ = make_cfg_d512_mxfp8(record)
    f = facts(
        causal=params.get("window_right") == 0,
        bottom_right=params.get("bottom_right", False),
        window_left=params.get("window_left"),
        thd=thd,
        padded=padded,
    )
    assert mx.retains_p(f, split) == d512_mxfp8_retains_p(cfg, split)


def test_other_heads_keep_their_explicit_owners():
    for dims, owner in (((128, 128), "sdpa_fwd_sm100_d128_mxfp8_multi_q"), ((256, 256), "sdpa_fwd_sm100_d256_mxfp8_single_q")):
        f = facts(d_qk=dims[0], d_v=dims[1])
        assert "owned by" in engines.mismatch(SPECS[mx.LEGACY_ROW].capabilities, f)
        assert engines.mismatch(SPECS[owner].capabilities, f) is None
        assert all(engines.mismatch(SPECS[name].capabilities, f) is not None for name in mx.IMPLEMENTATIONS)


@pytest.mark.parametrize("overrides", [{"device_cc": (10, 7)}, {"device_cc": (12, 0)}, {"is_mxfp8": False, "is_fp8": True}])
def test_other_lines_do_not_enter_the_owners(overrides):
    f = facts(**overrides)
    assert all(engines.mismatch(SPECS[name].capabilities, f) is not None for name in mx.IMPLEMENTATIONS)


@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
@pytest.mark.parametrize("overrides", [{}, {"causal": True}, SWA, dict(SWA, padded=True), dict(SWA, thd=True, padded=True)])
def test_candidate_order_and_budget_preserve_the_legacy_policy(kind, overrides):
    f = facts(**overrides)
    policy = heuristics.sm100_legacy_policy_spec(mx.LEGACY_ROW)
    old = heuristics._knob_sets(policy, f) if kind == "A" else [heuristics._fallback_knobs(policy, f)]
    expected = list(dict.fromkeys(mx.select_geometry(f, k).canonical_knobs(k) for k in old if engines.mismatch(policy.capabilities, f, k) is None))
    plans = heuristics.recommend(kind, f, OFFERED)
    assert plans and [p.knobs for p in plans] == expected
    ids = {v: k for k, v in OFFERED.items()}
    assert all(ids[p.engine_id] == mx.owner(f, p.knobs) for p in plans)


@pytest.mark.parametrize("overrides", [{}, SWA])
def test_placement_inherits_the_legacy_row(overrides):
    f = facts(**overrides)
    policy = heuristics.sm100_legacy_policy_spec(mx.LEGACY_ROW)
    knobs = mx.select_geometry(f, engines.SdpaFwdKnobs()).canonical_knobs(engines.SdpaFwdKnobs())
    name = mx.owner(f, knobs)
    assert placement.place(SPECS[name], f, knobs) == placement.place(policy, f, knobs)


@pytest.mark.parametrize("name,builder", [(mx.SLICE_RECOMPUTE, "_sm100_d512_mxfp8_slice_recompute_spec"), (mx.RETAINED_P, "_sm100_d512_mxfp8_retained_p_spec")])
def test_explicit_lowering_does_not_consult_candidate_selection(monkeypatch, name, builder):
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit record entered performance selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    calls = []
    monkeypatch.setattr(engines, "lower_dsl_prefill", lambda spec, f, k, *, selected_kernel: calls.append((k, selected_kernel)))
    spec = getattr(engines, builder)()
    spec.lower(spec, facts(**(SWA if name == mx.RETAINED_P else {})), engines.SdpaFwdKnobs())
    ((knobs, geometry),) = calls
    assert (geometry.implementation, geometry.lowering_row_name, geometry.kernel_file) == (
        name,
        mx.LEGACY_ROW,
        f"sm100/{name.removeprefix('sdpa_fwd_sm100_')}/kernel.py",
    )
    # A partial record keeps the scheduler unset: the adapter derives it for D512 MXFP8 as on the legacy row.
    assert (knobs.tile_m, knobs.cga, knobs.split_kv, knobs.sched_policy) == (128, 1, 1, None)


def test_compiled_identity_check_names_the_disagreement():
    geometry = mx.explicit_geometry(mx.RETAINED_P)
    params = SimpleNamespace(cta_mma=1, sched_policy=1, split_kv=1)
    assert geometry.compiled_mismatch("prefill_d512_mxfp8_retained_p", params, engines.SdpaFwdKnobs())[1] is None
    assert geometry.compiled_mismatch("prefill_d512_mxfp8", params, engines.SdpaFwdKnobs())[1] is not None
    assert geometry.compiled_mismatch("prefill_d512_mxfp8_retained_p", params, engines.SdpaFwdKnobs(sched_policy=0))[1] is not None
    split = SimpleNamespace(cta_mma=1, sched_policy=1, split_kv=2)
    assert geometry.compiled_mismatch("prefill_d512_mxfp8_retained_p", split, engines.SdpaFwdKnobs(split_kv=1))[1] is not None


@pytest.mark.parametrize(
    "params,file", [(dict(), "d512_mxfp8_slice_recompute/kernel.py"), (dict(window_left=127, window_right=0), "d512_mxfp8_retained_p/kernel.py")]
)
def test_standalone_loader_routes_by_the_kernel_rule(params, file):
    from cudnn.sdpa.fwd.api_dsl import _load_sm100_kernel_module

    mod = _load_sm100_kernel_module((512, 512), TemplateParams(dtype_qkv=0, dtype_o=2, cta_mma=1, **params), fp8=True, pertensor=False, rubin=False)
    assert mod.__file__.endswith(file)


@pytest.mark.parametrize("module", ["d512_mxfp8_slice_recompute", "d512_mxfp8_retained_p"])
def test_cold_host_import_does_not_import_registry_or_runtimes(module):
    source_root = str(Path(cudnn.__file__).resolve().parents[1])
    code = "\n".join(
        (
            "import json, sys",
            "sys.meta_path[:] = [f for f in sys.meta_path if 'editable' not in str(f).lower()]",
            f"sys.path.insert(0, {source_root!r})",
            f"import cudnn.sdpa.fwd.kernels.sm100.{module}.host",
            "print(json.dumps(sorted(sys.modules)))",
        )
    )
    loaded = set(json.loads(subprocess.check_output([sys.executable, "-c", code], text=True)))
    assert not {"cudnn.sdpa.fwd.engines", "cudnn.sdpa.fwd.api_dsl", "cutlass", "torch"} & loaded
