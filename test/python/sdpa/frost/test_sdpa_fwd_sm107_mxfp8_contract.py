# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for the SM107 MXFP8 owners: one block-scaled body per native head shape."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import cudnn
from cudnn.sdpa.fwd import engines, heuristics, placement
from cudnn.sdpa.fwd.config_sm107 import SM107_EPILOGUE_GATE_SHAPES, SM107_MXFP8_SPLIT_SHAPES
from cudnn.sdpa.fwd.config_sm100 import TemplateParams
from cudnn.sdpa.fwd.kernels.sm107 import mxfp8_contract as mx
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0
SPECS = {s.name: s for s in engines.ENGINE_SPECS}
OWNERS = {
    (128, 128): "sdpa_fwd_sm107_d128_mxfp8_multi_q",
    (192, 128): "sdpa_fwd_sm107_d192_mxfp8_multi_q",
    (256, 256): "sdpa_fwd_sm107_d256_mxfp8_single_q",
    (512, 512): "sdpa_fwd_sm107_d512_mxfp8_role_split",
}
OFFERED = {mx.LEGACY_ROW: 20516, **{name: 20565 + i for i, name in enumerate(OWNERS.values())}}
FEATURES = [{}, {"causal": True}, {"causal": True, "bottom_right": True}, {"causal": True, "window_left": 255}, {"has_sink": True}]


def facts(shape=(128, 128), **overrides):
    values = dict(
        b=2,
        h_q=8,
        h_kv=2,
        s_q=4096,
        s_kv=4096,
        d_qk=shape[0],
        d_v=shape[1],
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=cudnn.data_type.BFLOAT16,
        is_mxfp8=True,
        device_cc=(10, 7),
        device_sm_count=216,
    )
    values.update(overrides)
    return SdpaGraphFacts(**values)


def test_registry_slots_follow_the_row_they_split():
    from cudnn.engines.manifest import MANIFEST

    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    assert {name: family.engine_id + family.slots[name].slot for name in OFFERED} == OFFERED
    assert not any(family.slots[name].opt_in for name in OFFERED)
    assert mx.IMPLEMENTATIONS == frozenset(OWNERS.values()) and mx.SHAPES == frozenset(OWNERS)


@pytest.mark.parametrize("overrides", FEATURES)
@pytest.mark.parametrize("shape", sorted(OWNERS))
def test_the_head_shape_names_one_owner(shape, overrides):
    f = facts(shape, **overrides)
    owner = OWNERS[shape]
    assert mx.owner(f) == owner
    assert "owned by the explicit implementation engines" in engines.mismatch(SPECS[mx.LEGACY_ROW].capabilities, f)
    assert engines.mismatch(SPECS[owner].capabilities, f) is None
    assert all(engines.mismatch(SPECS[other].capabilities, f) is not None for other in mx.IMPLEMENTATIONS - {owner})


@pytest.mark.parametrize(
    "overrides",
    [{"device_cc": (10, 0)}, {"device_cc": (12, 0)}, {"is_mxfp8": False, "is_fp8": True}, {"d_qk": 64, "d_v": 64}, {"d_qk": 200, "d_v": 200}],
)
def test_other_lines_and_shapes_do_not_enter_the_owners(overrides):
    f = facts(**overrides)
    assert mx.owner(f) is None
    assert all(engines.mismatch(SPECS[name].capabilities, f) is not None for name in mx.IMPLEMENTATIONS)


@pytest.mark.parametrize("shape", sorted(OWNERS))
def test_owners_keep_the_row_claims_of_their_shape(shape):
    """Each owner claims the row's capabilities narrowed to its shape: split on D128 only, the epilogue gate on D256 only."""
    caps = SPECS[OWNERS[shape]].capabilities
    row = engines.sm100_legacy_policy_spec(mx.LEGACY_ROW).capabilities
    assert caps.d_shapes == frozenset({shape}) and caps.d_pad_multiple == 0 and caps.d_envelope_floors == ()
    assert (caps.dtypes, caps.out_dtypes, caps.thd_d_shapes, caps.paged_d_shapes) == (row.dtypes, row.out_dtypes, row.thd_d_shapes, row.paged_d_shapes)
    split = engines.mismatch(caps, facts(shape), engines.SdpaFwdKnobs(split_kv=2))
    assert (split is None) == (shape in SM107_MXFP8_SPLIT_SHAPES), split
    assert caps.epilogue_gate == (shape in SM107_EPILOGUE_GATE_SHAPES)
    assert caps.epilogue_gate_d_shapes is (SM107_EPILOGUE_GATE_SHAPES if caps.epilogue_gate else None)


@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
@pytest.mark.parametrize("overrides", FEATURES[:4] + [{"thd": True, "padded": True}, {"has_paged_kv": True, "padded": True, "page_size": 128}])
@pytest.mark.parametrize("shape", sorted(OWNERS))
def test_candidate_order_and_budget_preserve_the_legacy_policy(kind, overrides, shape):
    f = facts(shape, **overrides)
    policy = heuristics.sm100_legacy_policy_spec(mx.LEGACY_ROW)
    if engines.mismatch(policy.capabilities, f) is not None:
        assert heuristics.recommend(kind, f, OFFERED) == []
        return
    old = heuristics._knob_sets(policy, f) if kind == "A" else [heuristics._fallback_knobs(policy, f)]
    expected = list(dict.fromkeys(mx.select_geometry(f, k).canonical_knobs(k) for k in old if engines.mismatch(policy.capabilities, f, k) is None))
    plans = heuristics.recommend(kind, f, OFFERED)
    assert plans and [p.knobs for p in plans] == expected
    assert {p.engine_id for p in plans} == {OFFERED[OWNERS[shape]]}


@pytest.mark.parametrize("overrides", [{}, {"causal": True}])
@pytest.mark.parametrize("shape", sorted(OWNERS))
def test_placement_inherits_the_legacy_row(shape, overrides):
    f = facts(shape, **overrides)
    policy = heuristics.sm100_legacy_policy_spec(mx.LEGACY_ROW)
    knobs = engines.SdpaFwdKnobs()
    assert placement.place(SPECS[OWNERS[shape]], f, knobs) == placement.place(policy, f, knobs)


@pytest.mark.parametrize("shape", sorted(OWNERS))
def test_explicit_lowering_does_not_consult_candidate_selection(monkeypatch, shape):
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit record entered performance selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    calls = []
    monkeypatch.setattr(engines, "lower_dsl_prefill", lambda spec, f, k, *, selected_kernel: calls.append((k, selected_kernel)))
    name = OWNERS[shape]
    spec = next(s for s in engines._sm107_mxfp8_owner_specs() if s.name == name)
    spec.lower(spec, facts(shape), engines.SdpaFwdKnobs())
    ((knobs, geometry),) = calls
    assert (geometry.implementation, geometry.lowering_row_name, geometry.kernel_file) == (
        name,
        mx.LEGACY_ROW,
        f"sm107/{name.removeprefix('sdpa_fwd_sm107_')}/kernel.py",
    )
    # A partial record leaves every knob unset: the adapter derives them as on the legacy row; D256 keeps its CTA width a knob.
    assert knobs == engines.SdpaFwdKnobs()
    assert geometry.cta_group == (None if shape == (256, 256) else 2)


def test_compiled_identity_check_names_the_disagreement():
    params = SimpleNamespace(cta_mma=2, sched_policy=0, split_kv=1)
    d128 = mx.explicit_geometry(OWNERS[(128, 128)])
    assert d128.compiled_mismatch("prefill_d128_mxfp8", params, engines.SdpaFwdKnobs())[1] is None
    assert d128.compiled_mismatch("prefill_d256_mxfp8", params, engines.SdpaFwdKnobs())[1] is not None
    assert d128.compiled_mismatch("prefill_d128_mxfp8", SimpleNamespace(cta_mma=1, sched_policy=0, split_kv=1), engines.SdpaFwdKnobs())[1] is not None
    assert d128.compiled_mismatch("prefill_d128_mxfp8", params, engines.SdpaFwdKnobs(split_kv=2))[1] is not None
    d256 = mx.explicit_geometry(OWNERS[(256, 256)], engines.SdpaFwdKnobs(cga=1))
    assert d256.compiled_mismatch("prefill_d256_mxfp8", SimpleNamespace(cta_mma=1, sched_policy=0, split_kv=1), engines.SdpaFwdKnobs(cga=1))[1] is None
    assert d256.compiled_mismatch("prefill_d256_mxfp8", params, engines.SdpaFwdKnobs(cga=1))[1] is not None


@pytest.mark.parametrize("shape", sorted(OWNERS))
def test_standalone_loader_routes_by_head_shape(shape):
    from cudnn.sdpa.fwd.api_dsl import _SM107_MXFP8_KERNEL_FILES

    assert _SM107_MXFP8_KERNEL_FILES[shape] == mx.explicit_geometry(OWNERS[shape]).kernel_file


@pytest.mark.parametrize("module", ["d128_mxfp8_multi_q", "d192_mxfp8_multi_q", "d256_mxfp8_single_q", "d512_mxfp8_role_split"])
def test_cold_host_import_does_not_import_registry_or_runtimes(module):
    source_root = str(Path(cudnn.__file__).resolve().parents[1])
    code = "\n".join(
        (
            "import json, sys",
            "sys.meta_path[:] = [f for f in sys.meta_path if 'editable' not in str(f).lower()]",
            f"sys.path.insert(0, {source_root!r})",
            f"import cudnn.sdpa.fwd.kernels.sm107.{module}.host",
            "print(json.dumps(sorted(sys.modules)))",
        )
    )
    loaded = set(json.loads(subprocess.check_output([sys.executable, "-c", code], text=True)))
    assert not {"cudnn.sdpa.fwd.engines", "cudnn.sdpa.fwd.api_dsl", "cutlass", "torch"} & loaded
