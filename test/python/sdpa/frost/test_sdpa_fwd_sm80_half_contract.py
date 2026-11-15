# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for the SM80 half owners: K-prefetch, D256 K+V-prefetch and wide-head cooperative."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import cudnn
from cudnn.sdpa.fwd import config_sm80, engines, heuristics, placement
from cudnn.sdpa.fwd.kernels.sm80 import half_flavor_contract as half
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0
SPECS = {s.name: s for s in engines.ENGINE_SPECS}
K_PREFETCH, D256, WIDE = "sdpa_fwd_sm80_f16_k_prefetch", "sdpa_fwd_sm80_d256_kv_prefetch", "sdpa_fwd_sm80_wide_cooperative"
OFFERED = {half.LEGACY_ROW: 20508, K_PREFETCH: 20546, D256: 20547, WIDE: 20548}
IDS = {v: k for k, v in OFFERED.items()}


def facts(**overrides):
    values = dict(
        b=2,
        h_q=16,
        h_kv=4,
        s_q=256,
        s_kv=4096,
        d_qk=128,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        dtype_o=cudnn.data_type.BFLOAT16,
        device_cc=(8, 0),
        device_sm_count=108,
    )
    values.update(overrides)
    return SdpaGraphFacts(**values)


def test_flavor_table_matches_the_config():
    assert half.FLAVOR_DIMS == tuple((name, (cfg.D_QK, cfg.D_V)) for name, cfg in config_sm80.FLAVOR_CFGS.items())
    for q in range(8, 577, 8):
        for v in range(8, 513, 8):
            assert half.pick_flavor(q, v) == config_sm80.pick_flavor(q, v), (q, v)


def test_owners_partition_the_eight_aligned_domain():
    shapes = list(half.SHAPES.values())
    assert sum(map(len, shapes)) == len(frozenset().union(*shapes)) == 72 * 64
    assert len(half.SHAPES[K_PREFETCH]) == 24 * 16
    legacy = SPECS[half.LEGACY_ROW].capabilities
    assert {shape for _, shape in legacy.delegated_exact} == frozenset().union(*shapes)


def test_registry_slots():
    from cudnn.engines.manifest import MANIFEST

    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    assert {k: family.offered_ids()[k] for k in OFFERED} == OFFERED


@pytest.mark.parametrize(
    "dims,owner",
    [((8, 8), K_PREFETCH), ((128, 128), K_PREFETCH), ((192, 128), K_PREFETCH), ((200, 128), D256), ((128, 136), D256), ((256, 256), D256)]
    + [((264, 256), WIDE), ((512, 512), WIDE), ((576, 512), WIDE), ((520, 504), WIDE)],
)
def test_head_dims_name_one_owner(dims, owner):
    f = facts(d_qk=dims[0], d_v=dims[1])
    assert "owned by the explicit implementation engines" in engines.mismatch(SPECS[half.LEGACY_ROW].capabilities, f)
    assert engines.mismatch(SPECS[owner].capabilities, f) is None
    for other in half.IMPLEMENTATIONS - {owner}:
        assert engines.mismatch(SPECS[other].capabilities, f) is not None, other


@pytest.mark.parametrize("dims", [(100, 100), (584, 512), (576, 520)])
def test_unaligned_and_oversized_dims_have_no_owner(dims):
    f = facts(d_qk=dims[0], d_v=dims[1])
    assert all(engines.mismatch(SPECS[name].capabilities, f) is not None for name in OFFERED)


@pytest.mark.parametrize("kind", ["A", "FALLBACK"])
@pytest.mark.parametrize("dims", [(64, 64), (128, 128), (192, 128), (256, 256), (512, 512), (576, 512)])
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        dict(s_q=1, h_q=32, h_kv=8, s_kv=8192),
        dict(causal=True, bottom_right=True, s_kv=256),
        dict(window_left=128, causal=True, bottom_right=True),
        dict(s_q=4, has_paged_kv=True, padded=True, page_size=16),
    ],
)
def test_candidate_order_and_budget_preserve_the_legacy_policy(kind, dims, overrides):
    f = facts(d_qk=dims[0], d_v=dims[1], **overrides)
    policy = heuristics.sm100_legacy_policy_spec(half.LEGACY_ROW)
    old = heuristics._knob_sets(policy, f) if kind == "A" else [heuristics._fallback_knobs(policy, f)]
    expected = list(dict.fromkeys(k for k in old if engines.mismatch(policy.capabilities, f, k) is None))
    plans = heuristics.recommend(kind, f, OFFERED)
    assert [p.knobs for p in plans] == expected
    assert {IDS[p.engine_id] for p in plans} <= {half.implementation_for(*dims)}
    assert all(p.knobs.tile_m is None and p.knobs.tile_n is None for p in plans)


@pytest.mark.parametrize("dims", [(128, 128), (256, 256), (512, 512)])
def test_placement_and_trailing_inherit_the_legacy_row(dims):
    f = facts(s_q=8, h_q=32, h_kv=4, d_qk=dims[0], d_v=dims[1])
    owner = half.implementation_for(*dims)
    assert placement.place(SPECS[owner], f) == placement.place(heuristics.sm100_legacy_policy_spec(half.LEGACY_ROW), f)
    assert owner in placement.TRAIL_ONLY_ROWS and half.LEGACY_ROW in placement.TRAIL_ONLY_ROWS


@pytest.mark.parametrize(
    "owner,builder", [(K_PREFETCH, "_sm80_f16_k_prefetch_spec"), (D256, "_sm80_d256_kv_prefetch_spec"), (WIDE, "_sm80_wide_cooperative_spec")]
)
def test_explicit_lowering_does_not_consult_candidate_selection(monkeypatch, owner, builder):
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit record entered performance selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    calls = []
    monkeypatch.setattr(engines, "lower_dsl_prefill", lambda spec, f, k, *, api_type, selected_kernel: calls.append((api_type, k, selected_kernel)))
    spec = getattr(engines, builder)()
    spec.lower(spec, facts(), None)
    ((api_type, knobs, geometry),) = calls
    assert api_type == "SdpaFwdDslSm80"
    assert (knobs.tile_m, knobs.tile_n, geometry.implementation, geometry.lowering_row_name) == (None, None, owner, half.LEGACY_ROW)


def test_compiled_identity_check_names_the_disagreement():
    params = SimpleNamespace(d_qk=256, d_v=256, tile_m=64, tile_n=32, sched_policy=0, split_kv=3)
    knobs = engines.SdpaFwdKnobs(split_kv=3)
    geometry = half.Geometry(D256)
    assert geometry.compiled_mismatch("prefill_d256_f16", params, knobs)[1] is None
    assert geometry.compiled_mismatch("prefill_wide_f16", params, knobs)[1] is not None
    assert half.Geometry(K_PREFETCH).compiled_mismatch("prefill_d256_f16", params, knobs)[1] is not None
    assert geometry.compiled_mismatch("prefill_d256_f16", params, engines.SdpaFwdKnobs(split_kv=2))[1] is not None
    assert geometry.compiled_mismatch("prefill_d256_f16", params, engines.SdpaFwdKnobs(tile_m=128, split_kv=3))[1] is not None


@pytest.mark.parametrize("module", ["f16_k_prefetch", "d256_kv_prefetch", "wide_cooperative"])
def test_cold_host_import_does_not_import_registry_or_runtimes(module):
    source_root = str(Path(cudnn.__file__).resolve().parents[1])
    code = "\n".join(
        (
            "import json, sys",
            "sys.meta_path[:] = [f for f in sys.meta_path if 'editable' not in str(f).lower()]",
            f"sys.path.insert(0, {source_root!r})",
            f"import cudnn.sdpa.fwd.kernels.sm80.{module}.host",
            "print(json.dumps(sorted(sys.modules)))",
        )
    )
    loaded = set(json.loads(subprocess.check_output([sys.executable, "-c", code], text=True)))
    forbidden = {
        "torch",
        "cutlass",
        "cudnn.sdpa.fwd.engines",
        "cudnn.sdpa.fwd.api_dsl",
        "cudnn.sdpa.fwd.config_sm80",
        f"cudnn.sdpa.fwd.kernels.sm80.{module}.kernel",
    }
    assert not forbidden.intersection(loaded)
