# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts for the SM120 per-tensor FP8 and MXFP8 general-tile owners."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import cudnn
from cudnn.sdpa.fwd import config_sm120, engines, heuristics, placement
from cudnn.sdpa.fwd.kernels.sm120 import quantized_general_contract as quant
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0
SPECS = {s.name: s for s in engines.ENGINE_SPECS}
# owner, quantization flags, legacy id, owner id
OWNERS = {"fp8": (quant.FP8, dict(is_fp8=True), 20507, 20542), "mxfp8": (quant.MXFP8, dict(is_mxfp8=True), 20518, 20543)}


def facts(kind, **overrides):
    values = dict(b=2, h_q=16, h_kv=2, s_q=256, s_kv=4096, d_qk=128, d_v=128, dtype=cudnn.data_type.FP8_E4M3, dtype_o=cudnn.data_type.BFLOAT16)
    values.update(device_cc=(12, 0), device_sm_count=188, **OWNERS[kind][1])
    values.update(overrides)
    return SdpaGraphFacts(**values)


def offered(kind):
    owner, _, legacy_id, owner_id = OWNERS[kind]
    return {quant.explicit_geometry(owner, None).lowering_row_name: legacy_id, owner: owner_id}


def test_shared_tiles_match_the_config():
    assert quant.HEAD_TILES == config_sm120.FP8_GENERAL_HEAD_TILES


@pytest.mark.parametrize("kind", sorted(OWNERS))
def test_registry_slots(kind):
    from cudnn.engines.manifest import MANIFEST

    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    assert {k: family.offered_ids()[k] for k in offered(kind)} == offered(kind)


@pytest.mark.parametrize("kind", sorted(OWNERS))
@pytest.mark.parametrize("cc", [(12, 0), (12, 1)])
@pytest.mark.parametrize("dims", [(32, 32), (128, 128), (96, 160), (256, 64), (256, 256)])
def test_exact_general_tiles_have_one_owner(kind, cc, dims):
    owner, _, _, _ = OWNERS[kind]
    f = facts(kind, device_cc=cc, d_qk=dims[0], d_v=dims[1])
    legacy = quant.explicit_geometry(owner, None).lowering_row_name
    assert "owned by the explicit implementation engines" in engines.mismatch(SPECS[legacy].capabilities, f)
    assert engines.mismatch(SPECS[owner].capabilities, f) is None
    other = quant.MXFP8 if owner == quant.FP8 else quant.FP8
    assert engines.mismatch(SPECS[other].capabilities, f) is not None


@pytest.mark.parametrize("kind", sorted(OWNERS))
@pytest.mark.parametrize("dims", [(16, 16), (48, 48), (80, 112), (144, 128)])
def test_padded_heads_enter_the_owner_of_their_tile(kind, dims):
    owner, _, legacy_id, owner_id = OWNERS[kind]
    f = facts(kind, d_qk=dims[0], d_v=dims[1])
    legacy = quant.explicit_geometry(owner, None).lowering_row_name
    assert "padded to" in engines.mismatch(SPECS[legacy].capabilities, f)
    assert engines.mismatch(SPECS[owner].capabilities, f) is None


@pytest.mark.parametrize("kind", sorted(OWNERS))
@pytest.mark.parametrize("dims", [(512, 512), (8, 8), (40, 40), (264, 128)])
def test_wide_and_unaligned_heads_do_not_enter_the_owner(kind, dims):
    owner = OWNERS[kind][0]
    assert engines.mismatch(SPECS[owner].capabilities, facts(kind, d_qk=dims[0], d_v=dims[1])) is not None


@pytest.mark.parametrize("kind", sorted(OWNERS))
@pytest.mark.parametrize("tkind", ["A", "FALLBACK"])
@pytest.mark.parametrize(
    "overrides", [{}, dict(s_q=1, h_q=32, h_kv=8, s_kv=8192), dict(thd=True, padded=True, max_total_seq_len_q=512), dict(d_qk=192, d_v=128)]
)
def test_candidate_order_and_budget_preserve_the_legacy_policy(kind, tkind, overrides):
    owner, _, _, owner_id = OWNERS[kind]
    f = facts(kind, **overrides)
    policy = heuristics.sm100_legacy_policy_spec(quant.explicit_geometry(owner, None).lowering_row_name)
    old = heuristics._knob_sets(policy, f) if tkind == "A" else [heuristics._fallback_knobs(policy, f)]
    expected = list(dict.fromkeys(k for k in old if engines.mismatch(policy.capabilities, f, k) is None))
    plans = heuristics.recommend(tkind, f, offered(kind))
    assert [p.knobs for p in plans] == expected
    assert all(p.engine_id == owner_id for p in plans)


@pytest.mark.parametrize("kind", sorted(OWNERS))
def test_placement_inherits_the_legacy_study_keys(kind):
    owner = OWNERS[kind][0]
    f = facts(kind, s_q=1)
    knobs = engines.SdpaFwdKnobs(sched_policy=0, tile_m=64, tile_n=64, cga=1, pack_gqa=False, split_kv=1)
    legacy = quant.explicit_geometry(owner, None).lowering_row_name
    assert placement.place(SPECS[owner], f, knobs) == placement.place(heuristics.sm100_legacy_policy_spec(legacy), f)


@pytest.mark.parametrize("kind", sorted(OWNERS))
def test_explicit_lowering_does_not_consult_candidate_selection(monkeypatch, kind):
    owner = OWNERS[kind][0]

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit record entered performance selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    calls = []
    monkeypatch.setattr(engines, "lower_dsl_prefill", lambda spec, f, k, *, api_type, selected_kernel: calls.append((api_type, k, selected_kernel)))
    spec = {quant.FP8: engines._sm120_fp8_general_spec, quant.MXFP8: engines._sm120_mxfp8_general_spec}[owner]()
    spec.lower(spec, facts(kind), None)
    ((api_type, knobs, geometry),) = calls
    assert api_type == "SdpaFwdDslSm120"
    assert (knobs.tile_m, knobs.cga, geometry.implementation) == (128, 1, owner)


def test_compiled_identity_check_names_the_disagreement():
    geometry = quant.GeneralGeometry(quant.FP8, 64)
    knobs = geometry.canonical_knobs(engines.SdpaFwdKnobs(tile_n=128, split_kv=2))
    params = SimpleNamespace(q_tile=64, kv_tile=128, sched_policy=0, pack_gqa=False, split_kv=2)
    assert geometry.compiled_mismatch("prefill_fp8", params, knobs)[1] is None
    assert geometry.compiled_mismatch("prefill_mxfp8", params, knobs)[1] is not None
    assert geometry.compiled_mismatch("prefill_fp8", SimpleNamespace(**{**vars(params), "split_kv": 1}), knobs)[1] is not None


@pytest.mark.parametrize("module", ["fp8_general", "mxfp8_general"])
def test_cold_host_import_does_not_import_registry_or_runtimes(module):
    source_root = str(Path(cudnn.__file__).resolve().parents[1])
    code = "\n".join(
        (
            "import json, sys",
            "sys.meta_path[:] = [f for f in sys.meta_path if 'editable' not in str(f).lower()]",
            f"sys.path.insert(0, {source_root!r})",
            f"import cudnn.sdpa.fwd.kernels.sm120.{module}.host",
            "print(json.dumps(sorted(sys.modules)))",
        )
    )
    loaded = set(json.loads(subprocess.check_output([sys.executable, "-c", code], text=True)))
    forbidden = {
        "torch",
        "cutlass",
        "cudnn.sdpa.fwd.engines",
        "cudnn.sdpa.fwd.api_dsl",
        "cudnn.sdpa.fwd.config_sm120",
        f"cudnn.sdpa.fwd.kernels.sm120.{module}.kernel",
    }
    assert not forbidden.intersection(loaded)
