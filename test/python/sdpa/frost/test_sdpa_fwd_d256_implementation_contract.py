# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for explicit SM100 D256 main implementations.

GPU build/execution coverage belongs to the D256 forward suites. These tests
exercise ownership, public replay, candidate geometry and import isolation.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import cudnn
from cudnn.sdpa.fwd import engines, heuristics, placement
from cudnn.sdpa.fwd.kernels.sm100.d256_f16_contract import LOOKAHEAD, SWAP_AB, explicit_geometry
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0

SPECS = {s.name: s for s in engines.ENGINE_SPECS}
OFFERED = {"sdpa_fwd_prefill_sm100": 20511, LOOKAHEAD: 20519, SWAP_AB: 20520}


def facts(**overrides):
    return SdpaGraphFacts(
        **dict(
            dict(b=1, h_q=32, h_kv=8, s_q=1, s_kv=8192, d_qk=256, d_v=256, dtype=cudnn.data_type.BFLOAT16, causal=True, device_cc=(10, 0), device_sm_count=148),
            **overrides,
        )
    )


@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
def test_exact_d256_transfers_ownership_but_keeps_both_implementations_legal(cc):
    f = facts(device_cc=cc)
    assert "owned by" in engines.mismatch(SPECS["sdpa_fwd_prefill_sm100"].capabilities, f)
    for name in (LOOKAHEAD, SWAP_AB):
        assert engines.mismatch(SPECS[name].capabilities, f) is None
    # The old row still owns its other exact widths and D256 envelopes.
    for dq, dv in ((128, 128), (192, 128), (248, 248)):
        other = facts(device_cc=cc, d_qk=dq, d_v=dv)
        assert engines.mismatch(SPECS["sdpa_fwd_prefill_sm100"].capabilities, other) is None
        assert engines.mismatch(SPECS[LOOKAHEAD].capabilities, other) is not None
        assert engines.mismatch(SPECS[SWAP_AB].capabilities, other) is not None


@pytest.mark.parametrize("name, tile, cga", [(LOOKAHEAD, 128, 2), (SWAP_AB, 16, 1), (SWAP_AB, 32, 1)])
def test_explicit_public_config_does_not_apply_a_shape_crossover(name, tile, cga):
    knobs = engines.SdpaFwdKnobs(tile_m=tile, tile_n=128, cga=cga, sched_policy=0, split_kv=1, pack_gqa=False)
    replayed = engines.SdpaFwdKnobs.from_public(knobs.to_public())
    assert replayed == knobs
    assert knobs.to_public()[cudnn.knob_type.TILE_M] == tile
    for sq in (1, 17, 128):
        assert engines.mismatch(SPECS[name].capabilities, facts(s_q=sq), replayed) is None
        chosen = explicit_geometry(name, replayed)
        assert (chosen.implementation, chosen.query_tile, chosen.cga) == (name, tile, cga)


@pytest.mark.parametrize(
    "knobs",
    [
        engines.SdpaFwdKnobs(tile_m=128),
        engines.SdpaFwdKnobs(tile_m=16, cga=2),
        engines.SdpaFwdKnobs(tile_m=16, sched_policy=1),
        engines.SdpaFwdKnobs(tile_m=16, pack_gqa=True),
    ],
)
def test_swap_ab_declines_unimplemented_config_instead_of_ignoring_it(knobs):
    assert engines.mismatch(SPECS[SWAP_AB].capabilities, facts(h_q=32, h_kv=1), knobs) is not None


def test_thd_remains_lookahead_only():
    f = facts(thd=True, padded=True)
    assert engines.mismatch(SPECS[LOOKAHEAD].capabilities, f) is None
    assert "dense queries" in engines.mismatch(SPECS[SWAP_AB].capabilities, f)


def test_explicit_swap_ab_preserves_the_existing_packing_domain():
    packed = engines.SdpaFwdKnobs(tile_m=16, cga=1, pack_gqa=True)
    assert engines.mismatch(SPECS[SWAP_AB].capabilities, facts(h_q=12, h_kv=1), packed) is None
    assert "head-ratio domain" in engines.mismatch(SPECS[SWAP_AB].capabilities, facts(h_q=3, h_kv=1), packed)
    assert "wrong operation" in engines.mismatch(SPECS[SWAP_AB].capabilities, facts(), {})


def test_default_candidates_use_resolved_geometry_without_the_old_twin(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("migrated candidate consulted the old adapter-route predictor")

    monkeypatch.setattr(heuristics, "decode_d256_q_tile_for_row", forbidden)
    monkeypatch.setattr(engines, "decode_d256_q_tile_for_row", forbidden)
    plans = heuristics.recommend("A", facts(), OFFERED)
    assert plans and len(plans) <= heuristics._MAX_SETS_PER_ENGINE
    assert len({(p.engine_id, p.knobs) for p in plans}) == len(plans)
    assert all(p.engine_id == OFFERED[SWAP_AB] for p in plans)
    assert all(p.knobs.tile_m == 16 and p.knobs.cga == 1 and p.knobs.sched_policy == 0 for p in plans)


def test_new_names_and_explicit_alternatives_do_not_inherit_generic_lead():
    f = facts()
    assert placement.place(SPECS[LOOKAHEAD], f) == placement.TRAIL
    assert placement.place(SPECS[SWAP_AB], f) == placement.TRAIL
    lookahead = engines.SdpaFwdKnobs(tile_m=128, tile_n=128, cga=2, sched_policy=0, pack_gqa=False, split_kv=1)
    assert placement.place(SPECS[LOOKAHEAD], f, lookahead) == placement.TRAIL


@pytest.mark.parametrize("name,tile,cga", [(LOOKAHEAD, 128, 2), (SWAP_AB, 16, 1), (SWAP_AB, 32, 1)])
@pytest.mark.parametrize("direction,expected", [("always", placement.LEAD), ("never", placement.TRAIL)])
def test_explicit_implementation_study_precedes_legacy_mapping(monkeypatch, name, tile, cga, direction, expected):
    from cudnn.sdpa import placement_table as pt

    f = facts()
    knobs = engines.SdpaFwdKnobs(tile_m=tile, cga=cga, pack_gqa=False)
    monkeypatch.setitem(pt.TABLE, pt.class_key(name, f), pt.Entry(direction, None, "implementation-regression-control"))
    # Opposite legacy verdict: both True and False on the actual name must win,
    # even for a legal variant that the default policy would not choose.
    monkeypatch.setitem(pt.TABLE, pt.class_key("sdpa_fwd_prefill_sm100", f), pt.Entry("never" if direction == "always" else "always", None, "legacy-control"))
    assert placement.place(SPECS[name], f, knobs) == expected


@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
@pytest.mark.parametrize("sq", [1, 16, 128, 129, 512])
def test_pilot_shortcut_covers_every_eligible_row(cc, sq):
    offered = {s.name: i for i, s in enumerate(engines.ENGINE_SPECS)}
    eligible = {s.name for _, s in heuristics._eligible(facts(device_cc=cc, s_q=sq), offered)}
    assert eligible and eligible <= {LOOKAHEAD, SWAP_AB}


def test_dynamic_kernel_fixture_paths_exist():
    from cudnn.sdpa.fwd import api_dsl
    from service_register_release_checks import release_cases
    from test_sdpa_fwd_paged_sm100 import _KERNEL_FILES

    kernels = Path(api_dsl.__file__).parent / "kernels"
    paths = [f"sm100/{name}" for name in _KERNEL_FILES.values()]
    paths += [case.values[0] for arch in ("sm100", "sm107") for case in release_cases(arch)]
    for path in paths:
        assert (kernels / path).is_file(), path


def test_delegated_paged_split_retains_parent_study_and_placement():
    from types import SimpleNamespace
    from cudnn.sdpa import placement_table as pt

    pool = SimpleNamespace(get_stride=lambda: (8192, 4096, 256, 1))
    f = facts(
        h_q=16,
        h_kv=2,
        s_q=128,
        s_kv=2049,
        bottom_right=True,
        thd=True,
        padded=True,
        has_paged_kv=True,
        page_size=16,
        max_total_seq_len_q=128,
        k_t=pool,
        v_t=pool,
    )
    assert pt.class_key("sdpa_fwd_prefill_sm100", f)[8] == "gqa_split"
    assert pt.matched_entry("sdpa_fwd_prefill_sm100", f).study == "sm100_paged_d64_d256_prefill_20261111"
    first = heuristics.recommend("A", f, OFFERED)[0]
    spec = next(s for s in engines.ENGINE_SPECS if OFFERED.get(s.name) == first.engine_id)
    assert first.knobs.pack_gqa and first.knobs.split_kv > 1
    assert placement.place(spec, f, first.knobs) == placement.LEAD


def test_retired_recipe_is_an_explicit_cache_miss_and_can_be_rebuilt():
    import torch
    import cudnn._plan_serdes as ps

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("exact D256 ownership transition is qualified on SM100/SM103")

    def make_graph():
        g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        q, k, v = (g.tensor(dim=[1, 8, 16, 256], stride=[32768, 256, 2048, 1], name=n) for n in ("q", "k", "v"))
        o, _ = g.sdpa(q=q, k=k, v=v, generate_stats=False, attn_scale=0.0625)
        o.set_output(True).set_stride([32768, 256, 2048, 1])
        return g

    header = dict(
        format=ps.FORMAT,
        frontend_version="historical",
        graph=ps._encode_graph(make_graph()),
        plan=dict(engine_id=20511, engine_name="sdpa_fwd_prefill_sm100", knobs=[]),
        kernels=[],
    )
    encoded = json.dumps(header).encode()
    target = cudnn.pygraph()
    with pytest.raises((ValueError, RuntimeError), match="20511|owned by|not supported|No execution plans"):
        target.deserialize(ps.MAGIC + ps._LEN.pack(len(encoded)) + encoded)
    assert not target._nodes and not target._planning_done
    rebuilt = make_graph()
    rebuilt.validate()
    rebuilt.create_execution_plans([cudnn.heur_mode.A])
    ids = {rebuilt.get_engine_and_knobs_at_index(i)[0] for i in range(rebuilt.get_execution_plan_count())}
    assert 20511 not in ids and ids.intersection({20519, 20520})


@pytest.mark.parametrize(
    "module", ["d256_f16_contract", "d256_f16_lookahead.contract", "d256_f16_lookahead.config", "d256_f16_swap_ab.contract", "d256_f16_swap_ab.config"]
)
def test_implementation_contracts_import_without_other_bodies_or_heavy_runtimes(module):
    code = f"""import importlib, json, sys
sys.meta_path[:] = [f for f in sys.meta_path if 'editable' not in str(f).lower()]
sys.path.insert(0, {str(Path(cudnn.__file__).resolve().parents[1])!r})
importlib.import_module("cudnn.sdpa.fwd.kernels.sm100.{module}")
print(json.dumps(sorted(sys.modules)))
"""
    loaded = json.loads(subprocess.check_output([sys.executable, "-c", code], text=True))
    forbidden = (
        "torch",
        "cutlass",
        "cudnn.sdpa.fwd.api_dsl",
        "cudnn.sdpa.fwd.config_sm100",
        "cudnn.sdpa.fwd.kernels.sm100.d256_f16_lookahead.kernel",
        "cudnn.sdpa.fwd.kernels.sm100.d256_f16_swap_ab.kernel",
    )
    assert not set(forbidden).intersection(loaded)
