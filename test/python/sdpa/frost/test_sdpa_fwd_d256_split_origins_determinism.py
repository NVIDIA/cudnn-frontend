# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D256 split metadata preserves physical storage and fixed-plan output bytes."""

import pytest

from frost_test_utils import requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl]


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 7)])
def test_split_origins_require_matching_extension(monkeypatch, cc):
    import cudnn
    from cudnn.frost import buffers
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    facts = SdpaGraphFacts(
        b=4,
        h_q=8,
        h_kv=2,
        s_q=256,
        s_kv=256,
        d_qk=256,
        d_v=256,
        dtype=cudnn.data_type.BFLOAT16,
        thd=True,
        device_cc=cc,
        device_sm_count=148,
    )
    name = "sdpa_fwd_sm107_shared_d256_lookahead" if cc == (10, 7) else "sdpa_fwd_sm100_d256_lookahead"
    caps = next(spec.capabilities for spec in engines.ENGINE_SPECS if spec.name == name)
    knobs = engines.SdpaFwdKnobs(cga=2, split_kv=3, pack_gqa=0, sched_policy=0)
    assert engines.mismatch(caps, facts, knobs) is None
    binder = cudnn._pybind_module._SdpaThdBinder

    class PreviousExtension:
        def __getattr__(self, attr):
            if attr == "supports_nonpaged_d256_split_physical_origins":
                return False
            return getattr(binder, attr)

    monkeypatch.setattr(cudnn._pybind_module, "_SdpaThdBinder", PreviousExtension())
    assert "nonpaged D256 split physical origins" in engines.mismatch(caps, facts, knobs)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("stats", ["token", "head"])
@pytest.mark.parametrize("splits", [2, 5])
def test_origins_reference_replay_batch_and_golden(dtype, stats, splits):
    import torch

    from determinism_utils import assert_bitwise, assert_golden, assert_replays
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as check_origins
    from test_sdpa_fwd_thd_origins_sm120 import _check_invariance

    cc = torch.cuda.get_device_capability()
    if cc not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("SM100/SM103/SM107 D256 split origins")
    owner = 20570 if cc == (10, 7) else 20519
    engine = "sdpa_fwd_sm107_shared_d256_lookahead" if cc == (10, 7) else "sdpa_fwd_sm100_d256_lookahead"
    record = {}

    def extra(graph, nodes, values, state, workspace, overrides, flat, stats_layout):
        assert graph.get_engine_and_knobs_at_index(graph._plan_index)[0] == owner
        plan = graph._compiled_plans[graph._plan_index]
        assert plan._compiled.kernel_module.__file__.endswith("sm100/d256_f16_lookahead/kernel.py")
        assert plan._prepared.spec.native is not None
        assert plan._prepared.spec.origin_specs is not None
        _check_invariance(graph, nodes, values, state, workspace, overrides, flat, stats_layout)
        saved = {name: value.clone() for name, value in values.items() if name not in ("o", "stats")}

        def replay():
            # Fresh output and scratch storage on every call, with unchanged
            # pitches; concurrent work is supplied by assert_replays.
            outputs = {
                name: torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=value.device)
                for name, value in values.items()
                if name in ("o", "stats")
            }
            outputs["o"].fill_(-17)
            outputs["stats"].fill_(-29)
            scratch = torch.full_like(workspace, 255)
            graph.execute({nodes[name]: outputs.get(name, value) for name, value in values.items()}, scratch, **overrides)
            assert_bitwise({name: values[name] for name in saved}, saved, "input bytes")
            return outputs

        assert_replays(replay, repeats=4, allowed_inf={"stats": -torch.inf})
        ql, kl, origins = state
        if origins["q"][0] == 0:
            record["inputs"] = {
                name: flat(values[name])[origins[name][0] : origins[name][0] + (ql[0] if name == "q" else kl[0])].clone() for name in ("q", "k", "v")
            }
            stats_view = values["stats"].T if stats_layout == "head" else values["stats"]
            record["outputs"] = {
                "o": flat(values["o"])[origins["o"][0] : origins["o"][0] + ql[0]].clone(),
                "stats": stats_view[origins["stats"][0] : origins["stats"][0] + ql[0]].clone(),
            }
            _check_single_sequence(graph, nodes, values, origins, ql[0], stats_layout, record["outputs"])

    # The helper checks an independent FP64 oracle, changed lengths/pitches,
    # keyless rows, empty work and retained captures before recording anything.
    # Batch insertion in the callback needs capacity beyond the original sum
    # of live Q lengths; the separate origin suite covers bounded total hints.
    check_origins(256, dtype, stats, False, False, splits, engine_name=engine, cga=2, extra_checks=extra)
    assert_golden(f"sdpa-d256-thd-split{splits}-origins-{dtype}-{stats}", record["inputs"], record["outputs"])


def _check_single_sequence(graph, nodes, values, origins, nq, stats_layout, expected):
    """Rebuild the same public record at B=1 with the first sequence's bytes."""
    import cudnn
    import torch
    from determinism_utils import assert_bitwise

    solo = cudnn.pygraph(io_data_type=nodes["q"].get_data_type(), intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    bound = {name: value[:1] for name, value in values.items() if name in ("q", "k", "v", "ql", "kl")}
    bound.update({name: value[:2] for name, value in values.items() if name.startswith("ro_")})
    decl = {name: solo.tensor_like(value) for name, value in bound.items()}
    for name in ("q", "k", "v"):
        decl[name].set_ragged_offset(decl["ro_" + name]).set_ragged_offset_multiplier(nodes[name].get_ragged_offset_multiplier())
    decl["o"], decl["stats"] = solo.sdpa(
        q=decl["q"],
        k=decl["k"],
        v=decl["v"],
        attn_scale=256**-0.5,
        is_inference=False,
        diagonal_band_right_bound=0,
        use_padding_mask=True,
        seq_len_q=decl["ql"],
        seq_len_kv=decl["kl"],
        stats_use_log2=stats_layout == "head",
    )
    bound["o"] = torch.empty_like(values["o"][:1])
    stats = values["stats"][:, :256] if stats_layout == "head" else values["stats"][:256]
    bound["stats"] = torch.empty_strided(stats.shape, stats.stride(), dtype=stats.dtype, device=stats.device)
    decl["o"].set_output(True).set_dim(bound["o"].shape).set_stride(bound["o"].stride()).set_ragged_offset(decl["ro_o"]).set_ragged_offset_multiplier(
        nodes["o"].get_ragged_offset_multiplier()
    )
    decl["stats"].set_output(True).set_dim((1, values["q"].shape[1], 256, 1)).set_stride(nodes["stats"].get_stride()).set_ragged_offset(
        decl["ro_stats"]
    ).set_data_type(cudnn.data_type.FLOAT)
    solo.validate()
    solo.build_operation_graph()
    engine, knobs = graph.get_engine_and_knobs_at_index(graph._plan_index)
    solo.create_execution_plan(engine, knobs)
    solo.check_support()
    solo.build_plans()
    workspace = torch.empty(solo.get_workspace_size(), dtype=torch.uint8, device="cuda")
    solo.execute({decl[name]: value for name, value in bound.items()}, workspace)
    o = bound["o"].transpose(1, 2).reshape(256, values["q"].shape[1], 256)
    stats = bound["stats"].T if stats_layout == "head" else bound["stats"]
    assert_bitwise({"o": o[origins["o"][0] : origins["o"][0] + nq], "stats": stats[origins["stats"][0] : origins["stats"][0] + nq]}, expected, "B=1 versus B=4")
