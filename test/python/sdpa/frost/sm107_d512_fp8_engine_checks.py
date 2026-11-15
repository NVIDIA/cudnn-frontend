# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public FP8 D512 records fix the body across rebuild and retained capture."""

import json

import pytest
import torch

import cudnn
from frost_test_utils import requires_dsl, requires_rubin
import test_sdpa_prepared_fp8 as shared

pytestmark = [pytest.mark.L0, requires_dsl, requires_rubin]


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("split", [1, 4])
def test_public_record_rebuild_rebind_and_retained_capture(monkeypatch, dtype, output, split):
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    record = (20549, SdpaFwdKnobs(tile_m=128, tile_n=128, cga=4, sched_policy=0, pack_gqa=True, split_kv=split).to_public())

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit public replay consulted candidate selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)

    def build(saved):
        result = shared._case(arch="sm107", d=512, sq=17, skv=256, dtype=dtype, output_dtype=output, split_kv=split, record=saved)
        graph = result[0]
        assert graph.get_engine_and_knobs_at_index(graph._plan_index) == saved
        compiled = graph._compiled_plans[graph._plan_index]._compiled
        assert compiled.kernel_template == "prefill_d512_fp8"
        assert compiled.kernel_module.__file__.endswith("sm107/d512_fp8_role_split/kernel.py")
        assert type(compiled.kernel_module.CFG).__module__ == "cudnn.sdpa.fwd.kernels.sm107.d512_fp8_role_split.config"
        metadata = compiled.implementation_config
        assert graph.selected_engine.name == "sdpa_fwd_sm107_d512_fp8_role_split"
        assert (metadata["query_tile"], metadata["cluster_m"], metadata["cta_group"], metadata["split_kv"]) == (128, 4, 2, split)
        return result

    g, vp, ws, bufs, tensors = build(record)
    g.execute(vp, ws)
    shared._check(bufs, thd=False, sq=17, skv=256)

    with shared._cuda_graph() as capture:
        with torch.cuda.graph(capture):
            previous = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                g.execute(vp, ws)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
        rebuilt, rebound, scratch, other, _ = build(g.get_engine_and_knobs_at_index(g._plan_index))
        rebuilt.execute(rebound, scratch)
        shared._check(other, thd=False, sq=17, skv=256)
        bufs["descale_q"].fill_(0.5)
        bufs["scale_o"].fill_(1.7)
        for name in ("o", "lse", "amax_o"):
            bufs[name].fill_(float("nan"))
        capture.replay()
        shared._check(bufs, thd=False, sq=17, skv=256)

    # Eager execution rebinds caller storage and scales independently of the
    # retained executable's capture-time pointers.
    for name in ("q", "k", "v", "o", "lse", "amax_o", "descale_q", "scale_o"):
        bufs[name] = bufs[name].clone()
        vp[tensors[name]] = bufs[name]
    g.execute(vp, torch.empty_like(ws))
    shared._check(bufs, thd=False, sq=17, skv=256)


def test_retired_fp8_recipe_is_rejected_without_silent_substitution():
    import cudnn._plan_serdes as ps

    graph, _, _, _, _ = shared._case(arch="sm107", d=512, sq=17, skv=256)
    blob = graph.serialize(as_bytes=True, include_kernels=False)
    start = len(ps.MAGIC) + ps._LEN.size
    (length,) = ps._LEN.unpack_from(blob, len(ps.MAGIC))
    header = json.loads(blob[start : start + length])
    header["plan"].update(engine_id=20514, engine_name="sdpa_fwd_prefill_sm107_fp8", knobs=[])
    encoded = json.dumps(header).encode()
    target = cudnn.pygraph()
    with pytest.raises((ValueError, RuntimeError), match="20514|owned by|not supported|No execution plans"):
        target.deserialize(ps.MAGIC + ps._LEN.pack(len(encoded)) + encoded)
    assert not target._nodes and not target._planning_done
    # The same empty destination remains reusable for an eligible current recipe.
    target.deserialize(blob)
    assert target.get_engine_and_knobs_at_index(target._plan_index)[0] == 20549


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("stats", ["token", "head"])
@pytest.mark.parametrize("total_hints", [False, True])
def test_independent_thd_origins(dtype, stats, total_hints, monkeypatch):
    from determinism_utils import assert_replays
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as check
    from test_sdpa_fwd_thd_origins_fp8_sm120 import _check_routes_and_invariance

    def extra(graph, nodes, values, state, workspace, overrides, flat, stats_layout):
        plan = graph._compiled_plans[graph._plan_index]
        assert plan._compiled.kernel_module.__file__.endswith("sm107/d512_fp8_role_split/kernel.py")
        assert plan._prepared.spec.origin_specs is not None
        assert plan._prepared.spec.native is not None
        _check_routes_and_invariance(graph, nodes, values, state, workspace, overrides, flat, stats_layout)
        pack = {nodes[name]: value for name, value in values.items()}

        def run():
            values["o"].fill_(-17)
            values["stats"].fill_(-29)
            graph.execute(pack, workspace, **overrides)
            return {name: values[name] for name in ("o", "stats")}

        assert_replays(run, repeats=3, allowed_inf={"stats": -float("inf")})

    monkeypatch.setenv("FROST_THD_CLUSTERS", "2")
    check(512, dtype, stats, total_hints, False, 1, engine_name="sdpa_fwd_sm107_d512_fp8_role_split", cga=4, extra_checks=extra)


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group("gpu_exclusive")
def test_d512_fp8_refilled_stage_warm_stress():
    from warm_stress import StressCase, assert_warm_stress
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    record = (20549, SdpaFwdKnobs(tile_m=128, tile_n=128, cga=4, sched_policy=0, pack_gqa=True, split_kv=1).to_public())
    graph, vp, ws, bufs, _ = shared._case(arch="sm107", d=512, b=128, sq=17, skv=1024, record=record)
    graph.execute(vp, ws)
    shared._check(bufs, thd=False, b=128, sq=17, skv=1024)
    # Eight KV tiles cross the three-slot K/V ring and four-slot transfer ring.
    assert_warm_stress(
        StressCase(
            "SM107 D512 per-tensor FP8 role-split",
            lambda: graph.execute(vp, ws),
            {"o": bufs["o"].transpose(1, 2), "stats": bufs["lse"], "amax_o": bufs["amax_o"]},
            thread_blocks=1024,
            refills_per_launch=4,
        ),
        warm=256,
        poisoned=256,
    )
