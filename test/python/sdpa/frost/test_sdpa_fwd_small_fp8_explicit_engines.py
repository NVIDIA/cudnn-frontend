# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed small-head FP8 records survive rebuilding, rebinding and capture."""

import pytest
import torch

import cudnn
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
import test_sdpa_prepared_fp8 as shared

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]

OWNERS = {"multi_q": 20527, "dual_kv": 20528, "swap_ab": 20529}


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "variant,d,sq,cga,split,thd",
    [
        ("multi_q", 64, 17, 1, 1, False),
        ("multi_q", 128, 1, 1, 1, False),
        ("multi_q", 128, 129, 2, 4, False),
        ("multi_q", 128, 17, 1, 1, True),
        ("dual_kv", 128, 1, 1, 4, False),
        ("dual_kv", 128, 257, 1, 1, False),
        ("swap_ab", 128, 4, 1, 1, False),
        ("swap_ab", 128, 4, 1, 4, False),
    ],
)
def test_public_record_rebuild_rebind_and_retained_capture(monkeypatch, dtype, output, variant, d, sq, cga, split, thd):
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    tile_m = 16 if variant == "swap_ab" else 128
    record = (OWNERS[variant], SdpaFwdKnobs(tile_m=tile_m, tile_n=128, cga=cga, sched_policy=0, pack_gqa=not thd, split_kv=split).to_public())

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed record reentered body-selection policy")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    monkeypatch.setattr(heuristics, "select_sm100_small_fp8_geometry", forbidden)
    monkeypatch.setattr(SdpaFwdDslSm100, "_fp8_dual_kv_stream", forbidden)

    def build(saved):
        result = shared._case(d=d, sq=sq, skv=512, dtype=dtype, output_dtype=output, split_kv=split, thd=thd, record=saved)
        graph = result[0]
        assert graph.get_engine_and_knobs_at_index(graph._plan_index) == saved
        assert graph.selected_engine.name == "sdpa_fwd_sm100_d128_fp8_" + variant
        compiled = graph._compiled_plans[graph._plan_index]._compiled
        assert compiled.kernel_template == ("decode_d128_fp8" if variant == "swap_ab" else "prefill_d128_fp8")
        metadata = compiled.implementation_config
        assert (metadata["query_tile"], metadata["cluster_m"], metadata["cta_group"], metadata["split_kv"]) == (tile_m, cga, cga, split)
        return result

    graph, vp, workspace, buffers, tensors = build(record)
    graph.execute(vp, workspace)
    shared._check(buffers, thd=thd, sq=sq, skv=512)
    with shared._cuda_graph() as capture:
        with torch.cuda.graph(capture):
            previous = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                graph.execute(vp, workspace)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
        rebuilt, rebound, scratch, other, _ = build(graph.get_engine_and_knobs_at_index(graph._plan_index))
        rebuilt.execute(rebound, scratch)
        shared._check(other, thd=thd, sq=sq, skv=512)
        buffers["descale_q"].fill_(0.5)
        buffers["scale_o"].fill_(1.7)
        for name in ("o", "lse", "amax_o"):
            buffers[name].fill_(float("nan"))
        capture.replay()
        shared._check(buffers, thd=thd, sq=sq, skv=512)
    for name in ("q", "k", "v", "o", "lse", "amax_o", "descale_q", "scale_o"):
        buffers[name] = buffers[name].clone()
        vp[tensors[name]] = buffers[name]
    graph.execute(vp, torch.empty_like(workspace))
    shared._check(buffers, thd=thd, sq=sq, skv=512)


@pytest.mark.parametrize("d,sq,group", [(64, 3, 8), (64, 4, 8), (128, 1, 8)])
@pytest.mark.parametrize("split", [1, 4])
def test_paged_swap_ab_fixed_record_and_live_columns(monkeypatch, d, sq, group, split):
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
    import test_sdpa_fwd_paged_sm100 as paged

    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("paged small-head swap-AB retains its SM100-only contract")
    record = (20529, SdpaFwdKnobs(tile_m=16, tile_n=128, cga=1, sched_policy=0, pack_gqa=True, split_kv=split).to_public())

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed paged record reentered selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    monkeypatch.setattr(heuristics, "select_sm100_small_fp8_geometry", forbidden)
    graph, vp, workspace, buffers, _ = paged._run_graph_fp8(2, group * 2, 2, d, 16, 32, [497, 512], hnd=True, s_q=sq, record=record, return_case=True)
    assert graph.get_engine_and_knobs_at_index(graph._plan_index) == record
    assert graph.selected_engine.name == "sdpa_fwd_sm100_d128_fp8_swap_ab"
    expected = {name: buffers[name].clone() for name in ("o", "lse", "amax_o")}
    with shared._cuda_graph() as capture:
        with torch.cuda.graph(capture):
            graph.execute(vp, workspace)
        for name in expected:
            buffers[name].fill_(float("nan"))
        capture.replay()
        for name, output in expected.items():
            torch.testing.assert_close(buffers[name], output, atol=0, rtol=0)
