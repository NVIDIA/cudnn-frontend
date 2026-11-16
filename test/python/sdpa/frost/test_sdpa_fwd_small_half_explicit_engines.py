# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small-head half records retain their implementation through rebuild and capture."""

import math

import cudnn
import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from test_sdpa_fwd_d192_half_explicit_engines import _check_retired_record, _fixed_loader

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "folder,d,sq,cga,split,packed,causal",
    [
        ("d128_f16_multi_q", 32, 129, 1, 1, False, False),
        ("d128_f16_multi_q", 32, 129, 2, 1, False, False),
        ("d128_f16_multi_q", 64, 17, 1, 1, True, True),
        ("d128_f16_multi_q", 128, 129, 2, 4, True, True),
        ("d128_f16_single_q", 64, 33, 1, 1, False, False),
        ("d128_f16_single_q", 128, 17, 1, 3, True, True),
        ("d64_f16_quadrant", 32, 8, 1, 1, True, True),
        ("d64_f16_quadrant", 64, 8, 1, 3, True, True),
        ("d64_f16_quadrant", 64, 32, 1, 1, False, False),
        ("d128_f16_multi_q", (40, 56), 129, 1, 1, False, False),
        ("d128_f16_multi_q", (80, 112), 129, 2, 3, True, True),
        ("d128_f16_single_q", (40, 56), 33, 1, 3, False, False),
        ("d128_f16_single_q", (80, 112), 17, 1, 3, True, True),
        ("d64_f16_quadrant", (40, 56), 8, 1, 3, True, True),
    ],
)
def test_public_record_rebuild_rebind_and_capture(monkeypatch, dtype, folder, d, sq, cga, split, packed, causal):
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    seen = _fixed_loader(monkeypatch, folder)
    from cudnn.sdpa.fwd import heuristics

    monkeypatch.setattr(heuristics, "select_sm100_small_half_geometry", lambda *a, **kw: pytest.fail("fixed record reentered small-head selection"))
    torch.manual_seed(19231)
    b, h, hk, sk = 2, 8, 2, 512
    dq, dv = (d, d) if isinstance(d, int) else d
    owner = {"d128_f16_multi_q": 20535, "d128_f16_single_q": 20532, "d64_f16_quadrant": 20536}[folder]
    owner_name = {20535: "sdpa_fwd_sm100_d128_multi_q", 20532: "sdpa_fwd_sm100_d128_single_q", 20536: "sdpa_fwd_sm100_d64_quadrant"}[owner]
    tensors = {
        "q": torch.randn(b, sq, h, dq, device="cuda", dtype=dtype).transpose(1, 2),
        "k": torch.randn(b, sk, hk, dq, device="cuda", dtype=dtype).transpose(1, 2),
        "v": torch.randn(b, sk, hk, dv, device="cuda", dtype=dtype).transpose(1, 2),
        "o": torch.empty(b, sq, h, dv, device="cuda", dtype=dtype).transpose(1, 2),
        "stats": torch.empty(b, h, sq, 1, device="cuda", dtype=torch.float32),
    }
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    record = (owner, SdpaFwdKnobs(tile_m=128, tile_n=128, cga=cga, sched_policy=1, pack_gqa=packed, split_kv=split).to_public())

    def build(saved):
        graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        nodes = {name: graph.tensor_like(tensors[name]) for name in ("q", "k", "v")}
        nodes["o"], nodes["stats"] = graph.sdpa(q=nodes["q"], k=nodes["k"], v=nodes["v"], generate_stats=True, attn_scale=dq**-0.5, use_causal_mask=causal)
        for name in ("o", "stats"):
            nodes[name].set_output(True).set_dim(tensors[name].shape).set_stride(tensors[name].stride())
        nodes["stats"].set_data_type(cudnn.data_type.FLOAT)
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plan(*saved)
        graph.select_plan(0)
        graph.check_support()
        graph.build_plans()
        assert graph.selected_engine.name == owner_name
        assert graph.get_engine_and_knobs_at_index(0) == saved
        compiled = graph._compiled_plans[0]._compiled
        assert compiled.kernel_template == {20535: "prefill_d128_f16", 20532: "decode_d128_f16", 20536: "decode_d64_f16_quadrant"}[owner]
        metadata = compiled.implementation_config
        assert (metadata["query_tile"], metadata["cluster_m"], metadata["cta_group"], metadata["split_kv"]) == (128, cga, cga, split)
        assert seen and seen[-1].CFG.TILES_Q == (2 if owner == 20535 else 1)
        assert seen[-1].CFG.TILE_K == (32 if owner == 20535 and d == 32 and cga == 1 else 64 if max(dq, dv) <= 64 else 128)
        workspace = torch.empty(max(1, graph.get_workspace_size()), device="cuda", dtype=torch.uint8)
        return graph, nodes, workspace

    def check(values):
        scores = values["q"].float() @ values["k"].float().repeat_interleave(h // hk, dim=1).transpose(-1, -2) / math.sqrt(dq)
        if causal:
            scores.masked_fill_(torch.arange(sk, device="cuda")[None, :] > torch.arange(sq, device="cuda")[:, None], -float("inf"))
        expected = scores.softmax(-1) @ values["v"].float().repeat_interleave(h // hk, dim=1)
        torch.testing.assert_close(values["o"].float(), expected, atol=3e-2 if dtype == torch.bfloat16 else 1e-2, rtol=2e-2)
        torch.testing.assert_close(values["stats"].squeeze(-1), scores.logsumexp(-1), atol=5e-3, rtol=0)

    graph, nodes, workspace = build(record)
    bindings = {nodes[name]: value for name, value in tensors.items()}
    graph.execute(bindings, workspace)
    check(tensors)
    expected = {name: tensors[name].clone() for name in ("o", "stats")}
    handle = cudnn.create_handle()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    cudnn.set_stream(handle, stream.cuda_stream)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            graph.execute(bindings, workspace, handle=handle)
        stream.synchronize()
        with torch.cuda.graph(capture, stream=stream):
            previous = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                graph.execute(bindings, workspace, handle=handle)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
        rebuilt, other_nodes, scratch = build(graph.get_engine_and_knobs_at_index(0))
        rebuilt.execute({other_nodes[name]: value for name, value in tensors.items()}, scratch)
        torch.testing.assert_close(tensors["o"], expected["o"], atol=0, rtol=0)
        fresh = {name: value.clone() for name, value in tensors.items()}
        fresh["v"].mul_(0.25)
        fresh["o"].fill_(float("nan"))
        graph.execute({nodes[name]: value for name, value in fresh.items()}, torch.empty_like(workspace))
        check(fresh)
        # The older capture still owns its original pointers after eager rebind.
        tensors["v"].mul_(0.5)
        tensors["o"].fill_(float("nan"))
        tensors["stats"].fill_(float("nan"))
        workspace.fill_(255)
        capture.replay()
        torch.cuda.synchronize()
        check(tensors)
        torch.testing.assert_close(tensors["stats"], expected["stats"], atol=0, rtol=0)
        _check_retired_record(graph, owner)
    finally:
        torch.cuda.synchronize()
        capture.reset()
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "d,split,packed,owner,cga",
    [(64, 2, False, 20532, 1), (128, 3, False, 20532, 1), (128, 1, True, 20535, 2), (128, 2, True, 20535, 1), (128, 2, True, 20532, 1)],
)
def test_thd_origins_rebind_and_retained_capture(monkeypatch, dtype, d, split, packed, owner, cga):
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as run

    folder = "d128_f16_single_q" if owner == 20532 else "d128_f16_multi_q"
    name = "sdpa_fwd_sm100_d128_single_q" if owner == 20532 else "sdpa_fwd_sm100_d128_multi_q"
    seen = _fixed_loader(monkeypatch, folder)
    checked = []

    def check(graph, *args):
        if checked:
            return
        assert seen and seen[-1].CFG.SPLIT_KV == split
        assert graph.get_engine_and_knobs_at_index(graph._plan_index)[0] == owner
        _check_retired_record(graph, owner)
        checked.append(True)

    if owner == 20535 and split > 1:
        with pytest.raises((ValueError, RuntimeError), match="THD split-KV requires the single-Q implementation"):
            run(d, dtype, "head", True, packed, split, engine_name=name, cga=cga, extra_checks=check)
        assert not seen and not checked, "an unsupported body reached compilation"
        return
    run(d, dtype, "head", True, packed, split, engine_name=name, cga=cga, extra_checks=check)
    assert checked


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("split", [1, 3])
def test_paged_swap_ab_record_and_live_columns(monkeypatch, dtype, split):
    from cudnn.sdpa.fwd import heuristics
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
    import test_sdpa_fwd_paged_sm100 as paged

    seen = _fixed_loader(monkeypatch, "d256_f16_swap_ab")
    monkeypatch.setattr(heuristics, "select_sm100_small_half_geometry", lambda *a, **kw: pytest.fail("fixed record reentered small-head selection"))
    record = (20520, SdpaFwdKnobs(tile_m=16, tile_n=128, cga=1, sched_policy=0, pack_gqa=True, split_kv=split).to_public())
    checked = []

    def check(graph, *args):
        assert graph.get_engine_and_knobs_at_index(graph._plan_index) == record
        assert seen and seen[-1].CFG.N_Q == 16 and seen[-1].COLS == 8
        _check_retired_record(graph, 20520)
        checked.append(True)

    paged._run_graph(
        2,
        8,
        2,
        128,
        16,
        32,
        [497, 512],
        True,
        dtype=dtype,
        stats=True,
        s_q=2,
        pack_gqa=True,
        replay=True,
        record=record,
        extra_checks=check,
    )
    assert checked
