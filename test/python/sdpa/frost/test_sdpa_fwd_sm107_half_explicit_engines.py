# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rubin small/mid half records fix the body through rebuild, rebind and capture."""

import math

import cudnn
import pytest
import torch

from frost_test_utils import requires_dsl, requires_rubin

pytestmark = [pytest.mark.L0, requires_dsl, requires_rubin]


def _fixed_loader(monkeypatch, path):
    from cudnn.sdpa.fwd import api_dsl, config_sm107, heuristics

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed Rubin record reentered main-body selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    monkeypatch.setattr(heuristics, "select_sm107_half_geometry", forbidden)
    monkeypatch.setattr(config_sm107, "sm107_n_chunk_pick", forbidden)
    monkeypatch.setattr(api_dsl, "_load_sm100_kernel_module", forbidden)
    original = api_dsl.load_template
    seen = []

    def load(file, *args, **kwargs):
        module = original(file, *args, **kwargs)
        if str(file).endswith(path):
            seen.append(module)
        return module

    monkeypatch.setattr(api_dsl, "load_template", load)
    return seen


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "owner,folder,dq,dv,sq,sk,cga,tile,split,packed,bias",
    [
        (20559, "sm107/d128_f16_multi_q", 32, 32, 129, 512, 1, 128, 1, False, False),
        (20559, "sm107/d128_f16_multi_q", 64, 64, 129, 512, 2, 128, 1, True, False),
        (20559, "sm107/d128_f16_multi_q", 128, 128, 129, 512, 2, 128, 3, False, False),
        (20560, "sm107/small_f16_batch_pair", 32, 32, 128, 512, 1, 128, 1, False, True),
        (20560, "sm107/small_f16_batch_pair", 64, 64, 256, 256, 1, 128, 1, False, True),
        (20556, "sm107/d192_f16_multi_q", 192, 128, 33, 512, 2, 128, 1, False, False),
        (20557, "sm107/d256_f16_lookahead", 256, 256, 129, 512, 2, 128, 1, False, False),
        (20558, "sm107/d256_f16_swap_ab", 256, 256, 3, 512, 1, 16, 3, True, False),
        (20569, "sm100/d128_f16_single_q", 128, 128, 17, 512, 1, 128, 3, True, False),
    ],
)
def test_public_record_rebuild_rebind_and_capture(monkeypatch, dtype, owner, folder, dq, dv, sq, sk, cga, tile, split, packed, bias):
    from cudnn.engines.manifest import family_for
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    seen = _fixed_loader(monkeypatch, folder + "/kernel.py")
    torch.manual_seed(10731)
    b, h, hk = 4, 8, 8 if bias else 2
    values = {
        "q": torch.randn(b, sq, h, dq, device="cuda", dtype=dtype).transpose(1, 2),
        "k": torch.randn(b, sk, hk, dq, device="cuda", dtype=dtype).transpose(1, 2),
        "v": torch.randn(b, sk, hk, dv, device="cuda", dtype=dtype).transpose(1, 2),
        "o": torch.empty(b, sq, h, dv, device="cuda", dtype=dtype).transpose(1, 2),
        "stats": torch.empty(b, h, sq, 1, device="cuda", dtype=torch.float32),
    }
    if bias:
        values["bias"] = torch.randn(1, h, sq, sk, device="cuda", dtype=dtype) * 0.1
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    record = (owner, SdpaFwdKnobs(tile_m=tile, tile_n=128, cga=cga, sched_policy=0, pack_gqa=packed, split_kv=split, warp_spec_cfg=0).to_public())

    def build(saved):
        graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        nodes = {name: graph.tensor_like(value) for name, value in values.items() if name not in ("o", "stats")}
        nodes["o"], nodes["stats"] = graph.sdpa(
            q=nodes["q"],
            k=nodes["k"],
            v=nodes["v"],
            generate_stats=True,
            attn_scale=dq**-0.5,
            bias=nodes.get("bias"),
        )
        for name in ("o", "stats"):
            nodes[name].set_output(True).set_dim(values[name].shape).set_stride(values[name].stride())
        nodes["stats"].set_data_type(cudnn.data_type.FLOAT)
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plan(*saved)
        graph.select_plan(0)
        graph.check_support()
        graph.build_plans()
        assert family_for(graph).offered_ids()[graph.selected_engine.name] == owner
        assert graph.get_engine_and_knobs_at_index(0) == saved
        metadata = graph._compiled_plans[0]._compiled.implementation_config
        assert (metadata["query_tile"], metadata["cluster_m"], metadata["cta_group"], metadata["split_kv"]) == (tile, cga, cga, split)
        assert seen and seen[-1].__file__.endswith(folder + "/kernel.py")
        if owner in (20559, 20560):
            assert (seen[-1].CFG.N_CHUNK > 0) == (owner == 20560)
        return graph, nodes, torch.empty(max(1, graph.get_workspace_size()), device="cuda", dtype=torch.uint8)

    def check(tensors):
        scores = tensors["q"].float() @ tensors["k"].float().repeat_interleave(h // hk, dim=1).transpose(-1, -2) / math.sqrt(dq)
        if bias:
            scores += tensors["bias"].float()
        expected = scores.softmax(-1) @ tensors["v"].float().repeat_interleave(h // hk, dim=1)
        torch.testing.assert_close(tensors["o"].float(), expected, atol=3e-2 if dtype == torch.bfloat16 else 1e-2, rtol=2e-2)
        torch.testing.assert_close(tensors["stats"].squeeze(-1), scores.logsumexp(-1), atol=5e-3, rtol=0)

    graph, nodes, workspace = build(record)
    bindings = {nodes[name]: value for name, value in values.items()}
    graph.execute(bindings, workspace)
    check(values)
    expected = {name: values[name].clone() for name in ("o", "stats")}
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
        rebuilt.execute({other_nodes[name]: value for name, value in values.items()}, scratch)
        for name in expected:
            torch.testing.assert_close(values[name], expected[name], atol=0, rtol=0)
        fresh = {name: value.clone() for name, value in values.items()}
        fresh["v"].mul_(0.25)
        fresh["o"].fill_(float("nan"))
        graph.execute({nodes[name]: value for name, value in fresh.items()}, torch.empty_like(workspace))
        check(fresh)
        values["v"].mul_(0.5)
        values["o"].fill_(float("nan"))
        values["stats"].fill_(float("nan"))
        workspace.fill_(255)
        capture.replay()
        torch.cuda.synchronize()
        check(values)
        torch.testing.assert_close(values["stats"], expected["stats"], atol=0, rtol=0)
        with pytest.raises((ValueError, RuntimeError), match="owned by|not supported|No execution plans"):
            build((20515, record[1]))
    finally:
        torch.cuda.synchronize()
        capture.reset()
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "name,folder,dim,cga,split,packed,owner",
    [
        ("sdpa_fwd_sm107_d128_multi_q", "sm107/d128_f16_multi_q", 128, 2, 1, True, 20559),
        ("sdpa_fwd_sm107_d256_lookahead", "sm107/d256_f16_lookahead", 128, 1, 1, False, 20557),
        ("sdpa_fwd_sm107_d192_multi_q", "sm107/d192_f16_multi_q", (192, 128), 2, 1, False, 20556),
        ("sdpa_fwd_sm107_shared_single_q", "sm100/d128_f16_single_q", (192, 128), 1, 1, False, 20569),
        ("sdpa_fwd_sm107_shared_single_q", "sm100/d128_f16_single_q", (192, 128), 1, 3, False, 20569),
        ("sdpa_fwd_sm107_shared_single_q", "sm100/d128_f16_single_q", 128, 1, 3, False, 20569),
        ("sdpa_fwd_sm107_d256_lookahead", "sm107/d256_f16_lookahead", 256, 2, 1, False, 20557),
        ("sdpa_fwd_sm107_shared_d256_lookahead", "sm100/d256_f16_lookahead", 256, 2, 3, False, 20570),
    ],
)
def test_thd_origins_rebind_and_retained_capture(monkeypatch, dtype, name, folder, dim, cga, split, packed, owner):
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as run

    seen = _fixed_loader(monkeypatch, folder + "/kernel.py")
    checked = []

    def check(graph, *args):
        assert seen and seen[-1].CFG.SPLIT_KV == split
        assert graph.get_engine_and_knobs_at_index(graph._plan_index)[0] == owner
        checked.append(True)

    run(dim, dtype, "head", True, packed, split, engine_name=name, cga=cga, extra_checks=check)
    assert checked
