# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Half D192 public records fix the loaded body through rebuild and capture."""

import json
import math

import pytest
import torch

import cudnn
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


def _fixed_loader(monkeypatch, folder):
    from cudnn.sdpa.fwd import api_dsl, heuristics

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed implementation reentered legacy body selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    monkeypatch.setattr(heuristics, "select_sm100_d192_half_geometry", forbidden)
    monkeypatch.setattr(api_dsl, "_load_sm100_kernel_module", forbidden)
    original = api_dsl.load_template
    seen = []

    def load(path, *args, **kwargs):
        module = original(path, *args, **kwargs)
        if str(path).endswith(f"/{folder}/kernel.py"):
            seen.append(module)
        return module

    monkeypatch.setattr(api_dsl, "load_template", load)
    return seen


def _check_retired_record(graph, expected_id):
    import cudnn._plan_serdes as ps

    blob = graph.serialize(as_bytes=True, include_kernels=False)
    start = len(ps.MAGIC) + ps._LEN.size
    (length,) = ps._LEN.unpack_from(blob, len(ps.MAGIC))
    header = json.loads(blob[start : start + length])
    header["plan"].update(engine_id=20511, engine_name="sdpa_fwd_prefill_sm100", knobs=[])
    encoded = json.dumps(header).encode()
    target = cudnn.pygraph()
    with pytest.raises((ValueError, RuntimeError), match="20511|owned by|not supported|No execution plans"):
        target.deserialize(ps.MAGIC + ps._LEN.pack(len(encoded)) + encoded)
    assert not target._nodes and not target._planning_done
    target.deserialize(blob)
    assert target.get_engine_and_knobs_at_index(target._plan_index) == graph.get_engine_and_knobs_at_index(graph._plan_index)
    assert target.get_engine_and_knobs_at_index(target._plan_index)[0] == expected_id


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("dims", [(192, 128), (176, 112)], ids=["native", "envelope"])
@pytest.mark.parametrize("cga,split,packed,causal", [(1, 1, False, False), (1, 1, True, True), (2, 1, True, False), (2, 4, True, True)])
def test_multi_q_record_rebuild_rebind_and_capture(monkeypatch, dtype, cga, split, packed, causal, dims):
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    seen = _fixed_loader(monkeypatch, "d192_f16_multi_q")
    torch.manual_seed(19231)
    b, h, hk, sq, sk = 2, 8, 2, 17, 512
    dq, dv = dims
    tensors = {
        "q": torch.randn(b, sq, h, dq, device="cuda", dtype=dtype).transpose(1, 2),
        "k": torch.randn(b, sk, hk, dq, device="cuda", dtype=dtype).transpose(1, 2),
        "v": torch.randn(b, sk, hk, dv, device="cuda", dtype=dtype).transpose(1, 2),
        "o": torch.empty(b, sq, h, dv, device="cuda", dtype=dtype).transpose(1, 2),
        "stats": torch.empty(b, h, sq, 1, device="cuda", dtype=torch.float32),
    }
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    record = (20531, SdpaFwdKnobs(tile_m=128, tile_n=128, cga=cga, sched_policy=1, pack_gqa=packed, split_kv=split).to_public())

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
        assert graph.selected_engine.name == "sdpa_fwd_sm100_d192_multi_q"
        assert graph.get_engine_and_knobs_at_index(0) == saved
        compiled = graph._compiled_plans[0]._compiled
        assert compiled.kernel_template == "prefill_d192_d128_f16"
        metadata = compiled.implementation_config
        assert (metadata["query_tile"], metadata["cluster_m"], metadata["cta_group"], metadata["split_kv"]) == (128, cga, cga, split)
        assert seen and seen[-1].CFG.TILES_Q == 2
        assert seen[-1].CFG.K_SUB == (3 if cga == 1 else 1)
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
        _check_retired_record(graph, 20531)
    finally:
        torch.cuda.synchronize()
        capture.reset()
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("split", [1, 3])
def test_single_q_record_origins_and_retained_capture(monkeypatch, dtype, split):
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as run

    seen = _fixed_loader(monkeypatch, "d128_f16_single_q")
    checked = []

    def record_check(graph, *args):
        if checked:
            return
        assert seen and seen[-1].CFG.TILES_Q == 1
        assert seen[-1].CFG.TILE_K == 192 and seen[-1].CFG.STAGES_KV == 2
        assert seen[-1].CFG.SPLIT_KV == split
        assert graph.get_engine_and_knobs_at_index(graph._plan_index)[0] == 20532
        assert graph._compiled_plans[graph._plan_index]._compiled.kernel_template == "decode_d128_f16"
        _check_retired_record(graph, 20532)
        checked.append(True)

    # Independent physical origins and packed splits retain their exact-width
    # admission. The padded envelope is covered with compact THD below.
    run((192, 128), dtype, "head", True, False, split, engine_name="sdpa_fwd_sm100_d128_single_q", cga=1, extra_checks=record_check)
    assert checked


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("cga,engine_id,folder", [(1, 20532, "d128_f16_single_q"), (2, 20531, "d192_f16_multi_q")])
def test_envelope_compact_thd_record_rebind_and_capture(monkeypatch, dtype, cga, engine_id, folder):
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    seen = _fixed_loader(monkeypatch, folder)
    torch.manual_seed(176112)
    b, h, hk, sq, sk, dq, dv = 2, 4, 2, 129, 192, 176, 112
    q_lengths, k_lengths = (129, 17), (159, 35)
    q_starts, k_starts = (0, 129), (0, 159)
    values = {
        "q": torch.randn(b * sq, h, dq, device="cuda", dtype=dtype),
        "k": torch.randn(b * sk, hk, dq, device="cuda", dtype=dtype),
        "v": torch.randn(b * sk, hk, dv, device="cuda", dtype=dtype),
        "o": torch.full((b * sq, h, dv), float("nan"), device="cuda", dtype=dtype),
        "stats": torch.full((b * sq, h), float("nan"), device="cuda", dtype=torch.float32),
        "ql": torch.tensor(q_lengths, device="cuda", dtype=torch.int32).view(b, 1, 1, 1),
        "kl": torch.tensor(k_lengths, device="cuda", dtype=torch.int32).view(b, 1, 1, 1),
    }
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    nodes = {name: graph.tensor_like(values[name]) for name in ("ql", "kl")}
    for name, heads, width, maximum, starts, lengths in (
        ("q", h, dq, sq, q_starts, q_lengths),
        ("k", hk, dq, sk, k_starts, k_lengths),
        ("v", hk, dv, sk, k_starts, k_lengths),
        ("o", h, dv, sq, q_starts, q_lengths),
        ("stats", h, 1, sq, q_starts, q_lengths),
    ):
        values["ro_" + name] = (torch.tensor((*starts, sum(lengths)), device="cuda", dtype=torch.int64) * heads * width).view(b + 1, 1, 1, 1)
        nodes["ro_" + name] = graph.tensor_like(values["ro_" + name])
        if name in ("q", "k", "v"):
            nodes[name] = graph.tensor(dim=(b, heads, maximum, width), stride=(maximum * heads * width, width, heads * width, 1), data_type=io)
            nodes[name].set_ragged_offset(nodes["ro_" + name])
    nodes["o"], nodes["stats"] = graph.sdpa(
        q=nodes["q"],
        k=nodes["k"],
        v=nodes["v"],
        generate_stats=True,
        attn_scale=dq**-0.5,
        use_padding_mask=True,
        seq_len_q=nodes["ql"],
        seq_len_kv=nodes["kl"],
        max_total_seq_len_q=sum(q_lengths),
        max_total_seq_len_kv=sum(k_lengths),
    )
    nodes["o"].set_output(True).set_dim((b, h, sq, dv)).set_stride((sq * h * dv, dv, h * dv, 1)).set_ragged_offset(nodes["ro_o"])
    nodes["stats"].set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim((b, h, sq, 1)).set_stride((sq * h, 1, h, 1)).set_ragged_offset(
        nodes["ro_stats"]
    )
    graph.validate()
    graph.build_operation_graph()
    record = engine_id, SdpaFwdKnobs(tile_m=128, tile_n=128, cga=cga, pack_gqa=False, split_kv=1, sched_policy=0).to_public()
    graph.create_execution_plan(*record)
    graph.select_plan(0)
    graph.check_support()
    graph.build_plans()
    assert graph.get_engine_and_knobs_at_index(0) == record
    assert seen and seen[-1].CFG.TILES_Q == cga
    assert seen[-1].CFG.TILE_K == 192
    workspace = torch.empty(max(1, graph.get_workspace_size()), device="cuda", dtype=torch.uint8)

    def execute(bound, **kwargs):
        graph.execute({nodes[name]: tensor for name, tensor in bound.items()}, workspace, **kwargs)

    def check(bound):
        for qi, ki, nq, nk in zip(q_starts, k_starts, q_lengths, k_lengths):
            q = bound["q"][qi : qi + nq].float().transpose(0, 1)
            k = bound["k"][ki : ki + nk].float().transpose(0, 1).repeat_interleave(h // hk, dim=0)
            v = bound["v"][ki : ki + nk].float().transpose(0, 1).repeat_interleave(h // hk, dim=0)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(dq)
            torch.testing.assert_close(
                bound["o"][qi : qi + nq].float(), (scores.softmax(-1) @ v).transpose(0, 1), atol=3e-2 if dtype == torch.bfloat16 else 1e-2, rtol=2e-2
            )
            torch.testing.assert_close(bound["stats"][qi : qi + nq], scores.logsumexp(-1).transpose(0, 1), atol=5e-3, rtol=0)

    execute(values)
    check(values)
    handle, stream, capture = cudnn.create_handle(), torch.cuda.Stream(), torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    cudnn.set_stream(handle, stream.cuda_stream)
    try:
        with torch.cuda.stream(stream):
            execute(values, handle=handle)
        stream.synchronize()
        with torch.cuda.graph(capture, stream=stream):
            execute(values, handle=handle)
        fresh = {name: value.clone() for name, value in values.items()}
        fresh["v"].mul_(0.25)
        execute(fresh)
        check(fresh)
        values["v"].mul_(0.5)
        values["o"].fill_(float("nan"))
        values["stats"].fill_(float("nan"))
        capture.replay()
        torch.cuda.synchronize()
        check(values)
        _check_retired_record(graph, engine_id)
    finally:
        torch.cuda.synchronize()
        capture.reset()
        cudnn.destroy_handle(handle)
