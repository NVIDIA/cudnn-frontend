# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TransformerEngine's flex-attention graph: sdpa(score_mod=...) on a shape every
FROST prefill row admits WITHOUT the modifier (b2 h4 s16 d64, sbhd / bshd physical strides, stats
requested) is served by the cuDNN backend, and O carries the modifier. The score_mod is TE's
post-scale bias ``score + (row - col)``: dropped, it leaves 38 % of O outside a 5e-2 tolerance.
Rows run through the python engine router first, so a graph the router mislabels is executed by a
kernel that ignores the modifier -- which is what the analyzer's `fn`-only read of the callback did."""

import math

import cudnn
import pytest
import torch

from test_utils import torch_fork_set_rng


def _post_scale_bias(graph, score):
    row = graph.gen_index(input=score, axis=2)
    row.set_data_type(cudnn.data_type.INT32)
    col = graph.gen_index(input=score, axis=3)
    col.set_data_type(cudnn.data_type.INT32)
    bias = graph.sub(a=row, b=col, compute_data_type=cudnn.data_type.FLOAT)
    bias.set_data_type(cudnn.data_type.FLOAT)
    return graph.add(a=score, b=bias, compute_data_type=cudnn.data_type.FLOAT)


def _bhsd_dim_stride(t, layout):
    """TE declares BHSD dims with the strides of its physical sbhd / bshd buffer."""
    if layout == "sbhd":
        return (t.shape[1], t.shape[2], t.shape[0], t.shape[3]), (t.stride(1), t.stride(2), t.stride(0), t.stride(3))
    return (t.shape[0], t.shape[2], t.shape[1], t.shape[3]), (t.stride(0), t.stride(2), t.stride(1), t.stride(3))


def _reference(q, k, v, layout, scale):
    perm = (1, 2, 0, 3) if layout == "sbhd" else (0, 2, 1, 3)
    qb, kb, vb = (x.permute(*perm).float() for x in (q, k, v))
    s = torch.matmul(qb, kb.transpose(-2, -1)) * scale
    row = torch.arange(s.shape[-2], device=s.device, dtype=torch.float32).view(-1, 1)
    col = torch.arange(s.shape[-1], device=s.device, dtype=torch.float32).view(1, -1)
    o = torch.matmul(torch.softmax(s + (row - col), dim=-1), vb)
    return o.permute(2, 0, 1, 3) if layout == "sbhd" else o.permute(0, 2, 1, 3)


@pytest.mark.L0
@pytest.mark.parametrize("layout", ["sbhd", "bshd"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@torch_fork_set_rng(seed=0)
def test_te_score_mod_graph_is_served_by_the_backend(cudnn_handle, layout, dtype):
    if cudnn.backend_version() < 90600:
        pytest.skip("score_mod graphs need cuDNN 9.6+")
    b, h, s, d = 2, 4, 16, 64
    shape = (s, b, h, d) if layout == "sbhd" else (b, s, h, d)
    q, k, v = (0.1 * torch.randn(shape, dtype=dtype, device="cuda") for _ in range(3))
    o_gpu = torch.full_like(q, float("nan"))
    stats_gpu = torch.empty((b, h, s, 1), dtype=torch.float32, device="cuda")
    scale = 1.0 / math.sqrt(d)

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )
    gq, gk, gv = (graph.tensor(dim=dim, stride=stride, data_type=dtype) for dim, stride in (_bhsd_dim_stride(t, layout) for t in (q, k, v)))
    o, stats = graph.sdpa(name="te_score_mod_sdpa", q=gq, k=gk, v=gv, generate_stats=True, attn_scale=scale, use_causal_mask=False, score_mod=_post_scale_bias)
    o_dim, o_stride = _bhsd_dim_stride(o_gpu, layout)
    o.set_output(True).set_dim(o_dim).set_stride(o_stride)
    stats.set_output(True).set_dim(stats_gpu.size()).set_stride(stats_gpu.stride()).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    try:
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"TEST WAIVED: unsupported graph. {e}")
    graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)
    # No python engine implements score_mod: the selected plan must be the backend's.
    assert getattr(graph, "selected_engine", None) is None, f"a python engine admitted a score_mod graph: {graph.selected_engine}"

    workspace = torch.empty(max(graph.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    # The variant pack binds TE's physical buffers (the graph tensors carry their strides).
    graph.execute({gq: q, gk: k, gv: v, o: o_gpu, stats: stats_gpu}, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    ref = _reference(q, k, v, layout, scale)
    assert not torch.isnan(o_gpu).any()
    torch.testing.assert_close(o_gpu.float(), ref, atol=5e-2, rtol=5e-2)
