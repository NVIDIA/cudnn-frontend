# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM80 packed-THD forward through the graph API (issue #377)."""

from __future__ import annotations

import math

import pytest
import torch

import cudnn
from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa.fwd import engines as fwd_engines
from frost_test_utils import select_engine


_SM80 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="needs an SM80 (A100) GPU",
)
pytestmark = [pytest.mark.L0, _SM80]
_ENGINE = "sdpa_fwd_prefill_sm80"


def _make_graph(lens_q, lens_kv, *, h=2, d=128, dtype=torch.float16):
    b = len(lens_q)
    assert b == len(lens_kv)
    dev = "cuda"
    sq, skv = max(lens_q), max(lens_kv)
    tq, tkv = sum(lens_q), sum(lens_kv)
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    vp = {}
    cu_q = [0]
    cu_kv = [0]
    for nq, nkv in zip(lens_q, lens_kv, strict=True):
        cu_q.append(cu_q[-1] + nq)
        cu_kv.append(cu_kv[-1] + nkv)

    def port(name, smax, total, cu):
        stride = (smax * h * d, d, h * d, 1)
        node = graph.tensor(name=name, dim=(b, h, smax, d), stride=stride, data_type=io)
        offset = graph.tensor(name=f"{name}_ro", dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        node.set_ragged_offset(offset)
        vp[offset] = (torch.tensor(cu, dtype=torch.int64, device=dev) * h * d).view(b + 1, 1, 1, 1)
        vp[node] = torch.randn(1, total, h, d, dtype=dtype, device=dev)
        return node

    q = port("q", sq, tq, cu_q)
    q_ro_buf = vp[next(node for node in vp if node.get_name() == "q_ro")]
    k = port("k", skv, tkv, cu_kv)
    v = port("v", skv, tkv, cu_kv)
    q_len = graph.tensor(name="seq_len_q", dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
    kv_len = graph.tensor(name="seq_len_kv", dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
    vp[q_len] = torch.tensor(lens_q, dtype=torch.int32, device=dev).view(b, 1, 1, 1)
    vp[kv_len] = torch.tensor(lens_kv, dtype=torch.int32, device=dev).view(b, 1, 1, 1)
    o, stats = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        attn_scale=1 / math.sqrt(d),
        generate_stats=True,
        use_padding_mask=True,
        seq_len_q=q_len,
        seq_len_kv=kv_len,
        max_total_seq_len_q=tq,
        max_total_seq_len_kv=tkv,
    )
    o.set_output(True).set_data_type(io).set_dim((b, h, sq, d)).set_stride((sq * h * d, d, h * d, 1))
    o_ro = graph.tensor(name="o_ro", dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
    o.set_ragged_offset(o_ro)
    vp[o_ro] = q_ro_buf
    vp[o] = torch.full((1, tq, h, d), float("nan"), dtype=dtype, device=dev)

    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    stats.set_dim((b, h, sq, 1)).set_stride((h * tq, tq, 1, 1))
    stats_ro = graph.tensor(name="stats_ro", dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
    stats.set_ragged_offset(stats_ro)
    vp[stats_ro] = torch.tensor(cu_q, dtype=torch.int64, device=dev).view(b + 1, 1, 1, 1)
    vp[stats] = torch.full((1, h, tq), float("nan"), dtype=torch.float32, device=dev)
    return graph, vp, (q, k, v, o, stats), (cu_q, cu_kv)


def test_graph_thd_forward_unequal_lengths():
    torch.manual_seed(7)
    graph, vp, nodes, (cu_q, cu_kv) = _make_graph((144, 96), (160, 128))
    q, k, v, o, stats = nodes
    facts = ga.analyze(graph)
    assert facts is not None and facts.thd and facts.packed_layout
    caps = next(spec.capabilities for spec in fwd_engines.ENGINE_SPECS if spec.name == _ENGINE)
    assert fwd_engines.mismatch(caps, facts) is None

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    select_engine(graph, _ENGINE)
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    graph.execute(vp, workspace)
    torch.cuda.synchronize()

    scale = 1 / math.sqrt(128)
    for batch in range(2):
        q_slice = vp[q][0, cu_q[batch] : cu_q[batch + 1]].transpose(0, 1).double()
        k_slice = vp[k][0, cu_kv[batch] : cu_kv[batch + 1]].transpose(0, 1).double()
        v_slice = vp[v][0, cu_kv[batch] : cu_kv[batch + 1]].transpose(0, 1).double()
        scores = (q_slice @ k_slice.transpose(-1, -2)) * scale
        ref_o = (scores.softmax(-1) @ v_slice).transpose(0, 1)
        ref_lse = scores.logsumexp(-1)
        got_o = vp[o][0, cu_q[batch] : cu_q[batch + 1]]
        got_lse = vp[stats][0, :, cu_q[batch] : cu_q[batch + 1]]
        torch.testing.assert_close(got_o.float(), ref_o.float(), rtol=1e-2, atol=4e-3)
        torch.testing.assert_close(got_lse, ref_lse.float(), rtol=3e-2, atol=5e-2)
