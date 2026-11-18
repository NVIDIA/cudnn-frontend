# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunk-64 KDA against the FP64 recurrence."""

import pytest
import torch
from cudnn.linear_attention import ops
from linear_attention.reference_kda import kda_reference
from linear_attention.reference_gdn import rms_ratio
from linear_attention.cake.test_gdn_cake_det import cake_available

pytestmark = [pytest.mark.L0]


@pytest.fixture(scope="module", autouse=True)
def release_kda_plans():
    yield
    import gc
    from cudnn.linear_attention.ops import kda

    for cache in (kda.fprop_cache, kda.bprop_cache, kda.summary_cache):
        cache.clear()
    gc.collect()


def inputs(lengths, heads):
    torch.manual_seed(8401)
    total = sum(lengths)
    q, k = [torch.nn.functional.normalize(torch.randn(total, heads, 128, device="cuda"), dim=-1).to(torch.bfloat16) for _ in range(2)]
    v = torch.randn_like(q)
    g = -torch.rand(total, heads, 128, device="cuda") * 0.2
    beta = torch.rand(total, heads, device="cuda", dtype=torch.bfloat16)
    leaves = [x.requires_grad_() for x in (q, k, v, g, beta)]
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    return leaves, cu


def forward(leaves, cu):
    return ops.kimi_delta_attention(*leaves, cu, output_final_state=False, checkpoint_every_n_tokens=0, plan_name="kda_cake_det")[0]


@pytest.mark.parametrize("lengths,heads", [([33, 0, 32], 16), ([9], 32), ([9], 64), ([2] * 10, 16), ([2] * 3, 64)])
def test_training(cake_available, lengths, heads):
    leaves, cu = inputs(lengths, heads)
    out = forward(leaves, cu)
    do = torch.randn_like(out)
    grads = torch.autograd.grad(out, leaves, do)
    ref_leaves = [x.detach().double().requires_grad_() for x in leaves]
    ref, _ = kda_reference(*(x.unsqueeze(0) for x in ref_leaves), cu_seqlens=cu)
    ref_grads = torch.autograd.grad(ref, ref_leaves, do.unsqueeze(0).double())
    assert rms_ratio(out, ref[0]) < 0.025
    for name, got, want in zip(("dq", "dk", "dv", "dg", "dbeta"), grads, ref_grads):
        assert rms_ratio(got, want) < 0.035, name
    again = forward(leaves, cu)
    again_grads = torch.autograd.grad(again, leaves, do)
    assert torch.equal(out, again)
    assert all(torch.equal(a, b) for a, b in zip(grads, again_grads))


def test_capture(cake_available):
    leaves, cu = inputs([65], 16)
    forward(leaves, cu)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward(leaves, cu)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = forward(leaves, cu)
    with torch.no_grad():
        leaves[2].mul_(0.5)
    graph.replay()
    torch.testing.assert_close(out, forward(leaves, cu), rtol=0, atol=0)


def test_many_documents(cake_available):
    leaves, cu = inputs([1] * 257, 16)
    out = forward(leaves, cu)
    q, k, v, g, beta = leaves
    want = (q.double() * k.double()).sum(-1, keepdim=True) * v.double() * beta.double().unsqueeze(-1) / 128**0.5
    assert rms_ratio(out, want) < 0.025
