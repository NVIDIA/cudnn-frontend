# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunk-64 GDN parity and graph replay contracts."""

import pytest
import torch

from cudnn.linear_attention import ops
from linear_attention.reference_gdn import gdn_reference, rms_ratio

pytestmark = [pytest.mark.L0]


@pytest.fixture(scope="module", autouse=True)
def release_gdn_plans():
    yield
    import gc
    from cudnn.linear_attention.ops import gdn

    for cache in (gdn.fprop_cache, gdn.bprop_cache, gdn.summary_cache):
        cache.clear()
    gc.collect()


@pytest.fixture(autouse=True)
def cake_available(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires SM100")
    from cudnn._cake.loader import check_toolkit

    try:
        check_toolkit("sm_100a", "gdn_cake_det")
    except (RuntimeError, NotImplementedError) as error:
        pytest.skip(str(error))
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")


def inputs(lengths, h, hv, normalize):
    torch.manual_seed(4801)
    t = sum(lengths)
    q = torch.randn(t, h, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    if not normalize:
        q = torch.nn.functional.normalize(q.float(), dim=-1).to(q.dtype)
        k = torch.nn.functional.normalize(k.float(), dim=-1).to(k.dtype)
    v = torch.randn(t, hv, 128, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(t, hv, device="cuda") * 0.2
    beta = torch.rand(t, hv, device="cuda")
    leaves = [x.requires_grad_() for x in (q, k, v, g, beta)]
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    return leaves, cu


def forward(leaves, cu, normalize):
    return ops.gated_delta_net(*leaves, cu, output_final_state=False, use_qk_l2norm_in_kernel=normalize, checkpoint_every_n_tokens=0, plan_name="gdn_cake_det")[
        0
    ]


@pytest.mark.parametrize("lengths,h,hv,normalize", [([64, 64], 2, 2, False), ([31, 0, 66], 2, 4, True)])
def test_training(lengths, h, hv, normalize):
    leaves, cu = inputs(lengths, h, hv, normalize)
    out = forward(leaves, cu, normalize)
    do = torch.randn_like(out)
    grads = torch.autograd.grad(out, leaves, do)
    ref_leaves = [x.detach().double().requires_grad_() for x in leaves]
    q, k, v, g, beta = ref_leaves
    if normalize:
        q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    ref, _ = gdn_reference(*(x.unsqueeze(0) for x in (q, k, v, g, beta)), cu_seqlens=cu)
    ref_grads = torch.autograd.grad(ref, ref_leaves, do.unsqueeze(0).double())
    assert rms_ratio(out, ref[0]) < 0.025
    for name, got, want in zip(("dq", "dk", "dv", "dg", "dbeta"), grads, ref_grads):
        assert rms_ratio(got, want) < 0.035, name
    again = forward(leaves, cu, normalize)
    again_grads = torch.autograd.grad(again, leaves, do)
    assert torch.equal(out, again)
    assert all(torch.equal(a, b) for a, b in zip(grads, again_grads))


def test_capture_replay_changes_inputs():
    leaves, cu = inputs([65, 7], 2, 2, True)
    forward(leaves, cu, True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward(leaves, cu, True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = forward(leaves, cu, True)
    with torch.no_grad():
        leaves[2].mul_(0.5)
    graph.replay()
    want = forward(leaves, cu, True)
    torch.testing.assert_close(out, want, rtol=0, atol=0)


def test_executor_uses_only_caller_storage(monkeypatch):
    from cudnn._cake.loader import no_build
    from cudnn.frost.workspace import Workspace
    from cudnn.linear_attention.ops.gdn import fprop_cache

    leaves, cu = inputs([65, 7], 2, 2, True)
    forward(leaves, cu, True)
    compiled = next(
        plan.compiled
        for graph, _ in fprop_cache.values()
        for plan in graph._compiled_plans.values()
        if getattr(getattr(plan, "compiled", None), "plan_name", None) == "gdn_cake_det" and plan.compiled.t == 72
    )
    out = torch.empty_like(leaves[2])
    operands = dict(zip(("q", "k", "v", "g", "beta"), leaves), cu_seqlens=cu, O=out)
    scratch = torch.empty(compiled.workspace_size, dtype=torch.uint8, device="cuda")
    workspace = Workspace(scratch, compiled.workspace_size, compiled.plan_name, device=0)
    views = [operands[name] for name in compiled.names]
    stream = torch.cuda.current_stream().cuda_stream

    def forbidden(*args, **kwargs):
        raise AssertionError("executor allocated or synchronized")

    with monkeypatch.context() as patch, no_build():
        for name in ("empty", "empty_like", "zeros", "zeros_like", "ones", "ones_like"):
            patch.setattr(torch, name, forbidden)
        patch.setattr(torch.cuda, "synchronize", forbidden)
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "cpu", forbidden)
        compiled.run(views, workspace, stream)
    torch.testing.assert_close(out, forward(leaves, cu, True), rtol=0, atol=0)

    launches = []
    monkeypatch.setattr(compiled, "entries", {name: lambda *args: launches.append(args) for name in compiled.entries})
    operands["O"] = leaves[2]
    with pytest.raises(ValueError, match="overlaps"):
        compiled.run([operands[name] for name in compiled.names], workspace, stream)
    assert launches == []
