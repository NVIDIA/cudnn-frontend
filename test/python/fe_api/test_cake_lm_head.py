# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Cake LM-head logp and gradients."""

import pytest
import torch
from cudnn.ops import LmHeadLogprobPlan

pytestmark = [pytest.mark.L0]


@pytest.fixture
def available():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires SM100")
    from cudnn._cake.loader import check_toolkit

    try:
        check_toolkit("sm_100a", "LmHeadLogprobPlan")
    except (RuntimeError, NotImplementedError) as error:
        pytest.skip(str(error))


def case(rows, hidden, vocab):
    torch.manual_seed(8127)
    X = (torch.randn(rows, hidden, device="cuda") * 0.1).to(torch.bfloat16)
    W = (torch.randn(vocab, hidden, device="cuda") * 0.1).to(torch.bfloat16)
    labels = torch.randint(vocab, (rows,), device="cuda")
    labels[::3] = -100
    logp = torch.empty(rows, device="cuda")
    lse = torch.empty_like(logp)
    dlogp = torch.randn_like(logp)
    dX = torch.empty_like(X, dtype=torch.float32)
    dW = torch.empty_like(W, dtype=torch.float32)
    plan = LmHeadLogprobPlan(rows, hidden, vocab)
    ws = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device="cuda")
    return plan, (X, W, labels, logp, lse, dlogp, dX, dW, ws)


@pytest.mark.parametrize("rows,hidden,vocab", [(17, 512, 512), (257, 1024, 768)])
def test_parity_and_repeat(available, rows, hidden, vocab):
    plan, (X, W, labels, logp, lse, dlogp, dX, dW, ws) = case(rows, hidden, vocab)
    plan.forward(X, W, labels, logp, lse, ws)
    plan.backward(X, W, labels, lse, dlogp, dX, dW, ws)
    xr = X.double().requires_grad_()
    wr = W.double().requires_grad_()
    logits = xr @ wr.T
    ref = -torch.nn.functional.cross_entropy(logits, labels, reduction="none", ignore_index=-100)
    dx, dw = torch.autograd.grad(ref, (xr, wr), dlogp.double())
    torch.testing.assert_close(logp, ref.float(), atol=0.01, rtol=0.003)
    for got, want in ((dX, dx), (dW, dw)):
        assert ((got.double() - want).norm() / want.norm()).item() < 0.015
    torch.testing.assert_close(lse, logits.logsumexp(-1).float(), atol=0.01, rtol=0.003)
    saved = [t.clone() for t in (logp, lse, dX, dW)]
    ws.fill_(255)
    plan.forward(X, W, labels, logp, lse, ws)
    plan.backward(X, W, labels, lse, dlogp, dX, dW, ws)
    assert all(torch.equal(a, b) for a, b in zip(saved, (logp, lse, dX, dW)))


def test_capture_and_contract(available, monkeypatch):
    from cudnn._cake.loader import no_build

    plan, (X, W, labels, logp, lse, dlogp, dX, dW, ws) = case(17, 512, 512)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    plan.forward(X, W, labels, logp, lse, ws, stream=stream.cuda_stream)
    plan.backward(X, W, labels, lse, dlogp, dX, dW, ws, stream=stream.cuda_stream)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()

    def forbidden(*args, **kwargs):
        raise AssertionError("execute allocated or synchronized")

    with torch.cuda.graph(graph, stream=stream):
        with monkeypatch.context() as patch, no_build():
            patch.setattr(torch, "empty", forbidden)
            patch.setattr(torch.cuda, "synchronize", forbidden)
            patch.setattr(torch.Tensor, "cpu", forbidden)
            patch.setattr(torch.Tensor, "item", forbidden)
            plan.forward(X, W, labels, logp, lse, ws, stream=stream.cuda_stream)
            plan.backward(X, W, labels, lse, dlogp, dX, dW, ws, stream=stream.cuda_stream)
    X.mul_(0.5)
    graph.replay()
    actual = [t.clone() for t in (logp, lse, dX, dW)]
    plan.forward(X, W, labels, logp, lse, ws)
    plan.backward(X, W, labels, lse, dlogp, dX, dW, ws)
    assert all(torch.equal(a, b) for a, b in zip(actual, (logp, lse, dX, dW)))
    calls = []
    monkeypatch.setattr(plan, "launches", {key: (lambda *args: calls.append(args), template) for key, (_, template) in plan.launches.items()})
    with pytest.raises(ValueError, match="overlaps"):
        plan.forward(X, W, labels, logp, logp, ws)
    with pytest.raises(ValueError, match="workspace"):
        plan.forward(X, W, labels, logp, lse, ws[:8])
    with pytest.raises(ValueError, match="dw_acc must have shape"):
        plan.backward(X, W, labels, lse, dlogp, dX, dW[:1], ws)
    assert calls == []


@pytest.mark.parametrize("dims", [(0, 512, 512), (4097, 512, 512), (17, 256, 512), (17, 512, 255), (17, 512, 2**31)])
def test_unsupported(dims):
    with pytest.raises(NotImplementedError):
        LmHeadLogprobPlan.check_support(*dims)


def test_all_ignored(available):
    plan, (X, W, labels, logp, lse, dlogp, dX, dW, ws) = case(17, 512, 512)
    labels.fill_(-100)
    dX.fill_(float("nan"))
    dW.fill_(float("nan"))
    plan.forward(X, W, labels, logp, lse, ws)
    plan.backward(X, W, labels, lse, dlogp, dX, dW, ws)
    assert torch.count_nonzero(logp) == 0
    assert torch.count_nonzero(dX) == 0
    assert torch.count_nonzero(dW) == 0
