# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Single-process oracle checks (no GPUs / no dist required).

1. ReferenceMoE forward matches a naive per-token/per-k double loop (fp64).
2. torch.autograd.gradcheck on the full reference (x, w13, w2, topk_weights)
   in fp64 on tiny sizes.
3. Edge cases: duplicate expert in a token's top-k, experts with zero tokens.
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pt.reference import ReferenceMoE  # noqa: E402


def naive_moe(x, topk_ids, topk_weights, w13, w2):
    """Straight-line per-token loop; the most obviously-correct spelling."""
    num_tokens, top_k = topk_ids.shape
    intermediate = w2.shape[2]
    out = torch.zeros(num_tokens, x.shape[1], dtype=torch.float64)
    for t in range(num_tokens):
        for k in range(top_k):
            e = int(topk_ids[t, k])
            fc1 = x[t].double() @ w13[e].double().t()
            linear, gate = fc1[:intermediate], fc1[intermediate:]
            y = (F.silu(gate) * linear) @ w2[e].double().t()
            out[t] += topk_weights[t, k].double() * y
    return out.to(x.dtype)


def make_problem(seed=0, num_tokens=13, top_k=3, num_experts=5, hidden=8, intermediate=11):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(num_tokens, hidden, generator=gen, dtype=torch.float64)
    scores = torch.randn(num_tokens, num_experts, generator=gen)
    topk_weights, topk_ids = torch.topk(torch.softmax(scores, dim=-1), top_k)
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).double()
    w13 = torch.randn(num_experts, 2 * intermediate, hidden, generator=gen, dtype=torch.float64)
    w2 = torch.randn(num_experts, hidden, intermediate, generator=gen, dtype=torch.float64)
    return x, topk_ids, topk_weights, w13, w2


def test_reference_matches_naive():
    x, topk_ids, topk_weights, w13, w2 = make_problem()
    ref = ReferenceMoE(w13.clone(), w2.clone())
    out = ref(x, topk_ids, topk_weights)
    expected = naive_moe(x, topk_ids, topk_weights, w13, w2)
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)


def test_reference_duplicate_topk_and_empty_expert():
    x, _, topk_weights, w13, w2 = make_problem(num_experts=6)
    # Every token picks expert 1 twice and expert 2 once; experts 0,3,4,5 idle.
    topk_ids = torch.tensor([[1, 1, 2]]).repeat(x.shape[0], 1)
    ref = ReferenceMoE(w13.clone(), w2.clone())
    out = ref(x, topk_ids, topk_weights)
    expected = naive_moe(x, topk_ids, topk_weights, w13, w2)
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)


def test_reference_gradcheck():
    x, topk_ids, topk_weights, w13, w2 = make_problem(
        num_tokens=5, top_k=2, num_experts=4, hidden=6, intermediate=7
    )
    x = x.requires_grad_()
    topk_weights = topk_weights.requires_grad_()
    w13 = w13.requires_grad_()
    w2 = w2.requires_grad_()

    def fn(x_, tw_, w13_, w2_):
        return ReferenceMoE.forward(
            _make_module(w13_, w2_), x_, topk_ids, tw_
        )

    def _make_module(w13_, w2_):
        # Bypass nn.Parameter wrapping so gradcheck's raw tensors stay in
        # the graph; ReferenceMoE.forward only reads self.w13 / self.w2.
        m = ReferenceMoE.__new__(ReferenceMoE)
        torch.nn.Module.__init__(m)
        object.__setattr__(m, "w13", w13_)
        object.__setattr__(m, "w2", w2_)
        return m

    assert torch.autograd.gradcheck(
        fn, (x, topk_weights, w13, w2), eps=1e-6, atol=1e-8, rtol=1e-6
    )


def test_grad_flows_to_all_inputs():
    x, topk_ids, topk_weights, w13, w2 = make_problem()
    x = x.requires_grad_()
    topk_weights = topk_weights.requires_grad_()
    ref = ReferenceMoE(w13.clone(), w2.clone())
    out = ref(x, topk_ids, topk_weights)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert topk_weights.grad is not None and torch.isfinite(topk_weights.grad).all()
    assert ref.w13.grad is not None and torch.isfinite(ref.w13.grad).all()
    assert ref.w2.grad is not None and torch.isfinite(ref.w2.grad).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
