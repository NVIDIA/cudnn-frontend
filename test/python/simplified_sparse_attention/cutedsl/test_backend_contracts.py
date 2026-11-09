# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU autograd contracts; CUDA attention math is covered by GPU tests.

Only the kernels are replaced. The public entry point and its autograd Function
execute normally, including the packed-row mask in backward. A simple linear
kernel makes its derivative independently checkable with ordinary PyTorch.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import torch

from cudnn.simplified_sparse_attention.cutedsl import backend

pytestmark = pytest.mark.L0


@pytest.fixture
def mock_kernels(monkeypatch):
    def forward(q, k, v, qi, ki, layout, topk):
        output = (q + k + 2 * v).masked_fill(~layout.query_valid[:, None, None], 0)
        lengths = torch.ones(q.size(0), dtype=torch.int32)
        return (
            output,
            torch.zeros(q.shape[:2]),
            torch.zeros((q.size(0), topk)),
            torch.zeros(q.shape[:2]),
            torch.ones(q.shape[:2]),
            torch.zeros((q.size(0), topk), dtype=torch.int32),
            lengths,
            lengths,
            torch.zeros((q.size(0), topk)),
            layout,
        )

    def auxiliary(qi, ki, indices, lengths, predict, teacher, loss, dqi, dki, *args, **kwargs):
        loss.zero_()
        dqi.zero_()
        dki.zero_()

    def backward(q, k, v, output, dout, lse, indices, lengths, dq, dk, dv):
        assert dout.is_contiguous()
        dq.copy_(dout)
        dk.copy_(dout.sum(dim=1, keepdim=True))
        dv.copy_(2 * dout.sum(dim=1, keepdim=True))

    modules = {
        "indexer_aux_grad_precompute": SimpleNamespace(run=auxiliary),
        "sparse_attention_backward": SimpleNamespace(run=backward),
    }
    monkeypatch.setattr(backend, "_validate_production_environment", lambda: None)
    monkeypatch.setattr(backend, "_run_sparse_forward", forward)
    monkeypatch.setattr(backend, "_kernel_module", lambda name, **kwargs: modules[name])
    monkeypatch.setattr(backend, "_nvtx_range", lambda name: nullcontext())


def caller_gradient(layout):
    # Every element is nonzero, including rows the first backward must mask.
    # Keep the owner as well as the view to detect writes anywhere in storage.
    if layout == "contiguous":
        owner = torch.arange(1, 6 * 16 * 256 + 1, dtype=torch.float32).reshape(6, 16, 256)
        gradient = owner[1:-1]
        assert gradient.is_contiguous() and gradient.storage_offset() > 0
    elif layout == "strided":
        owner = torch.arange(1, 4 * 16 * 512 + 1, dtype=torch.float32).reshape(4, 16, 512)
        gradient = owner[:, :, ::2]
        assert not gradient.is_contiguous()
    else:
        owner = torch.arange(1, 16 * 256 + 1, dtype=torch.float32).reshape(1, 16, 256)
        gradient = owner.expand(4, -1, -1)
        assert not gradient.is_contiguous() and gradient.stride(0) == 0
    return owner, gradient


def run_backward(valid, caller_grad, oracle_grad):
    q = torch.arange(4 * 16 * 256, dtype=torch.float32).reshape(4, 16, 256).requires_grad_()
    k = torch.arange(4 * 256, dtype=torch.float32).reshape(4, 1, 256).requires_grad_()
    v = (k.detach() + 1).requires_grad_()
    reference_inputs = tuple(t.detach().clone().requires_grad_() for t in (q, k, v))
    rq, rk, rv = reference_inputs
    reference_output = torch.where(valid[:, None, None], rq + rk + 2 * rv, 0)
    # Compute the oracle before SSA sees the separate caller-owned gradient.
    expected = torch.autograd.grad(reference_output, reference_inputs, oracle_grad)
    layout = backend.RowMetadata(
        key_starts=torch.zeros(4, dtype=torch.int32),
        key_ends=torch.ones(4, dtype=torch.int32),
        query_valid=valid,
        loss_denominator=torch.ones(1),
        triangular_scores=False,
        mask_invalid_rows=True,
    )
    output, _ = backend.sparse_attention(q, k, v, torch.zeros((4, 128)), torch.zeros((4, 128)), layout, 512, loss_coeff=0.0)
    actual = torch.autograd.grad(output, (q, k, v), caller_grad)
    torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
    return actual, expected


@pytest.mark.parametrize("layout", ["contiguous", "strided", "expanded"])
def test_backward_preserves_caller_gradient_and_shared_storage(mock_kernels, layout):
    owner, gradient = caller_gradient(layout)
    owner_before = owner.clone()
    oracle_grad = gradient.detach().clone()
    valid = torch.tensor([False, True, False, True])

    actual, expected = run_backward(valid, gradient, oracle_grad)

    assert torch.equal(owner, owner_before), "backward modified caller-owned gradient storage"
    assert torch.equal(gradient, oracle_grad), "backward modified its incoming gradient"
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["contiguous", "strided", "expanded"])
def test_reused_gradient_matches_an_independent_oracle_across_masks(mock_kernels, layout):
    _, gradient = caller_gradient(layout)
    oracle_grad = gradient.detach().clone()
    first_valid = torch.tensor([False, True, False, True])
    results = []
    # The second call needs rows masked by the first call. It must see their
    # original upstream values, even when both calls share the same storage.
    for valid in (first_valid, ~first_valid):
        results.append(run_backward(valid, gradient, oracle_grad))

    for actual, expected in results:
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
