# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repeatability and sequence isolation of packed convolution and history gradients."""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_replays, snapshot
from test_causal_conv1d_packed import _dsl, reference

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def enable_deterministic_algorithms():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _run(x, weight, bias, initial, lengths, dy, dfinal):
    from functools import partial
    from cudnn.ops import causal_conv1d as _causal_conv1d

    causal_conv1d = partial(_causal_conv1d, backend="se")

    xx, ww, bb, ii = (t.detach().clone().requires_grad_() for t in (x, weight, bias, initial))
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device=x.device, dtype=torch.int32)
    y, final = causal_conv1d(xx.T[None], ww, bb, cu_seqlens=cu, initial_states=ii.transpose(1, 2), activation="silu", return_final_states=True)
    y, final = y[0].T, final.transpose(1, 2)
    dx, dw, db, di = torch.autograd.grad((y, final), (xx, ww, bb, ii), (dy, dfinal))
    return dict(y=y, final=final, dx=dx, dw=dw, db=db, di=di)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_public_packed_replay_all_outputs_and_gradients(dtype):
    _dsl()
    torch.manual_seed(1907)
    lengths, hidden, width = [0, 1, 2, 7, 65, 130, 0], 136, 5
    x = torch.randn(sum(lengths), hidden, device="cuda", dtype=dtype)
    weight = torch.randn(hidden, width, device="cuda", dtype=dtype)
    bias = torch.randn(hidden, device="cuda", dtype=dtype)
    initial = torch.randn(len(lengths), width - 1, hidden, device="cuda", dtype=dtype)
    dy, dfinal = torch.randn_like(x), torch.randn_like(initial)
    fixed = snapshot(dict(x=x, weight=weight, bias=bias, initial=initial, dy=dy, dfinal=dfinal))
    actual = assert_replays(lambda: _run(x, weight, bias, initial, lengths, dy, dfinal), repeats=4)
    xx, ww, bb, ii = (t.double().detach().requires_grad_() for t in (x, weight, bias, initial))
    y, final = reference(xx, ww, bb, ii, lengths, True)
    grads = torch.autograd.grad((y, final), (xx, ww, bb, ii), (dy.double(), dfinal.double()))
    atol, rtol = {torch.float16: (0.003, 0.002), torch.bfloat16: (0.03, 0.02), torch.float32: (2e-5, 2e-5)}[dtype]
    for name, want in zip(("y", "final", "dx", "dw", "db", "di"), (y, final, *grads)):
        torch.testing.assert_close(actual[name], want.to(dtype), atol=atol, rtol=rtol)
    assert_bitwise(dict(x=x, weight=weight, bias=bias, initial=initial, dy=dy, dfinal=dfinal), fixed, "inputs mutated")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_public_packed_sequence_batch_invariance(dtype):
    """Shared parameter sums are replay-tested; this check isolates sequence outputs."""
    _dsl()
    torch.manual_seed(319)
    hidden, width = 24, 8
    lengths = [33, 0, 1, 130]
    xs = [torch.randn(n, hidden, device="cuda", dtype=dtype) for n in lengths]
    initial = torch.randn(len(lengths), width - 1, hidden, device="cuda", dtype=dtype)
    dys = [torch.randn_like(x) for x in xs]
    dfinal = torch.randn_like(initial)
    weight = torch.randn(hidden, width, device="cuda", dtype=dtype)
    bias = torch.randn(hidden, device="cuda", dtype=dtype)
    target = _run(xs[0], weight, bias, initial[:1], [lengths[0]], dys[0], dfinal[:1])
    expected = snapshot({k: target[k] for k in ("y", "final", "dx", "di")})
    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 0, 1]):
        packed_lengths = [lengths[s] for s in order]
        result = _run(torch.cat([xs[s] for s in order]), weight, bias, initial[order], packed_lengths, torch.cat([dys[s] for s in order]), dfinal[order])
        first = 0
        for i, s in enumerate(order):
            if s == 0:
                actual = dict(
                    y=result["y"][first : first + lengths[s]],
                    final=result["final"][i : i + 1],
                    dx=result["dx"][first : first + lengths[s]],
                    di=result["di"][i : i + 1],
                )
                assert_bitwise(actual, expected, f"order={order}, copy={i}")
            first += lengths[s]
    # Change neighboring values independently while preserving the target and
    # its upstream gradients. New allocations also perturb physical placement.
    xs[3] = xs[3] * 3
    initial[3].add_(4)
    dys[3] = dys[3] * 2
    result = _run(torch.cat(xs), weight, bias, initial, lengths, torch.cat(dys), dfinal)
    assert_bitwise(dict(y=result["y"][:33], final=result["final"][:1], dx=result["dx"][:33], di=result["di"][:1]), expected, "changed neighbors")
