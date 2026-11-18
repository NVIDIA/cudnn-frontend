# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST cancellation regressions for BF16 GDN residuals, forward and backward."""

import pytest
import torch

from cudnn.linear_attention.ops import gated_delta_net, gated_delta_net_summary

from linear_attention.test_la import frost_la_unavailable

requirement_error = frost_la_unavailable()
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def cancellation_inputs(dim_k, dim_v, active, *, tokens=33):
    """Exactly representable inputs whose FP32 state projection is a BF16 midpoint.

    With unit decay, ``P = k @ state = 1 + 1/256`` rounds to BF16 1.  Subtracting
    that rounded projection from ``V = 1`` loses the residual; subtracting in FP32
    retains ``V - P = -1/256``.  At ``beta = 1/2`` the state correction is
    ``-1/512`` in the first key row and ``-1/131072`` in the second.
    """
    q = torch.zeros(tokens, 1, dim_k, dtype=torch.bfloat16, device="cuda")
    q[..., 0] = 1
    k = torch.zeros_like(q)
    k[active, :, 0], k[active, :, 1] = 1, 1 / 256
    v = torch.zeros(tokens, 1, dim_v, dtype=torch.bfloat16, device="cuda")
    v[active] = 1
    gate = torch.zeros(tokens, 1, dtype=torch.float32, device="cuda")
    beta = torch.full((tokens, 1), 0.5, dtype=torch.bfloat16, device="cuda")
    cu = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    seed = torch.zeros(1, 1, dim_v, dim_k, device="cuda")
    seed[..., :2] = 1
    expected = seed.double().clone()
    expected[..., 0] -= 1 / 512
    expected[..., 1] -= 1 / 131072
    return (q, k, v, gate, beta, cu), seed, expected


def summary_cancellation_inputs(dim_k, dim_v, active, *, tokens=97):
    """Build the midpoint projection in one chunk and consume it in the next."""
    projected = active + 64
    q = torch.zeros(tokens, 1, dim_k, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros_like(q)
    k[active, :, :2] = 1
    k[projected, :, 0], k[projected, :, 1] = 1, 1 / 256
    v = torch.zeros(tokens, 1, dim_v, dtype=torch.bfloat16, device="cuda")
    v[active], v[projected] = 1, 1
    gate = torch.zeros(tokens, 1, dtype=torch.float32, device="cuda")
    beta = torch.zeros(tokens, 1, dtype=torch.bfloat16, device="cuda")
    beta[active], beta[projected] = 1, 0.5
    cu = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    seed = torch.zeros(1, 1, dim_v, dim_k, device="cuda")
    expected = seed.double().clone()
    expected[..., :2] = 1
    expected[..., 0] -= 1 / 512
    expected[..., 1] -= 1 / 131072
    return (q, k, v, gate, beta, cu), seed, expected


def assert_projection_residual(dims, active, path, beta_dtype):
    """Check the small analytic correction without dilution by unchanged state."""
    inputs = summary_cancellation_inputs if path == "summary" else cancellation_inputs
    args, seed, expected = inputs(*dims, active)
    args = (*args[:4], args[4].to(beta_dtype), args[5])
    if path == "forward":
        _, actual = gated_delta_net(*args, initial_state=seed, output_final_state=True, batch_invariant=True, plan_name="gdn_frost")
    else:
        actual, _ = gated_delta_net_summary(*args[1:], initial_state=seed, batch_invariant=True, plan_name="gdn_summary_frost")
    torch.testing.assert_close(actual.double()[..., :2], expected[..., :2], rtol=0, atol=1e-7)
    assert torch.count_nonzero(actual[..., 2:]) == 0


@pytest.mark.L0
@pytest.mark.parametrize("path", ["forward", "summary"])
def test_projection_residual_cancellation_smoke(path):
    """Keep one small cancellation regression per public forward/summary path."""
    assert_projection_residual((64, 64), 0, path, torch.bfloat16)


@pytest.mark.L1
@pytest.mark.parametrize("dims", [(64, 64), (64, 128), (128, 64), (128, 128)])
@pytest.mark.parametrize("active", [0, 16], ids=["initial", "steady"])
@pytest.mark.parametrize("path", ["forward", "summary"])
@pytest.mark.parametrize("beta_dtype", [torch.bfloat16, torch.float32])
def test_projection_residual_retains_cancellation(dims, active, path, beta_dtype):
    """Cover dimension, initial/steady-state and beta-dtype intersections."""
    assert_projection_residual(dims, active, path, beta_dtype)


@pytest.mark.L0
@pytest.mark.parametrize("checkpoint", [0, 64], ids=["recompute", "saved"])
def test_backward_retains_projection_residual(checkpoint):
    """Both regenerated and saved-checkpoint backward paths use the FP32 residual."""
    active = 16
    args, seed, _ = cancellation_inputs(64, 64, active, tokens=65)
    leaves = [value.detach().clone().requires_grad_() for value in args[:-1]]
    result = gated_delta_net(
        *leaves,
        args[-1],
        initial_state=seed,
        output_final_state=True,
        batch_invariant=True,
        checkpoint_every_n_tokens=checkpoint,
        plan_name="gdn_frost",
    )
    final_state = result[1]
    (d_beta,) = torch.autograd.grad(final_state[0, 0, 0, 0], (leaves[4],))
    torch.testing.assert_close(d_beta[active, 0].float(), torch.tensor(-1 / 256, device="cuda"), rtol=0, atol=0)
    assert torch.count_nonzero(torch.cat((d_beta[:active], d_beta[active + 1 :]))) == 0
