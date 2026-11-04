# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST cancellation regressions for BF16 KDA residuals and recompute."""

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("KDA residual tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.linear_attention.ops.kda import kimi_delta_attention, kimi_delta_attention_summary

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.fixture(autouse=True)
def frost_device():
    """Match the production FROST LA eligibility gate on other CI devices."""
    from cudnn.frost.buffers import current_sm

    sm = current_sm()
    if sm is None or not (100 <= sm <= 103 or sm == 107):
        pytest.skip("requires SM100 / SM103 / SM107 FROST KDA")


def cancellation_inputs(dim_k, dim_v, active):
    """Exactly representable inputs whose FP32 projection is a BF16 midpoint.

    P = 1 + 1/256 rounds to BF16 1, so the old packed subtraction incorrectly
    produced zero. beta * (V - P) = -1/512 is exactly representable in BF16.
    Only one token has a nonzero key, isolating initial/steady-state Y staging
    from inverse approximation and preceding state-rounding errors.
    """
    tokens = 33
    q = torch.zeros(tokens, 1, dim_k, dtype=torch.bfloat16, device="cuda")
    q[..., 0] = 1
    k = torch.zeros_like(q)
    k[active, :, 0], k[active, :, 1] = 1, 1 / 256
    v = torch.zeros(tokens, 1, dim_v, dtype=torch.bfloat16, device="cuda")
    v[active] = 1
    gate = torch.zeros(tokens, 1, dim_k, dtype=torch.float32, device="cuda")
    beta = torch.full((tokens, 1), 0.5, dtype=torch.bfloat16, device="cuda")
    cu = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    seed = torch.zeros(1, 1, dim_v, dim_k, device="cuda")
    seed[..., :2] = 1
    expected = seed.double().clone()
    expected[..., 0] -= 1 / 512
    expected[..., 1] -= 1 / 131072
    return (q, k, v, gate, beta, cu), seed, expected


def assert_projection_residual(dims, active, path, beta_dtype):
    """Check the small analytic correction without dilution by unchanged state."""
    args, seed, expected = cancellation_inputs(*dims, active)
    args = (*args[:4], args[4].to(beta_dtype), args[5])
    if path == "forward":
        _, actual = kimi_delta_attention(*args, initial_state=seed, output_final_state=True, batch_invariant=True, plan_name="kda_frost")
    else:
        actual, _ = kimi_delta_attention_summary(*args[1:], initial_state=seed, batch_invariant=True, plan_name="kda_summary_frost")
    # Assert the correction itself, not a global relative error diluted by the
    # unchanged state rows. This must fail on the old early-cast implementation.
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
    """Cover dtype, dimension and initial/steady-state intersections at L1."""
    assert_projection_residual(dims, active, path, beta_dtype)


@pytest.mark.L1
@pytest.mark.parametrize("dim_k", [64, 128])
@pytest.mark.parametrize("active", [0, 16], ids=["initial", "steady"])
def test_prep_projection_residual_retains_cancellation(monkeypatch, dim_k, active):
    """Exercise prep-fed Y staging explicitly, independent of heuristic ranking."""
    from cudnn.linear_attention.frost.common import piece_chain
    from cudnn.linear_attention.frost import kda_engine
    from cudnn.linear_attention.ops.kda import fprop_cache

    fprop_cache.clear()
    choose = piece_chain.choose_pieces
    monkeypatch.setattr(piece_chain, "choose_pieces", lambda **kw: (0, choose(**kw)[1]))
    monkeypatch.setattr(piece_chain, "is_dv_split", lambda **kw: True)
    monkeypatch.setattr(kda_engine, "PREP_TILE_FRACTION", 1.0)
    args, seed, expected = cancellation_inputs(dim_k, 128, active)
    try:
        _, actual = kimi_delta_attention(*args, initial_state=seed, output_final_state=True, plan_name="kda_frost")
        torch.testing.assert_close(actual.double()[..., :2], expected[..., :2], rtol=0, atol=1e-7)
    finally:
        fprop_cache.clear()


@pytest.mark.L1
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("safe_gate", [False, True])
def test_backward_saved_checkpoints_and_recompute_agree(dtype, safe_gate):
    """Both backward entry paths rebuild Y with the forward's precision rule."""
    generator = torch.Generator(device="cpu").manual_seed(1731)
    tokens, heads, dim = 65, 2, 64
    q = torch.randn(tokens, heads, dim, generator=generator)
    k = torch.randn(tokens, heads, dim, generator=generator)
    k = k / k.square().sum(-1, keepdim=True).sqrt()
    v = torch.randn(tokens, heads, dim, generator=generator)
    g = torch.full_like(q, -6.0 if safe_gate else -0.001)
    beta = torch.rand(tokens, heads, generator=generator) * 0.5 + 0.1
    values = [x.to(device="cuda", dtype=torch.float32 if i == 3 else dtype) for i, x in enumerate((q, k, v, g, beta))]
    cu = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    do = torch.randn(tokens, heads, dim, generator=generator).to(device="cuda", dtype=dtype)
    results = []
    for checkpoint in (0, 16):
        leaves = [x.detach().clone().requires_grad_() for x in values]
        outputs = kimi_delta_attention(
            *leaves,
            cu,
            batch_invariant=True,
            checkpoint_every_n_tokens=checkpoint,
            safe_gate=safe_gate,
            gate_lower_bound=-5.0 if safe_gate else None,
            plan_name="kda_frost",
        )
        grads = torch.autograd.grad(outputs[0], leaves, do)
        results.append((outputs[0], grads))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for name, got, expected in zip(("q", "k", "v", "gate", "beta"), results[0][1], results[1][1]):
        relative_l2 = (got.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-20)
        assert relative_l2 < 0.003, f"{name}: checkpoint/recompute relative L2 {relative_l2}"
