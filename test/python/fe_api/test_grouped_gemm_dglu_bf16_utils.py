# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures and exact Torch reference for the BF16 dGLU API."""

from __future__ import annotations

from typing import Any

import torch


def make_grouped_gemm_dglu_bf16_problem(
    *,
    m: int = 512,
    n: int = 128,
    k: int = 128,
    experts: int = 2,
    discrete: bool = False,
    b_major: str = "k",
    alpha_values: tuple[float, ...] = (0.75, -1.25),
    beta_values: tuple[float, ...] = (1.5, -0.5),
) -> dict[str, Any]:
    """Create source-compatible BF16 dGLU inputs with clamp-sensitive C."""
    if m % (256 * experts) != 0:
        raise ValueError("m must give every expert a 256-aligned token count")
    if n % 32 != 0:
        raise ValueError("n must be divisible by the 32-column dGLU block")

    generator = torch.Generator(device="cuda").manual_seed(20260718 + m + int(discrete))
    a = (torch.randn((1, m, k), generator=generator, device="cuda", dtype=torch.bfloat16) * 0.125).permute(1, 2, 0)
    b_reference = (
        torch.randn(
            (experts, n, k),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    if b_major not in ("k", "n"):
        raise ValueError(f"b_major must be 'k' or 'n', got {b_major}")
    b_storage = b_reference if b_major == "k" else b_reference.transpose(1, 2).contiguous()
    b = b_reference.permute(1, 2, 0)
    b_ptrs = torch.tensor(
        [b_storage[index].data_ptr() for index in range(experts)],
        dtype=torch.int64,
        device="cuda",
    )

    group_m = m // experts
    offsets = torch.arange(group_m, m + 1, group_m, dtype=torch.int32, device="cuda")
    alpha = torch.tensor(alpha_values[:experts], dtype=torch.float32, device="cuda")
    beta = torch.tensor(beta_values[:experts], dtype=torch.float32, device="cuda")
    prob = torch.linspace(-0.75, 0.875, m, dtype=torch.float32, device="cuda").view(m, 1, 1)

    c = (
        torch.randn(
            (1, m, 2 * n),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.5
    ).permute(1, 2, 0)
    begin = 0
    for expert, end in enumerate(offsets.cpu().tolist()):
        beta_value = float(beta[expert].item())
        # After beta scaling, these values exercise every dGeGLU clamp mask.
        c[begin:end, 0, 0] = 8.5 / beta_value
        c[begin:end, 1, 0] = -8.5 / beta_value
        c[begin:end, 32, 0] = 8.25 / beta_value
        c[begin:end, 33, 0] = -8.25 / beta_value
        begin = end

    return {
        "a": a,
        "b": None if discrete else b,
        "b_storage": b_storage,
        "b_reference": b_reference,
        "b_ptrs": b_ptrs if discrete else None,
        "c": c,
        "offsets": offsets,
        "alpha": alpha,
        "beta": beta,
        "prob": prob,
        "dprob": torch.zeros((m, 1, 1), dtype=torch.float32, device="cuda"),
        "n": n,
        "k": k,
        "experts": experts,
    }


def _interleaved_indices(n: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Return gate/input destinations and the compact N source order."""
    columns_2n = torch.arange(2 * n, device=device).view((2 * n) // 32, 32)
    gate_columns = columns_2n[0::2].reshape(-1)
    input_columns = columns_2n[1::2].reshape(-1)
    compact_columns = torch.arange(n, device=device).view(n // 32, 32).reshape(-1)
    return gate_columns, input_columns, compact_columns


def grouped_gemm_dglu_bf16_reference(
    problem: dict[str, Any],
    *,
    act_func: str,
    linear_offset: float,
    generate_dbias: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return D, dprob, and optional dbias matching the reference oracle."""
    a = problem["a"].float().squeeze(-1)
    b_storage = problem["b_reference"].float()
    c = problem["c"].float()
    offsets = problem["offsets"].cpu().tolist()
    alpha = problem["alpha"].float()
    beta = problem["beta"].float()
    prob = problem["prob"].float()
    n = problem["n"]

    ref = torch.empty((a.shape[0], n, 1), dtype=torch.float32, device="cuda")
    c_scaled = torch.empty_like(c, dtype=torch.float32)
    begin = 0
    for expert, end in enumerate(offsets):
        ref[begin:end, :, 0] = torch.matmul(
            a[begin:end] * alpha[expert],
            (b_storage[expert] * alpha[expert]).t(),
        )
        c_scaled[begin:end, :, 0] = c[begin:end, :, 0] * beta[expert]
        begin = end

    gate_columns, input_columns, compact_columns = _interleaved_indices(n, c.device)
    gate_unclipped = c_scaled.index_select(1, gate_columns)
    input_unclipped = c_scaled.index_select(1, input_columns)

    if act_func == "dswiglu":
        gate = gate_unclipped
        input_value = input_unclipped
        sigmoid = torch.sigmoid(gate)
        swish = gate * sigmoid
        dprob_terms = swish * input_value * ref
        d_gate = ref * prob * input_value * sigmoid * (1.0 + gate * (1.0 - sigmoid))
        d_input = ref * prob * swish
    elif act_func == "dgeglu":
        gate = torch.clamp(gate_unclipped, max=7.0)
        input_value = torch.clamp(input_unclipped, min=-7.0, max=7.0)
        sigmoid = torch.sigmoid(1.702 * gate)
        swish = gate * sigmoid
        dprob_terms = swish * (input_value + linear_offset) * ref
        d_gate = ref * sigmoid * (1.0 + 1.702 * gate * (1.0 - sigmoid)) * (input_value + linear_offset) * prob
        d_input = ref * gate * sigmoid * prob

        gate_filter = gate_unclipped.clone()
        input_filter = input_unclipped.clone()
        gate_filter[gate_unclipped > 7.0] = 0.0
        input_filter[(input_unclipped > 7.0) | (input_unclipped < -7.0)] = 0.0
        d_gate = d_gate * gate_filter
        d_input = d_input * input_filter
    else:
        raise ValueError(f"unsupported activation {act_func}")

    chunk_sums = [chunk.sum(dim=1, keepdim=True) for chunk in torch.split(dprob_terms, 32, dim=1)]
    dprob = torch.cat(chunk_sums, dim=1).sum(dim=1, keepdim=True)

    d = torch.empty_like(c_scaled)
    d.index_copy_(1, gate_columns, d_gate.index_select(1, compact_columns))
    d.index_copy_(1, input_columns, d_input.index_select(1, compact_columns))

    dbias = None
    if generate_dbias:
        dbias = torch.zeros(
            (problem["experts"], 2 * n, 1),
            dtype=torch.bfloat16,
            device="cuda",
        )
        begin = 0
        for expert, end in enumerate(offsets):
            dbias[expert, :, 0] = d[begin:end, :, 0].sum(dim=0).to(torch.bfloat16)
            begin = end
    return d, dprob, dbias


def assert_grouped_gemm_dglu_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare D/dprob after output conversion with BF16 fast-math tolerance."""
    converted = expected.to(actual.dtype).float()
    torch.testing.assert_close(actual.float(), converted, rtol=3e-2, atol=8e-2)


def assert_grouped_gemm_dglu_dbias_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Use the upstream reduction-order-aware tolerance for BF16 atomics."""
    expected_bf16 = expected.to(torch.bfloat16).float()
    atol = max(expected_bf16.abs().max().item() * 0.008 * (4**0.5), 0.1)
    torch.testing.assert_close(actual.float(), expected_bf16, rtol=1e-2, atol=atol)
