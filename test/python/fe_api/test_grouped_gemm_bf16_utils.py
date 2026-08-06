# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures and Torch reference for the unfused BF16 grouped GEMM API."""

from __future__ import annotations

from typing import Any

import torch


def make_grouped_gemm_bf16_problem(
    *,
    m: int = 512,
    n: int = 256,
    k: int = 128,
    experts: int = 2,
    discrete: bool = False,
    enable_bias: bool = False,
    alpha_values: tuple[float, ...] = (0.75, -1.25),
) -> dict[str, Any]:
    """Create source-compatible K-major A/B and N-major output inputs."""
    if m % (256 * experts) != 0:
        raise ValueError("m must give every expert a 256-aligned token count")

    generator = torch.Generator(device="cuda").manual_seed(20260716 + m + int(discrete))
    a = (torch.randn((1, m, k), generator=generator, device="cuda", dtype=torch.bfloat16) * 0.125).permute(1, 2, 0)

    # Storage is (expert, n, k), so each expert matrix is contiguous K-major.
    b_storage = torch.randn((experts, n, k), generator=generator, device="cuda", dtype=torch.bfloat16) * 0.125
    b = b_storage.permute(1, 2, 0)
    b_ptrs = torch.tensor(
        [b_storage[i].data_ptr() for i in range(experts)],
        dtype=torch.int64,
        device="cuda",
    )

    group_m = m // experts
    offsets = torch.arange(group_m, m + 1, group_m, dtype=torch.int32, device="cuda")
    alpha = torch.tensor(alpha_values[:experts], dtype=torch.float32, device="cuda")
    prob = torch.linspace(0.25, 0.875, m, dtype=torch.float32, device="cuda").view(m, 1, 1)
    bias = None
    if enable_bias:
        bias = (torch.randn((experts, n), generator=generator, device="cuda", dtype=torch.bfloat16) * 0.125).t()

    return {
        "a": a,
        "b": None if discrete else b,
        "b_storage": b_storage,
        "b_ptrs": b_ptrs if discrete else None,
        "offsets": offsets,
        "alpha": alpha,
        "prob": prob,
        "bias": bias,
        "n": n,
        "k": k,
        "experts": experts,
    }


def grouped_gemm_bf16_reference(problem: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 C/D references matching the source kernel epilogue exactly."""
    a = problem["a"].float().squeeze(-1)
    b_storage = problem["b_storage"].float()
    offsets = problem["offsets"].cpu().tolist()
    alpha = problem["alpha"].float()
    prob = problem["prob"].float().squeeze(-1).squeeze(-1)
    bias = problem["bias"]

    c_ref = torch.empty((a.shape[0], problem["n"], 1), dtype=torch.float32, device="cuda")
    d_ref = torch.empty_like(c_ref)
    begin = 0
    for expert, end in enumerate(offsets):
        gemm = torch.matmul(a[begin:end], b_storage[expert].t())
        scaled = alpha[expert] * gemm
        if bias is None:
            c = scaled
            d = prob[begin:end, None] * scaled
        else:
            c = scaled + prob[begin:end, None] * bias[:, expert].float()[None, :]
            d = c
        c_ref[begin:end, :, 0] = c
        d_ref[begin:end, :, 0] = d
        begin = end
    return c_ref, d_ref


def assert_grouped_gemm_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare after output conversion with tolerances suitable for BF16 inputs."""
    converted = expected.to(actual.dtype).float()
    torch.testing.assert_close(actual.float(), converted, rtol=2e-2, atol=3e-2)
