# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures and Torch reference for the BF16 grouped GEMM GLU API."""

from __future__ import annotations

from typing import Any

import torch


def make_grouped_gemm_glu_bf16_problem(
    *,
    m: int = 512,
    n: int = 256,
    k: int = 128,
    experts: int = 2,
    discrete: bool = False,
    b_major: str = "k",
    enable_bias: bool = False,
    alpha_values: tuple[float, ...] = (0.75, -1.25),
) -> dict[str, Any]:
    """Create source-compatible BF16 inputs for GLU forward."""
    if m % (256 * experts) != 0:
        raise ValueError("m must give every expert a 256-aligned token count")
    if n % 64 != 0:
        raise ValueError("n must contain paired 32-column gate/up blocks")

    generator = torch.Generator(device="cuda").manual_seed(20260717 + m + int(discrete) + int(enable_bias))
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
    probability = torch.linspace(0.25, 0.875, m, dtype=torch.float32, device="cuda").view(m, 1, 1)
    bias = None
    if enable_bias:
        bias = (
            torch.randn(
                (experts, n),
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.125
        ).t()

    return {
        "a": a,
        "b": None if discrete else b,
        "b_storage": b_storage,
        "b_reference": b_reference,
        "b_ptrs": b_ptrs if discrete else None,
        "offsets": offsets,
        "alpha": alpha,
        "prob": probability,
        "bias": bias,
        "n": n,
        "k": k,
        "experts": experts,
    }


def _interleaved_gate_up(c_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split alternating 32-column blocks into gate and up tensors."""
    n = c_tensor.shape[1]
    columns = torch.arange(n, device=c_tensor.device).view(n // 32, 32)
    gate_columns = columns[0::2].reshape(-1)
    up_columns = columns[1::2].reshape(-1)
    return (
        c_tensor.index_select(1, gate_columns),
        c_tensor.index_select(1, up_columns),
    )


def grouped_gemm_glu_bf16_reference(
    problem: dict[str, Any],
    *,
    act_func: str,
    linear_offset: float,
    geglu_alpha: float = 1.702,
    glu_clamp_max: float = 7.0,
    glu_clamp_min: float = -7.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 C/D matching the upstream 32-column GLU epilogue."""
    a = problem["a"].float().squeeze(-1)
    b_storage = problem["b_reference"].float()
    offsets = problem["offsets"].cpu().tolist()
    alpha = problem["alpha"].float()
    probability = problem["prob"].float()
    bias = problem["bias"]

    c_reference = torch.empty((a.shape[0], problem["n"], 1), dtype=torch.float32, device="cuda")
    begin = 0
    for expert, end in enumerate(offsets):
        c_value = alpha[expert] * torch.matmul(a[begin:end], b_storage[expert].t())
        if bias is not None:
            c_value = c_value + bias[:, expert].float()[None, :]
        c_reference[begin:end, :, 0] = c_value
        begin = end

    gate, up = _interleaved_gate_up(c_reference)
    if act_func == "swiglu":
        activated = up * torch.nn.functional.silu(gate)
    elif act_func == "geglu":
        gate = torch.clamp(gate, max=glu_clamp_max)
        up = torch.clamp(up, min=glu_clamp_min, max=glu_clamp_max)
        activated = (up + linear_offset) * gate * torch.sigmoid(geglu_alpha * gate)
    else:
        raise ValueError(f"unsupported activation {act_func}")
    d_reference = activated * probability
    return c_reference, d_reference


def assert_grouped_gemm_glu_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare after output conversion with BF16/CuTe fast-math tolerances."""
    converted = expected.to(actual.dtype).float()
    torch.testing.assert_close(actual.float(), converted, rtol=2e-2, atol=4e-2)
