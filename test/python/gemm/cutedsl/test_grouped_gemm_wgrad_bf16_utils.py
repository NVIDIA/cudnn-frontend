# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact upstream-style fixtures and Torch reference for BF16 MoE wgrad."""

from __future__ import annotations

from typing import Any, Iterable

import torch


def _pack_ragged(
    logical: torch.Tensor,
    group_k_list: Iterable[int],
    *,
    operand: str,
    major: str,
) -> torch.Tensor:
    """Pack expert slices exactly as WgradTensormapConstructor expects."""
    pieces = []
    begin = 0
    for group_k in group_k_list:
        if operand == "a":
            part = logical[:, begin : begin + group_k]
            packed = part.contiguous() if major == "k" else part.t().contiguous()
        else:
            part = logical[begin : begin + group_k, :]
            packed = part.t().contiguous() if major == "k" else part.contiguous()
        pieces.append(packed.reshape(-1))
        begin += group_k
    storage = torch.cat(pieces) if pieces else logical.new_empty((0,))
    if operand == "a":
        m, tokens = logical.shape
        stride = (tokens, 1) if major == "k" else (1, m)
        return torch.as_strided(storage, (m, tokens), stride)
    tokens, n = logical.shape
    stride = (1, tokens) if major == "k" else (n, 1)
    return torch.as_strided(storage, (tokens, n), stride)


def make_grouped_gemm_wgrad_bf16_problem(
    *,
    group_k_list: tuple[int, ...] = (256, 0, 256),
    m: int = 128,
    n: int = 128,
    a_major: str = "k",
    b_major: str = "k",
    input_order: str = "tensor2d",
    output_dtype: torch.dtype = torch.bfloat16,
    discrete: bool = False,
    initial_value: float | None = None,
) -> dict[str, Any]:
    """Create logical inputs plus source-compatible 2-D/ragged physical views."""
    if a_major not in ("k", "m") or b_major not in ("k", "n"):
        raise ValueError("unsupported major mode")
    if any(group_k < 0 or group_k % 256 for group_k in group_k_list):
        raise ValueError("group K sizes must be non-negative multiples of 256")
    tokens = sum(group_k_list)
    generator = torch.Generator(device="cuda").manual_seed(20260719 + tokens + 11 * len(group_k_list) + int(discrete))
    logical_a = torch.randn((m, tokens), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.125
    logical_b = torch.randn((tokens, n), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.125
    a = _pack_ragged(
        logical_a,
        (tokens,) if input_order == "tensor2d" else group_k_list,
        operand="a",
        major=a_major,
    )
    b = _pack_ragged(
        logical_b,
        (tokens,) if input_order == "tensor2d" else group_k_list,
        operand="b",
        major=b_major,
    )
    offsets = torch.tensor(
        [sum(group_k_list[: index + 1]) for index in range(len(group_k_list))],
        dtype=torch.int32,
        device="cuda",
    )
    output = torch.empty((len(group_k_list), m, n), dtype=output_dtype, device="cuda")
    if initial_value is not None:
        output.fill_(initial_value)
    output_ptrs = torch.tensor(
        [output[index].data_ptr() for index in range(len(group_k_list))],
        dtype=torch.int64,
        device="cuda",
    )
    return {
        "a": a,
        "b": b,
        "logical_a": logical_a,
        "logical_b": logical_b,
        "offsets": offsets,
        "output": output,
        "output_ptrs": output_ptrs if discrete else None,
        "group_k_list": group_k_list,
        "m": m,
        "n": n,
        "experts": len(group_k_list),
        "a_major": a_major,
        "b_major": b_major,
        "input_order": input_order,
        "output_dtype": output_dtype,
        "discrete": discrete,
        "initial_value": initial_value,
    }


def grouped_gemm_wgrad_bf16_reference(problem: dict[str, Any], *, accumulate: bool = False) -> torch.Tensor:
    """Compute the exact per-expert grouped-mm oracle in FP32."""
    result = torch.empty(
        (problem["experts"], problem["m"], problem["n"]),
        dtype=torch.float32,
        device="cuda",
    )
    begin = 0
    for expert, end in enumerate(problem["offsets"].cpu().tolist()):
        if end == begin:
            result[expert].zero_()
        else:
            result[expert] = torch.matmul(
                problem["logical_a"][:, begin:end].float(),
                problem["logical_b"][begin:end, :].float(),
            )
        begin = end
    if accumulate and problem["initial_value"] is not None:
        result.add_(float(problem["initial_value"]))
    return result


def assert_grouped_gemm_wgrad_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare after the requested output conversion with upstream tolerance."""
    converted = expected.to(actual.dtype).float()
    torch.testing.assert_close(actual.float(), converted, rtol=3e-2, atol=8e-2)
