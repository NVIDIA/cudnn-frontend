# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Deterministic input data and quantization helpers for MoE EP tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_forward_inputs(device: torch.device):
    """Build one deterministic MXFP8 forward case."""

    generator = torch.Generator(device=device).manual_seed(20260811)
    experts, tokens, hidden, intermediate = 2, 5, 128, 256
    activation = quantize_mxfp8(
        torch.randn(tokens, hidden, generator=generator, device=device),
        axis=1,
    )
    fc1_weight = quantize_mxfp8(
        torch.randn(
            experts,
            hidden,
            2 * intermediate,
            generator=generator,
            device=device,
        )
        / 8,
        axis=1,
    )
    fc2_weight = quantize_mxfp8(
        torch.randn(
            experts,
            intermediate,
            hidden,
            generator=generator,
            device=device,
        )
        / 8,
        axis=1,
    )
    topk_idx = torch.tensor(
        [[0, 1], [1, 0], [0, -1], [1, 0], [0, 1]],
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.tensor(
        [
            [0.75, 0.25],
            [0.625, 0.375],
            [1.0, 0.0],
            [0.5, 0.5],
            [0.875, 0.125],
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    return activation, fc1_weight, fc2_weight, topk_idx, topk_weights


def make_distributed_forward_inputs(
    rank: int,
    world_size: int,
    device: torch.device,
):
    """Build rank-local inputs with one local and one remote route per token."""

    generator = torch.Generator(device=device).manual_seed(20260811 + rank)
    # Vary local shapes without exceeding the distributed tests'
    # max_tokens_per_rank=8 contract at EP sizes above seven.
    local_experts, tokens, hidden, intermediate = (
        2,
        rank % 7 + 2,
        128,
        256,
    )
    activation = quantize_mxfp8(
        torch.randn(tokens, hidden, generator=generator, device=device),
        axis=1,
    )
    fc1_weight = quantize_mxfp8(
        torch.randn(
            local_experts,
            hidden,
            2 * intermediate,
            generator=generator,
            device=device,
        )
        / 8,
        axis=1,
    )
    fc2_weight = quantize_mxfp8(
        torch.randn(
            local_experts,
            intermediate,
            hidden,
            generator=generator,
            device=device,
        )
        / 8,
        axis=1,
    )
    remote_rank = (rank + 1) % world_size
    topk_idx = torch.tensor(
        [
            [
                rank * local_experts + token % local_experts,
                remote_rank * local_experts + (token + 1) % local_experts,
            ]
            for token in range(tokens)
        ],
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.tensor(
        [[0.625, 0.375]],
        dtype=torch.bfloat16,
        device=device,
    ).expand(tokens, -1).contiguous()
    return activation, fc1_weight, fc2_weight, topk_idx, topk_weights


def quantize_mxfp8(tensor: torch.Tensor, *, axis: int = -1):
    """Return a public logical MXFP8 tensor (E4M3 payload + E8M0 scales)."""

    from cudnn import BlockScaledTensor

    axis = axis % tensor.ndim
    logical_shape = tuple(tensor.shape)
    logical_extent = logical_shape[axis]
    moved = tensor.float().movedim(axis, -1)
    block_count = (logical_extent + 31) // 32
    padded_extent = block_count * 32
    if padded_extent != logical_extent:
        moved = F.pad(moved, (0, padded_extent - logical_extent))

    blocks = moved.reshape(*moved.shape[:-1], block_count, 32)
    raw_scale = blocks.abs().amax(dim=-1) / 448.0
    safe_scale = torch.where(raw_scale > 0, raw_scale, 1.0)
    power_of_two_scale = torch.where(
        raw_scale > 0,
        torch.pow(2.0, torch.ceil(torch.log2(safe_scale))),
        torch.zeros_like(raw_scale),
    )
    scale = power_of_two_scale.to(torch.float8_e8m0fnu)
    reciprocal = torch.where(scale.float() > 0, scale.float().reciprocal(), 0.0)
    payload = (
        (blocks * reciprocal.unsqueeze(-1))
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(*moved.shape)[..., :logical_extent]
    )

    return BlockScaledTensor(
        data=payload.movedim(-1, axis).contiguous(),
        scale=scale.movedim(-1, axis).contiguous(),
        format="mxfp8",
        logical_shape=logical_shape,
        axis=axis,
    )


__all__ = [
    "make_distributed_forward_inputs",
    "make_forward_inputs",
    "quantize_mxfp8",
]
