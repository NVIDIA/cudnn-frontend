# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch references for width-four causal convolution."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def causal_conv1d_bulk_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return an FP32-compute reference and the full-width final state.

    ``x`` has native NWH layout ``[B, T, D]``.  When ``cu_seqlens`` is
    present, ``B`` is one and its token dimension contains the packed
    sequences.  State shifts are performed in the input dtype; only the
    four-term convolution and SiLU use FP32.
    """

    if x.ndim != 3:
        raise ValueError(f"x must be 3D, got {tuple(x.shape)}")
    if weight.shape != (x.shape[-1], 4):
        raise ValueError(f"weight must have shape {(x.shape[-1], 4)}, got {tuple(weight.shape)}")

    batch, tokens, channels = x.shape
    if cu_seqlens is None:
        num_sequences = batch
        ranges = [(row, 0, tokens) for row in range(batch)]
    else:
        if batch != 1:
            raise ValueError("packed x must have B=1")
        offsets = [int(offset) for offset in cu_seqlens.detach().cpu().tolist()]
        num_sequences = len(offsets) - 1
        ranges = [(0, offsets[sequence], offsets[sequence + 1]) for sequence in range(num_sequences)]

    if initial_state is None:
        state = torch.zeros((num_sequences, channels, 4), device=x.device, dtype=x.dtype)
    else:
        if initial_state.shape != (num_sequences, channels, 4):
            raise ValueError(f"initial_state must have shape {(num_sequences, channels, 4)}, got {tuple(initial_state.shape)}")
        state = initial_state.clone()

    output = torch.empty_like(x)
    weight_fp32 = weight.float()
    for sequence, (row, begin, end) in enumerate(ranges):
        sequence_state = state[sequence]
        for token in range(begin, end):
            sequence_state = torch.cat((sequence_state[:, 1:], x[row, token].unsqueeze(-1)), dim=-1)
            preactivation = (sequence_state.float() * weight_fp32).sum(dim=-1)
            output[row, token] = F.silu(preactivation).to(x.dtype)
        state[sequence] = sequence_state

    return output, state


def causal_conv1d_update_reference(
    x: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference one decode update using the bulk operation's state ABI."""

    output, final_state = causal_conv1d_bulk_reference(
        x.unsqueeze(1),
        weight,
        initial_state=state,
    )
    return output[:, 0], final_state
