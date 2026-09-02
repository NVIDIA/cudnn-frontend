# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Pure, zero-copy assembly of caller-owned WGrad operand views."""

from __future__ import annotations

import torch

from ..._types import (
    MoeEpTrainingBackwardOutputs,
    MoeEpTrainingWgradOperands,
)


def assemble_training_wgrad_operands(
    *,
    fc1_a: torch.Tensor,
    fc1_sfa: torch.Tensor,
    valid_route_counts: torch.Tensor,
    expert_offsets: torch.Tensor,
    backward: MoeEpTrainingBackwardOutputs,
) -> MoeEpTrainingWgradOperands:
    """Return non-owning views after producers wrote their final layouts."""

    required = (
        backward.fc1_b,
        backward.fc1_sfb,
        backward.fc2_a,
        backward.fc2_sfa,
        backward.fc2_b,
        backward.fc2_sfb,
    )
    if any(value is None for value in required):
        raise ValueError("all backward WGrad outputs are required for assembly")
    return MoeEpTrainingWgradOperands(
        fc1_a=fc1_a,
        fc1_sfa=fc1_sfa,
        fc1_b=backward.fc1_b,
        fc1_sfb=backward.fc1_sfb,
        fc2_a=backward.fc2_a,
        fc2_sfa=backward.fc2_sfa,
        fc2_b=backward.fc2_b,
        fc2_sfb=backward.fc2_sfb,
        expert_offsets=expert_offsets,
        valid_route_counts=valid_route_counts,
    )


__all__ = ["assemble_training_wgrad_operands"]
