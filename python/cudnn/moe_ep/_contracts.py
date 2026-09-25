# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private per-call contracts shared by MoeEP validation and its backend."""

from __future__ import annotations

from dataclasses import dataclass
import torch

from ._types import MoeTensor


@dataclass(frozen=True)
class _ForwardCall:
    """Dynamic inputs for one synchronously validated inference call."""

    activation: MoeTensor
    fc1_weight: MoeTensor
    fc2_weight: MoeTensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    token_count: int
    device: torch.device


__all__: list[str] = []
