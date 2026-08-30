# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Private data contracts shared by the MoE EP API, validation, and backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch

from ._tuning import MoeEpTuningConfig
from ._types import MoeTensor


@dataclass(frozen=True)
class ForwardConfig:
    """Static configuration snapshot for one ``MoeEp`` instance."""

    num_experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    experts_per_rank: int
    ep_size: int
    ep_rank: int
    ep_group: Any
    ep_global_ranks: tuple[int, ...]
    max_tokens_per_rank: Optional[int]
    output_format: str
    combine_format: str
    apply_topk_in_fc1: bool
    gate_up_clamp: Optional[float]
    generate_c: bool
    token_padding_size: int
    sf_padding_size: int
    tuning: MoeEpTuningConfig
    backward_wgrad_mode: Literal["none", "operands"] = "none"
    max_recv_size_per_rank: Optional[int] = None
    drop_on_overflow: bool = False
    weight_interleave_size: Optional[int] = None


@dataclass(frozen=True)
class ValidatedForwardRequest:
    """Runtime inputs that have passed the public forward contract."""

    config: ForwardConfig
    activation: MoeTensor
    fc1_weight: MoeTensor
    fc2_weight: MoeTensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    token_count: int
    device: torch.device


__all__ = [
    "ForwardConfig",
    "ValidatedForwardRequest",
]
