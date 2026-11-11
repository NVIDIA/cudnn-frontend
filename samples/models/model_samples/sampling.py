# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token choice on the GPU.

Greedy only for now: the id of the largest logit of every sequence, int32, without a host read.  Among exactly equal largest
logits the choice is PyTorch's; the tie-out counts such a step as a near-tie.  Sampling with temperature, top-k and top-p arrives
as one option with the stack that needs it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Sampling:
    kind: str = "greedy"


GREEDY = Sampling()


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """``[rows, vocabulary]`` logits -> ``[rows]`` int32 ids of the largest logit."""
    return torch.argmax(logits, dim=-1).to(torch.int32)


def margin(logits: torch.Tensor) -> torch.Tensor:
    """The gap between the largest and the second largest logit of every row, in fp32."""
    top = torch.topk(logits.float(), 2, dim=-1).values
    return top[..., 0] - top[..., 1]
