# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token choice on the GPU.

Greedy only for now: the id of the largest logit of every sequence, int32, without a host read.  Among exactly equal largest
logits the choice is PyTorch's; the tie-out counts such a step as a near-tie.  Sampling with temperature, top-k and top-p arrives
as one option with the stack that needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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


def accepted(windows: Sequence, greedy: Sequence) -> int:
    """How many drafted tokens a verify step keeps, by the greedy rule, from what its host read gave: ``windows`` the ids the
    step was fed and ``greedy`` the greedy id after each of them, as ``[sequences][tokens]`` lists (``[tokens]`` lists for
    one sequence).  Token 0 of a window is the token the run kept last; the tokens behind it are drafted.  A drafted token
    is kept while it equals the greedy token of the position in front of it, so a sequence keeps its leading drafted
    tokens with ``window[t + 1] == greedy[t]``.

    Returns the SMALLEST such count of the set: a step keeps one count for all its sequences (``Session.accept``), and only
    the smallest keeps nothing a sequence did not accept.  ``ValueError`` for lists of two shapes."""
    if windows and not isinstance(windows[0], (list, tuple)):
        windows, greedy = [windows], [greedy]
    if (
        len(windows) != len(greedy)
        or not windows
        or any(len(window) != len(chosen) or len(window) != len(windows[0]) for window, chosen in zip(windows, greedy))
    ):
        raise ValueError("the windows a verify step was fed and its greedy ids are lists of one shape, [sequences][tokens]; got two shapes")
    counts = []
    for window, chosen in zip(windows, greedy):
        kept = 0
        while kept + 1 < len(window) and int(window[kept + 1]) == int(chosen[kept]):
            kept += 1
        counts.append(kept)
    return min(counts)
