# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Grouped SiLU-GLU expert FFN, shared by the EP layer and the reference.

Weight layout matches flashinfer's canonical ``MoEWeightPack``:

- ``w13`` ``[num_experts, 2*intermediate, hidden]``; ``fc1 = x @ w13[e].T``
  splits as ``linear = fc1[:, :intermediate]``, ``gate = fc1[:, intermediate:]``
  and the activation is ``silu(gate) * linear`` (same split as
  flashinfer-moe_ep/tests/moe_ep/test_moe_ep_compute_correctness.py).
- ``w2`` ``[num_experts, hidden, intermediate]``; ``y = act @ w2[e].T``.

Fully autograd-native. Empty segments still run their (M=0) GEMMs so every
expert's weights participate in the graph and get well-defined zero grads.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def expert_ffn(x: torch.Tensor, w13_e: torch.Tensor, w2_e: torch.Tensor) -> torch.Tensor:
    """One expert's FFN on its ``[n, hidden]`` token slab (n may be 0)."""
    fc1 = x @ w13_e.t()  # [n, 2*intermediate]
    linear, gate = fc1.chunk(2, dim=-1)
    return (F.silu(gate) * linear) @ w2_e.t()  # [n, hidden]


def grouped_expert_ffn(
    x_grouped: torch.Tensor,
    tokens_per_expert: Sequence[int],
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """FFN over a token slab pre-grouped by expert.

    ``x_grouped`` is ``[N, hidden]`` where rows ``[offset_e, offset_e + n_e)``
    all belong to expert ``e`` (offsets from ``tokens_per_expert``). Returns
    the same layout.
    """
    if len(tokens_per_expert) != w13.shape[0]:
        raise ValueError(
            f"tokens_per_expert has {len(tokens_per_expert)} entries but "
            f"w13 has {w13.shape[0]} experts"
        )
    if sum(tokens_per_expert) != x_grouped.shape[0]:
        raise ValueError(
            f"tokens_per_expert sums to {sum(tokens_per_expert)} but "
            f"x_grouped has {x_grouped.shape[0]} rows"
        )
    outs = []
    start = 0
    for e, n in enumerate(tokens_per_expert):
        outs.append(expert_ffn(x_grouped[start : start + n], w13[e], w2[e]))
        start += n
    if not outs:
        return x_grouped.new_zeros((0, w2.shape[1]))
    return torch.cat(outs, dim=0)
