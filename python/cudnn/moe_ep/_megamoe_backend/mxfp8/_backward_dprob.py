# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Return pool-ordered dprob to the source ``(token, top-k)`` plane."""

from __future__ import annotations

import torch
import torch.distributed as dist

from ..._contracts import ValidatedBackwardRequest
from ._adapter import _decode_moe_tensor


def return_grad_topk_weights(
    request: ValidatedBackwardRequest,
    redispatched_grad_output: torch.Tensor,
) -> torch.Tensor:
    """Compute semantic route gradients and return them to every source.

    The fused dGLU kernel consumes an MXFP8 materialization of ``grad_output``.
    Using its in-kernel dprob would therefore expose quantization error through
    an operation whose public contract specifies straight-through semantics.
    Recompute only this scalar gradient from the original FP32 dY, the BF16
    forward stash, and the decoded FC2 weight.
    """

    config = request.config
    metadata = request.route_metadata.to(torch.int64)
    local_dprob = torch.zeros(
        (request.local_routes,),
        dtype=torch.float32,
        device=request.device,
    )
    fc2_weight = _decode_moe_tensor(request.fc2_weight)
    gate, up = request.fc1_c.float().split(
        config.intermediate_size,
        dim=-1,
    )
    if config.gate_up_clamp is not None:
        gate = gate.clamp(max=config.gate_up_clamp)
        up = up.clamp(
            min=-config.gate_up_clamp,
            max=config.gate_up_clamp,
        )
    hidden = (gate * torch.sigmoid(gate)) * up
    local_expert = metadata[:, 0]
    for expert in range(config.experts_per_rank):
        positions = torch.nonzero(
            local_expert == expert,
            as_tuple=False,
        ).flatten()
        if positions.numel() == 0:
            continue
        grad_output = redispatched_grad_output.index_select(0, positions)
        expert_hidden = hidden.index_select(0, positions)
        grad_hidden = grad_output @ fc2_weight[expert].transpose(0, 1)
        local_dprob.index_copy_(
            0,
            positions,
            (grad_hidden * expert_hidden).sum(dim=-1),
        )

    global_dprob = torch.zeros(
        (
            config.ep_size,
            int(config.max_tokens_per_rank),
            config.top_k,
        ),
        dtype=torch.float32,
        device=request.device,
    )
    if request.local_routes:
        global_dprob[
            metadata[:, 1],
            metadata[:, 2],
            metadata[:, 3],
        ] = local_dprob
    if config.ep_size > 1:
        dist.all_reduce(global_dprob, group=config.ep_group)
    return global_dprob[
        config.ep_rank,
        : request.token_count,
    ]


__all__ = ["return_grad_topk_weights"]
