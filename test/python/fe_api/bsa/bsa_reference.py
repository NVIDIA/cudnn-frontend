# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional

import torch


def block_sparse_mask(
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: torch.Tensor,
    seqlen_q: int,
    seqlen_k: int,
    block_size: int,
    q2k_block_nums: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Expand BSA block metadata to a token-level additive attention mask."""

    batch, heads, num_q_blocks, _ = q2k_block_index.shape
    mask = torch.full(
        (batch, heads, seqlen_q, seqlen_k),
        float("-inf"),
        dtype=torch.float32,
        device=q2k_block_index.device,
    )
    for b in range(batch):
        for h in range(heads):
            for q_block in range(num_q_blocks):
                count = block_sparse_num if q2k_block_nums is None else int(q2k_block_nums[b, h, q_block])
                q_start = q_block * block_size
                q_end = min(q_start + block_size, seqlen_q)
                for slot in range(count):
                    kv_block = int(q2k_block_index[b, h, q_block, slot])
                    k_start = kv_block * block_size
                    k_end = min(k_start + int(block_sizes[kv_block]), seqlen_k)
                    mask[b, h, q_start:q_end, k_start:k_end] = 0.0
    return mask


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    softmax_scale: Optional[float] = None,
):
    """FP32 block-sparse SDPA reference for tensors in BHSD layout."""

    softmax_scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    group_size = q.shape[1] // k.shape[1]
    k_expanded = k.float().repeat_interleave(group_size, dim=1)
    v_expanded = v.float().repeat_interleave(group_size, dim=1)
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float() * softmax_scale, k_expanded)
    scores = scores + mask
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = torch.nan_to_num(probabilities, nan=0.0)
    output = torch.einsum("bhqk,bhkd->bhqd", probabilities, v_expanded)
    lse = torch.logsumexp(scores, dim=-1)
    return output, lse


def attention_backward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    mask: torch.Tensor,
):
    """Return FP32 reference output/LSE and gradients for MHA."""

    q_ref = q.float().detach().requires_grad_()
    k_ref = k.float().detach().requires_grad_()
    v_ref = v.float().detach().requires_grad_()
    scores = torch.einsum("bhqd,bhkd->bhqk", q_ref / math.sqrt(q.shape[-1]), k_ref)
    scores = scores + mask
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bhkd->bhqd", probabilities, v_ref)
    lse = torch.logsumexp(scores, dim=-1)
    output.backward(do.float())
    return output.detach(), lse.detach(), q_ref.grad, k_ref.grad, v_ref.grad
