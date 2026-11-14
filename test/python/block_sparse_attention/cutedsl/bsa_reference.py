# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense torch references for the BSA tests.

``block_sparse_mask`` expands sparse metadata to an additive token mask; ``attention_reference`` evaluates masked
attention (forward, and the backward when ``do`` is given) on it.  ``fp8_online_reference`` models the Sage FP8
kernels' online softmax, including the FP8 cast of the probabilities.
"""

import math
from typing import Optional

import torch


def _block_extent(block_sizes: Optional[torch.Tensor], b: int, h: int, kv_block: int, block_size: int) -> int:
    if block_sizes is None:
        return block_size
    if block_sizes.ndim == 1:
        return int(block_sizes[kv_block])
    if block_sizes.ndim == 2:
        return int(block_sizes[b, kv_block])
    return int(block_sizes[b, h, kv_block])


def block_sparse_mask(
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor],
    seqlen_q: int,
    seqlen_k: int,
    block_size: int,
    q2k_block_nums: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Expand BSA block metadata to a ``[batch, heads, seqlen_q, seqlen_k]`` additive mask (0 or -inf).

    ``block_sizes`` may be ``None`` (full blocks, clipped to ``seqlen_k``) or of rank 1 ``[kv_blocks]``, 2
    ``[batch, kv_blocks]`` or 3 ``[batch, heads, kv_blocks]``.  Duplicate slots select a block once.
    """

    batch, heads, num_q_blocks, _ = q2k_block_index.shape
    mask = torch.full((batch, heads, seqlen_q, seqlen_k), float("-inf"), dtype=torch.float32, device=q2k_block_index.device)
    index = q2k_block_index.cpu()
    nums = q2k_block_nums.cpu() if q2k_block_nums is not None else None
    sizes = block_sizes.cpu() if block_sizes is not None else None
    for b in range(batch):
        for h in range(heads):
            for q_block in range(num_q_blocks):
                count = block_sparse_num if nums is None else int(nums[b, h, q_block])
                q_start = q_block * block_size
                q_end = min(q_start + block_size, seqlen_q)
                for slot in range(count):
                    kv_block = int(index[b, h, q_block, slot])
                    k_start = kv_block * block_size
                    k_end = min(k_start + _block_extent(sizes, b, h, kv_block, block_size), seqlen_k)
                    mask[b, h, q_start:q_end, k_start:k_end] = 0.0
    return mask


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    softmax_scale: Optional[float] = None,
    *,
    do: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
):
    """Masked attention in ``dtype`` for BHSD tensors (MHA, or GQA/MQA with dK/dV summed over each head group).

    Returns ``(o, lse)``, or ``(o, lse, dq, dk, dv)`` when ``do`` is given.  A query row with no visible key gets
    O = 0, LSE = -inf and zero gradients, as in the kernels.
    """

    softmax_scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    group_size = q.shape[1] // k.shape[1]
    q_ref, k_ref, v_ref = (t.detach().to(dtype).requires_grad_(do is not None) for t in (q, k, v))
    visible = torch.isfinite(mask)
    empty = ~visible.any(-1, keepdim=True)
    scores = torch.einsum("bhqd,bhkd->bhqk", q_ref * softmax_scale, k_ref.repeat_interleave(group_size, dim=1)) + mask.to(dtype)
    probabilities = torch.softmax(scores.masked_fill(empty, 0), dim=-1).masked_fill(~visible, 0)
    output = torch.einsum("bhqk,bhkd->bhqd", probabilities, v_ref.repeat_interleave(group_size, dim=1))
    lse = torch.logsumexp(scores, dim=-1)
    if do is None:
        return output.detach(), lse.detach()
    output.backward(do.to(dtype))
    return output.detach(), lse.detach(), q_ref.grad, k_ref.grad, v_ref.grad


def fp8_online_reference(
    quantized,
    indices: torch.Tensor,
    counts,
    softmax_scale: float,
    *,
    block_size: int,
    block_sizes: Optional[torch.Tensor] = None,
    k_quant_block_size: int = 16,
    v_layout: str = "bhsd",
    dtype: torch.dtype = torch.float32,
):
    """Model a Sage FP8 forward: walk each list from its last slot, scale P by 256 and round it to FP8 before P @ V.

    ``quantized`` is ``(q8, k8, v8, q_scale, k_scale, v_scale)`` with Q scales per row, K scales per
    ``k_quant_block_size`` keys and V scales per channel (applied to the normalized output).  ``v8`` is BHSD or, with
    ``v_layout="bhds"``, BHDS.  ``counts`` is an int (fixed count) or a ``[batch, heads, q_blocks]`` tensor;
    ``block_sizes`` uses the ranks of ``block_sparse_mask``.  Returns ``(o, lse)`` in ``dtype``; a query block with no
    visible key gets O = 0 and LSE = -inf.
    """

    q8, k8, v8, qs, ks, vs = quantized
    if v_layout == "bhds":
        v8 = v8.transpose(-1, -2)
    batch, heads, seqlen_q, _ = q8.shape
    seqlen_k = k8.shape[2]
    q = q8.to(dtype) * qs.reshape(batch, heads, seqlen_q, 1).to(dtype)
    k = k8.to(dtype) * ks.reshape(batch, heads, -1).repeat_interleave(k_quant_block_size, dim=-1)[..., :seqlen_k, None].to(dtype)
    v = v8.to(dtype)
    vs = vs.reshape(heads, -1).to(dtype)
    output = torch.zeros(q.shape[:-1] + (v.shape[-1],), device=q.device, dtype=dtype)
    lse = torch.full(q.shape[:-1], -math.inf, device=q.device, dtype=dtype)
    ids = indices.cpu()
    nums = counts.cpu() if isinstance(counts, torch.Tensor) else None
    sizes = block_sizes.cpu() if block_sizes is not None else None
    for b in range(batch):
        for h in range(heads):
            for qb in range(ids.shape[2]):
                rows = slice(qb * block_size, min((qb + 1) * block_size, seqlen_q))
                q_tile = q[b, h, rows]
                row_max = torch.full((q_tile.shape[0], 1), -math.inf, device=q.device, dtype=dtype)
                row_sum = torch.zeros_like(row_max)
                acc = torch.zeros((q_tile.shape[0], v.shape[-1]), device=q.device, dtype=dtype)
                count = int(counts) if nums is None else int(nums[b, h, qb])
                for slot in reversed(range(count)):
                    kb = int(ids[b, h, qb, slot])
                    valid = min(_block_extent(sizes, b, h, kb, block_size), seqlen_k - kb * block_size)
                    if valid <= 0:
                        continue
                    cols = slice(kb * block_size, kb * block_size + valid)
                    scores = (q_tile @ k[b, h, cols].T) * softmax_scale
                    new_max = torch.maximum(row_max, scores.amax(dim=-1, keepdim=True))
                    rescale = torch.exp(row_max - new_max)
                    p = torch.exp(scores - new_max) * 256.0
                    acc = acc * rescale + p.to(torch.float8_e4m3fn).to(dtype) @ v[b, h, cols]
                    row_sum = row_sum * rescale + p.sum(dim=-1, keepdim=True)
                    row_max = new_max
                output[b, h, rows] = torch.where(row_sum > 0, acc / row_sum, 0) * vs[h]
                lse[b, h, rows] = (row_max + torch.log(row_sum) - math.log(256.0)).squeeze(-1)
    return output, lse


def assert_fp8_quantization_matches_fp32(actual: torch.Tensor, scaled: torch.Tensor) -> None:
    """Allow only FP32 one-ULP midpoint flips, never a general FP8 tolerance."""
    expected = scaled.to(actual.dtype).float()
    different = actual.float() != expected
    if different.any():
        midpoint = scaled[different]
        lower = torch.nextafter(midpoint, torch.full_like(midpoint, -torch.inf))
        upper = torch.nextafter(midpoint, torch.full_like(midpoint, torch.inf))
        values = actual.float()[different]
        assert ((values == lower.to(actual.dtype).float()) | (values == upper.to(actual.dtype).float())).all()
