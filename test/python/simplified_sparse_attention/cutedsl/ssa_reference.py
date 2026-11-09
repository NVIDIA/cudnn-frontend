# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent selected-support reference and fixed packed inputs."""

import torch

from cudnn.ops import SimplifiedSparseAttentionMetadata, simplified_sparse_attention
from determinism_utils import snapshot


def sample(length, heads=16, seed=17):
    generator = torch.Generator().manual_seed(seed)
    shapes = ((length, heads, 256), (length, 1, 256), (length, 1, 256), (length, 128), (length, 128))
    tensors = [(0.2 * torch.randn(shape, generator=generator)).bfloat16().cuda() for shape in shapes]
    upstream = torch.randn(shapes[0], generator=generator).bfloat16().cuda()
    return tensors, upstream


def pack(samples, denominator=257.0):
    tensors = [torch.cat([s[0][i] for s in samples]) for i in range(5)]
    upstream = torch.cat([s[1] for s in samples])
    starts, ends, valid = [], [], []
    offset = 0
    for inputs, _ in samples:
        length = inputs[0].shape[0]
        rows = torch.arange(length, device="cuda")
        live = rows % 7 != 0
        starts.append(torch.full((length,), offset, device="cuda", dtype=torch.int32))
        ends.append(torch.where(live, offset + rows + 1, offset).to(torch.int32))
        valid.append(live)
        offset += length
    metadata = SimplifiedSparseAttentionMetadata(
        torch.cat(starts),
        torch.cat(ends),
        torch.cat(valid),
        torch.tensor([denominator], device="cuda"),
        False,
        True,
    )
    return tensors, metadata, upstream


def run(inputs, backward=False, topk=512):
    tensors, metadata, upstream = inputs
    local = [t.detach().clone().requires_grad_(backward) for t in tensors]
    output, loss = simplified_sparse_attention(*local, metadata, topk)
    result = {"output": output, "indexer_loss": loss}
    if backward:
        grads = torch.autograd.grad((output, loss), local, (upstream, torch.ones_like(loss)))
        result.update(zip(("dq", "dk", "dv", "dq_idx", "dk_idx"), grads))
    return snapshot(result)


def reference(inputs, backward=False, topk=512):
    tensors, metadata, upstream = inputs
    q, k, v, iq, ik = [t.detach().double().requires_grad_(backward) for t in tensors]
    ids = torch.arange(k.shape[0], device=k.device)
    valid = metadata.query_valid
    allowed = (ids[None, :] >= metadata.key_starts[:, None]) & (ids[None, :] < metadata.key_ends[:, None])
    safe = (valid[:, None] & allowed) | (~valid[:, None] & (ids[None, :] == 0))
    scores = iq @ ik.T / (128**0.5)
    if k.shape[0] > topk:
        selected = scores.detach().masked_fill(~safe, -torch.inf).topk(topk, dim=-1).indices
        support = torch.zeros_like(safe).scatter_(1, selected, True)
        safe = safe & support
    p = (torch.einsum("qhd,kd->hqk", q, k[:, 0]) / 16).masked_fill(~safe[None], -torch.inf).softmax(-1)
    output = torch.einsum("hqk,kd->qhd", p, v[:, 0]).masked_fill(~valid[:, None, None], 0)
    target = p.detach().mean(0)
    student = scores.masked_fill(~safe, -torch.inf).softmax(-1)
    loss = (target * ((target + 1e-10).log() - (student + 1e-10).log())).sum(-1)
    loss = loss.masked_fill(~valid, 0).sum() * 0.1 / metadata.loss_denominator[0]
    result = {"output": output.detach(), "indexer_loss": loss.detach()}
    if backward:
        grads = torch.autograd.grad((output, loss), (q, k, v, iq, ik), (upstream.double(), torch.ones_like(loss)))
        result.update(zip(("dq", "dk", "dv", "dq_idx", "dk_idx"), grads))
    return result


def check_reference(inputs, actual, backward=False, topk=512):
    expected = reference(inputs, backward, topk)
    for name, value in actual.items():
        want = expected[name]
        assert torch.isfinite(value).all(), name
        error = (value.double() - want).norm() / want.norm().clamp_min(1e-12)
        assert error < 0.01, f"{name}: relative L2 error {error.item()}"
        torch.testing.assert_close(value.double(), want, rtol=0.03, atol=0.005 * want.abs().max().item() + 1e-9)
