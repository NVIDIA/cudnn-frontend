# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent full-softmax teacher and autograd reference for Phase 1."""

import torch

from cudnn.ops import dense_indexer_kl
from determinism_utils import assert_bitwise, snapshot

COEFFICIENT = 0.17
UPSTREAM = 0.7


def sample(rows, keys, heads=16, seed=17, device="cuda", dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(seed)
    shapes = ((rows, heads, 256), (keys, 1, 256), (rows, 128), (keys, 128))
    tensors = [torch.randn(shape, generator=generator).to(device=device, dtype=dtype) for shape in shapes]
    starts = torch.zeros(rows, device=device, dtype=torch.int32)
    ends = torch.arange(rows, device=device, dtype=torch.int32) % keys + 1
    ends[1::3] = keys
    starts[1::3] = keys // 3
    valid = torch.ones(rows, device=device, dtype=torch.bool)
    valid[-1] = False
    ends[-1] = starts[-1]
    return tensors, (starts, ends, valid, torch.tensor([71.0], device=device))


def pack(samples, target):
    tensors = [torch.cat([s[0][i] for s in samples]) for i in range(4)]
    starts, ends, valid = [], [], []
    offset = 0
    for i, (values, metadata) in enumerate(samples):
        starts.append(metadata[0] + offset)
        ends.append(metadata[1] + offset)
        valid.append(metadata[2] if i == target else torch.zeros_like(metadata[2]))
        offset += values[1].shape[0]
    return tensors, (torch.cat(starts), torch.cat(ends), torch.cat(valid), samples[target][1][3])


def reference(inputs):
    (q, k, iq, ik), (starts, ends, valid, denominator) = inputs
    loss = torch.zeros((), device=q.device, dtype=torch.float64)
    dq = torch.zeros_like(iq, dtype=torch.float64)
    dk = torch.zeros_like(ik, dtype=torch.float64)
    lse = torch.zeros(q.shape[:2], device=q.device, dtype=torch.float32)
    positions = torch.arange(k.shape[0], device=q.device)
    # Bound reference memory without splitting the key reduction.
    for begin in range(0, q.shape[0], 16):
        end = min(begin + 16, q.shape[0])
        mask = (positions >= starts[begin:end, None]) & (positions < ends[begin:end, None])
        mask &= valid[begin:end, None]
        safe_mask = mask | (~valid[begin:end, None] & (positions == 0))
        scores = (q[begin:end].double() @ k[:, 0].double().T) / 16
        scores = scores.masked_fill(~safe_mask[:, None, :], -torch.inf)
        lse[begin:end] = scores.logsumexp(-1).float().masked_fill(~valid[begin:end, None], 0)
        teacher = scores.softmax(-1).mean(1).detach()
        qi = iq[begin:end].double().detach().requires_grad_()
        ki = ik.double().detach().requires_grad_()
        log_student = ((qi @ ki.T) / (128**0.5)).masked_fill(~safe_mask, -torch.inf).log_softmax(-1)
        terms = teacher * (teacher.clamp_min(1e-300).log() - log_student.masked_fill(~safe_mask, 0))
        chunk_loss = terms.masked_fill(~mask, 0).sum() * COEFFICIENT / denominator.double().reshape(())
        chunk_dq, chunk_dk = torch.autograd.grad(chunk_loss * UPSTREAM, (qi, ki))
        loss += chunk_loss.detach()
        dq[begin:end] = chunk_dq
        dk += chunk_dk
    return {"loss": loss.float(), "dq_idx": dq.to(iq.dtype), "dk_idx": dk.to(ik.dtype)}, lse


def run(inputs, use_kf, lse, backward=True):
    tensors = [x.detach().clone().requires_grad_() for x in inputs[0]]
    metadata = [x.clone() for x in inputs[1]]
    before = snapshot(dict(enumerate((*tensors, *metadata, lse))))
    starts, ends, valid, denominator = metadata
    loss = dense_indexer_kl(
        *tensors,
        key_starts=starts,
        key_ends=ends,
        query_valid=valid,
        loss_denominator=denominator,
        teacher_softmax_scale=1 / 16,
        student_softmax_scale=128**-0.5,
        loss_coeff=COEFFICIENT,
        teacher_lse=lse,
        use_kf=use_kf,
    )
    result = {"loss": loss}
    if backward:
        result.update(zip(("saved_dq_idx", "saved_dk_idx"), loss.grad_fn.saved_tensors))
        grads = torch.autograd.grad(loss * UPSTREAM, tensors, allow_unused=True)
        assert grads[:2] == (None, None), "Teacher must stay frozen"
        result.update(dq_idx=grads[2], dk_idx=grads[3])
    assert_bitwise(snapshot(dict(enumerate((*tensors, *metadata, lse)))), before, "input mutation")
    return snapshot(result)


def check_reference(actual, expected):
    torch.testing.assert_close(actual["loss"], expected["loss"], rtol=2e-4, atol=2e-7)
    for name in ("dq_idx", "dk_idx"):
        if name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=0.02, atol=2e-6)
