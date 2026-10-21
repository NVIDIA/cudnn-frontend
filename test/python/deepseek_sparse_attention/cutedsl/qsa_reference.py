# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gather-based fp32 references for Qwen Sparse Attention (QSA) and its index layout."""

import torch

GROUP = 12  # query heads per KV head. The kernels take 2 KV heads (the model) or 1 (one of two tensor-parallel ranks).
D = 256
BLOCK = 4
TOPK_BLOCKS = 512
W = TOPK_BLOCKS * BLOCK + BLOCK - 1
SCALE = D**-0.5
CHUNK = 512  # query rows per gather, which bounds the fp32 K / V gathers to a few GB


def make_topk_idxs(seq_len, generator):
    """Index rows in the QSA indexer layout, selecting uniformly random complete blocks."""
    device = generator.device
    pos = torch.arange(seq_len, device=device)
    ncb = (pos + 1) // BLOCK
    nsel = ncb.clamp(max=TOPK_BLOCKS)
    idx = torch.full((seq_len, W + 1), -1, dtype=torch.int64, device=device)
    for r0 in range(0, seq_len, 4096):  # rows per batch, which bounds the random-score tensor
        r1 = min(r0 + 4096, seq_len)
        nb = int(ncb[r1 - 1])
        if nb == 0:
            continue
        scores = torch.rand((r1 - r0, nb), device=device, generator=generator)
        scores.masked_fill_(torch.arange(nb, device=device)[None, :] >= ncb[r0:r1, None], -1.0)
        top = scores.topk(min(TOPK_BLOCKS, nb), dim=1).indices
        top = top.masked_fill(torch.arange(top.shape[1], device=device)[None, :] >= nsel[r0:r1, None], -1)
        tok = torch.where(top[:, :, None] >= 0, top[:, :, None] * BLOCK + torch.arange(BLOCK, device=device), -1)
        idx[r0:r1, : tok.shape[1] * BLOCK] = tok.flatten(1)
    j = torch.arange(BLOCK - 1, device=device)
    in_tail = j[None, :] < ((pos + 1) % BLOCK)[:, None]
    idx.scatter_(1, torch.where(in_tail, nsel[:, None] * BLOCK + j, W), torch.where(in_tail, ncb[:, None] * BLOCK + j, -1))
    return idx[:, :W].to(torch.int32).contiguous()


def reference_forward(q, k, v, topk_idxs):
    """bf16 output and fp32 natural-log LSE of the scaled scores, the backward's saved tensors."""
    h_kv = k.shape[1]
    h_q = GROUP * h_kv
    out = torch.empty_like(q)
    lse = torch.empty((q.shape[0], h_q), dtype=torch.float32, device=q.device)
    kf, vf = k.float(), v.float()
    for r0 in range(0, q.shape[0], CHUNK):
        r1 = min(r0 + CHUNK, q.shape[0])
        idx = topk_idxs[r0:r1].long()
        flat = idx.clamp(min=0).reshape(-1)
        ks = kf.index_select(0, flat).view(*idx.shape, h_kv, D)
        vs = vf.index_select(0, flat).view(*idx.shape, h_kv, D)
        s = torch.einsum("nhgd,nwhd->nhgw", q[r0:r1].float().view(-1, h_kv, GROUP, D), ks) * SCALE
        s = s.masked_fill(~(idx >= 0)[:, None, None, :], float("-inf"))
        row_lse = s.logsumexp(-1)
        o = torch.einsum("nhgw,nwhd->nhgd", torch.exp(s - row_lse[..., None]), vs)
        out[r0:r1] = o.reshape(-1, h_q, D).to(out.dtype)
        lse[r0:r1] = row_lse.reshape(-1, h_q)
    return out, lse


def reference_backward(q, k, v, out, dout, lse, topk_idxs):
    """fp32 (dq, dk, dv) from the saved bf16 ``out`` and fp32 ``lse``, the operands the backward consumes.

    Not autograd: rows with few keys cancel in dP - Delta, so a Delta from the unrounded O misses the tolerance.
    """
    h_kv = k.shape[1]
    h_q = GROUP * h_kv
    kf, vf = k.float(), v.float()
    dq = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    dk = torch.zeros(k.shape, dtype=torch.float32, device=q.device)
    dv = torch.zeros(v.shape, dtype=torch.float32, device=q.device)
    for r0 in range(0, q.shape[0], CHUNK):
        r1 = min(r0 + CHUNK, q.shape[0])
        idx = topk_idxs[r0:r1].long()
        valid = idx >= 0
        flat = idx.clamp(min=0).reshape(-1)
        ks = kf.index_select(0, flat).view(*idx.shape, h_kv, D)
        vs = vf.index_select(0, flat).view(*idx.shape, h_kv, D)
        qf = q[r0:r1].float().view(-1, h_kv, GROUP, D)
        dof = dout[r0:r1].float().view(-1, h_kv, GROUP, D)
        s = torch.einsum("nhgd,nwhd->nhgw", qf, ks) * SCALE
        p = torch.exp(s.masked_fill(~valid[:, None, None, :], float("-inf")) - lse[r0:r1].view(-1, h_kv, GROUP, 1))
        delta = (dof * out[r0:r1].float().view(-1, h_kv, GROUP, D)).sum(-1, keepdim=True)
        ds = p * (torch.einsum("nhgd,nwhd->nhgw", dof, vs) - delta)
        dq[r0:r1] = (torch.einsum("nhgw,nwhd->nhgd", ds, ks) * SCALE).reshape(-1, h_q, D)
        dk.index_add_(0, flat, (torch.einsum("nhgw,nhgd->nwhd", ds, qf) * SCALE).reshape(-1, h_kv, D))
        dv.index_add_(0, flat, torch.einsum("nhgw,nhgd->nwhd", p, dof).reshape(-1, h_kv, D))
    return dq, dk, dv


def make_inputs(seq_len, seed, grad_scale=1.0, device="cuda", h_kv=2):
    """Random QSA backward operands. ``grad_scale`` shrinks dO to training-like gradient magnitudes."""
    generator = torch.Generator(device=device).manual_seed(seed)
    h_q = GROUP * h_kv

    def randn(*shape):
        return torch.randn(shape, device=device, generator=generator, dtype=torch.float32).to(torch.bfloat16)

    q, k, v = randn(seq_len, h_q, D), randn(seq_len, h_kv, D), randn(seq_len, h_kv, D)
    dout = (torch.randn((seq_len, h_q, D), device=device, generator=generator) * grad_scale).to(torch.bfloat16)
    topk_idxs = make_topk_idxs(seq_len, generator)
    out, lse = reference_forward(q, k, v, topk_idxs)
    return {"q": q, "k": k, "v": v, "out": out, "dout": dout, "lse": lse, "topk_idxs": topk_idxs}


def assert_grads_close(name, actual, expected, rel_l2_tol=1e-2, row_rel_tol=5e-2, row_floor=0.1):
    """Global relative L2 error plus a per-(token, head) row bound that catches localized errors.

    Both bounds are relative because gradient magnitudes vary by orders of magnitude across rows.
    """
    a, e = actual.float(), expected.float()
    assert torch.isfinite(a).all(), f"{name} has non-finite values"
    err = a - e
    rel_l2 = float(err.norm() / e.norm().clamp_min(1e-30))
    row_ref = e.norm(dim=-1)
    row_rel = err.norm(dim=-1) / (row_ref + row_floor * row_ref.pow(2).mean().sqrt()).clamp_min(1e-30)
    worst = float(row_rel.max())
    assert rel_l2 <= rel_l2_tol, f"{name}: relative L2 error {rel_l2:.3e} > {rel_l2_tol}"
    assert worst <= row_rel_tol, f"{name}: worst row error {worst:.3e} > {row_rel_tol} at row {divmod(int(row_rel.argmax()), row_rel.shape[1])}"
