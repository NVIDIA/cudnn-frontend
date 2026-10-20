# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from cudnn.flex_attention import create_mask_plan, flex_attn_func


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.L2
@pytest.mark.parametrize(
    "head_dim,kv_heads,mask_kind,dtype",
    [
        (128, 4, "causal", torch.bfloat16),
        (256, 4, "causal", torch.bfloat16),
        (128, 2, "intervals", torch.bfloat16),
        (256, 4, "intervals", torch.bfloat16),
        (128, 4, "mixed", torch.bfloat16),
        (256, 2, "mixed", torch.bfloat16),
        (64, 2, "mixed", torch.float16),
    ],
)
def test_sm90_varlen_mask_pipeline_reuse(head_dim, kv_heads, mask_kind, dtype):
    """Exercise mask-slot reuse, including full public calls under racecheck."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("This mask pipeline regression targets SM90")

    torch.manual_seed(2035)
    q_lengths, k_lengths = (193, 65), (513, 257)
    heads = 4
    q = torch.randn((sum(q_lengths), heads, head_dim), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn((sum(k_lengths), kv_heads, head_dim), device="cuda", dtype=q.dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    dout = torch.randn_like(q)
    cu_q = torch.tensor([0, q_lengths[0], sum(q_lengths)], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, k_lengths[0], sum(k_lengths)], device="cuda", dtype=torch.int32)

    endpoints, masks = [], []
    for sq, sk in zip(q_lengths, k_lengths):
        rows = torch.arange(sq, dtype=torch.int32)
        if mask_kind == "causal":
            ep = (rows + 1).clamp(max=sk).view(1, 1, sq).expand(heads, -1, -1).clone()
        else:
            # Four separated intervals force more than two partial KV blocks.
            ep = torch.zeros((heads, 9, sq), dtype=torch.int32)
            for h in range(heads):
                for i, start in enumerate((16, 144, 272, 400)):
                    ep[h, 2 * i + 1] = (start + (rows + h) % 7).clamp(max=sk)
                    ep[h, 2 * i + 2] = (start + 31 + (rows + h) % 7).clamp(max=sk)
            if mask_kind == "mixed":
                # A full prefix, partial blocks, skipped blocks, and an empty Q tile.
                ep[:, :3] = 128
                ep[:, :, rows >= 128] = 0
            else:
                # Include empty rows without introducing undefined softmax gradients.
                ep[:, :, rows % 11 == 0] = 0
        keys = torch.arange(sk)
        visible = keys < ep[:, 0, :, None]
        for i in range(1, ep.shape[1], 2):
            visible |= (keys >= ep[:, i, :, None]) & (keys < ep[:, i + 1, :, None])
        endpoints.append(ep)
        masks.append(visible)
    plan = create_mask_plan(
        torch.cat(endpoints, dim=-1).to("cuda"),
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        build_backward=True,
    )

    def run():
        out, lse = flex_attn_func(q, k, v, mask_plan=plan, return_lse=True)
        return out, lse, *torch.autograd.grad(out, (q, k, v), dout)

    def check(actual):
        # Keep the independent reference on CPU so sanitizer covers only the DUT.
        qr, kr, vr = [t.detach().cpu().float().requires_grad_() for t in (q, k, v)]
        parts, stats = [], []
        qo = ko = 0
        for sq, sk, visible in zip(q_lengths, k_lengths, masks):
            ks = kr[ko : ko + sk].repeat_interleave(heads // kv_heads, dim=1)
            vs = vr[ko : ko + sk].repeat_interleave(heads // kv_heads, dim=1)
            scores = torch.einsum("qhd,khd->hqk", qr[qo : qo + sq], ks) / math.sqrt(head_dim)
            scores = scores.masked_fill(~visible, float("-inf"))
            has_keys = visible.any(-1, keepdim=True)
            probabilities = torch.softmax(torch.where(has_keys, scores, 0.0), dim=-1)
            probabilities = torch.where(has_keys, probabilities, 0.0)
            parts.append(torch.einsum("hqk,khd->qhd", probabilities, vs))
            stats.append(torch.logsumexp(scores, dim=-1))
            qo += sq
            ko += sk
        out_ref = torch.cat(parts)
        grads = torch.autograd.grad(out_ref, (qr, kr, vr), dout.cpu().float())
        for i, (got, expected) in enumerate(zip(actual, (out_ref, torch.cat(stats, dim=1), *grads))):
            tolerance = 3e-2 if i < 2 else 5e-2
            torch.testing.assert_close(got.cpu().float(), expected, atol=tolerance, rtol=tolerance)

    check(run())
    if mask_kind != "causal":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                captured = run()
            for _ in range(2):
                with torch.no_grad():
                    for t in (q, k, v, dout):
                        t.copy_(torch.randn_like(t))
                graph.replay()
                check(captured)
        finally:
            torch.cuda.synchronize()
            graph.reset()
