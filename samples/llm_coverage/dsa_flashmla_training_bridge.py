# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal B200 training example for the optional FlashMLA/cuDNN DSA bridge.

The forward is dynamically provided by the separately installed, official
deepseek-ai/FlashMLA package.  cuDNN Frontend supplies backward and attention
score recompute; this sample does not contain a forward kernel implementation.
"""

import math

import torch

from cudnn import DSA


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("this prototype sample requires SM100")

    torch.manual_seed(123)
    device = torch.device("cuda")
    s_q, s_kv, heads, head_dim, topk = 4, 96, 32, 576, 65
    scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(s_q, heads, head_dim, dtype=torch.bfloat16, device=device, requires_grad=True)
    kv = torch.randn(s_kv, head_dim, dtype=torch.bfloat16, device=device, requires_grad=True)
    attn_sink = torch.randn(heads, dtype=torch.float32, device=device, requires_grad=True)
    indices = torch.stack([torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]).to(torch.int32)
    topk_length = torch.tensor([65, 64, 33, 1], dtype=torch.int32, device=device)
    inactive = torch.arange(topk, device=device)[None, :] >= topk_length[:, None]
    indices.masked_fill_(inactive, -1)

    # H32 is padded to the official FlashMLA H64 launch, K65 to K128.  The
    # returned tensors and all cuDNN gradients retain the original H32/K65 ABI.
    # Trusted metadata requires a bounded active prefix AND an invalid
    # (-1) inactive suffix, constructed above. This lets the adapter skip
    # the safe-default metadata scan and compactification.
    result = DSA.sparse_attention(
        q,
        kv,
        indices,
        attn_sink,
        softmax_scale=scale,
        topk_length=topk_length,
        trusted_compact_metadata=True,
    )
    result["output"].float().square().mean().backward()

    score = DSA.sparse_attention_score_recompute(
        q.detach(),
        kv.detach(),
        result["lse"],
        indices,
        softmax_scale=scale,
        topk_length=topk_length,
    )["target"]

    print("output/lse/target:", result["output"].shape, result["lse"].shape, score.shape)
    print("dq/dkv/d_sink:", q.grad.shape, kv.grad.shape, attn_sink.grad.shape)


if __name__ == "__main__":
    main()
