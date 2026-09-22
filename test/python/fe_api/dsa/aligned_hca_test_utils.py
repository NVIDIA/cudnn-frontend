# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent physical HCA mask and chunked autograd reference."""

import torch

from fe_api.dsa.dsa_reference import ref_sparse_attention_forward


def physical_indices(rank):
    position = rank * 4096 + torch.arange(4096, device="cuda")
    window_count = (position + 1).clamp(max=128)
    compressed_count = (position + 1) // 128
    slot = torch.arange(640, device="cuda")[None, :]
    window = (position - 127).clamp(min=0)[:, None] + slot - rank * 4096 + 128
    compressed = (slot - window_count[:, None]).clamp(min=0)
    owner, local = compressed // 32, compressed % 32
    compressed = 4224 + owner * 33 + local + (owner > 0)
    indices = torch.where(slot < window_count[:, None], window, compressed)
    return torch.where(slot < (window_count + compressed_count)[:, None], indices, -1).to(torch.int32)


def reference_case(rank, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (4096, 128, 512)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    kv = torch.randn((4752, 512), device="cuda", dtype=torch.bfloat16, generator=generator)
    sink = torch.randn(128, device="cuda", generator=generator)
    dout = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    out, lse = torch.empty_like(q), torch.empty((4096, 128), device="cuda")
    dq, dkv, dsink = torch.empty_like(q, dtype=torch.float32), torch.zeros_like(kv, dtype=torch.float32), torch.zeros_like(sink)
    indices = physical_indices(rank)
    kv_ref, sink_ref = kv.float().requires_grad_(), sink.detach().requires_grad_()
    prior_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for start in range(0, 4096, 32):
            end = start + 32
            q_ref = q[start:end].float().requires_grad_()
            out_ref, lse_ref = ref_sparse_attention_forward(q_ref, kv_ref, sink_ref, indices[start:end], softmax_scale=512**-0.5)
            grads = torch.autograd.grad(out_ref, (q_ref, kv_ref, sink_ref), dout[start:end].float())
            with torch.no_grad():
                out[start:end].copy_(out_ref)
                lse[start:end].copy_(lse_ref)
                dq[start:end].copy_(grads[0])
                dkv.add_(grads[1])
                dsink.add_(grads[2])
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_tf32
    return (q, kv, out, dout, lse, sink), dict(dq=dq, dkv=dkv, d_sink=dsink)


def assert_reference(actual, expected):
    for name in expected:
        got, want = actual[name].float(), expected[name]
        assert torch.isfinite(got).all(), name
        relative_l2 = (got - want).norm() / want.norm().clamp_min(1e-30)
        matched = ((got - want).abs() <= 0.05 + 0.05 * want.abs()).float().mean()
        assert relative_l2 < 0.02, (name, relative_l2.item())
        assert matched >= 0.99, (name, matched.item())
    unused = torch.tensor([4224 + 32] + [4224 + rank * 33 for rank in range(1, 16)], device="cuda")
    assert torch.count_nonzero(actual["dkv"][unused]) == 0
