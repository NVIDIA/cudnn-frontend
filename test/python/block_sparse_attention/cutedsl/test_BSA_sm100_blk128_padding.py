# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM100 blk128 padding, CSR buckets and compile specialization regression."""

import pytest
import torch

from cudnn import BSA


@pytest.mark.L0
@pytest.mark.parametrize("arch", [107, 110])
def test_blk128_padding_rejects_unsupported_arch_before_dispatch(monkeypatch, arch):
    from cudnn.block_sparse_attention import _interface, api

    q = torch.zeros(1, 128, 1, 128, device="cuda", dtype=torch.bfloat16)
    lse = torch.zeros(1, 1, 128, device="cuda", dtype=torch.float32)
    ids = torch.zeros(1, 1, 1, 1, device="cuda", dtype=torch.int32)
    counts = torch.ones(1, 1, 1, device="cuda", dtype=torch.int32)
    sizes = torch.ones(1, device="cuda", dtype=torch.int32)
    monkeypatch.setattr(api, "_device_arch", lambda _: arch)

    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("Unsupported padding reached kernel dispatch")

    monkeypatch.setattr(_interface, "bsa_attn_bwd", unexpected_dispatch)
    with pytest.raises(NotImplementedError, match=f"SM{arch} blk128 backward does not yet support block_sizes"):
        BSA.block_sparse_attention_backward(q, q, q, q, q, lse, ids, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=128, layout="bshd")


@pytest.mark.L0
@pytest.mark.parametrize("bucket", [1, 128])
@pytest.mark.parametrize("sq,sk,dim,layout", [(512, 512, 128, "bshd"), (389, 517, 128, "bhsd"), (257, 257, 64, "bshd")])
def test_sm100_blk128_padding(bucket, sq, sk, dim, layout):
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Requires SM100/SM103")
    torch.manual_seed(13)
    q = torch.randn(2, sq, 2, dim, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(2, sk, 2, dim, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    do = torch.randn_like(q)
    nq, nk = (sq + 127) // 128, (sk + 127) // 128
    sizes = torch.tensor(([17, 128, 69, 1, 0])[:nk], device="cuda", dtype=torch.int32)
    sizes[-1] = min(1, sk - 128 * (nk - 1))
    sizes = torch.stack((sizes, torch.zeros_like(sizes)), dim=1)[:, 0]  # Strided metadata.
    ids = torch.arange(nk, device="cuda", dtype=torch.int32).view(1, 1, 1, nk).expand(2, 2, nq, nk).contiguous()
    counts = torch.tensor([1, nk, 0, nk][:nq], device="cuda", dtype=torch.int32).view(1, 1, nq).expand(2, 2, nq).contiguous()
    # A physical KV tail must be declared to forward even when other blocks are full.
    full_sizes = torch.full_like(sizes, 128)
    full_sizes[-1] = sk - 128 * (nk - 1)
    full_sizes = None if sk % 128 == 0 else full_sizes
    # None -> partial -> None on aligned inputs exercises separate specializations.
    for block_sizes in (full_sizes, sizes, full_sizes):
        qpos, kpos = torch.arange(sq, device="cuda"), torch.arange(sk, device="cuda")
        visible = kpos[None, :] // 128 < counts[:, :, qpos // 128, None]
        if block_sizes is not None:
            visible = visible & (kpos % 128 < block_sizes[kpos // 128])
        qr, kr, vr = [t.double().requires_grad_(True) for t in (q, k, v)]
        score = (qr.transpose(1, 2) @ kr.transpose(1, 2).transpose(-1, -2)) / dim**0.5
        score = score.masked_fill(~visible, -float("inf"))
        prob = torch.where(visible.any(-1, keepdim=True), score, 0).softmax(-1).masked_fill(~visible, 0)
        ref = (prob @ vr.transpose(1, 2)).transpose(1, 2)
        ref_grads = torch.autograd.grad(ref, (qr, kr, vr), do.double())
        inputs = [t.transpose(1, 2).contiguous() for t in (q, k, v, do)] if layout == "bhsd" else (q, k, v, do)
        qin, kin, vin, din = inputs
        out, lse = BSA.block_sparse_attention_forward(
            qin, kin, vin, ids, q2k_block_nums=counts, block_sizes=block_sizes, sparse_block_size=128, layout=layout, allow_empty_block_nums=True
        )
        grads = BSA.block_sparse_attention_backward(
            din, qin, kin, vin, out, lse, ids, q2k_block_nums=counts, block_sizes=block_sizes, sparse_block_size=128, layout=layout, bucket_size_blocks=bucket
        )
        for name, actual, expected in zip(("O", "dQ", "dK", "dV"), (out, *grads), (ref, *ref_grads)):
            actual = actual.transpose(1, 2) if layout == "bhsd" else actual
            torch.testing.assert_close(actual.double(), expected, atol=0.03, rtol=0.03, msg=lambda msg: name + ": " + msg)


@pytest.mark.L0
def test_sm100_blk128_padding_graph_replay():
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Requires SM100/SM103")
    torch.manual_seed(19)
    q, k, v, do = [torch.randn(1, 256, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    ids = torch.tensor([0, 1], device="cuda", dtype=torch.int32).view(1, 1, 1, 2).expand(1, 2, 2, 2).contiguous()
    sizes = torch.tensor([17, 69], device="cuda", dtype=torch.int32)

    def run():
        out, lse = BSA.block_sparse_attention_forward(q, k, v, ids, 2, sizes, sparse_block_size=128, layout="bshd")
        grads = BSA.block_sparse_attention_backward(do, q, k, v, out, lse, ids, 2, sizes, sparse_block_size=128, layout="bshd")
        return (out, *grads)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = run()
        for lengths in ([128, 1], [17, 69]):
            sizes.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
            graph.replay()
            for actual, expected in zip(captured, run()):
                torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    finally:
        graph.reset()


@pytest.mark.L1
def test_sm100_blk128_auto_bucket_cache_transitions():
    """One visible token gives an exact oracle across single/multiple Q groups."""
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Requires SM100/SM103")
    k = torch.zeros(1, 128, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.full_like(k, -7)
    v[:, 0] = 2
    sizes = torch.ones(1, device="cuda", dtype=torch.int32)
    # Reuse the same process and caches across the automatic bucket boundary.
    for blocks in (1499, 1500, 2049, 1500, 1499):
        q = torch.zeros(1, blocks * 128, 2, 128, device="cuda", dtype=torch.bfloat16)
        do = torch.ones_like(q)
        ids = torch.zeros(1, 2, blocks, 4, device="cuda", dtype=torch.int32)
        counts = torch.ones(1, 2, blocks, device="cuda", dtype=torch.int32)
        out, lse = BSA.block_sparse_attention_forward(q, k, v, ids, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=128, layout="bshd")
        dq, dk, dv = BSA.block_sparse_attention_backward(
            do, q, k, v, out, lse, ids, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=128, layout="bshd"
        )
        expected_dv = torch.zeros_like(v)
        expected_dv[:, 0] = blocks * 128
        for actual, expected in ((out, torch.full_like(q, 2)), (dq, torch.zeros_like(q)), (dk, torch.zeros_like(k)), (dv, expected_dv)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.L0
def test_sm100_blk128_kv_prefix_boundaries():
    """Partial KV prefixes must preserve all gradients across internal MMA boundaries."""
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Requires SM100/SM103")
    torch.manual_seed(123)
    q, k, v, do = [torch.randn(1, 256, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    ids = torch.tensor([0, 1], device="cuda", dtype=torch.int32).view(1, 1, 1, 2).expand(1, 2, 2, 2).contiguous()
    kpos = torch.arange(256, device="cuda")
    for prefix in (0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 77, 78, 79, 80, 81, 88, 94, 95, 96, 97, 104, 111, 112, 113, 127, 128):
        sizes = torch.tensor([prefix, 128 - prefix], device="cuda", dtype=torch.int32)
        visible = kpos % 128 < sizes[kpos // 128]
        qr, kr, vr = [t.double().requires_grad_(True) for t in (q, k, v)]
        score = (qr.transpose(1, 2) @ kr.transpose(1, 2).transpose(-1, -2)) / 128**0.5
        ref = (score.masked_fill(~visible, -float("inf")).softmax(-1) @ vr.transpose(1, 2)).transpose(1, 2)
        ref_grads = torch.autograd.grad(ref, (qr, kr, vr), do.double())
        out, lse = BSA.block_sparse_attention_forward(q, k, v, ids, 2, sizes, sparse_block_size=128, layout="bshd")
        grads = BSA.block_sparse_attention_backward(do, q, k, v, out, lse, ids, 2, sizes, sparse_block_size=128, layout="bshd")
        for name, actual, expected in zip(("O", "dQ", "dK", "dV"), (out, *grads), (ref, *ref_grads)):
            torch.testing.assert_close(actual.double(), expected, atol=0.03, rtol=0.03, msg=lambda msg: name + ": " + msg)
