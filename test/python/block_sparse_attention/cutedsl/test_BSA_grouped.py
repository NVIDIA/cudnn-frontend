# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cudnn
from cudnn.frost.buffers import cutedsl_requirement_error

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.fixture(autouse=True)
def supported():
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
        pytest.skip("grouped BSA requires SM100/SM103")
    error = cutedsl_requirement_error("grouped BSA tests")
    if error:
        pytest.skip(error)


def metadata(*, unused=False, identity=False):
    """Build the oracle graph first, then expand and independently permute it."""
    rng = torch.Generator().manual_seed(94)
    coarse = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0 if unused else 1]], dtype=torch.bool)
    dense = coarse.repeat_interleave(2, 0).repeat_interleave(2, 1)
    routes = torch.full((2, 2, 6, 8), -999, dtype=torch.int32)
    counts = torch.empty((2, 2, 6), dtype=torch.int32)
    masks = torch.empty((2, 2, 6, 8), dtype=torch.bool)
    for b in range(2):
        for h in range(2):
            qp = torch.arange(6) if identity else torch.randperm(6, generator=rng)
            kp = torch.arange(8) if identity else torch.randperm(8, generator=rng)
            mask = dense[qp][:, kp]
            masks[b, h] = mask
            for q in range(6):
                indices = torch.where(mask[q])[0].int()
                # Equal neighborhoods may have different route orders.
                if q % 2:
                    indices = indices.flip(0)
                counts[b, h, q] = len(indices)
                routes[b, h, q, : len(indices)] = indices
    return routes.cuda(), counts.cuda(), masks


def inputs():
    torch.manual_seed(26)
    q = torch.randn((2, 384, 2, 128), device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn((2, 512, 2, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    return q, k, v, torch.randn_like(q)


def reference(q, k, v, dout, masks, scale=128**-0.5):
    mask = masks.repeat_interleave(64, -2).repeat_interleave(64, -1).cuda()
    # NaNs in unused K/V are intentional. Remove them only in the oracle.
    active = mask.any(-2).transpose(1, 2).unsqueeze(-1)
    qr, kr, vr = [x.double().detach().requires_grad_(True) for x in (q, torch.where(active, k, 0), torch.where(active, v, 0))]
    scores = (qr.transpose(1, 2) @ kr.transpose(1, 2).transpose(-1, -2)) * scale
    scores = scores.masked_fill(~mask, -torch.inf)
    lse = scores.logsumexp(-1)
    out = (scores.softmax(-1) @ vr.transpose(1, 2)).transpose(1, 2)
    grads = torch.autograd.grad(out, (qr, kr, vr), dout.double())
    return out, lse, grads


def assert_numerics(actual, expected, *, grad=False):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, atol=0.03 if grad else 0.02, rtol=0.03 if grad else 0.02)
    relative = (actual.double() - expected).norm() / expected.norm().clamp_min(1e-10)
    assert relative < (0.012 if grad else 0.006)


@pytest.mark.parametrize("bucket", [1, 3])
def test_exact_graph_and_csr(bucket):
    routes, counts, mask = metadata()
    grouping = cudnn.block_sparse_attention_grouped_prepare(routes, num_kv_blocks=8, q2k_block_nums=counts, bucket_size_blocks=bucket)
    qp, kp, coarse, nums, offsets, edges = [
        x.cpu()
        for x in (
            grouping.q_permutation,
            grouping.kv_permutation,
            grouping.q2k_block_index,
            grouping.q2k_block_nums,
            grouping.bucketed_k2q_offsets,
            grouping.bucketed_k2q_indices,
        )
    ]
    for b in range(2):
        for h in range(2):
            assert sorted(qp[b, h].tolist()) == list(range(6))
            assert sorted(kp[b, h].tolist()) == list(range(8))
            reconstructed = torch.zeros_like(mask[b, h])
            reverse = torch.zeros((3, 4), dtype=torch.bool)
            for q in range(3):
                for kv in coarse[b, h, q, : nums[b, h, q]]:
                    for qi in qp[b, h, 2 * q : 2 * q + 2]:
                        reconstructed[qi, kp[b, h, 2 * kv : 2 * kv + 2]] = True
                    reverse[q, kv] = True
            assert torch.equal(reconstructed, mask[b, h])
            from_csr = torch.zeros_like(reverse)
            for group in range(offsets.shape[2]):
                for kv in range(4):
                    begin, end = offsets[b, h, group, kv : kv + 2]
                    neighbors = edges[b, h, begin:end]
                    assert all(group * bucket <= int(q) < min((group + 1) * bucket, 3) for q in neighbors)
                    from_csr[neighbors, kv] = True
            assert torch.equal(from_csr, reverse)
    assert grouping.edges64 == int(mask.sum()) == 4 * int(nums.sum())
    # The caller may reuse/mutate its original buffers; preparation is a snapshot.
    routes.fill_(-1)
    counts.zero_()
    assert torch.equal(grouping.q2k_block_index.cpu(), coarse)


@pytest.mark.parametrize("bucket,unused,identity", [(1, False, False), (3, True, False), (3, False, True)])
def test_forward_backward_reference_and_changed_input_replay(bucket, unused, identity):
    routes, counts, masks = metadata(unused=unused, identity=identity)
    grouping = cudnn.block_sparse_attention_grouped_prepare(routes, num_kv_blocks=8, q2k_block_nums=counts, bucket_size_blocks=bucket)
    q, k, v, dout = inputs()
    if unused:
        active = masks.any(-2).repeat_interleave(64, -1).transpose(1, 2).unsqueeze(-1).cuda()
        k.masked_fill_(~active, float("nan"))
        v.masked_fill_(~active, float("nan"))
    scale = 0.17

    def run():
        out = cudnn.BSA.block_sparse_attention_grouped_forward(q, k, v, grouping, softmax_scale=scale)
        grads = cudnn.BSA.block_sparse_attention_grouped_backward(dout, q, k, v, out[0], out[1], grouping, softmax_scale=scale)
        return out, grads

    eager, grads = run()
    expected = reference(q, k, v, dout, masks, scale)
    assert_numerics(eager[0], expected[0])
    torch.testing.assert_close(eager[1].double(), expected[1], atol=2e-5, rtol=2e-6)
    for actual, ref in zip(grads, expected[2]):
        assert_numerics(actual, ref, grad=True)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured, captured_grads = run()
        for repeat in range(2):
            q.mul_(0.75)
            dout.mul_(-0.5)
            expected = reference(q, k, v, dout, masks, scale)
            for output in (*captured, *captured_grads):
                output.fill_(float("nan"))
            graph.replay()
            assert_numerics(captured[0], expected[0])
            torch.testing.assert_close(captured[1].double(), expected[1], atol=2e-5, rtol=2e-6)
            for actual, ref in zip(captured_grads, expected[2]):
                assert_numerics(actual, ref, grad=True)
            if unused:
                for g in (captured_grads[1], captured_grads[2]):
                    assert torch.count_nonzero(g.masked_select(~active)) == 0
    finally:
        graph.reset()


@pytest.mark.parametrize("bad", ["negative", "range", "duplicate", "empty", "capacity", "unpaired_q", "unpaired_kv"])
def test_reject_invalid_graph(bad):
    routes, counts, _ = metadata(identity=True)
    if bad == "negative":
        routes[0, 0, 0, 0] = -1
    elif bad == "range":
        routes[0, 0, 0, 0] = 8
    elif bad == "duplicate":
        routes[0, 0, 0, 1] = routes[0, 0, 0, 0]
    elif bad == "empty":
        counts[0, 0, 0] = 0
    elif bad == "capacity":
        counts[0, 0, 0] = 9
    elif bad == "unpaired_q":
        routes[0, 0, 0, :2] = torch.tensor([6, 7], device="cuda")
    else:
        counts.fill_(1)
        routes[..., 0] = 0
    with pytest.raises(ValueError):
        cudnn.block_sparse_attention_grouped_prepare(routes, num_kv_blocks=8, q2k_block_nums=counts)


def test_validation_and_capture_preparation_guard():
    routes, counts, _ = metadata()
    with pytest.raises(ValueError, match="int32 token indexing"):
        cudnn.block_sparse_attention_grouped_prepare(routes, num_kv_blocks=2**25, q2k_block_nums=counts)
    group = cudnn.block_sparse_attention_grouped_prepare(routes, num_kv_blocks=8, q2k_block_nums=counts)
    q, k, v, dout = inputs()
    with pytest.raises(ValueError, match="shape"):
        cudnn.block_sparse_attention_grouped_forward(q[:, :-1], k, v, group)
    with pytest.raises(ValueError, match="dtype"):
        cudnn.block_sparse_attention_grouped_forward(q.float(), k, v, group)
    with pytest.raises(ValueError, match="finite and positive"):
        cudnn.block_sparse_attention_grouped_forward(q, k, v, group, softmax_scale=0)
    o, lse = cudnn.block_sparse_attention_grouped_forward(q, k, v, group)
    with pytest.raises(ValueError, match="lse"):
        cudnn.block_sparse_attention_grouped_backward(dout, q, k, v, o, lse.half(), group)
    graph = torch.cuda.CUDAGraph()
    try:
        with pytest.raises(RuntimeError, match="before CUDA Graph capture"):
            with torch.cuda.graph(graph):
                cudnn.block_sparse_attention_grouped_prepare(routes, num_kv_blocks=8, q2k_block_nums=counts)
    finally:
        graph.reset()


def test_side_stream_and_exports():
    from cudnn import BSA
    import cudnn.torch
    from cudnn.block_sparse_attention import api

    for name in ("block_sparse_attention_grouped_prepare", "block_sparse_attention_grouped_forward", "block_sparse_attention_grouped_backward"):
        assert getattr(cudnn, name) is getattr(BSA, name) is getattr(cudnn.torch, name)
        assert name in api.__all__
    routes, counts, mask = metadata()
    group = BSA.block_sparse_attention_grouped_prepare(routes, num_kv_blocks=8, q2k_block_nums=counts)
    q, k, v, do = inputs()
    BSA.block_sparse_attention_grouped_forward(q, k, v, group)  # Compile before stream/replay checks.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        q.mul_(0.5)
        out = BSA.block_sparse_attention_grouped_forward(q, k, v, group)
        grads = BSA.block_sparse_attention_grouped_backward(do, q, k, v, *out, group)
    torch.cuda.current_stream().wait_stream(stream)
    expected = reference(q, k, v, do, mask)
    assert_numerics(out[0], expected[0])
    for actual, ref in zip(grads, expected[2]):
        assert_numerics(actual, ref, grad=True)


def test_warm_execution_does_not_read_metadata_to_host(monkeypatch):
    routes, counts, _ = metadata()
    group = cudnn.block_sparse_attention_grouped_prepare(routes, num_kv_blocks=8, q2k_block_nums=counts)
    q, k, v, do = inputs()
    out = cudnn.block_sparse_attention_grouped_forward(q, k, v, group)
    cudnn.block_sparse_attention_grouped_backward(do, q, k, v, *out, group)

    def forbidden(*args, **kwargs):
        raise AssertionError("execute must not read device data to the host")

    with monkeypatch.context() as patch:
        for name in ("cpu", "item", "numpy", "tolist"):
            patch.setattr(torch.Tensor, name, forbidden)
        patch.setattr(torch.cuda, "synchronize", forbidden)
        out = cudnn.block_sparse_attention_grouped_forward(q, k, v, group)
        grads = cudnn.block_sparse_attention_grouped_backward(do, q, k, v, *out, group)
    assert all(torch.isfinite(x).all() for x in (*out, *grads))


def test_dependency_gate_precedes_kernel_import(monkeypatch):
    import builtins
    import cudnn.frost.buffers

    routes, counts, _ = metadata()
    group = cudnn.block_sparse_attention_grouped_prepare(routes, num_kv_blocks=8, q2k_block_nums=counts)
    q, k, v, do = inputs()
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {"_interface", "_grouped_reorder"} or name.startswith("csrc."):
            raise AssertionError("a kernel was imported before the DSL version gate")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(cudnn.frost.buffers, "cutedsl_requirement_error", lambda what: f"{what} requires CuTe DSL >= 4.7.0; installed 4.6.2")
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(RuntimeError, match="installed 4.6.2"):
        cudnn.block_sparse_attention_grouped_forward(q, k, v, group)
    with pytest.raises(RuntimeError, match="installed 4.6.2"):
        cudnn.block_sparse_attention_grouped_backward(do, q, k, v, q, torch.empty((2, 2, 384), device="cuda"), group)
