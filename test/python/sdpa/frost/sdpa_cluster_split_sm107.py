# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native cluster split keeps the graph and standalone output contracts."""

import math

import pytest

import cudnn


def _device():
    import torch
    from frost_test_utils import _dsl_installed

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("requires native SM107")
    if not _dsl_installed():
        pytest.skip("requires supported CuTe DSL")
    return torch


def _buffers(torch, sk, pitched, seed):
    torch.manual_seed(seed)
    b, h, hk, d = 2, 32, 2, 256
    storage = []

    def tensor(seq, heads, output=False):
        width = d + (16 if pitched else 0)
        guard = 17 if output else 16  # final O requires only element alignment
        slab = torch.full((b * seq * heads * width + 2 * guard,), 77.0, device="cuda", dtype=torch.bfloat16)
        view = slab[guard:-guard].view(b, seq, heads, width)
        if not output:
            view[..., :d].normal_()
        storage.append((slab, view, guard))
        return view[..., :d].transpose(1, 2)

    q, k, v, o = tensor(1, h), tensor(sk, hk), tensor(sk, hk), tensor(1, h, True)
    slab = torch.full((b * h * 2 + 32,), 77.0, device="cuda", dtype=torch.float32)
    lse = slab[16:-16].view(b, h, 2)[:, :, :1]
    return (q, k, v, o, lse), storage, slab


def _reference(torch, buffers):
    q, k, v, _, _ = buffers
    b, h, sq, d = q.shape
    hk = k.shape[1]
    scores = torch.einsum("bhgqd,bhkd->bhgqk", q.double().reshape(b, hk, h // hk, sq, d), k.double()) / math.sqrt(d)
    return torch.einsum("bhgqk,bhkd->bhgqd", scores.softmax(-1), v.double()).reshape(q.shape), scores.logsumexp(-1).reshape(b, h, sq)


def _check(torch, buffers, storage, lse_storage, reference, stats):
    torch.cuda.synchronize()
    o, lse = buffers[3:]
    expected, expected_lse = reference
    delta = o.double() - expected
    assert torch.isfinite(o).all()
    assert torch.linalg.vector_norm(delta) <= 0.03 * torch.linalg.vector_norm(expected)
    assert delta.abs().max() <= max(2e-5, 0.03 * expected.abs().max().item())
    if stats:
        torch.testing.assert_close(lse.double(), expected_lse, atol=6e-4, rtol=0)
    else:
        assert (lse == 77).all()
    for slab, view, guard in storage:
        assert (slab[:guard] == 77).all() and (slab[-guard:] == 77).all()
        assert (view[..., 256:] == 77).all()
    assert (lse_storage[:16] == 77).all() and (lse_storage[-16:] == 77).all()
    assert (lse_storage[16:-16].view(2, 32, 2)[:, :, 1] == 77).all()


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("python_binding", [False, True])
@pytest.mark.parametrize("sk,pitched", [(128, False), (2048, True)])
@pytest.mark.parametrize("split", [4, 8], ids=["separate", "fused"])
def test_cluster_split_standalone_rebind_and_retained_replay(stats, python_binding, sk, pitched, split):
    torch = _device()
    from cuda.bindings import driver as cuda
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    buffers, storage, lse_storage = _buffers(torch, sk, pitched, 351)
    q, k, v, o, lse = buffers
    api = SdpaFwdDslSm100(q, k, v, o, lse if stats else None, scale_softmax=1 / 16, cga=2, split_kv=split, pack_gqa=True)
    assert api.check_support()
    api.compile()
    assert api._dense_spec.fused_split == (split == 8)
    assert (api._dense_spec.combine.fn is None) == (split == 8)
    if python_binding:
        api._dense_spec.native = None
    size = api.scratch_workspace_bytes()
    slab = torch.full((size + 512,), 0xB7, device="cuda", dtype=torch.uint8)
    workspace = slab[256:-256]
    stream = torch.cuda.Stream()

    def run():
        q, k, v, o, lse = buffers
        api.execute(q, k, v, o, lse if stats else None, workspace=workspace, current_stream=cuda.CUstream(stream.cuda_stream))

    for seed in (None, 352):
        if seed is not None:
            buffers, storage, lse_storage = _buffers(torch, sk, pitched, seed)
        reference = _reference(torch, buffers)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        _check(torch, buffers, storage, lse_storage, reference, stats)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=stream):
        run()
    buffers[0].mul_(0.7)
    buffers[1].mul_(-0.4)
    buffers[2].mul_(-0.7)
    workspace.fill_(255)
    buffers[3].fill_(float("nan"))
    if stats:
        buffers[4].fill_(float("nan"))
    reference = _reference(torch, buffers)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        capture.replay()
    _check(torch, buffers, storage, lse_storage, reference, stats)
    saved = [tensor.contiguous().view(torch.uint8).clone() for tensor in buffers[3:] if stats or tensor is buffers[3]]
    stream.wait_stream(torch.cuda.current_stream())
    for _ in range(3):
        with torch.cuda.stream(stream):
            capture.replay()
        stream.synchronize()
        actual = [tensor.contiguous().view(torch.uint8) for tensor in buffers[3:] if stats or tensor is buffers[3]]
        assert all(torch.equal(a, b) for a, b in zip(actual, saved))
    capture.reset()
    assert (slab[:256] == 0xB7).all() and (slab[-256:] == 0xB7).all()
    before = buffers[3].clone()
    with pytest.raises(ValueError, match="workspace"):
        api.execute(*buffers[:4], buffers[4] if stats else None, workspace=workspace[:0])
    assert torch.equal(before.view(torch.uint8), buffers[3].view(torch.uint8))


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("ordered", [False, True])
def test_cluster_split_graph_contract(stats, ordered):
    torch = _device()
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    buffers, storage, lse_storage = _buffers(torch, 256, True, 353)
    q, k, v, o, lse = buffers
    handle = cudnn.create_handle()
    capture = None
    try:
        graph = cudnn.pygraph(
            handle=handle, io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT
        )
        inputs = [graph.tensor_like(t).set_uid(i + 1) for i, t in enumerate((q, k, v))]
        out, st = graph.sdpa(q=inputs[0], k=inputs[1], v=inputs[2], generate_stats=stats, attn_scale=1 / 16)
        out.set_uid(4).set_output(True).set_dim(o.shape).set_stride(o.stride())
        if stats:
            st.set_uid(5).set_output(True).set_dim((*lse.shape, 1)).set_stride((*lse.stride(), 1)).set_data_type(cudnn.data_type.FLOAT)
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plan(20515, SdpaFwdKnobs(sched_policy=0, tile_m=128, tile_n=128, cga=2, pack_gqa=True, split_kv=8))
        graph.build_plan_at_index(0)
        spec = graph._compiled_plans[graph._plan_index]._prepared.spec
        assert spec.fused_split and spec.native is not None and spec.combine.fn is None
        workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
        stream = torch.cuda.Stream()
        cudnn.set_stream(handle, stream.cuda_stream)

        def run():
            vals = (*buffers[:4], buffers[4].unsqueeze(-1)) if stats else buffers[:4]
            if ordered:
                graph.execute(vals, workspace, handle=handle, tensor_uids=tuple(range(1, len(vals) + 1)))
            else:
                graph.execute(dict(zip(range(1, len(vals) + 1), vals)), workspace, handle=handle)

        for seed in (None, 354):
            if seed is not None:
                buffers, storage, lse_storage = _buffers(torch, 256, True, seed)
            reference = _reference(torch, buffers)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            _check(torch, buffers, storage, lse_storage, reference, stats)
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture, stream=stream):
            run()
        buffers[2].mul_(-0.7)
        reference = _reference(torch, buffers)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            capture.replay()
        _check(torch, buffers, storage, lse_storage, reference, stats)
    finally:
        if capture is not None:
            capture.reset()
        cudnn.destroy_handle(handle)
