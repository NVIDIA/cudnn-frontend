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


def case(pattern, *, identity=False):
    rows = {
        "both": [[0, 1], [0, 1, 2, 3], [2, 3, 4, 5], [4, 5]] * 2,
        "q": [[0, 1], [1, 2], [2, 3], [3, 4]] * 2,
        "kv": [[0, 1], [2, 3], [4, 5], [6, 7], [0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [0, 1, 6, 7]],
        "neither": [[i, (i + 1) % 8] for i in range(8)],
    }[pattern]
    mask = torch.zeros((8, 8), dtype=torch.bool)
    for i, row in enumerate(rows):
        mask[i, row] = True
    masks = torch.empty((2, 2, 8, 8), dtype=torch.bool)
    routes = torch.full((2, 2, 8, 8), -999, dtype=torch.int32)
    counts = torch.empty((2, 2, 8), dtype=torch.int32)
    rng = torch.Generator().manual_seed(125)
    for b in range(2):
        for h in range(2):
            qp, kp = [torch.arange(8) if identity else torch.randperm(8, generator=rng) for _ in range(2)]
            masks[b, h] = mask[qp][:, kp]
            for i in range(8):
                ids = torch.where(masks[b, h, i])[0]
                routes[b, h, i, : len(ids)] = ids.flip(0) if i % 2 else ids
                counts[b, h, i] = len(ids)
    torch.manual_seed(4593)
    q, k, v, do = [torch.randn((2, 512, 2, 128), device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    return q, k, v, do, routes.cuda(), counts.cuda(), masks


@pytest.mark.parametrize("pattern,expected", [("both", (True, True)), ("q", (True, False)), ("kv", (False, True)), ("neither", (False, False))])
@pytest.mark.parametrize("bucket", [3, 1024])
def test_independent_directions_numerics_and_replay(pattern, expected, bucket):
    q, k, v, do, routes, counts, masks = case(pattern)
    plan = cudnn.block_sparse_attention_blk64_prepare(routes, num_kv_blocks=8, q2k_block_nums=counts, bucket_size_blocks=bucket)
    assert (plan.forward_enabled, plan.backward_enabled) == expected
    scale = 0.17
    kwargs = dict(q2k_block_nums=counts, sparse_block_size=64, layout="bshd", softmax_scale=scale, blk64_plan=plan)
    grads = [torch.empty_like(t) for t in (q, k, v)]

    def run(q=q, k=k, v=v, do=do):
        out = cudnn.block_sparse_attention_forward(q, k, v, routes, **kwargs)
        actual = cudnn.block_sparse_attention_backward(
            do,
            q,
            k,
            v,
            *out,
            routes,
            **kwargs,
            bucket_size_blocks=bucket,
            dq_tensor=grads[0],
            dk_tensor=grads[1],
            dv_tensor=grads[2],
        )
        # The native fallback can return layout views of caller outputs.
        assert all(a.data_ptr() == b.data_ptr() for a, b in zip(actual, grads))
        if expected[1]:
            assert all(a is b for a, b in zip(actual, grads))
        return out, actual

    def check(out, actual, qs=q, ks=k, vs=v, dos=do):
        expected_out, expected_lse, expected_grads = reference(qs, ks, vs, dos, masks, scale)
        assert_numerics(out[0], expected_out)
        torch.testing.assert_close(out[1].double(), expected_lse, atol=2e-5, rtol=2e-6)
        for a, e in zip(actual, expected_grads):
            assert_numerics(a, e, grad=True)

    check(*run())
    # Compiled artifacts must bind fresh pointers, not retain warmup inputs.
    fresh = [t.clone().mul_(0.75) for t in (q, k, v, do)]
    check(*run(*fresh), *fresh)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            out, actual = run()
        for _ in range(2):
            q.mul_(0.75)
            do.mul_(-0.5)
            for t in (*out, *actual):
                t.fill_(float("nan"))
            graph.replay()
            check(out, actual)
    finally:
        graph.reset()


def test_identity_fixed_counts_and_unused_kv():
    q, k, v, do, _, _, _ = case("both")
    routes = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int32).expand(2, 2, 8, 4).contiguous()
    plan = cudnn.block_sparse_attention_blk64_prepare(routes, num_kv_blocks=8)
    assert plan._q_identity and plan._kv_identity
    mask = torch.zeros((2, 2, 8, 8), dtype=torch.bool)
    mask[..., :4] = True
    k[:, 256:].fill_(float("nan"))
    v[:, 256:].fill_(float("nan"))
    kwargs = dict(sparse_block_size=64, layout="bshd", blk64_plan=plan)
    out = cudnn.block_sparse_attention_forward(q, k, v, routes, **kwargs)
    grads = cudnn.block_sparse_attention_backward(do, q, k, v, *out, routes, **kwargs)
    expected = reference(q, k, v, do, mask)
    assert_numerics(out[0], expected[0])
    for a, e in zip(grads, expected[2]):
        assert_numerics(a, e, grad=True)
    assert torch.count_nonzero(grads[1][:, 256:]) == 0
    assert torch.count_nonzero(grads[2][:, 256:]) == 0


def test_binding_and_capture_guards():
    q, k, v, do, routes, counts, _ = case("both")
    make = lambda: cudnn.block_sparse_attention_blk64_prepare(routes, num_kv_blocks=8, q2k_block_nums=counts)
    plan = make()
    kwargs = dict(sparse_block_size=64, layout="bshd", q2k_block_nums=counts, blk64_plan=plan)
    with pytest.raises(ValueError, match="same routing"):
        cudnn.block_sparse_attention_forward(q, k, v, routes.clone(), **kwargs)
    routes.copy_(routes)
    with pytest.raises(ValueError, match="metadata changed"):
        cudnn.block_sparse_attention_forward(q, k, v, routes, **kwargs)
    plan = make()
    kwargs["blk64_plan"] = plan
    out = cudnn.block_sparse_attention_forward(q, k, v, routes, **kwargs)
    counts.copy_(counts)
    with pytest.raises(ValueError, match="metadata changed"):
        cudnn.block_sparse_attention_backward(do, q, k, v, *out, routes, **kwargs)
    graph = torch.cuda.CUDAGraph()
    try:
        with pytest.raises(RuntimeError, match="before CUDA Graph capture"):
            with torch.cuda.graph(graph):
                make()
    finally:
        graph.reset()


def test_warm_execution_side_stream_without_host_reads(monkeypatch):
    from cudnn import BSA
    import cudnn.torch
    from cudnn.block_sparse_attention import api

    name = "block_sparse_attention_blk64_prepare"
    assert getattr(cudnn, name) is getattr(BSA, name) is getattr(cudnn.torch, name)
    assert name in api.__all__
    q, k, v, do, routes, counts, mask = case("both")
    plan = cudnn.block_sparse_attention_blk64_prepare(routes, num_kv_blocks=8, q2k_block_nums=counts)
    kwargs = dict(sparse_block_size=64, layout="bshd", q2k_block_nums=counts, blk64_plan=plan)

    def run():
        out = cudnn.block_sparse_attention_forward(q, k, v, routes, **kwargs)
        grads = cudnn.block_sparse_attention_backward(do, q, k, v, *out, routes, **kwargs)
        return out, grads

    run()

    def forbidden(*args, **kwargs):
        raise AssertionError("execution must not read routing metadata to the host or synchronize")

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with monkeypatch.context() as patch:
        for method in ("cpu", "item", "tolist"):
            patch.setattr(torch.Tensor, method, forbidden)
        patch.setattr(torch.cuda, "synchronize", forbidden)
        with torch.cuda.stream(side):
            q.mul_(0.5)
            out, grads = run()
    torch.cuda.current_stream().wait_stream(side)
    expected = reference(q, k, v, do, mask)
    assert_numerics(out[0], expected[0])
    for a, e in zip(grads, expected[2]):
        assert_numerics(a, e, grad=True)


def test_declined_plan_and_no_plan_preserve_native_calls(monkeypatch):
    from cudnn.block_sparse_attention import _interface

    q, k, v, do, routes, counts, _ = case("neither")
    plan = cudnn.block_sparse_attention_blk64_prepare(routes, num_kv_blocks=8, q2k_block_nums=counts)
    calls = []
    out, lse = torch.empty_like(q), torch.empty((2, 2, 512), device=q.device)

    def fwd(*args, **kwargs):
        calls.append((args, kwargs))
        return out, lse

    def bwd(*args, **kwargs):
        calls.append((args, kwargs))
        return q, k, v

    monkeypatch.setattr(_interface, "bsa_attn_fwd_blk64_cutedsl", fwd)
    monkeypatch.setattr(_interface, "bsa_attn_bwd", bwd)
    for method in ("fwd", "bwd"):
        for p in (None, plan):
            kwargs = dict(sparse_block_size=64, layout="bshd", q2k_block_nums=counts, blk64_plan=p)
            if method == "fwd":
                cudnn.block_sparse_attention_forward(q, k, v, routes, **kwargs)
            else:
                cudnn.block_sparse_attention_backward(do, q, k, v, out, lse, routes, **kwargs)
        assert calls[-2] == calls[-1]


@pytest.mark.parametrize("kind", ["layout", "tail", "block_sizes", "kv_splits", "use_clc", "bucket", "strided_grads"])
def test_unsupported_declarations_preserve_native_dispatch(kind, monkeypatch):
    from cudnn.block_sparse_attention import _interface

    q, k, v, do, routes, counts, _ = case("both")
    plan = cudnn.block_sparse_attention_blk64_prepare(routes, num_kv_blocks=8, q2k_block_nums=counts)
    assert plan.forward_enabled and plan.backward_enabled
    kwargs = dict(sparse_block_size=64, layout="bshd", q2k_block_nums=counts, blk64_plan=plan)
    backward_only = kind in ("bucket", "strided_grads")
    if kind == "layout":
        q, k, v, do = [x.transpose(1, 2).contiguous() for x in (q, k, v, do)]
        kwargs["layout"] = "bhsd"
    elif kind == "tail":
        q, do = q[:, :-1], do[:, :-1]
    elif kind == "block_sizes":
        kwargs["block_sizes"] = torch.full((8,), 64, device=q.device, dtype=torch.int32)
    elif kind == "kv_splits":
        kwargs["kv_splits"] = 2
    elif kind == "use_clc":
        kwargs["use_clc"] = False
    elif kind == "bucket":
        kwargs["bucket_size_blocks"] = 3
    else:
        kwargs["dq_tensor"] = torch.empty((2, 1024, 2, 128), device=q.device, dtype=q.dtype)[:, ::2]
    n = q.shape[2 if kwargs["layout"] == "bhsd" else 1]
    lse = torch.empty((2, 2, n), device=q.device)
    calls = []

    def native_fwd(*args, **kw):
        calls.append("fwd")
        return q, lse

    def native_bwd(*args, **kw):
        calls.append("bwd")
        return q, k, v

    monkeypatch.setattr(_interface, "bsa_attn_fwd_blk64_cutedsl", native_fwd)
    monkeypatch.setattr(_interface, "bsa_attn_bwd", native_bwd)
    if not backward_only:
        cudnn.block_sparse_attention_forward(q, k, v, routes, **kwargs)
        assert calls == ["fwd"]
    if kind not in ("kv_splits", "use_clc"):
        cudnn.block_sparse_attention_backward(do, q, k, v, q, lse, routes, **kwargs)
        assert calls[-1] == "bwd"


@pytest.mark.parametrize("kind", ["negative", "range", "duplicate", "counts", "overflow", "bucket"])
def test_invalid_plan_metadata(kind):
    _, _, _, _, routes, counts, _ = case("both")
    kwargs = dict(num_kv_blocks=8, q2k_block_nums=counts)
    if kind == "negative":
        routes[0, 0, 0, 0] = -1
    elif kind == "range":
        routes[0, 0, 0, 0] = 8
    elif kind == "duplicate":
        routes[0, 0, 0, 1] = routes[0, 0, 0, 0]
    elif kind == "counts":
        counts[0, 0, 0] = 9
    elif kind == "overflow":
        kwargs["num_kv_blocks"] = 2**25
    else:
        kwargs["bucket_size_blocks"] = 0
    with pytest.raises(ValueError):
        cudnn.block_sparse_attention_blk64_prepare(routes, **kwargs)


def test_unequal_lengths_and_runtime_scale_rebinding():
    q, k, v, do, routes, counts, masks = case("both")
    # Q512 / KV768; the extra unused columns must remain zero in dK/dV,
    # including when unused K/V contain NaNs.
    k, v = [torch.cat((x, torch.full((2, 256, 2, 128), float("nan"), device=x.device, dtype=x.dtype)), dim=1) for x in (k, v)]
    masks = torch.cat((masks, torch.zeros((2, 2, 8, 4), dtype=torch.bool)), dim=-1)
    plan = cudnn.block_sparse_attention_blk64_prepare(routes, num_kv_blocks=12, q2k_block_nums=counts, bucket_size_blocks=3)
    assert plan.forward_enabled and plan.backward_enabled
    for scale in (0.17, 0.09):
        kwargs = dict(sparse_block_size=64, layout="bshd", q2k_block_nums=counts, blk64_plan=plan, softmax_scale=scale)
        out = cudnn.block_sparse_attention_forward(q, k, v, routes, **kwargs)
        grads = cudnn.block_sparse_attention_backward(do, q, k, v, *out, routes, **kwargs)
        expected = reference(q, k, v, do, masks, scale)
        assert_numerics(out[0], expected[0])
        torch.testing.assert_close(out[1].double(), expected[1], atol=2e-5, rtol=2e-6)
        for a, e in zip(grads, expected[2]):
            assert_numerics(a, e, grad=True)
        for g in (grads[1], grads[2]):
            assert torch.count_nonzero(g[:, 512:]) == 0
