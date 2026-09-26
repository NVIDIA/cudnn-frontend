# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sink limits, mask isolation, and large finite gradients through the public HCA API."""

import pytest
import torch

pytestmark = [pytest.mark.L1, pytest.mark.gpu_exclusive]


@pytest.fixture(autouse=True)
def require_supported_device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)):
        pytest.skip("Aligned HCA requires GB300 or Rubin")
    pytest.importorskip("cutlass")
    pytest.importorskip("triton")
    from cudnn import AlignedHCABackward

    if not AlignedHCABackward.supports_configuration(4096, 16, "cuda"):
        pytest.skip("Aligned HCA on Rubin requires Triton >=3.8.0")


def constant_case(rank):
    from cudnn import AlignedHCABackward
    from fe_api.dsa.aligned_hca_test_utils import physical_indices

    q = torch.zeros((4096, 128, 512), device="cuda", dtype=torch.bfloat16)
    kv = torch.full((4752, 512), 1 / 64, device="cuda", dtype=torch.bfloat16)
    out, dout = torch.empty_like(q), torch.full_like(q, 1 / 32)
    sink = torch.empty(128, device="cuda", dtype=torch.float32)
    indices = physical_indices(rank)
    counts = (indices >= 0).sum(dim=1)
    lse = counts.float().log()[:, None].expand(4096, 128).contiguous()
    inputs = (q, kv, out, dout, lse, sink)
    outputs = (torch.empty_like(q), torch.empty_like(kv), torch.empty_like(sink))
    api = AlignedHCABackward(*inputs, *outputs, cp_rank=rank)
    api.compile()
    workspace = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    return api, inputs, outputs, workspace, indices


def assert_identical(outputs, expected):
    for name, actual, baseline in zip(("dq", "dkv", "d_sink"), outputs, expected):
        assert torch.isfinite(actual).all(), name
        assert torch.equal(actual, baseline), name


@pytest.mark.parametrize("rank", [0, 3, 15])
def test_positive_infinity_sink_limit(rank):
    api, inputs, outputs, workspace, _ = constant_case(rank)
    _, _, out, _, lse, sink = inputs
    graph = None
    for mixed in (False, True):
        # Finite control for the +inf limit.
        sink.fill_(2.4e38)
        if mixed:
            sink[1::3] = float("-inf")
            sink[2::3] = 0
        out.copy_((torch.sigmoid(lse - sink[None, :]) / 64)[:, :, None])
        api.execute(*inputs, *outputs, workspace)
        expected = tuple(value.clone() for value in outputs)
        assert all(torch.isfinite(value).all() for value in expected)
        if not mixed:
            assert all(torch.count_nonzero(value) == 0 for value in expected)
        else:
            assert torch.count_nonzero(expected[1]) > 0
            assert torch.count_nonzero(expected[2][2::3]) > 0
        if graph is None:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                api.execute(*inputs, *outputs, workspace)
        sink[:: 3 if mixed else 1] = float("inf")
        # Replay must use the updated sink.
        for run in (lambda: api.execute(*inputs, *outputs, workspace), graph.replay):
            for value in outputs:
                value.fill_(float("nan"))
            run()
            torch.cuda.synchronize()
            assert_identical(outputs, expected)


@pytest.mark.parametrize("rank", [0, 3, 15])
def test_unused_kv_storage_invariance(rank):
    api, inputs, outputs, workspace, indices = constant_case(rank)
    _, kv, out, _, _, sink = inputs
    sink.fill_(float("-inf"))
    out.fill_(1 / 64)
    used = torch.zeros(kv.shape[0], device="cuda", dtype=torch.bool)
    used[indices[indices >= 0].long()] = True
    unused = ~used
    if rank == 0:
        assert unused[:128].all()
    api.execute(*inputs, *outputs, workspace)
    expected = tuple(value.clone() for value in outputs)
    assert all(torch.isfinite(value).all() for value in expected)
    assert torch.count_nonzero(expected[1][unused]) == 0
    assert torch.count_nonzero(expected[1][used]) > 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        api.execute(*inputs, *outputs, workspace)
    # Poison only independently identified unused rows.
    for poison in (float("nan"), float("inf"), float("-inf"), 3e38, -3e38, 1 / 64):
        kv[unused] = poison
        for run in (lambda: api.execute(*inputs, *outputs, workspace), graph.replay):
            for value in outputs:
                value.fill_(float("nan"))
            run()
            torch.cuda.synchronize()
            assert_identical(outputs, expected)
            assert torch.count_nonzero(outputs[1][unused]) == 0


@pytest.mark.parametrize("rank", [0, 3, 15])
@pytest.mark.parametrize("gradient", [1.0, 2.0**60], ids=["unit_gradient", "large_gradient"])
def test_negative_lse_large_gradient(rank, gradient):
    api, inputs, outputs, workspace, indices = constant_case(rank)
    q, kv, out, dout, lse, sink = inputs
    q.fill_(16)
    kv.fill_(-1)
    out.fill_(-1)
    sink.fill_(float("-inf"))
    counts = (indices >= 0).sum(dim=1)
    lse.copy_((counts.float().log() - 16 * (512**0.5))[:, None])
    assert (lse < -300).all()

    # Uniform attention: dQ=0 and dKV=sum(128*dO/count).
    valid = indices >= 0
    weights = (128.0 / counts.double())[:, None].expand_as(indices)
    expected_dkv = torch.zeros(kv.shape[0], device="cuda", dtype=torch.float64)
    expected_dkv.scatter_add_(0, indices[valid].long(), weights[valid])
    used = expected_dkv > 0
    expected_dkv = expected_dkv[:, None].expand_as(kv)

    # Replay with large dO must not overflow masked dS.
    dout.fill_(1)
    api.execute(*inputs, *outputs, workspace)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        api.execute(*inputs, *outputs, workspace)
    dout.fill_(gradient)
    for run in (lambda: api.execute(*inputs, *outputs, workspace), graph.replay):
        for value in outputs:
            value.fill_(float("nan"))
        run()
        torch.cuda.synchronize()
        for name, actual in zip(("dq", "dkv", "d_sink"), outputs):
            assert torch.isfinite(actual).all(), name
        assert torch.count_nonzero(outputs[0]) == 0
        assert torch.count_nonzero(outputs[2]) == 0
        assert torch.count_nonzero(outputs[1][~used]) == 0
        # Normalize dO and allow BF16 rounding.
        torch.testing.assert_close(outputs[1].double() / gradient, expected_dkv, rtol=0.02, atol=0)
