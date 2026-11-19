# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public SE dispatch, persistent backward arithmetic, and staging lifetime."""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from test_causal_conv1d_packed import reference

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.fixture(autouse=True)
def persistent_arch():
    from cudnn.causal_conv1d_se.packed import _require_dsl

    _require_dsl()
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("persistent SE backward requires SM100 or SM103")


@pytest.fixture
def dense_arch():
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("the public dense SE adapter is qualified on SM103")


def _inputs(lengths, channels, bias=True):
    generator = torch.Generator().manual_seed(19019)

    def fixed(shape, scale=0.25):
        return (torch.randn(shape, generator=generator) * scale).to(device="cuda", dtype=torch.bfloat16)

    x = fixed((sum(lengths), channels)).requires_grad_()
    w = fixed((channels, 4)).requires_grad_()
    b = fixed((channels,)).requires_grad_() if bias else None
    dy = fixed(x.shape, 1.0)
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device="cuda", dtype=torch.int32)
    return x, w, b, cu, dy


def _public(args, dense_shape=None):
    from cudnn.ops import causal_conv1d

    x, w, b, cu, dy = args
    public_x = x.T[None] if dense_shape is None else x.view(*dense_shape).transpose(1, 2)
    y = causal_conv1d(public_x, w, b, activation="silu", cu_seqlens=cu if dense_shape is None else None, backend="se")
    public_dy = dy.T[None] if dense_shape is None else dy.view(*dense_shape).transpose(1, 2)
    grads = torch.autograd.grad(y, (x, w) if b is None else (x, w, b), public_dy)
    return dict(zip(("y", "dx", "dw", "db"), (y.transpose(1, 2).reshape_as(x), *grads)))


def _reference_check(actual, args, lengths):
    x, w, b, _, dy = args
    indices = torch.linspace(0, x.shape[1] - 1, 8, device=x.device).long()
    xx = x[:, indices].double().detach().requires_grad_()
    ww = w[indices].double().detach().requires_grad_()
    bb = None if b is None else b[indices].double().detach().requires_grad_()
    y, _ = reference(xx, ww, bb, None, lengths, True)
    grads = torch.autograd.grad(y, (xx, ww) if bb is None else (xx, ww, bb), dy[:, indices].double())
    for name, expected in zip(actual, (y, *grads)):
        assert torch.isfinite(actual[name]).all(), name
        selected = actual[name][:, indices] if name in ("y", "dx") else actual[name][indices]
        torch.testing.assert_close(selected, expected.to(selected.dtype), atol=0.03, rtol=0.02)


@pytest.mark.parametrize("lengths,channels", [([8192], 768), ([0, 8191, 0, 1, 0], 2048), ([3, 1, 8188], 768)])
@pytest.mark.parametrize("bias", [False, True])
def test_public_persistent_gradients_and_parent_dx(monkeypatch, lengths, channels, bias):
    from cudnn.causal_conv1d_se import packed

    args = _inputs(lengths, channels, bias)
    x, w, b, cu, dy = args
    assert packed._backward_schedule(x, w, b, None, dy, None, True)[0] == "persistent"
    actual = _public(args)
    _reference_check(actual, args, lengths)
    monkeypatch.setattr(packed, "_backward_schedule", lambda *args: ("tiled", 64))
    parent = _public(args)
    assert_bitwise({"dx": actual["dx"]}, {"dx": parent["dx"]}, "persistent versus tiled dX")


def test_public_persistent_replay_and_golden():
    args = _inputs([0, 4097, 1, 4094], 768)
    inputs = dict(zip(("x", "w", "b", "cu", "dy"), args))
    before = snapshot(inputs)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        actual = assert_replays(lambda: _public(args), repeats=3)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    _reference_check(actual, args, [0, 4097, 1, 4094])
    assert_bitwise(inputs, before, "persistent inputs")
    from cutlass.base_dsl.version_info import CUDA_VERSION

    assert_golden(f"causal_conv1d_se_persistent_bf16_cuda{CUDA_VERSION.major}_{CUDA_VERSION.minor}", inputs, actual)


def test_public_persistent_nondefault_graph_changed_offsets():
    args = _inputs([1, 8191], 768)
    x, w, b, cu, dy = args
    _public(args)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            _public(args)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            old = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                captured = _public(args)
            finally:
                torch.cuda.set_sync_debug_mode(old)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.no_grad():
            x.neg_()
            dy.mul_(0.5)
            cu.copy_(torch.tensor([0, 4097, 8192], device=x.device, dtype=torch.int32))
        graph.replay()
        actual = snapshot(captured)
        expected = _public(args)
        assert_bitwise(actual, expected, "changed offsets on graph replay")
        _reference_check(actual, args, [4097, 4095])
    finally:
        graph.reset()


@pytest.mark.parametrize("dense_shape", [(2, 32, 256), (8, 8192, 2048)])
@pytest.mark.usefixtures("dense_arch")
def test_public_dense_gradients_and_dispatch(dense_shape):
    from cudnn.causal_conv1d_se.dense import _persistent_shape

    batch, tokens, channels = dense_shape
    args = _inputs([tokens] * batch, channels)
    assert _persistent_shape(dense_shape, (10, 3)) == (tokens == 8192)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    if tokens == 8192:
        torch.use_deterministic_algorithms(True)
    try:
        actual = _public(args, dense_shape)
        _reference_check(actual, args, [tokens] * batch)
        if tokens == 8192:
            for _ in range(3):
                assert_bitwise(actual, _public(args, dense_shape), "dense persistent replay")
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.usefixtures("dense_arch")
def test_dense_tiled_deterministic_mode_declines():
    args = _inputs([32, 32], 256)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with pytest.raises(RuntimeError, match="atomic parameter reductions"):
            _public(args, (2, 32, 256))
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize("mode", ["packed", "dense"])
def test_persistent_warm_stress(mode, request):
    if mode == "dense":
        request.getfixturevalue("dense_arch")
    from cudnn.causal_conv1d_se.packed import _compiled, _backward_schedule
    from cuda.bindings import driver as cuda
    from warm_stress import StressCase, assert_warm_stress

    if mode == "packed":
        tokens, channels = 8192, 768
        x, w, b, cu, dy = _inputs([tokens], channels)
        route, splits = _backward_schedule(x, w, b, None, dy, None, True)
        assert route == "persistent"
        dw_storage = torch.empty_like(w)
        dw = dw_storage
    else:
        from cudnn.causal_conv1d_se import dense

        tokens, channels = 65536, 2048
        x, w, b, cu, dy = _inputs([8192] * 8, channels)
        assert dense._persistent_shape((8, 8192, channels), (10, 3))
        w = w.T.contiguous().T
        splits = max(1, 2 * torch.cuda.get_device_properties(x.device).multi_processor_count // 8)
        dw_storage = torch.empty_like(w.T)
        dw = dw_storage.T
    dx, db = torch.empty_like(x), torch.empty_like(b)
    dz = torch.empty_like(x, dtype=torch.float32)
    partials = torch.empty((splits, 5, channels), device=x.device)
    args = (x, w, b, cu, None, dy, None, dz, dx, dw, db, None, partials)
    fn = _compiled("backward", args, 4, True) if mode == "packed" else dense._compiled("persistent", args)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    case = StressCase(
        "causal_conv1d_se_persistent_bf16_" + mode,
        lambda: fn(*args, stream),
        dict(dx=dx, dw=dw_storage, db=db, partials=partials),
        thread_blocks=((channels + 255) // 256) * splits,
        refills_per_launch=((tokens + 4 * splits - 1) // (4 * splits) + 10) // 8 - 2,
    )
    assert_warm_stress(case, warm=3000, poisoned=3000)


def test_persistent_sequence_batch_invariance():
    """Repacking retains each sequence's output and dX bytes; parameter sums can regroup."""
    x, w, b, cu, dy = _inputs([8192], 768)
    expected = snapshot({k: v for k, v in _public((x, w, b, cu, dy)).items() if k in ("y", "dx")})
    neighbor, _, _, _, grad_neighbor = _inputs([4097], 768)
    for order in ((0, 1), (1, 0, 0)):
        inputs = [x if i == 0 else neighbor for i in order]
        gradients = [dy if i == 0 else grad_neighbor for i in order]
        lengths = [t.shape[0] for t in inputs]
        offsets = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device=x.device, dtype=torch.int32)
        actual = _public((torch.cat(inputs), w, b, offsets, torch.cat(gradients)))
        start = 0
        for i, length in zip(order, lengths):
            if i == 0:
                assert_bitwise({k: actual[k][start : start + length] for k in ("y", "dx")}, expected, "persistent sequence repacking")
            start += length


@pytest.mark.usefixtures("dense_arch")
def test_dense_fullgraph_autograd():
    from cudnn.ops import causal_conv1d

    args = _inputs([32, 32], 256)
    x, w, b, _, dy = args
    public_x = x.view(2, 32, 256).transpose(1, 2)

    def call(x, w, b):
        return causal_conv1d(x, w, b, activation="silu", backend="se")

    compiled = torch.compile(call, backend="aot_eager", fullgraph=True, dynamic=True)
    expected = call(public_x, w, b)
    actual = compiled(public_x, w, b)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    grad = dy.view(2, 32, 256).transpose(1, 2)
    got = torch.autograd.grad(actual, (x, w, b), grad)
    want = torch.autograd.grad(expected, (x, w, b), grad)
    for a, ref in zip(got, want):
        torch.testing.assert_close(a, ref, atol=0.03, rtol=0.02)


@pytest.mark.parametrize("shape", [(65536, 1, 8), (1, 524281, 8), (1, 1, 2**31)])
@pytest.mark.usefixtures("dense_arch")
def test_dense_launch_limits_reject_before_compilation(shape):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from cudnn.ops import causal_conv1d

    with FakeTensorMode():
        x = torch.empty(shape, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        w = torch.empty((shape[2], 4), device=x.device, dtype=x.dtype)
        with pytest.raises(NotImplementedError, match="CUDA grid limits"):
            causal_conv1d(x, w, activation="silu", backend="se")


@pytest.mark.parametrize("width,channels,dtype,silu", [(4, 1032, torch.bfloat16, True), (8, 384, torch.bfloat16, False), (8, 320, torch.float32, False)])
def test_tiled_staging_warm_stress(width, channels, dtype, silu):
    from cuda.bindings import driver as cuda
    from cudnn.causal_conv1d_se.packed import _compiled, _backward_schedule
    from warm_stress import StressCase, assert_warm_stress

    tokens = 16384
    generator = torch.Generator().manual_seed(23119)

    def fixed(shape):
        return (torch.randn(shape, generator=generator) * 0.25).to(device="cuda", dtype=dtype)

    x, w, b, dy = fixed((tokens, channels)), fixed((channels, width)), fixed((channels,)), fixed((tokens, channels))
    cu = torch.tensor([0, 8191, tokens], device=x.device, dtype=torch.int32)
    route, splits = _backward_schedule(x, w, b, None, dy, None, silu)
    assert route == "tiled"
    dx, dw, db = torch.empty_like(x), torch.empty_like(w), torch.empty_like(b)
    dz = torch.empty_like(x, dtype=torch.float32)
    partials = torch.empty((splits, width + 1, channels), device=x.device, dtype=torch.float64 if dtype == torch.float32 else torch.float32)
    args = (x, w, b, cu, None, dy, None, dz, dx, dw, db, None, partials)
    fn = _compiled("backward", args, width, silu)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    tile_channels = 256 if width == 4 else 64
    stages = 2 if width == 8 and dtype != torch.float32 else 1
    case = StressCase(
        f"causal_conv1d_se_tiled_{dtype}_k{width}",
        lambda: fn(*args, stream),
        dict(dx=dx, dw=dw, db=db, partials=partials),
        thread_blocks=((channels + tile_channels - 1) // tile_channels) * splits,
        refills_per_launch=(tokens + 64 * splits - 1) // (64 * splits) - stages,
    )
    assert_warm_stress(case, warm=3000, poisoned=3000)


@pytest.mark.usefixtures("dense_arch")
def test_dense_signed_zero_tail_matches_parent(monkeypatch):
    """Preserve the parent's zero sign after padded-tail arithmetic."""
    from cudnn.causal_conv1d_se import dense

    shape = (8, 8192, 2048)
    args = _inputs([8192] * 8, 2048)
    x, w, b, cu, dy = args
    with torch.no_grad():
        x.zero_()
        w.zero_()
        w[:, 3] = 2.0**-126
        b.zero_()
        dy.zero_()
        dy.view(shape)[:, -1].fill_(-(2.0**-24))
    assert dense._persistent_shape(shape, (10, 3))
    actual = _public(args, shape)
    monkeypatch.setattr(dense, "_persistent_shape", lambda *args: False)
    parent = _public(args, shape)
    assert_bitwise({"dx": actual["dx"]}, {"dx": parent["dx"]}, "dense signed-zero tail")
    assert (actual["dx"] == 0).all()


@pytest.mark.parametrize(
    "name,lengths,channels",
    [
        ("tiny", [1] * 64, 256),
        ("tail", [0, 1031, 0, 1031, 0], 1032),
        ("ragged", [0, 1, 3, 29, 1025, 4097, 0], 256),
        ("split_boundary", [4095, 1], 256),
        ("before_persistent", [8191], 768),
        ("after_persistent", [1, 8192], 768),
    ],
)
@pytest.mark.parametrize("bias", [False, True])
def test_public_tuned_schedule_boundaries(name, lengths, channels, bias):
    args = _inputs(lengths, channels, bias)
    actual = assert_replays(lambda: _public(args), repeats=3)
    _reference_check(actual, args, lengths)


@pytest.mark.parametrize("name,lengths,channels", [("tiny", [1] * 64, 256), ("tail", [1031, 1031], 1032), ("ragged", [1, 3, 29, 1025, 4097], 256)])
def test_public_tuned_tiled_golden(name, lengths, channels):
    from cutlass.base_dsl.version_info import CUDA_VERSION

    args = _inputs(lengths, channels)
    inputs = dict(zip(("x", "w", "b", "cu", "dy"), args))
    before = snapshot(inputs)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        actual = assert_replays(lambda: _public(args), repeats=3)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    _reference_check(actual, args, lengths)
    assert_bitwise(inputs, before, "tuned tiled inputs")
    assert_golden(f"causal_conv1d_se_tuned_{name}_bf16_cuda{CUDA_VERSION.major}_{CUDA_VERSION.minor}", inputs, actual)
