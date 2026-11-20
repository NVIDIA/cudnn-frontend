# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP32 accumulation accuracy, workspace bounds, and changed-input graph replay."""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_replays, snapshot
from test_causal_conv1d_packed import _dsl, reference

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def deterministic_algorithms():
    _dsl()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _run(x, weight, bias, initial, lengths, dy, dfinal, silu=False):
    from functools import partial
    from cudnn.ops import causal_conv1d as _causal_conv1d

    causal_conv1d = partial(_causal_conv1d, backend="se")

    xx, ww, bb, ii = (None if t is None else t.detach().clone().requires_grad_() for t in (x, weight, bias, initial))
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device=x.device, dtype=torch.int32)
    result = causal_conv1d(
        xx.T[None],
        ww,
        bb,
        cu_seqlens=cu,
        initial_states=None if ii is None else ii.transpose(1, 2),
        activation="silu" if silu else None,
        return_final_states=dfinal is not None,
    )
    if dfinal is None:
        y = result[0].T
        outputs, upstream = y, dy
        actual = dict(y=y)
    else:
        y, final = result[0][0].T, result[1].transpose(1, 2)
        outputs, upstream = (y, final), (dy, dfinal)
        actual = dict(y=y, final=final)
    inputs = (xx, ww, bb) if ii is None else (xx, ww, bb, ii)
    grads = torch.autograd.grad(outputs, inputs, upstream)
    actual.update(zip(("dx", "dw", "db", "di"), grads))
    return actual


def _check_reference(actual, x, weight, bias, initial, lengths, dy, dfinal, silu=False):
    xx, ww, bb, ii = (None if t is None else t.detach().double().requires_grad_() for t in (x, weight, bias, initial))
    y, final = reference(xx, ww, bb, ii, lengths, silu)
    inputs = (xx, ww, bb) if ii is None else (xx, ww, bb, ii)
    outputs = y if dfinal is None else (y, final)
    upstream = dy.double() if dfinal is None else (dy.double(), dfinal.double())
    grads = torch.autograd.grad(outputs, inputs, upstream)
    expected = dict(y=y)
    if dfinal is not None:
        expected["final"] = final
        assert_bitwise({"final": actual["final"]}, {"final": final.float()}, "raw history")
    expected.update(zip(("dx", "dw", "db", "di"), grads))
    for name, want in expected.items():
        torch.testing.assert_close(actual[name], want.float(), atol=2e-5, rtol=2e-5, msg=lambda message: name + ": " + message)


@pytest.mark.parametrize("silu", [False, True], ids=["identity", "silu"])
@pytest.mark.parametrize("seed", [1907, 319])
@pytest.mark.parametrize("width", [2, 4, 8])
@pytest.mark.parametrize("lengths", [[0, 1, 2, 7, 65, 4097, 0], [0, 1, 7, 8192, 8211, 0]], ids=["T4172", "T16411"])
@pytest.mark.parametrize("layout", ["HK", "KH"])
def test_long_fp32_parameter_gradients(seed, width, lengths, layout, silu):
    torch.manual_seed(seed)
    hidden = 136
    x = torch.randn(sum(lengths), hidden, device="cuda")
    weight = torch.randn(hidden, width, device="cuda")
    if layout == "KH":
        weight = weight.T.contiguous().T
    bias = torch.randn(hidden, device="cuda")
    initial = torch.randn(len(lengths), width - 1, hidden, device="cuda")
    dy, dfinal = torch.randn_like(x), torch.randn_like(initial)
    inputs = dict(x=x, weight=weight, bias=bias, initial=initial, dy=dy, dfinal=dfinal)
    before = snapshot(inputs)
    actual = assert_replays(lambda: _run(x, weight, bias, initial, lengths, dy, dfinal, silu=silu), repeats=3)
    _check_reference(actual, x, weight, bias, initial, lengths, dy, dfinal, silu=silu)
    assert_bitwise(inputs, before, "long-sequence inputs mutated")


@pytest.mark.parametrize("with_initial,with_final", [(True, True), (False, False), (False, True)], ids=["history", "no-history", "final-only"])
def test_fp32_parameter_workspace_budget(monkeypatch, with_initial, with_final):
    from cudnn.causal_conv1d_se import packed as adapter

    torch.manual_seed(1907)
    lengths, hidden, width = [0, 1, 2, 7, 65, 4097, 0], 8192, 8
    x = torch.randn(sum(lengths), hidden, device="cuda")
    weight = torch.randn(hidden, width, device="cuda").T.contiguous().T
    bias = torch.randn(hidden, device="cuda")
    initial = torch.randn(len(lengths), width - 1, hidden, device="cuda") if with_initial else None
    dy = torch.randn_like(x)
    dfinal = torch.randn(len(lengths), width - 1, hidden, device="cuda") if with_final else None
    inputs = {name: value for name, value in dict(x=x, weight=weight, bias=bias, initial=initial, dy=dy, dfinal=dfinal).items() if value is not None}
    before = snapshot(inputs)
    original_launch = adapter._launch
    captured = []

    def launch(direction, tensors, kernel_width, silu):
        if direction == "backward":
            partials = tensors[-1]
            assert partials.numel() * partials.element_size() <= 16 * 1024 * 1024
            captured.append(True)
        return original_launch(direction, tensors, kernel_width, silu)

    monkeypatch.setattr(adapter, "_launch", launch)
    actual = assert_replays(lambda: _run(x, weight, bias, initial, lengths, dy, dfinal), repeats=3)
    assert len(captured) == 3
    _check_reference(actual, x, weight, bias, initial, lengths, dy, dfinal)
    assert_bitwise(inputs, before, "workspace-budget inputs mutated")


def _nohistory_grad_reference(x, weight, bias, lengths, dy):
    """Keep nonfinite values out of artificial zero-valued autograd anchors."""
    xx, ww, bb = (value.detach().double().requires_grad_() for value in (x, weight, bias))
    outputs = []
    start = 0
    for length in lengths:
        if length:
            padded = torch.nn.functional.pad(xx[start : start + length].T[None], (weight.shape[1] - 1, 0))
            outputs.append(torch.nn.functional.conv1d(padded, ww[:, None], bb, groups=x.shape[1])[0].T)
        start += length
    grads = torch.autograd.grad(torch.cat(outputs), (xx, ww, bb), dy.double())
    return dict(zip(("dx", "dw", "db"), grads))


def _check_graph_fallback(lengths, layout, kind, *, hidden=136, changes=3):
    from cuda.bindings import driver as cuda
    from cudnn.causal_conv1d_se import packed as adapter

    torch.manual_seed(1907)
    total, width = sum(lengths), 8
    x = torch.randn(total, hidden, device="cuda") * 0.125
    weight = torch.randn(hidden, width, device="cuda") * 0.125
    if layout == "KH":
        weight = weight.T.contiguous().T
    bias = torch.zeros(hidden, device="cuda")
    dy = torch.randn_like(x) * 0.125
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device="cuda", dtype=torch.int32)
    dx, dw, db = torch.empty_like(x), torch.empty_like(weight), torch.empty_like(bias)
    partials = torch.empty(64, width + 1, hidden, device="cuda", dtype=torch.float64)
    tensors = (x, weight, bias, cu, None, dy, None, torch.empty_like(x), dx, dw, db, None, partials)
    compiled = adapter._compiled("backward", tensors, width, False)

    def call():
        compiled(*tensors, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for change in range(changes):
        x.copy_(torch.randn_like(x) * 0.125)
        dy.copy_(torch.randn_like(dy) * 0.125)
        if change == 1:
            # This row is inside the long sequence, away from zero padding.
            row = 4096
            if kind == "large":
                x[row], dy[row] = 2**44, 2**44
            elif kind == "nan":
                x[row] = float("nan")
            elif kind == "inf":
                dy[row] = float("inf")
        graph.replay()
        actual = {name: value.clone() for name, value in dict(dx=dx, dw=dw, db=db, partials=partials).items()}
        graph.replay()
        assert_bitwise(dict(dx=dx, dw=dw, db=db, partials=partials), actual, f"{kind}, change {change}")
        expected = _nohistory_grad_reference(x, weight, bias, lengths, dy)
        for name, want in expected.items():
            torch.testing.assert_close(actual[name], want.float(), atol=2e-5, rtol=2e-5, equal_nan=True, msg=lambda message: name + ": " + message)


@pytest.mark.parametrize("lengths", [[0, 1, 2, 7, 65, 4097, 0], [0, 1, 7, 8192, 8211, 0]], ids=["T4172", "T16411"])
@pytest.mark.parametrize("layout", ["HK", "KH"])
@pytest.mark.parametrize("kind", ["normal", "large", "nan", "inf"])
def test_fp32_changed_input_graph_fallback(lengths, layout, kind):
    _check_graph_fallback(lengths, layout, kind)


def test_fp32_large_token_extent():
    _check_graph_fallback([0, 1, 1048575, 1, 0], "HK", "normal", hidden=8, changes=1)


@pytest.mark.parametrize("contiguous", [True, False], ids=["contiguous", "strided"])
def test_fp32_product_overflow_cancellation(contiguous):
    from functools import partial
    from cudnn.ops import causal_conv1d as _causal_conv1d

    causal_conv1d = partial(_causal_conv1d, backend="se")

    hidden = 8
    x_values = torch.tensor([-3e19, 2e19], device="cuda")[:, None].expand(2, hidden)
    dy_values = torch.tensor([1e19, 2e19], device="cuda")[:, None].expand(2, hidden)
    x_storage, dy_storage = torch.zeros(2, hidden * 2, device="cuda"), torch.zeros(2, hidden * 2, device="cuda")
    x_storage[:, ::2], dy_storage[:, ::2] = x_values, dy_values
    x = x_values.clone() if contiguous else x_storage[:, ::2]
    dy = dy_values.clone() if contiguous else dy_storage[:, ::2]
    x.requires_grad_()
    weight = torch.zeros(hidden, 2, device="cuda", requires_grad=True)
    bias = torch.zeros(hidden, device="cuda", requires_grad=True)
    cu = torch.tensor([0, 2], device="cuda", dtype=torch.int32)

    def run():
        y = causal_conv1d(x.T[None], weight, bias, cu_seqlens=cu, activation=None)[0].T
        grads = torch.autograd.grad(y, (x, weight, bias), dy)
        return dict(zip(("dx", "dw", "db"), grads))

    # One filter tap overflows legitimately; the cancelling tap must stay finite.
    actual = run()
    for _ in range(2):
        assert_bitwise(run(), actual, "overflow cancellation replay")
    xx, dd = x.detach().double(), dy.double()
    expected = torch.stack((xx[0] * dd[1], (xx * dd).sum(0)), dim=1).float()
    assert torch.isfinite(actual["dw"][:, 1]).all()
    torch.testing.assert_close(actual["dw"], expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual["dx"], torch.zeros_like(x), atol=0, rtol=0)
    torch.testing.assert_close(actual["db"], dd.sum(0).float(), atol=2e-5, rtol=2e-5)
