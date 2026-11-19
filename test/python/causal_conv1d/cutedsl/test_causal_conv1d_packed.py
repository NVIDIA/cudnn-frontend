# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed sequence, boundary, and state-gradient contracts for direct convolution."""

import pytest
import torch

pytestmark = pytest.mark.L0


def _dsl():
    from cudnn.frost.buffers import CUTEDSL_MIN_VERSION, cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip(f"requires CuTe DSL >= {CUTEDSL_MIN_VERSION}")
    if torch.cuda.get_device_capability() not in {(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0)}:
        pytest.skip("requires a target in the packed-convolution qualification matrix")
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack
    from cuda.bindings import driver as cuda

    return cute, from_dlpack, cuda


def reference(x, weight, bias, initial, lengths, silu):
    """Independent FP64 grouped-convolution oracle, including short/empty history."""
    width = weight.shape[1]
    anchor = x.sum() * 0 + weight.sum() * 0
    if bias is not None:
        anchor = anchor + bias.sum() * 0
    if initial is not None:
        anchor = anchor + initial.sum() * 0
    outputs, finals = [], []
    start = 0
    for seq, length in enumerate(lengths):
        prefix = x.new_zeros((width - 1, x.shape[1])) if initial is None else initial[seq]
        joined = torch.cat((prefix, x[start : start + length]), dim=0)
        if length:
            out = torch.nn.functional.conv1d(joined.T[None], weight[:, None, :], bias, groups=x.shape[1])[0].T
            outputs.append(torch.nn.functional.silu(out) if silu else out)
        finals.append(joined[-(width - 1) :])
        start += length
    y = torch.cat(outputs) if outputs else x * 0
    return y + anchor, torch.stack(finals) + anchor


def _compile(fn, args, width, silu, fast=None):
    cute, dlpack, cuda = _dsl()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    views = [None if t is None else dlpack(t.detach(), assumed_align=16) for t in args]
    extra = () if fast is None else (fast,)
    compiled = cute.compile(fn, *views, stream, width, silu, *extra, options="--enable-tvm-ffi")
    return compiled, stream


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", range(2, 9))
@pytest.mark.parametrize("silu", [False, True])
def test_packed_outputs_and_all_gradients(dtype, width, silu):
    _dsl()
    from cudnn.causal_conv1d_se.packed_kernel import launch_packed_forward, launch_packed_backward

    torch.manual_seed(20261105)
    lengths, hidden = [0, 1, 2, 0, 7, 33, 0, 129, 257, 0], 24
    x = torch.randn(sum(lengths), hidden, device="cuda", dtype=dtype)
    w = torch.randn(width, hidden, device="cuda", dtype=dtype).T
    b = torch.randn(hidden, device="cuda", dtype=dtype)
    initial = torch.randn(len(lengths), width - 1, hidden, device="cuda", dtype=dtype)
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), device="cuda", dtype=torch.int32)
    y, final = torch.full_like(x, float("nan")), torch.full_like(initial, float("nan"))
    fargs = (x, w, b, cu, initial, y, final)
    forward, stream = _compile(launch_packed_forward, fargs, width, silu, fast=True)
    forward(*fargs, stream)
    xx, ww, bb, ii = (t.double().detach().requires_grad_() for t in (x, w, b, initial))
    expected_y, expected_final = reference(xx, ww, bb, ii, lengths, silu)
    atol, rtol = {torch.float16: (0.003, 0.002), torch.bfloat16: (0.03, 0.02), torch.float32: (2e-5, 2e-5)}[dtype]
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, expected_y.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(final, expected_final.to(dtype), atol=0, rtol=0)
    dy, dfinal = torch.randn_like(x), torch.randn_like(initial)
    dz, dx = torch.empty_like(x, dtype=torch.float32), torch.full_like(x, float("nan"))
    dw, db, di = (torch.full_like(t, float("nan")) for t in (w, b, initial))
    partials = torch.empty(4, width + 1, hidden, device="cuda", dtype=torch.float64 if dtype == torch.float32 else torch.float32)
    bargs = (x, w, b, cu, initial, dy, dfinal, dz, dx, dw, db, di, partials)
    backward, stream = _compile(launch_packed_backward, bargs, width, silu, fast=True)
    backward(*bargs, stream)
    expected = torch.autograd.grad((expected_y, expected_final), (xx, ww, bb, ii), (dy.double(), dfinal.double()))
    for actual, want in zip((dx, dw, db, di), expected):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, want.to(dtype), atol=atol, rtol=rtol)
    dy.zero_()
    dfinal.zero_()
    backward(*bargs, stream)
    for actual in (dx, dw, db, di):
        assert torch.count_nonzero(actual).item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_bias,with_initial", [(False, False), (True, True)])
@pytest.mark.parametrize("lengths", [[0, 0, 0], [2, 0], [1, 1, 1], [0, 1, 7, 33, 0]])
def test_packed_adapter_optional_state_and_autograd(dtype, with_bias, with_initial, lengths):
    _dsl()
    from cudnn.causal_conv1d_se.packed import packed_causal_conv1d

    torch.manual_seed(817)
    hidden, width = 24, 4
    x = torch.randn(sum(lengths), hidden, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden, width, device="cuda", dtype=dtype, requires_grad=True)
    if with_bias:
        weight = weight.T.contiguous().T
    bias = torch.randn(hidden, device="cuda", dtype=dtype, requires_grad=True) if with_bias else None
    initial = torch.randn(len(lengths), width - 1, hidden, device="cuda", dtype=dtype, requires_grad=True) if with_initial else None
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), device="cuda", dtype=torch.int32)
    y, final = packed_causal_conv1d(x, weight, cu, bias, initial, silu=True, return_final_states=True)
    inputs = [t for t in (x, weight, bias, initial) if t is not None]
    copies = [None if t is None else t.detach().double().requires_grad_() for t in (x, weight, bias, initial)]
    expected_y, expected_final = reference(*copies, lengths, True)
    atol, rtol = {torch.float16: (0.003, 0.002), torch.bfloat16: (0.03, 0.02), torch.float32: (2e-5, 2e-5)}[dtype]
    torch.testing.assert_close(y, expected_y.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(final, expected_final.to(dtype), atol=0, rtol=0)
    dy, dfinal = torch.randn_like(y), torch.randn_like(final)
    got = torch.autograd.grad((y, final), inputs, (dy, dfinal))
    want = torch.autograd.grad((expected_y, expected_final), [t for t in copies if t is not None], (dy.double(), dfinal.double()))
    for actual, target in zip(got, want):
        torch.testing.assert_close(actual, target.to(dtype), atol=atol, rtol=rtol)


def test_packed_dynamic_cache_and_changed_boundary_graph_replay(monkeypatch):
    cute, _, _ = _dsl()
    from cudnn.causal_conv1d_se.packed import packed_causal_conv1d

    torch.manual_seed(918)
    weight = torch.randn(24, 3, device="cuda", dtype=torch.float32)
    cu = torch.tensor([0, 1, 9], device="cuda", dtype=torch.int32)
    x = torch.randn(9, 24, device="cuda")
    packed_causal_conv1d(x, weight, cu, silu=False)

    def forbidden_compile(*args, **kwargs):
        raise AssertionError("packed token/sequence extents caused a new compilation")

    monkeypatch.setattr(cute, "compile", forbidden_compile)
    longer = torch.randn(13, 24, device="cuda")
    more_sequences = torch.tensor([0, 0, 3, 13], device="cuda", dtype=torch.int32)
    got, _ = packed_causal_conv1d(longer, weight, more_sequences, silu=False)
    want, _ = reference(longer.double(), weight.double(), None, None, [0, 3, 10], False)
    torch.testing.assert_close(got, want.float(), atol=2e-5, rtol=2e-5)
    work = torch.cuda.Stream()
    work.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(work):
            with torch.cuda.graph(graph, stream=work):
                torch.cuda.set_sync_debug_mode("error")
                try:
                    captured, _ = packed_causal_conv1d(x, weight, cu, silu=False)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
        torch.cuda.current_stream().wait_stream(work)
        cu.copy_(torch.tensor([0, 5, 9], device="cuda", dtype=torch.int32))
        x.add_(2)
        graph.replay()
        expected, _ = reference(x.double(), weight.double(), None, None, [5, 4], False)
        torch.testing.assert_close(captured, expected.float(), atol=2e-5, rtol=2e-5)
    finally:
        graph.reset()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_packed_decode_padded_filter_alignment_cache(dtype):
    """The same unit-stride pattern can have aligned or misaligned tap columns."""
    _dsl()
    from cudnn.causal_conv1d_se.packed import packed_causal_conv1d

    torch.manual_seed(917)
    hidden, width, lengths = 24, 4, [1, 1, 1]
    x = torch.randn(3, hidden, device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(hidden, device="cuda", dtype=dtype, requires_grad=True)
    initial = torch.randn(3, width - 1, hidden, device="cuda", dtype=dtype, requires_grad=True)
    cu = torch.arange(4, device="cuda", dtype=torch.int32)
    atol, rtol = {torch.float16: (0.003, 0.002), torch.bfloat16: (0.03, 0.02), torch.float32: (2e-5, 2e-5)}[dtype]
    # Neither padded view is contiguous; both have stride(0)==1. Warming the
    # aligned case must not make the next call reuse its vector-load signature.
    for pitch in (32, 25, 32):
        weight = torch.randn(width, pitch, device="cuda", dtype=dtype)[:, :hidden].T.detach().requires_grad_()
        y, final = packed_causal_conv1d(x, weight, cu, bias, initial, silu=True, return_final_states=True)
        xx, ww, bb, ii = (t.detach().double().requires_grad_() for t in (x, weight, bias, initial))
        expected_y, expected_final = reference(xx, ww, bb, ii, lengths, True)
        torch.testing.assert_close(y, expected_y.to(dtype), atol=atol, rtol=rtol)
        torch.testing.assert_close(final, expected_final.to(dtype), atol=0, rtol=0)
        dy, dfinal = torch.randn_like(y), torch.randn_like(final)
        got = torch.autograd.grad((y, final), (x, weight, bias, initial), (dy, dfinal))
        want = torch.autograd.grad((expected_y, expected_final), (xx, ww, bb, ii), (dy.double(), dfinal.double()))
        for actual, expected in zip(got, want):
            torch.testing.assert_close(actual, expected.to(dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided_tokens", [False, True])
@pytest.mark.parametrize("loss_outputs", ["tokens", "history", "both"])
def test_packed_strides_and_partial_output_gradients(dtype, strided_tokens, loss_outputs):
    """Exercise arbitrary histories/gradients and losses using either output."""
    _dsl()
    from cudnn.causal_conv1d_se.packed import packed_causal_conv1d

    torch.manual_seed(912)
    hidden, width, lengths = 24, 5, [0, 1, 5, 0, 129, 257, 0]
    total, sequences = sum(lengths), len(lengths)
    x = torch.randn(total, hidden * (2 if strided_tokens else 1), device="cuda", dtype=dtype)
    if strided_tokens:
        x = x[:, ::2]
    x.requires_grad_()
    weight = torch.randn(hidden, width * 2, device="cuda", dtype=dtype)[:, ::2].requires_grad_()
    bias = torch.randn(hidden * 2, device="cuda", dtype=dtype)[::2]
    if not strided_tokens:
        bias = bias.contiguous()
    bias.requires_grad_()
    initial = torch.randn(sequences, hidden, width - 1, device="cuda", dtype=dtype).transpose(1, 2).requires_grad_()
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), device="cuda", dtype=torch.int32)
    y, final = packed_causal_conv1d(x, weight, cu, bias, initial, silu=True, return_final_states=True)
    copies = [t.detach().double().requires_grad_() for t in (x, weight, bias, initial)]
    expected_y, expected_final = reference(*copies, lengths, True)
    atol, rtol = {torch.float16: (0.003, 0.002), torch.bfloat16: (0.03, 0.02), torch.float32: (2e-5, 2e-5)}[dtype]
    torch.testing.assert_close(y, expected_y.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(final, expected_final.to(dtype), atol=0, rtol=0)
    dy = torch.randn(hidden, total, device="cuda", dtype=dtype).T
    if not strided_tokens:
        dy = dy.contiguous()
    dfinal = torch.randn(sequences, hidden, width - 1, device="cuda", dtype=dtype).transpose(1, 2)
    if loss_outputs == "tokens":
        outputs, expected_outputs, grads = (y,), (expected_y,), (dy,)
    elif loss_outputs == "history":
        outputs, expected_outputs, grads = (final,), (expected_final,), (dfinal,)
    else:
        outputs, expected_outputs, grads = (y, final), (expected_y, expected_final), (dy, dfinal)
    got = torch.autograd.grad(outputs, (x, weight, bias, initial), grads)
    want = torch.autograd.grad(expected_outputs, copies, tuple(g.double() for g in grads))
    for actual, target in zip(got, want):
        torch.testing.assert_close(actual, target.to(dtype), atol=atol, rtol=rtol)


def test_packed_history_only_gradient_ignores_nonfinite_filter():
    """A history-only loss has no mathematical dependency on filter or bias."""
    _dsl()
    from cudnn.causal_conv1d_se.packed import packed_causal_conv1d

    lengths, hidden, width = [0, 1, 5, 0], 24, 3
    x = torch.randn(sum(lengths), hidden, device="cuda", requires_grad=True)
    weight = torch.full((hidden, width), float("nan"), device="cuda", requires_grad=True)
    bias = torch.full((hidden,), float("inf"), device="cuda", requires_grad=True)
    initial = torch.randn(len(lengths), width - 1, hidden, device="cuda", requires_grad=True)
    cu = torch.tensor([0, 0, 1, 6, 6], device="cuda", dtype=torch.int32)
    _, final = packed_causal_conv1d(x, weight, cu, bias, initial, silu=True, return_final_states=True)
    dfinal = torch.randn_like(final)
    dx, dw, db, di = torch.autograd.grad(final, (x, weight, bias, initial), dfinal)
    start, histories = 0, []
    for seq, length in enumerate(lengths):
        histories.append(torch.cat((initial[seq], x[start : start + length]))[-(width - 1) :])
        start += length
    expected_dx, expected_di = torch.autograd.grad(torch.stack(histories), (x, initial), dfinal)
    torch.testing.assert_close(dx, expected_dx, atol=0, rtol=0)
    torch.testing.assert_close(di, expected_di, atol=0, rtol=0)
    assert torch.count_nonzero(dw).item() == 0
    assert torch.count_nonzero(db).item() == 0


@pytest.mark.parametrize("name,change", [("dy", "shape"), ("dy", "dtype"), ("dfinal", "shape"), ("dfinal", "device")])
def test_packed_rejects_invalid_output_gradient(name, change):
    _dsl()
    from cudnn.causal_conv1d_se.packed import _backward

    x = torch.randn(9, 24, device="cuda")
    weight = torch.randn(24, 3, device="cuda")
    cu = torch.tensor([0, 1, 9], device="cuda", dtype=torch.int32)
    gradients = {"dy": torch.randn_like(x), "dfinal": torch.randn(2, 2, 24, device="cuda")}
    if change == "shape":
        gradients[name] = gradients[name][..., :-1]
    elif change == "dtype":
        gradients[name] = gradients[name].half()
    else:
        gradients[name] = gradients[name].cpu()
    with pytest.raises(ValueError, match=name):
        _backward(x, weight, cu, None, None, gradients["dy"], gradients["dfinal"], False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packed_long_backward_changed_boundaries_graph_replay(dtype):
    """Reuse backward buffers across long sequences and changed graph inputs."""
    _, _, cuda = _dsl()
    from cudnn.causal_conv1d_se.packed import _compiled

    torch.manual_seed(7713)
    lengths = [0, 1, 7, 4096, 4115, 0]
    changed = [0, 2048, 0, 1, 6170, 0]
    hidden, width = 24, 8
    x = torch.randn(sum(lengths), hidden, device="cuda", dtype=dtype) * 0.125
    weight = torch.randn(hidden, width, device="cuda", dtype=dtype) * 0.125
    bias = torch.randn(hidden, device="cuda", dtype=dtype) * 0.125
    initial = torch.randn(len(lengths), width - 1, hidden, device="cuda", dtype=dtype) * 0.125
    dy, dfinal = torch.randn_like(x) * 0.125, torch.randn_like(initial) * 0.125
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device="cuda", dtype=torch.int32)
    dx, dw, db, di = (torch.empty_like(value) for value in (x, weight, bias, initial))
    dz = torch.empty_like(x, dtype=torch.float32)
    partials = torch.empty(64, width + 1, hidden, device="cuda", dtype=torch.float32)
    tensors = (x, weight, bias, cu, initial, dy, dfinal, dz, dx, dw, db, di, partials)
    backward = _compiled("backward", tensors, width, False)

    def call():
        backward(*tensors, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            call()
        for current in (lengths, changed, lengths):
            for value in (x, weight, bias, initial, dy, dfinal):
                value.copy_(torch.randn_like(value) * 0.125)
            cu.copy_(torch.tensor([0] + torch.tensor(current).cumsum(0).tolist(), device="cuda", dtype=torch.int32))
            for output in (dx, dw, db, di):
                output.fill_(float("nan"))
            graph.replay()
            xx, ww, bb, ii = (value.detach().double().requires_grad_() for value in (x, weight, bias, initial))
            y, final = reference(xx, ww, bb, ii, current, False)
            expected = torch.autograd.grad((y, final), (xx, ww, bb, ii), (dy.double(), dfinal.double()))
            atol, rtol = (0.03, 0.02) if dtype == torch.bfloat16 else (0.003, 0.002)
            for actual, want in zip((dx, dw, db, di), expected):
                torch.testing.assert_close(actual, want.to(dtype), atol=atol, rtol=rtol)
    finally:
        graph.reset()
