# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical, causality, and graph-replay contracts for the direct SE candidates."""

import warnings

import pytest
import torch

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def _dsl():
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip("requires nvidia-cutlass-dsl>=4.7.0")
    if torch.cuda.get_device_capability() not in {(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0)}:
        pytest.skip("architecture has not been admitted by the candidate GPU checks")
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack
    from cuda.bindings import driver as cuda

    return cute, from_dlpack, cuda


def _reference(x, weight, bias, silu, nwh):
    xx = x.transpose(1, 2) if nwh else x
    ww = weight.T if nwh else weight
    result = bias.double()[None, :, None].expand_as(xx).clone()
    for k in range(ww.shape[1]):
        shift = ww.shape[1] - 1 - k
        if shift < xx.shape[2]:
            result[:, :, shift:] += xx[:, :, : xx.shape[2] - shift].double() * ww[:, k].double()[None, :, None]
    if silu:
        result = torch.nn.functional.silu(result)
    return result.transpose(1, 2) if nwh else result


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("nwh", [False, True])
@pytest.mark.parametrize("silu", [False, True])
def test_direct_forward_and_all_gradients(dtype, nwh, silu):
    cute, dlpack, cuda = _dsl()
    from cudnn.causal_conv1d_se.kernel import launch_fwd
    from cudnn.causal_conv1d_se.backward_kernel import launch_bwd

    torch.manual_seed(20261022)
    x = torch.randn((2, 65, 7) if nwh else (2, 7, 65), device="cuda", dtype=dtype)
    w = torch.randn((4, 7) if nwh else (7, 4), device="cuda", dtype=dtype)
    b = torch.randn(7, device="cuda", dtype=dtype)
    y = torch.full_like(x, float("nan"))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    forward = cute.compile(launch_fwd, *(dlpack(t, assumed_align=16) for t in (x, w, b, y)), stream, 4, silu, nwh, 8, 1, False, options="--enable-tvm-ffi")
    forward(x, w, b, y, stream)
    atol, rtol = {torch.bfloat16: (0.03, 0.02), torch.float16: (0.003, 0.002), torch.float32: (2e-5, 2e-5), torch.float64: (1e-12, 1e-12)}[dtype]
    torch.testing.assert_close(y, _reference(x, w, b, silu, nwh).to(dtype), atol=atol, rtol=rtol)

    compute = torch.float64 if dtype == torch.float64 else torch.float32
    dy = torch.randn_like(x)
    dz, dx = torch.empty_like(x, dtype=compute), torch.full_like(x, float("nan"))
    dw, db = torch.full_like(w, float("nan"), dtype=compute), torch.full_like(b, float("nan"), dtype=compute)
    backward = cute.compile(
        launch_bwd, *(dlpack(t, assumed_align=16) for t in (x, w, b, dy, dz, dx, dw, db)), stream, 4, silu, nwh, 32 if nwh else 4, options="--enable-tvm-ffi"
    )
    backward(x, w, b, dy, dz, dx, dw, db, stream)
    xx, ww, bb = (t.double().requires_grad_() for t in (x, w, b))
    refs = torch.autograd.grad(_reference(xx, ww, bb, silu, nwh), (xx, ww, bb), dy.double())
    torch.testing.assert_close(dx, refs[0].to(dtype), atol=max(atol, 1e-4) if dtype != torch.float64 else atol, rtol=rtol)
    for out, ref in zip((dw, db), refs[1:]):
        torch.testing.assert_close(out, ref.to(compute), atol=2e-3 if compute == torch.float32 else 1e-10, rtol=2e-4 if compute == torch.float32 else 1e-10)
    # A second execution must overwrite all outputs, including parameter accumulators.
    dy.zero_()
    backward(x, w, b, dy, dz, dx, dw, db, stream)
    for out in (dx, dw, db):
        assert torch.count_nonzero(out).item() == 0


@pytest.mark.parametrize(
    "nwh,width,length,channels,vector,rows",
    [
        (False, 4, 264, 7, 8, 8),
        (False, 256, 257, 3, 1, 8),
        (True, 33, 257, 16, 2, 4),
        (True, 128, 257, 7, 1, 4),
    ],
)
def test_causality_tail_and_mutable_graph_replay(nwh, width, length, channels, vector, rows):
    cute, dlpack, cuda = _dsl()
    from cudnn.causal_conv1d_se.kernel import launch_fwd

    torch.manual_seed(331)
    x = torch.randn((1, length, channels) if nwh else (1, channels, length), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((width, channels) if nwh else (channels, width), device="cuda", dtype=x.dtype)
    b = torch.randn(channels, device="cuda", dtype=x.dtype)
    y = torch.empty_like(x)
    fn = cute.compile(
        launch_fwd,
        *(dlpack(t, assumed_align=16) for t in (x, w, b, y)),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        width,
        True,
        nwh,
        rows,
        vector,
        False,
        options="--enable-tvm-ffi",
    )
    work = torch.cuda.Stream()
    work.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(work):
        fn(x, w, b, y, cuda.CUstream(work.cuda_stream))
    torch.cuda.current_stream().wait_stream(work)
    before = y.clone()
    split = length // 2
    if nwh:
        x[:, split:, :].add_(3)
    else:
        x[:, :, split:].add_(3)
    graph = torch.cuda.CUDAGraph()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="The CUDA Graph is empty.*")
        with torch.cuda.graph(graph):
            fn(x, w, b, y, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    graph.replay()
    torch.testing.assert_close(y, _reference(x, w, b, True, nwh).to(y.dtype), atol=0.03, rtol=0.02)
    if nwh:
        torch.testing.assert_close(y[:, :split], before[:, :split], atol=0, rtol=0)
    else:
        torch.testing.assert_close(y[:, :, :split], before[:, :, :split], atol=0, rtol=0)
    x.zero_()
    b.fill_(2)
    graph.replay()
    torch.testing.assert_close(y, torch.full_like(y, torch.nn.functional.silu(torch.tensor(2.0)).item()), atol=0, rtol=0)


@pytest.mark.parametrize("projection_width,mixer_width", [(3, 8), (32, 256)])
def test_back_to_back_saved_output_backward(projection_width, mixer_width):
    cute, dlpack, cuda = _dsl()
    from cudnn.causal_conv1d_se.b2b_kernel import launch_b2b_fwd, launch_b2b_bwd

    torch.manual_seed(37)
    dtype = torch.float64
    x = torch.randn(1, 6, 257, device="cuda", dtype=dtype) * 0.25
    wp = torch.randn(6, projection_width, device="cuda", dtype=dtype) * 0.25
    wm = torch.randn(2, mixer_width, device="cuda", dtype=dtype) * 0.25
    skip = torch.randn(2, device="cuda", dtype=dtype) * 0.25
    y = torch.empty(1, 2, 257, device="cuda", dtype=dtype)
    gated = torch.empty_like(y)
    dy = torch.randn_like(y)
    dx, dwp, dwm, ds = (torch.empty_like(t) for t in (x, wp, wm, skip))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fwd = cute.compile(
        launch_b2b_fwd, *(dlpack(t, assumed_align=16) for t in (x, wp, wm, skip, y, gated)), stream, projection_width, mixer_width, options="--enable-tvm-ffi"
    )
    bwd = cute.compile(
        launch_b2b_bwd,
        *(dlpack(t, assumed_align=16) for t in (x, wp, wm, skip, y, dy, dx, dwp, dwm, ds)),
        stream,
        projection_width,
        mixer_width,
        options="--enable-tvm-ffi",
    )
    fwd(x, wp, wm, skip, y, gated, stream)
    xx, wpp, wmm, ss = (t.detach().requires_grad_() for t in (x, wp, wm, skip))
    projected = _reference(xx, wpp, torch.zeros_like(wp[:, 0]), False, False).reshape(1, 2, 3, 257)
    p1, p2, pv = projected.unbind(2)
    product = p2 * pv
    ref_y = _reference(product, wmm, torch.zeros_like(skip), False, False) + product * ss[None, :, None]
    torch.testing.assert_close(y, ref_y, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(gated, p1 * ref_y, atol=1e-11, rtol=1e-11)
    refs = torch.autograd.grad(p1 * ref_y, (xx, wpp, wmm, ss), dy)
    bwd(x, wp, wm, skip, y, dy, dx, dwp, dwm, ds, stream)
    for out, ref in zip((dx, dwp, dwm, ds), refs):
        torch.testing.assert_close(out, ref, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("family", ["ncw", "nwh", "b2b"])
def test_parameter_gradient_accumulation(family):
    """The additive backend contract must preserve preexisting parameter gradients."""
    cute, dlpack, cuda = _dsl()
    from cudnn.causal_conv1d_se.backward_kernel import launch_bwd
    from cudnn.causal_conv1d_se.b2b_kernel import launch_b2b_fwd, launch_b2b_bwd
    from se_bitwise import bitwise_difference, parameter_difference

    torch.manual_seed(611)
    nwh, b2b = family == "nwh", family == "b2b"
    channels, length, width = (1024 if nwh else 7), (1024 if nwh else 65), 3
    shape = (1, length, channels) if nwh else (1, (3 if b2b else 1) * channels, length)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.25
    w = torch.randn((width, channels) if nwh else ((3 if b2b else 1) * channels, width), device="cuda", dtype=x.dtype) * 0.25
    bias = torch.randn(channels, device="cuda", dtype=x.dtype) * 0.25
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if b2b:
        wm = torch.randn(channels, 4, device="cuda", dtype=x.dtype) * 0.25
        y, gated, dy = (torch.empty((1, channels, length), device="cuda", dtype=x.dtype) for _ in range(3))
        dy.normal_()
        forward = cute.compile(launch_b2b_fwd, *(dlpack(t, assumed_align=16) for t in (x, w, wm, bias, y, gated)), stream, width, 4, options="--enable-tvm-ffi")
        forward(x, w, wm, bias, y, gated, stream)
        outputs = [torch.empty_like(x)] + [torch.empty_like(t, dtype=torch.float32) for t in (w, wm, bias)]
        tensors = [x, w, wm, bias, y, dy] + outputs
        function, constants = launch_b2b_bwd, [width, 4]
    else:
        dy, dz = torch.randn_like(x), torch.empty_like(x, dtype=torch.float32)
        outputs = [torch.empty_like(x)] + [torch.empty_like(t, dtype=torch.float32) for t in (w, bias)]
        tensors = [x, w, bias, dy, dz] + outputs
        function, constants = launch_bwd, [width, True, nwh, 32 if nwh else 4]
    clear, add = (
        cute.compile(function, *(dlpack(t, assumed_align=16) for t in tensors), stream, *constants, accumulate, options="--enable-tvm-ffi")
        for accumulate in (False, True)
    )
    clear(*tensors, stream)
    first = [t.clone() for t in outputs]
    for value in outputs[1:]:
        value.fill_(0.25)
    add(*tensors, stream)
    add(*tensors, stream)
    assert bitwise_difference(outputs[0], first[0])["equal"]
    for value, reference in zip(outputs[1:], first[1:]):
        assert parameter_difference(value, 0.25 + 2 * reference, 2 * length)["accepted"]
    clear(*tensors, stream)
    for value, reference in zip(outputs[1:], first[1:]):
        assert parameter_difference(value, reference, length)["accepted"]
