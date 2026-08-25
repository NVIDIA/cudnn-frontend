# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import torch.nn.functional as F

import cudnn

pytestmark = pytest.mark.L0


@pytest.mark.parametrize(
    "binding_name,args,kernel_size_index,api_name,argument_name,max_kernel_size",
    [
        (
            "causal_conv1d_forward",
            (0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0),
            8,
            "causal_conv1d",
            "kernel_size",
            256,
        ),
        (
            "causal_conv1d_backward",
            (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0),
            11,
            "causal_conv1d",
            "kernel_size",
            256,
        ),
        (
            "causal_conv1d_nwh_forward",
            (0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0),
            8,
            "causal_conv1d_nwh",
            "kernel_size",
            128,
        ),
        (
            "causal_conv1d_nwh_backward",
            (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0),
            11,
            "causal_conv1d_nwh",
            "kernel_size",
            128,
        ),
        (
            "b2b_causal_conv1d_forward",
            (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 2, 0),
            10,
            "b2b_causal_conv1d",
            "kernel_size_proj",
            32,
        ),
        (
            "b2b_causal_conv1d_backward",
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 2, 0, 0),
            14,
            "b2b_causal_conv1d",
            "kernel_size_proj",
            32,
        ),
        (
            "b2b_causal_conv1d_forward",
            (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 0, 0),
            11,
            "b2b_causal_conv1d",
            "kernel_size_mixer",
            256,
        ),
        (
            "b2b_causal_conv1d_backward",
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 0, 0, 0),
            15,
            "b2b_causal_conv1d",
            "kernel_size_mixer",
            256,
        ),
    ],
)
@pytest.mark.parametrize("boundary", ["below", "above"])
def test_causal_conv1d_rejects_unsupported_kernel_size(binding_name, args, kernel_size_index, api_name, argument_name, max_kernel_size, boundary):
    binding = getattr(cudnn, binding_name, None)
    if binding is None:
        pytest.skip(f"{binding_name} is unavailable in this cuDNN frontend build")

    kernel_size = 1 if boundary == "below" else max_kernel_size + 1
    args = list(args)
    args[kernel_size_index] = kernel_size

    message = rf"{api_name} {argument_name} must be between 2 and {max_kernel_size}, inclusive; got {kernel_size}"
    with pytest.raises(ValueError, match=message):
        binding(*args)


_TOLERANCES = {
    torch.float64: (5e-12, 5e-12),
    torch.float32: (5e-6, 5e-6),
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (1e-2, 1e-2),
}

# Mirrors the dtype, activation, and odd/even filter-width coverage in the
# causal conv1d NHW/NWH notebooks, extended with the new FP64 support.
_CAUSAL_CONV1D_CASES = [
    pytest.param(torch.float32, 4, "silu", id="fp32-k4-silu"),
    pytest.param(torch.float16, 4, "silu", id="fp16-k4-silu"),
    pytest.param(torch.bfloat16, 4, "silu", id="bf16-k4-silu"),
    pytest.param(torch.bfloat16, 3, "silu", id="bf16-k3-silu"),
    pytest.param(torch.float16, 7, "silu", id="fp16-k7-silu"),
    pytest.param(torch.bfloat16, 4, "identity", id="bf16-k4-identity"),
    pytest.param(torch.float64, 4, "silu", id="fp64-k4-silu"),
    pytest.param(torch.float64, 4, "identity", id="fp64-k4-identity"),
]


def _require_variant(required_symbols, dtype, minimum_version):
    is_sm107 = torch.cuda.get_device_capability() == (10, 7)
    required_version = max(minimum_version, 92600 if dtype == torch.float64 or is_sm107 else 0)
    if cudnn.backend_version() < required_version or any(getattr(cudnn, name, None) is None for name in required_symbols):
        pytest.skip(f"SE causal conv1d case requires cuDNN {required_version // 10000}.{required_version // 100 % 100}")


def test_require_variant_skips_sm107_before_cudnn_926(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 7))
    monkeypatch.setattr(cudnn, "backend_version", lambda: 92500)

    with pytest.raises(pytest.skip.Exception, match=r"requires cuDNN 9\.26"):
        _require_variant((), torch.float32, 92200)


def _make_tensor(shape, dtype, scale=1.0):
    return (scale * torch.randn(*shape, device="cuda")).to(dtype)


def _causal_conv1d_reference(x, weight, bias=None, activation="identity"):
    dim, kernel_size = weight.shape
    y = F.conv1d(F.pad(x, (kernel_size - 1, 0)), weight.unsqueeze(1), bias=bias, groups=dim)
    return F.silu(y) if activation == "silu" else y


def _causal_conv1d_nwh_reference(x, weight, bias=None, activation="identity"):
    return _causal_conv1d_reference(
        x.transpose(1, 2),
        weight.transpose(0, 1),
        bias,
        activation,
    ).transpose(1, 2)


def _b2b_causal_conv1d_reference(x, weights_proj, weights_mixer, skip_bias):
    projected = _causal_conv1d_reference(x, weights_proj)
    x1, x2, value = projected[:, 0::3], projected[:, 1::3], projected[:, 2::3]
    gated = x2 * value
    mixed = _causal_conv1d_reference(gated, weights_mixer) + skip_bias[None, :, None] * gated
    return mixed * x1


def _assert_close(actual, expected, dtype):
    atol, rtol = _TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected.to(dtype), atol=atol, rtol=rtol)


@pytest.mark.L0
@pytest.mark.parametrize("dtype,kernel_size,activation", _CAUSAL_CONV1D_CASES)
def test_causal_conv1d_autograd(dtype, kernel_size, activation):
    _require_variant(("causal_conv1d_forward", "causal_conv1d_backward"), dtype, 92200)
    torch.manual_seed(42)
    batch, dim, seq_len = 2, 16, 128
    x_data = _make_tensor((batch, dim, seq_len), dtype, scale=0.1)
    weight_data = _make_tensor((dim, kernel_size), dtype, scale=1.0 / math.sqrt(kernel_size))
    bias_data = _make_tensor((dim,), dtype, scale=0.1)
    grad_out = _make_tensor((batch, dim, seq_len), dtype, scale=0.1)

    x = x_data.detach().requires_grad_(True)
    weight = weight_data.detach().requires_grad_(True)
    bias = bias_data.detach().requires_grad_(True)
    actual = cudnn.ops.causal_conv1d(x, weight, bias, activation=activation)
    actual.backward(grad_out)

    x_ref = x_data.double().detach().requires_grad_(True)
    weight_ref = weight_data.double().detach().requires_grad_(True)
    bias_ref = bias_data.double().detach().requires_grad_(True)
    expected = _causal_conv1d_reference(x_ref, weight_ref, bias_ref, activation)
    expected.backward(grad_out.double())

    _assert_close(actual, expected, dtype)
    _assert_close(x.grad, x_ref.grad, dtype)
    _assert_close(weight.grad, weight_ref.grad, dtype)
    _assert_close(bias.grad, bias_ref.grad, dtype)


@pytest.mark.L0
@pytest.mark.parametrize("dtype,kernel_size,activation", _CAUSAL_CONV1D_CASES)
def test_causal_conv1d_nwh_autograd(dtype, kernel_size, activation):
    _require_variant(("causal_conv1d_nwh_forward", "causal_conv1d_nwh_backward"), dtype, 92400)
    torch.manual_seed(42)
    batch, dim, seq_len = 2, 16, 128
    x_data = _make_tensor((batch, seq_len, dim), dtype, scale=0.1)
    weight_data = _make_tensor((kernel_size, dim), dtype, scale=1.0 / math.sqrt(kernel_size))
    bias_data = _make_tensor((dim,), dtype, scale=0.1)
    grad_out = _make_tensor((batch, seq_len, dim), dtype, scale=0.1)

    x = x_data.detach().requires_grad_(True)
    weight = weight_data.detach().requires_grad_(True)
    bias = bias_data.detach().requires_grad_(True)
    actual = cudnn.ops.causal_conv1d_nwh(x, weight, bias, activation=activation)
    actual.backward(grad_out)

    x_ref = x_data.double().detach().requires_grad_(True)
    weight_ref = weight_data.double().detach().requires_grad_(True)
    bias_ref = bias_data.double().detach().requires_grad_(True)
    expected = _causal_conv1d_nwh_reference(x_ref, weight_ref, bias_ref, activation)
    expected.backward(grad_out.double())

    _assert_close(actual, expected, dtype)
    _assert_close(x.grad, x_ref.grad, dtype)
    _assert_close(weight.grad, weight_ref.grad, dtype)
    _assert_close(bias.grad, bias_ref.grad, dtype)


@pytest.mark.L0
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, id="fp32"),
        pytest.param(torch.float16, id="fp16"),
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float64, id="fp64"),
    ],
)
def test_b2b_causal_conv1d_autograd(dtype):
    _require_variant(("b2b_causal_conv1d_forward", "b2b_causal_conv1d_backward"), dtype, 92400)
    torch.manual_seed(42)
    batch, dim, seq_len = 2, 8, 128
    kernel_size_proj, kernel_size_mixer = 4, 7
    x_data = _make_tensor((batch, 3 * dim, seq_len), dtype, scale=0.1)
    weights_proj_data = _make_tensor(
        (3 * dim, kernel_size_proj),
        dtype,
        scale=1.0 / math.sqrt(kernel_size_proj),
    )
    weights_mixer_data = _make_tensor(
        (dim, kernel_size_mixer),
        dtype,
        scale=1.0 / math.sqrt(kernel_size_mixer),
    )
    skip_bias_data = _make_tensor((dim,), dtype, scale=0.1)
    grad_out = _make_tensor((batch, dim, seq_len), dtype, scale=0.1)

    x = x_data.detach().requires_grad_(True)
    weights_proj = weights_proj_data.detach().requires_grad_(True)
    weights_mixer = weights_mixer_data.detach().requires_grad_(True)
    skip_bias = skip_bias_data.detach().requires_grad_(True)
    actual = cudnn.ops.b2b_causal_conv1d(x, weights_proj, weights_mixer, skip_bias)
    actual.backward(grad_out)

    x_ref = x_data.double().detach().requires_grad_(True)
    weights_proj_ref = weights_proj_data.double().detach().requires_grad_(True)
    weights_mixer_ref = weights_mixer_data.double().detach().requires_grad_(True)
    skip_bias_ref = skip_bias_data.double().detach().requires_grad_(True)
    expected = _b2b_causal_conv1d_reference(
        x_ref,
        weights_proj_ref,
        weights_mixer_ref,
        skip_bias_ref,
    )
    expected.backward(grad_out.double())

    _assert_close(actual, expected, dtype)
    _assert_close(x.grad, x_ref.grad, dtype)
    _assert_close(weights_proj.grad, weights_proj_ref.grad, dtype)
    _assert_close(weights_mixer.grad, weights_mixer_ref.grad, dtype)
    _assert_close(skip_bias.grad, skip_bias_ref.grad, dtype)


@pytest.mark.L0
@pytest.mark.parametrize("layout", ["nhw", "nwh"])
def test_causal_conv1d_compiled_autograd(layout):
    symbols = ("causal_conv1d_forward", "causal_conv1d_backward") if layout == "nhw" else ("causal_conv1d_nwh_forward", "causal_conv1d_nwh_backward")
    _require_variant(symbols, torch.bfloat16, 92200 if layout == "nhw" else 92400)
    torch.manual_seed(42)
    batch, dim, seq_len, kernel_size = 1, 16, 64, 4
    if layout == "nhw":
        x_data = _make_tensor((batch, dim, seq_len), torch.bfloat16, scale=0.1)
        weight_data = _make_tensor((dim, kernel_size), torch.bfloat16, scale=0.5)
        op = cudnn.ops.causal_conv1d
    else:
        x_data = _make_tensor((batch, seq_len, dim), torch.bfloat16, scale=0.1)
        weight_data = _make_tensor((kernel_size, dim), torch.bfloat16, scale=0.5)
        op = cudnn.ops.causal_conv1d_nwh
    bias_data = _make_tensor((dim,), torch.bfloat16, scale=0.1)
    grad_out = _make_tensor(x_data.shape, torch.bfloat16, scale=0.1)

    def train_step(x, weight, bias, dy):
        x = x.detach().requires_grad_(True)
        weight = weight.detach().requires_grad_(True)
        bias = bias.detach().requires_grad_(True)
        op(x, weight, bias, activation="silu").backward(dy)
        return x.grad, weight.grad, bias.grad

    eager = train_step(x_data, weight_data, bias_data, grad_out)
    compiled = torch.compile(train_step)(x_data, weight_data, bias_data, grad_out)
    for eager_grad, compiled_grad in zip(eager, compiled):
        torch.testing.assert_close(compiled_grad, eager_grad, atol=0, rtol=0)
