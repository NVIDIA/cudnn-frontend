import math

import pytest
import torch
import torch.nn.functional as F

import cudnn
from cudnn.ops.fft_causal_conv1d import _long_buffer_size_bytes, fft_causal_conv1d


def _require_fft_causal_conv1d():
    required_symbols = (
        "fft_causal_conv1d_forward",
        "fft_causal_conv1d_backward",
        "long_fft_causal_conv1d_get_buffer_sizes",
        "long_fft_causal_conv1d_forward",
        "long_fft_causal_conv1d_backward",
    )
    if cudnn.backend_version() < 92600 or any(getattr(cudnn, name, None) is None for name in required_symbols):
        pytest.skip("FFT causal conv1d requires cuDNN 9.26.0 or newer bindings and backend")


def _reference(x, weight):
    dim, kernel_size = weight.shape
    return F.conv1d(F.pad(x, (kernel_size - 1, 0)), weight.flip(-1).unsqueeze(1), groups=dim)


def _make_inputs(batch, dim, seq_len, kernel_size, dtype):
    x = (0.1 * torch.randn(batch, dim, seq_len, device="cuda")).to(dtype)
    weight = (torch.randn(dim, kernel_size, device="cuda") / math.sqrt(kernel_size)).to(dtype)
    return x, weight


@pytest.mark.L0
@pytest.mark.parametrize(
    "batch,dim,seq_len,kernel_size,dtype,atol,rtol",
    [
        (2, 4, 750, 192, torch.float32, 2e-6, 2e-6),  # Medium path pads both input and filter.
        # FP64 filters above K=4096 select long FFT on every supported architecture.
        (1, 1, 8192, 8192, torch.float64, 5e-11, 5e-11),
    ],
)
def test_fft_causal_conv1d_forward_and_backward(batch, dim, seq_len, kernel_size, dtype, atol, rtol):
    _require_fft_causal_conv1d()
    x_data, weight_data = _make_inputs(batch, dim, seq_len, kernel_size, dtype)
    grad_out = 0.1 * torch.randn_like(x_data)

    x = x_data.detach().requires_grad_(True)
    weight = weight_data.detach().requires_grad_(True)
    actual = fft_causal_conv1d(x, weight)
    actual.backward(grad_out)

    # FP64 avoids TF32 convolution/reduction error masking the FFT error.
    x_ref = x_data.double().detach().requires_grad_(True)
    weight_ref = weight_data.double().detach().requires_grad_(True)
    expected = _reference(x_ref, weight_ref)
    expected.backward(grad_out.double())

    torch.testing.assert_close(actual, expected.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(x.grad, x_ref.grad.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(weight.grad, weight_ref.grad.to(dtype), atol=atol, rtol=rtol)


@pytest.mark.L0
@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (torch.float64, 5e-12, 5e-12),
        (torch.float32, 2e-6, 2e-6),
        (torch.float16, 5e-3, 5e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
def test_medium_fft_supported_dtypes(dtype, atol, rtol):
    _require_fft_causal_conv1d()
    x_data, weight_data = _make_inputs(1, 2, 256, 128, dtype)
    grad_out = (0.1 * torch.randn(1, 2, 256, device="cuda")).to(dtype)

    x = x_data.detach().requires_grad_(True)
    weight = weight_data.detach().requires_grad_(True)
    actual = fft_causal_conv1d(x, weight)
    actual.backward(grad_out)

    x_ref = x_data.double().detach().requires_grad_(True)
    weight_ref = weight_data.double().detach().requires_grad_(True)
    expected = _reference(x_ref, weight_ref)
    expected.backward(grad_out.double())

    torch.testing.assert_close(actual, expected.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(x.grad, x_ref.grad.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(weight.grad, weight_ref.grad.to(dtype), atol=atol, rtol=rtol)


@pytest.mark.L0
def test_long_fft_fake_buffer_formula_matches_backend_query():
    _require_fft_causal_conv1d()
    batch, dim, seq_len, kernel_size = 2, 3, 4096, 4096
    _, reserve_size = cudnn.long_fft_causal_conv1d_get_buffer_sizes(batch, dim, seq_len, kernel_size, 0)  # CUDNN_DATA_FLOAT
    assert reserve_size == _long_buffer_size_bytes(batch, dim, kernel_size, torch.float32)
