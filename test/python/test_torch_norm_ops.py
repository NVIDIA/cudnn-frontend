"""Correctness tests for the experimental PyTorch normalization ops."""

import cudnn
import pytest
import torch

from cudnn.experimental.ops import layer_norm, rms_norm


SHAPES = [(128, 768), (32, 4096)]
DTYPES = [torch.float16, torch.bfloat16]


def _rms_norm_reference(input, weight, bias=None, eps=1e-5):
    normalized = input.float() * torch.rsqrt(input.float().square().mean(dim=-1, keepdim=True) + eps)
    output = normalized * weight.float()
    if bias is not None:
        output = output + bias.float()
    return output.to(input.dtype)


@pytest.mark.L0
@pytest.mark.skipif(cudnn.backend_version() < 80906, reason="RMSNorm requires cuDNN >= 8.9.6")
@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: "x".join(map(str, shape)))
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "bias"])
def test_rms_norm_custom_op(shape, dtype, has_bias):
    torch.manual_seed(42)
    input = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn_like(weight, requires_grad=True) if has_bias else None

    actual = rms_norm(input, weight, bias, eps=1e-3)
    reference_input = input.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference_bias = bias.detach().clone().requires_grad_(True) if bias is not None else None
    expected = _rms_norm_reference(reference_input, reference_weight, reference_bias, eps=1e-3)

    tolerance = 0.03125
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(input.grad, reference_input.grad, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(weight.grad, reference_weight.grad, atol=tolerance, rtol=tolerance)
    if bias is not None:
        torch.testing.assert_close(bias.grad, reference_bias.grad, atol=tolerance, rtol=tolerance)


@pytest.mark.L0
@pytest.mark.skipif(cudnn.backend_version() < 80905, reason="LayerNorm requires cuDNN >= 8.9.5")
@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: "x".join(map(str, shape)))
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("has_parameters", [False, True], ids=["no_parameters", "parameters"])
def test_layer_norm_custom_op(shape, dtype, has_parameters):
    torch.manual_seed(42)
    input = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype, requires_grad=True) if has_parameters else None
    bias = torch.randn(shape[-1], device="cuda", dtype=dtype, requires_grad=True) if has_parameters else None

    actual = layer_norm(input, (shape[-1],), weight, bias)
    reference_input = input.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True) if weight is not None else None
    reference_bias = bias.detach().clone().requires_grad_(True) if bias is not None else None
    expected = torch.nn.functional.layer_norm(reference_input, (shape[-1],), reference_weight, reference_bias)

    tolerance = 0.03125 if dtype == torch.bfloat16 else 0.015625
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(input.grad, reference_input.grad, atol=tolerance, rtol=tolerance)
    if weight is not None:
        torch.testing.assert_close(weight.grad, reference_weight.grad, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(bias.grad, reference_bias.grad, atol=tolerance, rtol=tolerance)


@pytest.mark.L0
def test_layer_norm_rejects_multi_dimensional_normalized_shape():
    input = torch.empty(2, 3, 4, device="cuda", dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="one-dimensional"):
        layer_norm(input, (3, 4))
