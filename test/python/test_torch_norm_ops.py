# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the experimental PyTorch normalization ops."""

from concurrent.futures import ThreadPoolExecutor

import cudnn
import pytest
import torch

from cudnn.experimental.ops import layer_norm, rms_norm

SHAPES = [(128, 768), (32, 4096)]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _rms_norm_reference(input, weight, bias=None, eps=1e-5):
    normalized = input.float() * torch.rsqrt(input.float().square().mean(dim=-1, keepdim=True) + eps)
    output = normalized * weight.float()
    if bias is not None:
        output = output + bias.float()
    return output.to(input.dtype)


@pytest.mark.L0
@pytest.mark.skipif(cudnn.backend_version() < 80906, reason="RMSNorm requires cuDNN >= 8.9.6")
@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: "x".join(map(str, shape)))
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16", "fp32"])
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

    tolerance = 0.03125 if dtype != torch.float32 else 1e-5
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
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16", "fp32"])
@pytest.mark.parametrize(
    "has_weight,has_bias",
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["no_parameters", "weight", "bias", "weight_and_bias"],
)
def test_layer_norm_custom_op(shape, dtype, has_weight, has_bias):
    torch.manual_seed(42)
    input = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype, requires_grad=True) if has_weight else None
    bias = torch.randn(shape[-1], device="cuda", dtype=dtype, requires_grad=True) if has_bias else None

    actual = layer_norm(input, (shape[-1],), weight, bias)
    reference_input = input.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True) if weight is not None else None
    reference_bias = bias.detach().clone().requires_grad_(True) if bias is not None else None
    expected = torch.nn.functional.layer_norm(reference_input, (shape[-1],), reference_weight, reference_bias)

    tolerance = 1e-5 if dtype == torch.float32 else (0.03125 if dtype == torch.bfloat16 else 0.015625)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(input.grad, reference_input.grad, atol=tolerance, rtol=tolerance)
    if weight is not None:
        torch.testing.assert_close(weight.grad, reference_weight.grad, atol=tolerance, rtol=tolerance)
    if bias is not None:
        torch.testing.assert_close(bias.grad, reference_bias.grad, atol=tolerance, rtol=tolerance)


@pytest.mark.L0
def test_layer_norm_rejects_multi_dimensional_normalized_shape():
    input = torch.empty(2, 3, 4, device="cuda", dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="one-dimensional"):
        layer_norm(input, (3, 4))


@pytest.mark.L0
def test_layer_norm_supports_fp32_parameters_with_low_precision_input():
    input = torch.randn(4, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(64, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(64, device="cuda", dtype=torch.float32, requires_grad=True)
    actual = layer_norm(input, (64,), weight, bias)
    expected = torch.nn.functional.layer_norm(input, (64,), weight, bias)
    torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.015625)
    actual.sum().backward()
    assert weight.grad is not None and weight.grad.dtype == torch.float32
    assert bias.grad is not None and bias.grad.dtype == torch.float32


@pytest.mark.L0
def test_norm_public_apis_reject_cpu_inputs():
    input = torch.randn(2, 64)
    weight = torch.randn(64)
    with pytest.raises(ValueError, match="CUDA"):
        layer_norm(input, (64,), weight)
    with pytest.raises(ValueError, match="CUDA"):
        rms_norm(input, weight)


@pytest.mark.L0
@pytest.mark.parametrize("op_name", ["layer_norm", "rms_norm"])
def test_norm_public_apis_reject_zero_rows(op_name):
    input = torch.empty(0, 64, device="cuda", dtype=torch.float16)
    weight = torch.empty(64, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="at least one normalization row"):
        if op_name == "layer_norm":
            layer_norm(input, (64,), weight)
        else:
            rms_norm(input, weight)


@pytest.mark.L0
def test_norm_public_apis_reject_invalid_parameter_dtypes():
    input = torch.randn(2, 64, device="cuda", dtype=torch.float32)
    half_weight = torch.randn(64, device="cuda", dtype=torch.float16)
    with pytest.raises(TypeError, match="weight dtype"):
        layer_norm(input, (64,), half_weight)
    with pytest.raises(ValueError, match="same device and dtype"):
        rms_norm(input, half_weight)

    low_precision_input = input.half()
    float_bias = torch.randn(64, device="cuda", dtype=torch.float32)
    with pytest.raises(TypeError, match="weight and bias must have the same dtype"):
        layer_norm(low_precision_input, (64,), half_weight, float_bias)


@pytest.mark.L0
@pytest.mark.parametrize("shape", [(64,), (2, 4, 64)])
@pytest.mark.parametrize("op_name", ["layer_norm", "rms_norm"])
def test_norm_custom_ops_support_public_rank_contract(shape, op_name):
    input = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float32, requires_grad=True)
    if op_name == "layer_norm":
        actual = layer_norm(input, (shape[-1],), weight)
        expected = torch.nn.functional.layer_norm(input, (shape[-1],), weight)
    else:
        actual = rms_norm(input, weight)
        expected = _rms_norm_reference(input, weight)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.L0
@pytest.mark.parametrize("op_name", ["layer_norm", "rms_norm"])
def test_norm_custom_ops_accept_noncontiguous_input(op_name):
    input = torch.randn(2, 64, 3, device="cuda", dtype=torch.float32).transpose(1, 2).requires_grad_(True)
    weight = torch.randn(64, device="cuda", dtype=torch.float32, requires_grad=True)
    if op_name == "layer_norm":
        actual = layer_norm(input, (64,), weight)
        expected = torch.nn.functional.layer_norm(input, (64,), weight)
    else:
        actual = rms_norm(input, weight)
        expected = _rms_norm_reference(input, weight)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    actual.sum().backward()
    assert input.grad is not None


@pytest.mark.L0
def test_norm_stats_are_non_differentiable():
    x = torch.randn(4, 64, 1, 1, device="cuda", dtype=torch.float16, requires_grad=True)
    layer_scale = torch.randn(1, 64, 1, 1, device="cuda", dtype=torch.float32, requires_grad=True)
    layer_bias = torch.randn_like(layer_scale, requires_grad=True)
    _y, mean, inv_var = torch.ops.cudnn.layernorm(x, layer_scale, layer_bias, 1e-5)
    assert not mean.requires_grad
    assert not inv_var.requires_grad

    rms_scale = torch.randn(1, 64, 1, 1, device="cuda", dtype=torch.float16, requires_grad=True)
    _y, inv_var = torch.ops.cudnn.rmsnorm(x, rms_scale, 1e-5)
    assert not inv_var.requires_grad


@pytest.mark.L0
def test_norm_custom_ops_opcheck():
    x = torch.randn(4, 64, 1, 1, device="cuda", dtype=torch.float16, requires_grad=True)
    layer_scale = torch.randn(1, 64, 1, 1, device="cuda", dtype=torch.float32, requires_grad=True)
    layer_bias = torch.randn_like(layer_scale, requires_grad=True)
    torch.library.opcheck(torch.ops.cudnn.layernorm, (x, layer_scale, layer_bias, 1e-5))

    rms_scale = torch.randn(1, 64, 1, 1, device="cuda", dtype=torch.float16, requires_grad=True)
    torch.library.opcheck(torch.ops.cudnn.rmsnorm, (x, rms_scale, 1e-5))


@pytest.mark.L0
@pytest.mark.parametrize("op_name", ["layer_norm", "rms_norm"])
def test_norm_custom_ops_compile(op_name):
    input = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, device="cuda", dtype=torch.float16)
    eager = layer_norm(input, (64,), weight) if op_name == "layer_norm" else rms_norm(input, weight)
    compiled_fn = torch.compile(layer_norm if op_name == "layer_norm" else rms_norm, fullgraph=True)
    actual = compiled_fn(input, (64,), weight) if op_name == "layer_norm" else compiled_fn(input, weight)
    torch.testing.assert_close(actual, eager, atol=0.015625, rtol=0.015625)


@pytest.mark.L1
@pytest.mark.parametrize("op_name", ["layer_norm", "rms_norm"])
def test_norm_custom_ops_are_thread_and_stream_safe(op_name):
    def run(seed):
        torch.cuda.set_device(0)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            generator = torch.Generator(device="cuda").manual_seed(seed)
            input = torch.randn(32, 64, device="cuda", dtype=torch.float16, generator=generator)
            weight = torch.randn(64, device="cuda", dtype=torch.float16, generator=generator)
            output = layer_norm(input, (64,), weight) if op_name == "layer_norm" else rms_norm(input, weight)
            expected = torch.nn.functional.layer_norm(input, (64,), weight) if op_name == "layer_norm" else _rms_norm_reference(input, weight)
        stream.synchronize()
        torch.testing.assert_close(output, expected, atol=0.015625, rtol=0.015625)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run, (1, 2)))


@pytest.mark.L1
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
@pytest.mark.parametrize("op_name", ["layer_norm", "rms_norm"])
def test_norm_custom_ops_create_handles_on_input_device(op_name):
    outputs = []
    for device_index in (0, 1):
        device = torch.device("cuda", device_index)
        input = torch.randn(4, 64, device=device, dtype=torch.float16)
        weight = torch.randn(64, device=device, dtype=torch.float16)
        with torch.cuda.device(1 - device_index):
            output = layer_norm(input, (64,), weight) if op_name == "layer_norm" else rms_norm(input, weight)
        outputs.append(output)
    assert [output.device.index for output in outputs] == [0, 1]
