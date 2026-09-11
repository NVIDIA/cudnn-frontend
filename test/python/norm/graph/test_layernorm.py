# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
from typing import Optional

import cudnn
import pytest
import torch

from norm_test_utils import (
    NormPlanDiscoveryUnsupportedError,
    block_scale_quantize_reference,
    block_scale_quantize_skip_reason,
    dequantize_block_scaled,
    execute_layernorm_backward,
    execute_layernorm_block_scaled_forward,
    execute_layernorm_forward,
    make_bsh_block_scale_quantize_specs,
    make_seeded_randn,
)
from test_utils import torch_fork_set_rng

embedding_dim_options = [768, 1024, 1280, 1600]
input_type_options = [torch.bfloat16, torch.float16]

all_options = [elem for elem in itertools.product(*[embedding_dim_options, input_type_options])]

_LLM_FORWARD_CASES = (
    pytest.param(
        cudnn.norm_forward_phase.TRAINING,
        (2, 8192, 4096),
        (1, 1, 4096),
        torch.bfloat16,
        1e-5,
        1001,
        8e-3,
        marks=pytest.mark.L0,
        id="train_b2_s8192_h4096_bf16",
    ),
    pytest.param(
        cudnn.norm_forward_phase.INFERENCE,
        (2, 8192, 4096),
        (1, 1, 4096),
        torch.bfloat16,
        1e-5,
        1002,
        8e-3,
        marks=pytest.mark.L0,
        id="infer_b2_s8192_h4096_bf16",
    ),
    pytest.param(
        cudnn.norm_forward_phase.TRAINING,
        (2, 8192, 4096),
        (1, 1, 4096),
        torch.float32,
        1e-5,
        1004,
        1e-3,
        marks=pytest.mark.L1,
        id="train_b2_s8192_h4096_fp32",
    ),
)
_LLM_BACKWARD_CASES = (
    pytest.param(
        (2, 8192, 4096),
        (1, 1, 4096),
        torch.bfloat16,
        1e-5,
        1003,
        8e-3,
        marks=pytest.mark.L1,
        id="backward_b2_s8192_h4096_bf16",
    ),
)
_LLM_MXFP8_FORWARD_CASES = (
    pytest.param(
        cudnn.norm_forward_phase.TRAINING,
        True,
        (2, 8192, 4096),
        (1, 1, 4096),
        1e-5,
        1101,
        marks=pytest.mark.L1,
        id="train_b2_s8192_h4096_mxfp8",
    ),
    pytest.param(
        cudnn.norm_forward_phase.INFERENCE,
        False,
        (2, 8192, 4096),
        (1, 1, 4096),
        1e-5,
        1102,
        marks=pytest.mark.L1,
        id="infer_b2_s8192_h4096_mxfp8",
    ),
)

pytestmark = pytest.mark.skipif(cudnn.backend_version() < 8905, reason="LayerNorm requires cuDNN 8.9.5 or newer")


@pytest.fixture(params=all_options)
def param_extract(request):
    return request.param


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_layernorm(param_extract, cudnn_handle):

    embedding_dim, input_type = param_extract

    if input_type == torch.bfloat16:
        atol, rtol = 0.125, 0.125
    else:
        atol, rtol = 1e-2, 1e-2

    batch_size, seq_size = 16, 128

    epsilon_value = 1e-3

    x_gpu = 3 * torch.randn(batch_size, seq_size, embedding_dim, requires_grad=True, device="cuda", dtype=input_type) - 0.5
    scale_gpu = 5 * torch.randn(1, 1, embedding_dim, requires_grad=True, device="cuda", dtype=input_type) - 1
    bias_gpu = 7 * torch.randn(1, 1, embedding_dim, requires_grad=True, device="cuda", dtype=input_type) - 2
    epsilon_cpu = torch.full(
        (1, 1, 1),
        epsilon_value,
        requires_grad=False,
        device="cpu",
        dtype=torch.float32,
    )

    Y_expected = torch.nn.functional.layer_norm(
        x_gpu,
        [embedding_dim],
        weight=scale_gpu.reshape(-1),
        bias=bias_gpu.reshape(-1),
        eps=epsilon_value,
    )
    mean_expected = x_gpu.to(torch.float32).mean(dim=-1, keepdim=True)
    inv_var_expected = torch.rsqrt(torch.var(x_gpu.to(torch.float32), dim=-1, keepdim=True, correction=0) + epsilon_value)

    try:
        forward = execute_layernorm_forward(
            phase=cudnn.norm_forward_phase.TRAINING,
            x=x_gpu,
            scale=scale_gpu,
            bias=bias_gpu,
            epsilon=epsilon_cpu,
            cudnn_handle=cudnn_handle,
        )
    except NormPlanDiscoveryUnsupportedError as e:
        print(f"TEST WAIVED: unsupported graph. {e}")
        pytest.skip("TEST WAIVED: unsupported graph.")

    torch.testing.assert_close(Y_expected, forward.y, atol=atol, rtol=rtol)
    torch.testing.assert_close(mean_expected, forward.mean, atol=atol, rtol=rtol)
    torch.testing.assert_close(inv_var_expected, forward.inv_variance, atol=atol, rtol=rtol)

    target = torch.randn_like(Y_expected)
    criterion = torch.nn.MSELoss()
    loss = criterion(Y_expected, target)

    Y_expected.retain_grad()
    x_gpu.retain_grad()
    scale_gpu.retain_grad()
    bias_gpu.retain_grad()

    loss.backward()

    try:
        backward = execute_layernorm_backward(
            x=x_gpu,
            scale=scale_gpu,
            grad=Y_expected.grad,
            mean=forward.mean,
            inv_variance=forward.inv_variance,
            cudnn_handle=cudnn_handle,
        )
    except NormPlanDiscoveryUnsupportedError as e:
        print(f"TEST WAIVED: unsupported graph. {e}")
        pytest.skip("TEST WAIVED: unsupported graph.")

    torch.testing.assert_close(x_gpu.grad, backward.dx, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(scale_gpu.grad, backward.dscale, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(bias_gpu.grad, backward.dbias, atol=2e-4, rtol=2e-4)


def _manual_layernorm(x: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor], epsilon_value: float):
    x_float = x.float()
    mean = x_float.mean(dim=-1, keepdim=True)
    centered = x_float - mean
    inv_variance = torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + epsilon_value)
    y = centered * inv_variance * scale.float()
    if bias is not None:
        y = y + bias.float()
    return y, mean, inv_variance


@pytest.mark.parametrize("phase,input_shape,parameter_shape,input_type,epsilon_value,seed,data_tolerance", _LLM_FORWARD_CASES)
def test_layernorm_llm_forward(phase, input_shape, parameter_shape, input_type, epsilon_value, seed, data_tolerance, cudnn_handle):
    x_gpu = make_seeded_randn(input_shape, input_type, seed * 10, scale=2.0, shift=-0.5)
    scale_gpu = make_seeded_randn(parameter_shape, input_type, seed * 10 + 1, scale=1.5, shift=0.25)
    bias_gpu = make_seeded_randn(parameter_shape, input_type, seed * 10 + 2, shift=-0.125)
    epsilon_cpu = torch.full((1, 1, 1), epsilon_value, device="cpu", dtype=torch.float32)
    y_expected, mean_expected, inv_variance_expected = _manual_layernorm(x_gpu, scale_gpu, bias_gpu, epsilon_value)

    outputs = execute_layernorm_forward(
        phase=phase,
        x=x_gpu,
        scale=scale_gpu,
        bias=bias_gpu,
        epsilon=epsilon_cpu,
        cudnn_handle=cudnn_handle,
    )

    torch.testing.assert_close(outputs.y, y_expected.to(input_type), atol=data_tolerance, rtol=data_tolerance)
    if phase == cudnn.norm_forward_phase.TRAINING:
        torch.testing.assert_close(outputs.mean, mean_expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(outputs.inv_variance, inv_variance_expected, atol=1e-5, rtol=1e-5)
    else:
        assert outputs.mean is None
        assert outputs.inv_variance is None


@pytest.mark.parametrize("input_shape,parameter_shape,input_type,epsilon_value,seed,data_tolerance", _LLM_BACKWARD_CASES)
def test_layernorm_llm_backward(input_shape, parameter_shape, input_type, epsilon_value, seed, data_tolerance, cudnn_handle):
    x_gpu = make_seeded_randn(input_shape, input_type, seed * 10, scale=2.0, shift=-0.5)
    scale_gpu = make_seeded_randn(parameter_shape, input_type, seed * 10 + 1, scale=1.5, shift=0.25)
    bias_gpu = make_seeded_randn(parameter_shape, input_type, seed * 10 + 2, shift=-0.125)
    grad_gpu = make_seeded_randn(input_shape, input_type, seed * 10 + 3, scale=0.5)

    x_ref = x_gpu.detach().float().requires_grad_(True)
    scale_ref = scale_gpu.detach().float().requires_grad_(True)
    bias_ref = bias_gpu.detach().float().requires_grad_(True)
    y_ref, mean, inv_variance = _manual_layernorm(x_ref, scale_ref, bias_ref, epsilon_value)
    dx_expected, dscale_expected, dbias_expected = torch.autograd.grad(y_ref, (x_ref, scale_ref, bias_ref), grad_outputs=grad_gpu.float())

    outputs = execute_layernorm_backward(
        x=x_gpu,
        scale=scale_gpu,
        grad=grad_gpu,
        mean=mean.detach(),
        inv_variance=inv_variance.detach(),
        cudnn_handle=cudnn_handle,
    )

    torch.testing.assert_close(outputs.dx, dx_expected.to(input_type), atol=data_tolerance, rtol=data_tolerance)
    torch.testing.assert_close(outputs.dscale, dscale_expected.to(input_type), atol=data_tolerance, rtol=data_tolerance)
    torch.testing.assert_close(outputs.dbias, dbias_expected.to(input_type), atol=data_tolerance, rtol=data_tolerance)


@pytest.mark.parametrize("phase,include_column_output,input_shape,parameter_shape,epsilon_value,seed", _LLM_MXFP8_FORWARD_CASES)
def test_layernorm_llm_mxfp8_output(phase, include_column_output, input_shape, parameter_shape, epsilon_value, seed, cudnn_handle):
    skip_reason = block_scale_quantize_skip_reason("mxfp8")
    if skip_reason is not None:
        pytest.skip(skip_reason)

    x_gpu = make_seeded_randn(input_shape, torch.bfloat16, seed * 10, scale=2.0, shift=-0.5)
    scale_gpu = make_seeded_randn(parameter_shape, torch.bfloat16, seed * 10 + 1, scale=1.5, shift=0.25)
    bias_gpu = make_seeded_randn(parameter_shape, torch.bfloat16, seed * 10 + 2, shift=-0.125)
    epsilon_cpu = torch.full((1, 1, 1), epsilon_value, device="cpu", dtype=torch.float32)
    y_expected, mean_expected, inv_variance_expected = _manual_layernorm(x_gpu, scale_gpu, bias_gpu, epsilon_value)

    quantize_specs = make_bsh_block_scale_quantize_specs("mxfp8", include_column_output=include_column_output)

    outputs = execute_layernorm_block_scaled_forward(
        phase=phase,
        x=x_gpu,
        scale=scale_gpu,
        bias=bias_gpu,
        epsilon=epsilon_cpu,
        quantize_specs=quantize_specs,
        cudnn_handle=cudnn_handle,
    )

    for quantized, block_scales, spec in zip(outputs.quantized, outputs.scales, quantize_specs):
        quantized_expected, scales_expected = block_scale_quantize_reference(y_expected, spec)
        scale_match_rate = (block_scales.float() == scales_expected.float()).float().mean().item()
        quantized_match_rate = (quantized.float() == quantized_expected).float().mean().item()
        assert scale_match_rate > 0.999
        assert quantized_match_rate > 0.99
        torch.testing.assert_close(dequantize_block_scaled(quantized, block_scales, spec), y_expected, atol=0.05, rtol=0.25)

    if phase == cudnn.norm_forward_phase.TRAINING:
        torch.testing.assert_close(outputs.mean, mean_expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(outputs.inv_variance, inv_variance_expected, atol=1e-5, rtol=1e-5)
    else:
        assert outputs.mean is None
        assert outputs.inv_variance is None
