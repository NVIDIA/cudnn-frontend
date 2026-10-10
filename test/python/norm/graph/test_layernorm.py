# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
from typing import Optional

import cudnn
import pytest
import torch

from norm_test_utils import (
    NormPlanDiscoveryUnsupportedError,
    assert_finite,
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
        dequantized = dequantize_block_scaled(quantized, block_scales, spec)
        assert_finite(
            quantized=quantized,
            block_scales=block_scales,
            quantized_reference=quantized_expected,
            scale_reference=scales_expected,
            dequantized=dequantized,
            reference=y_expected,
        )
        scale_match_rate = (block_scales.float() == scales_expected.float()).float().mean().item()
        quantized_match_rate = (quantized.float() == quantized_expected).float().mean().item()
        assert scale_match_rate > 0.999
        assert quantized_match_rate > 0.99
        torch.testing.assert_close(dequantized, y_expected, atol=0.05, rtol=0.25)

    if phase == cudnn.norm_forward_phase.TRAINING:
        torch.testing.assert_close(outputs.mean, mean_expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(outputs.inv_variance, inv_variance_expected, atol=1e-5, rtol=1e-5)
    else:
        assert outputs.mean is None
        assert outputs.inv_variance is None


_PHASES = [cudnn.norm_forward_phase.INFERENCE, cudnn.norm_forward_phase.TRAINING]


@pytest.mark.L0
@pytest.mark.parametrize("input_type", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("phase", _PHASES, ids=["inference", "training"])
def test_layernorm_forward_without_bias(phase, input_type, cudnn_handle):
    """``bias=None`` (#188, ``nn.LayerNorm(bias=False)``): the C++ node used to dereference the absent bias."""
    x_gpu = make_seeded_randn((2, 512, 1024), input_type, 2101, scale=2.0, shift=-0.5)
    scale_gpu = make_seeded_randn((1, 1, 1024), input_type, 2102, scale=1.5, shift=0.25)
    epsilon_cpu = torch.full((1, 1, 1), 1e-5, device="cpu", dtype=torch.float32)
    y_expected, mean_expected, inv_variance_expected = _manual_layernorm(x_gpu, scale_gpu, None, 1e-5)

    outputs = execute_layernorm_forward(phase=phase, x=x_gpu, scale=scale_gpu, bias=None, epsilon=epsilon_cpu, cudnn_handle=cudnn_handle)

    tolerance = 8e-3 if input_type == torch.bfloat16 else 1e-4
    torch.testing.assert_close(outputs.y, y_expected.to(input_type), atol=tolerance, rtol=tolerance)
    if phase == cudnn.norm_forward_phase.TRAINING:
        torch.testing.assert_close(outputs.mean, mean_expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(outputs.inv_variance, inv_variance_expected, atol=1e-5, rtol=1e-5)


def _layernorm_graph(cudnn_handle, x, *, scale_shape=None, bias_shape=None, phase=cudnn.norm_forward_phase.TRAINING, ada=False):
    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=cudnn_handle)
    X = graph.tensor(name="X", dim=list(x.shape), stride=list(x.stride()), data_type=x.dtype)

    def param(name, shape):
        return None if shape is None else graph.tensor(name=name, dim=list(shape), stride=list(torch.empty(shape).stride()), data_type=x.dtype)

    S, Bt = param("scale", scale_shape), param("bias", bias_shape)
    eps = graph.tensor(name="eps", dim=[1] * x.dim(), stride=[1] * x.dim(), is_pass_by_value=True, data_type=cudnn.data_type.FLOAT)
    norm = graph.adalayernorm if ada else graph.layernorm
    Y, mean, inv_var = norm(norm_forward_phase=phase, input=X, scale=S, bias=Bt, epsilon=eps)
    Y.set_output(True).set_data_type(x.dtype)
    for t in (mean, inv_var):
        if t is not None:
            t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    return graph, X, S, eps, Y, mean, inv_var


def _execute(graph, cudnn_handle, x, variant_pack, stats):
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    cudnn.set_stream(handle=cudnn_handle, stream=torch.cuda.current_stream().cuda_stream)
    out = {t: torch.full([int(d) for d in t.get_dim()], float("nan"), device="cuda") for t in stats if t is not None}
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute({**variant_pack, **out}, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    return out


@pytest.mark.L0
@pytest.mark.parametrize("phase", _PHASES, ids=["inference", "training"])
def test_layernorm_forward_without_scale_or_bias(phase, cudnn_handle):
    """No affine at all: ``{rows, H, 1, 1}`` states the normalization axes, so neither scale nor bias is needed."""
    rows, H = 64, 128
    x = make_seeded_randn((rows, H, 1, 1), torch.float32, 2103)
    graph, X, _, eps, Y, mean, inv_var = _layernorm_graph(cudnn_handle, x, phase=phase)
    y = torch.full_like(x, float("nan"))
    stats = _execute(graph, cudnn_handle, x, {X: x, eps: torch.full((1, 1, 1, 1), 1e-5), Y: y}, (mean, inv_var))
    y_expected, mean_expected, inv_variance_expected = _manual_layernorm(x.view(rows, H), torch.ones(H, device="cuda"), None, 1e-5)
    torch.testing.assert_close(y, y_expected.view(rows, H, 1, 1), atol=1e-5, rtol=1e-5)
    if phase == cudnn.norm_forward_phase.TRAINING:
        assert [int(d) for d in mean.get_dim()] == [int(d) for d in inv_var.get_dim()] == [rows, 1, 1, 1]
        torch.testing.assert_close(stats[mean].view(rows, 1), mean_expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(stats[inv_var].view(rows, 1), inv_variance_expected, atol=1e-5, rtol=1e-5)


@pytest.mark.L0
@pytest.mark.parametrize("phase", _PHASES, ids=["inference", "training"])
def test_layernorm_refuses_bias_without_scale(phase, cudnn_handle):
    """The backend rejects a bias without a scale (``hasBiasDesc() && !hasScaleDesc()``); the node says so at validate()."""
    x = make_seeded_randn((2, 8, 64), torch.float32, 2104)
    graph = _layernorm_graph(cudnn_handle, x, bias_shape=(1, 1, 64), phase=phase)[0]
    with pytest.raises(Exception, match="requires a scale"):
        graph.validate()


@pytest.mark.L0
@pytest.mark.parametrize("phase", _PHASES, ids=["inference", "training"])
def test_layernorm_without_scale_refuses_ambiguous_axes(phase, cudnn_handle):
    """``{B, S, H}`` with no affine: the backend normalizes H in inference but S and H for ``{B, 1, 1}`` stats, so refuse."""
    x = make_seeded_randn((2, 8, 64), torch.float32, 2105)
    graph = _layernorm_graph(cudnn_handle, x, phase=phase)[0]
    with pytest.raises(Exception, match="ambiguous"):
        graph.validate()


@pytest.mark.L0
@pytest.mark.parametrize("set_stat", ["mean", "inv_var"])
def test_layernorm_without_scale_one_stat_dims_state_the_axes(set_stat, cudnn_handle):
    """Training ``{B, S, H}`` with no affine: dims set on one stat are enough, and the other stat takes them."""
    B, S, H = 2, 8, 64
    x = make_seeded_randn((B, S, H), torch.float32, 2108)
    graph, X, _, eps, Y, mean, inv_var = _layernorm_graph(cudnn_handle, x)
    (mean if set_stat == "mean" else inv_var).set_dim([B, S, 1]).set_stride([S, 1, 1])
    graph.validate()
    # callers allocate the stats from these dims right after validate()
    assert [int(d) for d in mean.get_dim()] == [int(d) for d in inv_var.get_dim()] == [B, S, 1]
    y = torch.full_like(x, float("nan"))
    stats = _execute(graph, cudnn_handle, x, {X: x, eps: torch.full((1, 1, 1), 1e-5), Y: y}, (mean, inv_var))
    assert [int(d) for d in mean.get_dim()] == [int(d) for d in inv_var.get_dim()] == [B, S, 1]
    y_expected, mean_expected, inv_variance_expected = _manual_layernorm(x, torch.ones(H, device="cuda"), None, 1e-5)
    torch.testing.assert_close(y, y_expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(stats[mean], mean_expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(stats[inv_var], inv_variance_expected, atol=1e-5, rtol=1e-5)


@pytest.mark.L0
@pytest.mark.parametrize("scale_shape", [None, (1, 1, 64)], ids=["no_scale", "scale"])
def test_layernorm_refuses_stats_with_different_dims(scale_shape, cudnn_handle):
    """``mean`` and ``inv_var`` with different dims are refused at validate(), not at build_operation_graph()."""
    x = make_seeded_randn((2, 8, 64), torch.float32, 2109)
    graph, _, _, _, _, mean, inv_var = _layernorm_graph(cudnn_handle, x, scale_shape=scale_shape)
    mean.set_dim([2, 8, 1]).set_stride([8, 1, 1])
    inv_var.set_dim([2, 1, 1]).set_stride([1, 1, 1])
    with pytest.raises(Exception, match="MEAN and INV_VARIANCE dims differ"):
        graph.validate()


@pytest.mark.L0
@pytest.mark.parametrize("phase", _PHASES, ids=["inference", "training"])
def test_adalayernorm_forward_without_bias(phase, cudnn_handle):
    """``adalayernorm``'s binding has always defaulted ``bias`` to ``None``; the C++ node used to dereference it."""
    B, S, H = 2, 64, 256
    x = make_seeded_randn((B, S, H), torch.float32, 2106)
    scale = make_seeded_randn((B, 1, H), torch.float32, 2107, scale=1.5, shift=0.25)
    graph, X, Sc, eps, Y, mean, inv_var = _layernorm_graph(cudnn_handle, x, scale_shape=(B, 1, H), phase=phase, ada=True)
    y = torch.full_like(x, float("nan"))
    stats = _execute(graph, cudnn_handle, x, {X: x, Sc: scale, eps: torch.full((1, 1, 1), 1e-5), Y: y}, (mean, inv_var))
    y_expected, mean_expected, inv_variance_expected = _manual_layernorm(x, scale, None, 1e-5)
    torch.testing.assert_close(y, y_expected, atol=1e-4, rtol=1e-4)
    if phase == cudnn.norm_forward_phase.TRAINING:
        torch.testing.assert_close(stats[mean].view(B, S, 1), mean_expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(stats[inv_var].view(B, S, 1), inv_variance_expected, atol=1e-5, rtol=1e-5)
