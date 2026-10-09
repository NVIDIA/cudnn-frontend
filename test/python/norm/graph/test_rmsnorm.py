# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import math

import cudnn
import pytest
import torch
import torch.nn as nn

from norm_test_utils import (
    NormPlanDiscoveryUnsupportedError,
    NormSupportUnsupportedError,
    assert_finite,
    block_scale_quantize_reference,
    block_scale_quantize_skip_reason,
    dequantize_block_scaled,
    execute_rmsnorm_backward,
    execute_rmsnorm_block_scaled_forward,
    execute_rmsnorm_forward,
    make_bsh_block_scale_quantize_specs,
    make_seeded_randn,
    unpack_last_dim_fp4,
)
from test_utils import torch_fork_set_rng


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        inv_var = torch.rsqrt(norm_x.float() + self.eps)
        x_normed = x * inv_var.to(x.dtype)
        x_scaled = weight * x_normed
        if bias is not None:
            x_scaled += bias
        return x_scaled, inv_var


embedding_dim_options = [768, 1024, 1280, 1600]
input_type_options = [torch.float16, torch.bfloat16]
bias_options = [True, False]

all_options = [elem for elem in itertools.product(*[embedding_dim_options, input_type_options, bias_options])]

_LLM_FORWARD_CASES = (
    pytest.param(
        cudnn.norm_forward_phase.TRAINING,
        (2, 8192, 4096),
        (1, 1, 4096),
        torch.bfloat16,
        1e-5,
        2001,
        8e-3,
        marks=pytest.mark.L1,
        id="train_b2_s8192_h4096_bf16",
    ),
    pytest.param(
        cudnn.norm_forward_phase.INFERENCE,
        (2, 8192, 4096),
        (1, 1, 4096),
        torch.bfloat16,
        1e-5,
        2002,
        8e-3,
        marks=pytest.mark.L0,
        id="infer_b2_s8192_h4096_bf16",
    ),
)
_LLM_BACKWARD_CASES = (
    pytest.param(
        (2, 8192, 4096),
        (1, 1, 4096),
        torch.bfloat16,
        1e-5,
        2003,
        8e-3,
        8e-3,
        marks=pytest.mark.L0,
        id="backward_b2_s8192_h4096_bf16",
    ),
    pytest.param(
        (2, 8192, 4096),
        (1, 1, 4096),
        torch.float32,
        1e-5,
        2004,
        2e-4,
        5e-4,
        marks=pytest.mark.L1,
        id="backward_b2_s8192_h4096_fp32",
    ),
)
_LLM_BLOCK_SCALED_FORWARD_CASES = (
    pytest.param(
        "mxfp8",
        cudnn.norm_forward_phase.TRAINING,
        True,
        (2, 8192, 4096),
        (1, 1, 4096),
        1e-5,
        2101,
        marks=pytest.mark.L1,
        id="train_b2_s8192_h4096_mxfp8",
    ),
    pytest.param(
        "nvfp4",
        cudnn.norm_forward_phase.INFERENCE,
        False,
        (2, 8192, 4096),
        (1, 1, 4096),
        1e-5,
        2102,
        marks=pytest.mark.L1,
        id="infer_b2_s8192_h4096_nvfp4",
    ),
)

pytestmark = pytest.mark.skipif(cudnn.backend_version() < 8906, reason="RMSNorm requires cuDNN 8.9.6 or newer")


@pytest.fixture(params=all_options)
def param_extract(request):
    return request.param


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_rmsnorm(param_extract, cudnn_handle):

    embedding_dim, input_type, has_bias = param_extract

    batch_size, seq_size = 16, 128

    epsilon_value = 1e-3

    x_gpu = 2 * torch.randn(batch_size, seq_size, embedding_dim, requires_grad=True, device="cuda", dtype=input_type) - 1.25
    scale_gpu = 3 * torch.randn(1, 1, embedding_dim, requires_grad=True, device="cuda", dtype=input_type) - 2.75
    bias_gpu = torch.randn(1, 1, embedding_dim, requires_grad=True, device="cuda", dtype=input_type)
    epsilon_cpu = torch.full(
        (1, 1, 1),
        epsilon_value,
        requires_grad=False,
        device="cpu",
        dtype=torch.float32,
    )

    print("Running reference")

    model = RMSNorm(eps=epsilon_value, dim=-1).float()
    Y_expected, inv_var_expected = model(x_gpu, scale_gpu, bias_gpu if has_bias else None)

    print("Running cudnn graph")
    try:
        forward = execute_rmsnorm_forward(
            phase=cudnn.norm_forward_phase.TRAINING,
            x=x_gpu,
            scale=scale_gpu,
            bias=bias_gpu if has_bias else None,
            epsilon=epsilon_cpu,
            cudnn_handle=cudnn_handle,
        )
    except (NormPlanDiscoveryUnsupportedError, NormSupportUnsupportedError) as e:
        print(f"TEST WAIVED: unsupported graph. {e}")
        pytest.skip("TEST WAIVED: unsupported graph.")

    torch.testing.assert_close(Y_expected, forward.y, atol=0.03125, rtol=0.03125)
    torch.testing.assert_close(inv_var_expected, forward.inv_variance, atol=0.005, rtol=0.005)

    target = torch.randn_like(Y_expected)
    criterion = nn.MSELoss()
    loss = criterion(Y_expected, target)

    Y_expected.retain_grad()
    x_gpu.retain_grad()
    scale_gpu.retain_grad()
    bias_gpu.retain_grad()

    loss.backward()

    print("Running cudnn backward graph")
    backward = execute_rmsnorm_backward(
        x=x_gpu,
        scale=scale_gpu,
        grad=Y_expected.grad,
        inv_variance=forward.inv_variance,
        has_dbias=has_bias,
        cudnn_handle=cudnn_handle,
    )

    print("Comparing with reference")
    torch.testing.assert_close(x_gpu.grad, backward.dx, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(scale_gpu.grad, backward.dscale, atol=5e-4, rtol=5e-4)
    if has_bias:
        torch.testing.assert_close(bias_gpu.grad, backward.dbias, atol=5e-4, rtol=5e-4)
    print("Success!!")


def _manual_rmsnorm(x: torch.Tensor, scale: torch.Tensor, epsilon_value: float):
    x_float = x.float()
    inv_variance = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + epsilon_value)
    return x_float * inv_variance * scale.float(), inv_variance


@pytest.mark.parametrize("phase,input_shape,parameter_shape,input_type,epsilon_value,seed,data_tolerance", _LLM_FORWARD_CASES)
def test_rmsnorm_llm_forward(phase, input_shape, parameter_shape, input_type, epsilon_value, seed, data_tolerance, cudnn_handle):
    x_gpu = make_seeded_randn(input_shape, input_type, seed * 10, scale=2.0, shift=-0.5)
    scale_gpu = make_seeded_randn(parameter_shape, input_type, seed * 10 + 1, scale=1.5, shift=0.25)
    epsilon_cpu = torch.full((1, 1, 1), epsilon_value, device="cpu", dtype=torch.float32)
    y_expected, inv_variance_expected = _manual_rmsnorm(x_gpu, scale_gpu, epsilon_value)

    outputs = execute_rmsnorm_forward(
        phase=phase,
        x=x_gpu,
        scale=scale_gpu,
        bias=None,
        epsilon=epsilon_cpu,
        cudnn_handle=cudnn_handle,
    )

    torch.testing.assert_close(outputs.y, y_expected.to(input_type), atol=data_tolerance, rtol=data_tolerance)
    if phase == cudnn.norm_forward_phase.TRAINING:
        torch.testing.assert_close(outputs.inv_variance, inv_variance_expected, atol=1e-5, rtol=1e-5)
    else:
        assert outputs.inv_variance is None
    assert outputs.mean is None


@pytest.mark.parametrize("input_shape,parameter_shape,input_type,epsilon_value,seed,dx_tolerance,dscale_tolerance", _LLM_BACKWARD_CASES)
def test_rmsnorm_llm_backward(input_shape, parameter_shape, input_type, epsilon_value, seed, dx_tolerance, dscale_tolerance, cudnn_handle):
    x_gpu = make_seeded_randn(input_shape, input_type, seed * 10, scale=2.0, shift=-0.5)
    scale_gpu = make_seeded_randn(parameter_shape, input_type, seed * 10 + 1, scale=1.5, shift=0.25)
    grad_gpu = make_seeded_randn(input_shape, input_type, seed * 10 + 3, scale=0.5)

    x_ref = x_gpu.detach().float().requires_grad_(True)
    scale_ref = scale_gpu.detach().float().requires_grad_(True)
    y_ref, inv_variance = _manual_rmsnorm(x_ref, scale_ref, epsilon_value)
    dx_expected, dscale_expected = torch.autograd.grad(y_ref, (x_ref, scale_ref), grad_outputs=grad_gpu.float())

    outputs = execute_rmsnorm_backward(
        x=x_gpu,
        scale=scale_gpu,
        grad=grad_gpu,
        inv_variance=inv_variance.detach(),
        has_dbias=False,
        cudnn_handle=cudnn_handle,
    )

    torch.testing.assert_close(outputs.dx, dx_expected.to(input_type), atol=dx_tolerance, rtol=dx_tolerance)
    torch.testing.assert_close(outputs.dscale, dscale_expected.to(input_type), atol=dscale_tolerance, rtol=dscale_tolerance)
    assert outputs.dbias is None


@pytest.mark.parametrize("quantization,phase,include_column_output,input_shape,parameter_shape,epsilon_value,seed", _LLM_BLOCK_SCALED_FORWARD_CASES)
def test_rmsnorm_llm_block_scaled_output(quantization, phase, include_column_output, input_shape, parameter_shape, epsilon_value, seed, cudnn_handle):
    skip_reason = block_scale_quantize_skip_reason(quantization)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    x_gpu = make_seeded_randn(input_shape, torch.bfloat16, seed * 10, scale=2.0, shift=-0.5)
    scale_gpu = make_seeded_randn(parameter_shape, torch.bfloat16, seed * 10 + 1, scale=1.5, shift=0.25)
    epsilon_cpu = torch.full((1, 1, 1), epsilon_value, device="cpu", dtype=torch.float32)
    y_expected, inv_variance_expected = _manual_rmsnorm(x_gpu, scale_gpu, epsilon_value)

    quantize_specs = make_bsh_block_scale_quantize_specs(quantization, include_column_output=include_column_output)

    outputs = execute_rmsnorm_block_scaled_forward(
        phase=phase,
        x=x_gpu,
        scale=scale_gpu,
        bias=None,
        epsilon=epsilon_cpu,
        quantize_specs=quantize_specs,
        cudnn_handle=cudnn_handle,
    )

    for quantized_storage, block_scales, spec in zip(outputs.quantized, outputs.scales, quantize_specs):
        quantized = unpack_last_dim_fp4(quantized_storage, input_shape) if spec.output_data_type == cudnn.data_type.FP4_E2M1 else quantized_storage.float()
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
        if spec.scale_data_type == cudnn.data_type.FP8_E8M0:
            scale_matches = block_scales.float() == scales_expected.float()
        else:
            scale_matches = torch.isclose(block_scales.float(), scales_expected.float(), rtol=0.07, atol=0)
        scale_match_rate = scale_matches.float().mean().item()
        quantized_match_rate = (quantized == quantized_expected).float().mean().item()
        assert scale_match_rate > 0.999
        assert quantized_match_rate > 0.99
        if quantization == "mxfp8":
            torch.testing.assert_close(dequantized, y_expected, atol=0.05, rtol=0.25)
        else:
            error = (dequantized - y_expected).abs()
            # Exact quantized values and scales are checked above; this secondary
            # bound allows the expected E2M1 dequantization error.
            tolerance = 0.34 * y_expected.abs() + 0.05 * y_expected.abs().amax()
            assert (error <= tolerance).float().mean().item() > 0.999

    assert outputs.mean is None
    if phase == cudnn.norm_forward_phase.TRAINING:
        torch.testing.assert_close(outputs.inv_variance, inv_variance_expected, atol=1e-5, rtol=1e-5)
    else:
        assert outputs.inv_variance is None


def _run_norm_leaving_stats_dims_to_inference(cudnn_handle, x, scale, *, layernorm=False, phase=cudnn.norm_forward_phase.TRAINING):
    """Build a forward norm whose stats dims are left to the graph, allocate every stats buffer from ``get_dim()`` with a
    NaN guard band behind it, execute, and return ``(y, {name: (dims after validate(), dims after build, buffer, guard_written)})``."""
    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=cudnn_handle)
    cudnn.set_stream(handle=cudnn_handle, stream=torch.cuda.current_stream().cuda_stream)
    X = graph.tensor(name="X", dim=list(x.shape), stride=list(x.stride()), data_type=x.dtype)
    S = None if scale is None else graph.tensor(name="scale", dim=list(scale.shape), stride=list(scale.stride()), data_type=scale.dtype)
    eps = graph.tensor(name="eps", dim=[1] * x.dim(), stride=[1] * x.dim(), is_pass_by_value=True, data_type=cudnn.data_type.FLOAT)
    if layernorm:
        bias = torch.zeros_like(scale)
        Bt = graph.tensor(name="bias", dim=list(bias.shape), stride=list(bias.stride()), data_type=bias.dtype)
        Y, mean, inv_var = graph.layernorm(norm_forward_phase=phase, input=X, scale=S, bias=Bt, epsilon=eps)
        stats = {"mean": mean, "inv_var": inv_var}
    else:
        Y, inv_var = graph.rmsnorm(norm_forward_phase=phase, input=X, scale=S, epsilon=eps)
        stats = {"inv_var": inv_var}
    stats = {k: t for k, t in stats.items() if t is not None}
    Y.set_output(True).set_data_type(x.dtype)
    for t in stats.values():
        t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    graph.validate()
    validated_dims = {name: [int(d) for d in t.get_dim()] for name, t in stats.items()}
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    y = torch.full_like(x, float("nan"))
    variant_pack = {X: x, eps: torch.full((1,) * x.dim(), 1e-5), Y: y}
    if S is not None:
        variant_pack[S] = scale
    if layernorm:
        variant_pack[Bt] = bias
    buffers = {}
    for name, t in stats.items():
        dims = [int(d) for d in t.get_dim()]
        n = math.prod(dims)
        guarded = torch.full((n + x.numel(),), float("nan"), device="cuda")
        variant_pack[t] = guarded[:n].view(dims)
        buffers[name] = (dims, guarded[:n], guarded[n:])
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    return y, {name: (validated_dims[name], dims, buf, int((~guard.isnan()).sum())) for name, (dims, buf, guard) in buffers.items()}


@pytest.mark.L0
@pytest.mark.parametrize("layernorm", [False, True], ids=["rmsnorm", "layernorm"])
def test_norm_forward_inferred_stats_dims_cover_what_the_kernel_writes(layernorm, cudnn_handle):
    """``{B, S, H}`` with a ``{1, 1, H}`` scale: after ``validate()`` the graph used to report ``{B, 1, 1}`` stats while the
    kernel writes ``B * S`` rows -- an allocation sized from ``get_dim()`` there was overrun ``S``-fold.  The inferred dims
    now follow the C++ node (input dims, 1 wherever the scale is not)."""
    B, S, H = 2, 8, 64
    x = make_seeded_randn((B, S, H), torch.float32, 7)
    scale = torch.ones(1, 1, H, device="cuda")
    y, stats = _run_norm_leaving_stats_dims_to_inference(cudnn_handle, x, scale, layernorm=layernorm)
    centered = x - x.mean(-1, keepdim=True) if layernorm else x
    inv_var = torch.rsqrt(centered.square().mean(-1, keepdim=True) + 1e-5)
    for name, (validated_dims, dims, buf, guard_written) in stats.items():
        assert validated_dims == dims == [B, S, 1], name
        assert guard_written == 0, f"{name}: the kernel wrote {guard_written} elements past the inferred extent"
    torch.testing.assert_close(stats["inv_var"][2].view(B, S, 1), inv_var, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(y, centered * inv_var, atol=1e-5, rtol=1e-5)


@pytest.mark.L0
@pytest.mark.parametrize("phase", [cudnn.norm_forward_phase.INFERENCE, cudnn.norm_forward_phase.TRAINING], ids=["inference", "training"])
def test_rmsnorm_without_scale(phase, cudnn_handle):
    """``scale=None`` (#188): one non-unit axis after the first states the normalization axes, so no scale is needed."""
    rows, H = 64, 128
    x = make_seeded_randn((rows, H, 1, 1), torch.float32, 11)
    y, stats = _run_norm_leaving_stats_dims_to_inference(cudnn_handle, x, None, phase=phase)
    inv_var = torch.rsqrt(x.square().mean(1, keepdim=True) + 1e-5)
    torch.testing.assert_close(y, x * inv_var, atol=1e-5, rtol=1e-5)
    if phase == cudnn.norm_forward_phase.TRAINING:
        validated_dims, dims, buf, guard_written = stats["inv_var"]
        assert validated_dims == dims == [rows, 1, 1, 1] and guard_written == 0
        torch.testing.assert_close(buf.view(rows, 1, 1, 1), inv_var, atol=1e-5, rtol=1e-5)


@pytest.mark.L0
def test_rmsnorm_without_scale_refuses_ambiguous_axes(cudnn_handle):
    """``{B, S, H}`` with no scale: the backend's inference default (H only) and the stats default (S and H) disagree, so
    the node refuses rather than pick one."""
    x = make_seeded_randn((2, 8, 64), torch.float32, 13)
    with pytest.raises(Exception, match="ambiguous"):
        _run_norm_leaving_stats_dims_to_inference(cudnn_handle, x, None, phase=cudnn.norm_forward_phase.INFERENCE)
