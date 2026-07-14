# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from fe_api.moe_ep.moe_ep_reference import (
    BlockScaledTensor,
    MoeEpReference,
    MoeFormat,
    quantize_blockwise,
)

pytestmark = pytest.mark.L0


def _output_as_float(output):
    if isinstance(output, torch.Tensor):
        return output.float()
    return output.dequantize()


def _naive_reference(
    activation,
    fc1_weight,
    fc2_weight,
    topk_idx,
    topk_weights,
    *,
    apply_topk_in_fc1,
    clamp=None,
    combine_format=MoeFormat.BF16,
):
    token_count, top_k = topk_idx.shape
    hidden_size = activation.shape[1]
    intermediate_size = fc2_weight.shape[1]
    combine = torch.zeros(token_count, top_k, hidden_size, dtype=torch.float32)
    for token in range(token_count):
        for slot in range(top_k):
            expert = int(topk_idx[token, slot])
            if expert == -1:
                continue
            gate_up = activation[token].float() @ fc1_weight[expert].float()
            gate, up = gate_up.split(intermediate_size)
            if clamp is not None:
                gate = gate.clamp(max=clamp)
                up = up.clamp(-clamp, clamp)
            intermediate = F.silu(gate) * up
            route_weight = topk_weights[token, slot].float()
            if apply_topk_in_fc1:
                intermediate = intermediate * route_weight
            result = intermediate @ fc2_weight[expert].float()
            if not apply_topk_in_fc1:
                result = result * route_weight
            if combine_format is MoeFormat.BF16:
                result = result.to(torch.bfloat16).float()
            else:
                result = quantize_blockwise(result, combine_format).dequantize()
            combine[token, slot] = result
    return combine.sum(dim=1).to(torch.bfloat16)


@pytest.mark.parametrize(
    "format,expected_data_shape,expected_scale_shape,scale_dtype",
    [
        (MoeFormat.MXFP8, (3, 64), (3, 2), torch.float8_e8m0fnu),
        (MoeFormat.NVFP4, (3, 32), (3, 4), torch.float8_e4m3fn),
    ],
)
def test_block_scaled_round_trip(format, expected_data_shape, expected_scale_shape, scale_dtype):
    values = torch.linspace(-4.0, 4.0, 3 * 64).reshape(3, 64)
    quantized = quantize_blockwise(values, format)

    assert isinstance(quantized, BlockScaledTensor)
    assert quantized.format is format
    assert quantized.logical_shape == (3, 64)
    assert tuple(quantized.data.shape) == expected_data_shape
    assert tuple(quantized.scale.shape) == expected_scale_shape
    assert quantized.scale.dtype == scale_dtype
    assert quantized.dequantize().shape == values.shape
    assert torch.isfinite(quantized.dequantize()).all()


def test_quantization_along_weight_reduction_axis():
    values = torch.randn(2, 64, 48)
    quantized = quantize_blockwise(values, MoeFormat.NVFP4, axis=1)

    assert quantized.logical_shape == (2, 64, 48)
    assert quantized.data.shape == (2, 32, 48)
    assert quantized.scale.shape == (2, 4, 48)
    assert quantized.dequantize().shape == values.shape


@pytest.mark.parametrize("apply_topk_in_fc1", [False, True])
def test_single_rank_bf16_matches_naive(apply_topk_in_fc1):
    torch.manual_seed(7)
    experts, tokens, hidden, intermediate, top_k = 4, 5, 32, 24, 2
    activation = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    fc1_weight = torch.randn(experts, hidden, 2 * intermediate, dtype=torch.bfloat16) / 8
    fc2_weight = torch.randn(experts, intermediate, hidden, dtype=torch.bfloat16) / 8
    topk_idx = torch.tensor([[0, 3], [2, 1], [1, -1], [3, 0], [2, 0]], dtype=torch.int64)
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.2, 0.8], [1.0, 0.0], [0.55, 0.45], [0.6, 0.4]],
        dtype=torch.float32,
    )

    op = MoeEpReference(
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=top_k,
        combine_format="bf16",
        output_format="bf16",
        apply_topk_in_fc1=apply_topk_in_fc1,
        gate_up_clamp=2.5,
    )
    actual = op(activation, fc1_weight, fc2_weight, topk_idx, topk_weights)
    expected = _naive_reference(
        activation,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights,
        apply_topk_in_fc1=apply_topk_in_fc1,
        clamp=2.5,
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_generate_c_single_rank_matches_naive():
    """fc1_c is the pre-clamp, unweighted gate+up accumulator, grouped by expert."""

    torch.manual_seed(37)
    experts, tokens, hidden, intermediate, top_k = 4, 5, 32, 24, 2
    activation = torch.randn(tokens, hidden)
    fc1_weight = torch.randn(experts, hidden, 2 * intermediate) / 8
    fc2_weight = torch.randn(experts, intermediate, hidden) / 8
    topk_idx = torch.tensor([[0, 3], [2, 1], [1, -1], [3, 0], [2, 0]], dtype=torch.int64)
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.2, 0.8], [1.0, 0.0], [0.55, 0.45], [0.6, 0.4]],
        dtype=torch.float32,
    )

    op = MoeEpReference(
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=top_k,
        gate_up_clamp=2.5,
        generate_c=True,
    )
    output, fc1_c, route_metadata = op(activation, fc1_weight, fc2_weight, topk_idx, topk_weights)

    expected_rows = []
    expected_metadata = []
    for expert in range(experts):
        for token in range(tokens):
            for slot in range(top_k):
                if int(topk_idx[token, slot]) == expert:
                    expected_rows.append(activation[token] @ fc1_weight[expert])
                    expected_metadata.append([expert, 0, token, slot])
    expected_fc1_c = torch.stack(expected_rows).to(torch.bfloat16)

    assert fc1_c.dtype == torch.bfloat16
    assert fc1_c.shape == (int((topk_idx != -1).sum()), 2 * intermediate)
    torch.testing.assert_close(fc1_c, expected_fc1_c, atol=0, rtol=0)

    assert route_metadata.dtype == torch.int32
    assert route_metadata.tolist() == expected_metadata

    expected_output = _naive_reference(
        activation,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights,
        apply_topk_in_fc1=True,
        clamp=2.5,
    )
    torch.testing.assert_close(output, expected_output, atol=0, rtol=0)


def _autograd_replica_grads(
    activation,
    fc1_weight,
    fc2_weight,
    topk_idx,
    topk_weights,
    grad_output,
    *,
    apply_topk_in_fc1=True,
    clamp=None,
):
    """Autograd gradients of a per-route replica of the reference forward.

    The FC1 accumulator is rounded to bf16 with a straight-through gradient,
    matching a backward pass that recomputes SwiGLU from the bf16 fc1_c stash.
    Combine/output round-trips are omitted (straight-through), as in
    ``MoeEpReference.backward``.
    """

    a = activation.detach().clone().float().requires_grad_()
    w1 = fc1_weight.detach().clone().float().requires_grad_()
    w2 = fc2_weight.detach().clone().float().requires_grad_()
    tw = topk_weights.detach().clone().float().requires_grad_()
    token_count, top_k = topk_idx.shape
    intermediate = w2.shape[1]
    rows = []
    for token in range(token_count):
        acc = torch.zeros(a.shape[1], dtype=torch.float32)
        for slot in range(top_k):
            expert = int(topk_idx[token, slot])
            if expert == -1:
                continue
            c = a[token] @ w1[expert]
            c = c + (c.to(torch.bfloat16).float() - c).detach()
            gate, up = c.split(intermediate)
            if clamp is not None:
                gate = gate.clamp(max=clamp)
                up = up.clamp(-clamp, clamp)
            h = F.silu(gate) * up
            weight = tw[token, slot]
            if apply_topk_in_fc1:
                h = h * weight
            y = h @ w2[expert]
            if not apply_topk_in_fc1:
                y = y * weight
            acc = acc + y
        rows.append(acc)
    torch.stack(rows).backward(grad_output.float())
    return a.grad, w1.grad, w2.grad, tw.grad


@pytest.mark.parametrize("apply_topk_in_fc1", [False, True])
def test_backward_single_rank_matches_autograd(apply_topk_in_fc1):
    torch.manual_seed(43)
    experts, tokens, hidden, intermediate, top_k = 4, 5, 32, 24, 2
    activation = torch.randn(tokens, hidden)
    fc1_weight = torch.randn(experts, hidden, 2 * intermediate) / 8
    fc2_weight = torch.randn(experts, intermediate, hidden) / 8
    topk_idx = torch.tensor([[0, 3], [2, 1], [1, -1], [3, 0], [2, 0]], dtype=torch.int64)
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.2, 0.8], [1.0, 0.0], [0.55, 0.45], [0.6, 0.4]],
        dtype=torch.float32,
    )
    grad_output = torch.randn(tokens, hidden)

    op = MoeEpReference(
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=top_k,
        apply_topk_in_fc1=apply_topk_in_fc1,
        gate_up_clamp=2.5,
        generate_c=True,
    )
    _, fc1_c, route_metadata = op(activation, fc1_weight, fc2_weight, topk_idx, topk_weights)
    grad_activation, grad_fc1, grad_fc2, grad_topk = op.backward(
        grad_output,
        activation,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights,
        fc1_c,
        route_metadata,
    )
    expected_ga, expected_g1, expected_g2, expected_gw = _autograd_replica_grads(
        activation,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights,
        grad_output,
        apply_topk_in_fc1=apply_topk_in_fc1,
        clamp=2.5,
    )

    torch.testing.assert_close(grad_activation, expected_ga, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(grad_fc1, expected_g1, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(grad_fc2, expected_g2, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(grad_topk, expected_gw, rtol=1e-4, atol=1e-6)
    assert grad_topk[2, 1].item() == 0.0  # -1 slot receives no gradient


@pytest.mark.parametrize("input_format", [MoeFormat.MXFP8, MoeFormat.NVFP4])
def test_backward_quantized_inputs_matches_autograd(input_format):
    """Backward with block-scaled inputs and quantized combine/output formats.

    Gradients are straight-through with respect to the dequantized values, so
    the expected gradients come from the autograd replica run on the decoded
    tensors; the combine/output encodes must not perturb backward.
    """

    torch.manual_seed(47)
    experts, tokens, hidden, intermediate, top_k = 2, 4, 32, 16, 2
    device = torch.device("cpu")
    activation, fc1_weight, fc2_weight = _block_scaled_inputs(input_format, experts, tokens, hidden, intermediate, device)
    topk_idx = torch.tensor([[0, 1], [1, 0], [0, -1], [1, 0]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [1.0, 0.0], [0.55, 0.45]])
    grad_output = torch.randn(tokens, hidden)

    op = MoeEpReference(
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=top_k,
        output_format=input_format,
        combine_format=input_format,
        generate_c=True,
    )
    _, fc1_c, route_metadata = op(activation, fc1_weight, fc2_weight, topk_idx, topk_weights)
    grad_activation, grad_fc1, grad_fc2, grad_topk = op.backward(
        grad_output,
        activation,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights,
        fc1_c,
        route_metadata,
    )
    expected_ga, expected_g1, expected_g2, expected_gw = _autograd_replica_grads(
        activation.dequantize(),
        fc1_weight.dequantize(),
        fc2_weight.dequantize(),
        topk_idx,
        topk_weights,
        grad_output,
    )

    torch.testing.assert_close(grad_activation, expected_ga, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(grad_fc1, expected_g1, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(grad_fc2, expected_g2, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(grad_topk, expected_gw, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("combine_format", [MoeFormat.MXFP8, MoeFormat.NVFP4])
def test_quantized_combine_round_trip(combine_format):
    torch.manual_seed(13)
    experts, tokens, hidden, intermediate = 2, 4, 32, 16
    activation = torch.randn(tokens, hidden)
    fc1_weight = torch.randn(experts, hidden, 2 * intermediate) / 8
    fc2_weight = torch.randn(experts, intermediate, hidden) / 8
    topk_idx = torch.tensor([[0, 1], [1, 0], [0, -1], [1, 0]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [1.0, 0.0], [0.55, 0.45]])
    op = MoeEpReference(
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=2,
        combine_format=combine_format,
    )

    actual = op(activation, fc1_weight, fc2_weight, topk_idx, topk_weights)
    expected = _naive_reference(
        activation,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights,
        apply_topk_in_fc1=True,
        combine_format=combine_format,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_quantized_inputs_and_weights_use_logical_shapes():
    torch.manual_seed(19)
    experts, tokens, hidden, intermediate = 2, 3, 32, 32
    activation = torch.randn(tokens, hidden)
    fc1_weight = torch.randn(experts, hidden, 2 * intermediate) / 8
    fc2_weight = torch.randn(experts, intermediate, hidden) / 8
    q_activation = quantize_blockwise(activation, "mxfp8", axis=1)
    q_fc1 = quantize_blockwise(fc1_weight, "mxfp8", axis=1)
    q_fc2 = quantize_blockwise(fc2_weight, "nvfp4", axis=1)
    topk_idx = torch.tensor([[0], [1], [0]], dtype=torch.int64)
    topk_weights = torch.ones(tokens, 1)
    op = MoeEpReference(
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=1,
    )

    actual = op(q_activation, q_fc1, q_fc2, topk_idx, topk_weights)
    expected = _naive_reference(
        q_activation.dequantize(),
        q_fc1.dequantize(),
        q_fc2.dequantize(),
        topk_idx,
        topk_weights,
        apply_topk_in_fc1=True,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("output_format", ["bf16", "mxfp8", "nvfp4"])
@pytest.mark.xfail(
    strict=True,
    reason="MoeEp.__call__ only allocates output; the device implementation is not wired yet",
)
def test_moe_ep_api_matches_reference(output_format):
    """Exercise the public API and reference through the same tensor contract."""

    from cudnn import MoeEp

    torch.manual_seed(23)
    device = torch.device("cuda")
    experts, tokens, hidden, intermediate, top_k = 2, 4, 32, 16, 2
    activation = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    fc1_weight = (
        torch.randn(
            experts,
            hidden,
            2 * intermediate,
            dtype=torch.bfloat16,
            device=device,
        )
        / 8
    )
    fc2_weight = (
        torch.randn(
            experts,
            intermediate,
            hidden,
            dtype=torch.bfloat16,
            device=device,
        )
        / 8
    )
    topk_idx = torch.tensor(
        [[0, 1], [1, 0], [0, -1], [1, 0]],
        dtype=torch.int64,
        device=device,
    )
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [1.0, 0.0], [0.55, 0.45]],
        dtype=torch.float32,
        device=device,
    )

    kwargs = dict(
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=top_k,
        output_format=output_format,
        combine_format="bf16",
    )
    api = MoeEp(**kwargs)
    reference = MoeEpReference(**kwargs)

    actual = api(activation, fc1_weight, fc2_weight, topk_idx, topk_weights)
    expected = reference(activation, fc1_weight, fc2_weight, topk_idx, topk_weights)

    actual_shape = tuple(actual.shape) if isinstance(actual, torch.Tensor) else actual.logical_shape
    assert actual_shape == (tokens, hidden)
    torch.testing.assert_close(
        _output_as_float(actual),
        _output_as_float(expected),
        atol=0,
        rtol=0,
    )


def _block_scaled_inputs(input_format, experts, tokens, hidden, intermediate, device):
    """Quantized inputs whose scale factors ride inside each BlockScaledTensor.

    Everything is block-scaled along the GEMM reduction axis: activation along
    hidden, fc1_weight along hidden, fc2_weight along intermediate.
    """

    activation = quantize_blockwise(
        torch.randn(tokens, hidden, device=device),
        input_format,
        axis=1,
    )
    fc1_weight = quantize_blockwise(
        torch.randn(experts, hidden, 2 * intermediate, device=device) / 8,
        input_format,
        axis=1,
    )
    fc2_weight = quantize_blockwise(
        torch.randn(experts, intermediate, hidden, device=device) / 8,
        input_format,
        axis=1,
    )
    return activation, fc1_weight, fc2_weight


@pytest.mark.parametrize("input_format", [MoeFormat.MXFP8, MoeFormat.NVFP4])
def test_moe_ep_api_accepts_block_scaled_inputs(input_format):
    """The API takes data+scale bundles where the kernel takes separate sf args."""

    from cudnn import MoeEp

    torch.manual_seed(29)
    device = torch.device("cuda")
    experts, tokens, hidden, intermediate = 2, 4, 32, 16
    activation, fc1_weight, fc2_weight = _block_scaled_inputs(input_format, experts, tokens, hidden, intermediate, device)

    api = MoeEp(num_experts=experts, hidden_size=hidden, intermediate_size=intermediate, top_k=1)
    output = api(
        activation,
        fc1_weight,
        fc2_weight,
        torch.tensor([[0], [1], [0], [1]], dtype=torch.int64, device=device),
        torch.ones(tokens, 1, device=device),
    )

    assert isinstance(output, torch.Tensor)
    assert output.shape == (tokens, hidden)
    assert output.dtype == torch.bfloat16


def test_moe_ep_api_generate_c_allocates_fc1_c():
    """generate_c=True returns (output, fc1_c) sized by this rank's valid routes."""

    from cudnn import MoeEp

    device = torch.device("cuda")
    experts, tokens, hidden, intermediate = 2, 4, 32, 16
    api = MoeEp(
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=2,
        generate_c=True,
    )
    topk_idx = torch.tensor([[0, 1], [1, 0], [0, -1], [1, 0]], dtype=torch.int64, device=device)
    output, fc1_c, route_metadata = api(
        torch.empty(tokens, hidden, dtype=torch.bfloat16, device=device),
        torch.empty(experts, hidden, 2 * intermediate, dtype=torch.bfloat16, device=device),
        torch.empty(experts, intermediate, hidden, dtype=torch.bfloat16, device=device),
        topk_idx,
        torch.ones(tokens, 2, dtype=torch.float32, device=device),
    )

    valid_routes = int((topk_idx != -1).sum())
    assert output.shape == (tokens, hidden)
    assert fc1_c.dtype == torch.bfloat16
    assert fc1_c.shape == (valid_routes, 2 * intermediate)
    assert fc1_c.device.type == "cuda"
    assert route_metadata.dtype == torch.int32
    assert route_metadata.shape == (valid_routes, 4)
    assert route_metadata.device.type == "cuda"


@pytest.mark.xfail(
    strict=True,
    reason="MoeEp.__call__ only allocates output; the device implementation is not wired yet",
)
def test_moe_ep_api_generate_c_matches_reference():
    from cudnn import MoeEp

    torch.manual_seed(41)
    device = torch.device("cuda")
    experts, tokens, hidden, intermediate, top_k = 2, 4, 32, 16, 2
    activation = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    fc1_weight = torch.randn(experts, hidden, 2 * intermediate, dtype=torch.bfloat16, device=device) / 8
    fc2_weight = torch.randn(experts, intermediate, hidden, dtype=torch.bfloat16, device=device) / 8
    topk_idx = torch.tensor([[0, 1], [1, 0], [0, -1], [1, 0]], dtype=torch.int64, device=device)
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [1.0, 0.0], [0.55, 0.45]],
        dtype=torch.float32,
        device=device,
    )

    kwargs = dict(
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=top_k,
        generate_c=True,
    )
    actual, actual_fc1_c, actual_metadata = MoeEp(**kwargs)(activation, fc1_weight, fc2_weight, topk_idx, topk_weights)
    expected, expected_fc1_c, expected_metadata = MoeEpReference(**kwargs)(activation, fc1_weight, fc2_weight, topk_idx, topk_weights)

    torch.testing.assert_close(_output_as_float(actual), _output_as_float(expected), atol=0, rtol=0)
    torch.testing.assert_close(actual_fc1_c, expected_fc1_c, atol=0, rtol=0)
    torch.testing.assert_close(actual_metadata, expected_metadata, atol=0, rtol=0)


def test_moe_ep_api_backward_allocates_grads():
    """MoeEp.backward returns the four gradient allocations of the contract."""

    from cudnn import MoeEp

    torch.manual_seed(53)
    device = torch.device("cuda")
    experts, tokens, hidden, intermediate, top_k = 2, 4, 32, 16, 2
    api = MoeEp(
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=top_k,
        generate_c=True,
    )
    activation = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    fc1_weight = torch.randn(experts, hidden, 2 * intermediate, dtype=torch.bfloat16, device=device)
    fc2_weight = torch.randn(experts, intermediate, hidden, dtype=torch.bfloat16, device=device)
    topk_idx = torch.tensor([[0, 1], [1, 0], [0, -1], [1, 0]], dtype=torch.int64, device=device)
    topk_weights = torch.ones(tokens, top_k, device=device)

    _, fc1_c, route_metadata = api(activation, fc1_weight, fc2_weight, topk_idx, topk_weights)
    grads = api.backward(
        torch.randn(tokens, hidden, device=device),
        activation,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights,
        fc1_c,
        route_metadata,
    )

    expected_shapes = [
        (tokens, hidden),
        (experts, hidden, 2 * intermediate),
        (experts, intermediate, hidden),
        (tokens, top_k),
    ]
    assert [tuple(grad.shape) for grad in grads] == expected_shapes
    assert all(grad.dtype == torch.float32 for grad in grads)
    assert all(grad.device.type == "cuda" for grad in grads)


@pytest.mark.xfail(
    strict=True,
    reason="MoeEp.backward only allocates gradients; the device implementation is not wired yet",
)
def test_moe_ep_api_backward_matches_reference():
    from cudnn import MoeEp

    torch.manual_seed(59)
    device = torch.device("cuda")
    experts, tokens, hidden, intermediate, top_k = 2, 4, 32, 16, 2
    activation = torch.randn(tokens, hidden, device=device)
    fc1_weight = torch.randn(experts, hidden, 2 * intermediate, device=device) / 8
    fc2_weight = torch.randn(experts, intermediate, hidden, device=device) / 8
    topk_idx = torch.tensor([[0, 1], [1, 0], [0, -1], [1, 0]], dtype=torch.int64, device=device)
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [1.0, 0.0], [0.55, 0.45]],
        dtype=torch.float32,
        device=device,
    )
    grad_output = torch.randn(tokens, hidden, device=device)

    kwargs = dict(
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=top_k,
        generate_c=True,
    )
    api = MoeEp(**kwargs)
    reference = MoeEpReference(**kwargs)
    forward_args = (activation, fc1_weight, fc2_weight, topk_idx, topk_weights)
    _, ref_fc1_c, ref_metadata = reference(*forward_args)
    _, api_fc1_c, api_metadata = api(*forward_args)

    actual = api.backward(grad_output, *forward_args, api_fc1_c, api_metadata)
    expected = reference.backward(grad_output, *forward_args, ref_fc1_c, ref_metadata)
    for actual_grad, expected_grad in zip(actual, expected):
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)


@pytest.mark.parametrize("input_format", ["mxfp8", "nvfp4"])
@pytest.mark.xfail(
    strict=True,
    reason="MoeEp.__call__ only allocates output; the device implementation is not wired yet",
)
def test_moe_ep_api_quantized_inputs_match_reference(input_format):
    """Quantized activations/weights carry their scale factors through both paths."""

    from cudnn import MoeEp

    torch.manual_seed(31)
    device = torch.device("cuda")
    experts, tokens, hidden, intermediate, top_k = 2, 4, 32, 16, 2
    activation, fc1_weight, fc2_weight = _block_scaled_inputs(input_format, experts, tokens, hidden, intermediate, device)
    topk_idx = torch.tensor([[0, 1], [1, 0], [0, -1], [1, 0]], dtype=torch.int64, device=device)
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [1.0, 0.0], [0.55, 0.45]],
        dtype=torch.float32,
        device=device,
    )

    kwargs = dict(num_experts=experts, hidden_size=hidden, intermediate_size=intermediate, top_k=top_k)
    actual = MoeEp(**kwargs)(activation, fc1_weight, fc2_weight, topk_idx, topk_weights)
    expected = MoeEpReference(**kwargs)(activation, fc1_weight, fc2_weight, topk_idx, topk_weights)

    torch.testing.assert_close(
        _output_as_float(actual),
        _output_as_float(expected),
        atol=0,
        rtol=0,
    )


def _distributed_worker(rank, world_size, init_file):
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        experts, hidden, intermediate, top_k = 2 * world_size, 32, 16, 2
        torch.manual_seed(1234)
        global_fc1 = torch.randn(experts, hidden, 2 * intermediate) / 8
        global_fc2 = torch.randn(experts, intermediate, hidden) / 8
        experts_per_rank = experts // world_size
        begin = rank * experts_per_rank
        end = begin + experts_per_rank

        generator = torch.Generator().manual_seed(2000 + rank)
        token_count = 3 + rank
        activation = torch.randn(token_count, hidden, generator=generator)
        topk_idx = torch.tensor(
            [[(rank + token) % experts, (rank + token + 2) % experts] for token in range(token_count)],
            dtype=torch.int64,
        )
        topk_weights = torch.tensor([[0.625, 0.375]]).expand(token_count, -1).contiguous()

        op = MoeEpReference(
            num_experts=experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            top_k=top_k,
            ep_group=dist.group.WORLD,
            combine_format="bf16",
            output_format="bf16",
            generate_c=True,
        )
        actual, fc1_c, route_metadata = op(
            activation.to(device),
            global_fc1[begin:end].contiguous().to(device),
            global_fc2[begin:end].contiguous().to(device),
            topk_idx.to(device),
            topk_weights.to(device),
        )
        assert actual.device.type == "cuda"
        expected = _naive_reference(
            activation,
            global_fc1,
            global_fc2,
            topk_idx,
            topk_weights,
            apply_topk_in_fc1=True,
        )
        # GPU and CPU float32 GEMMs accumulate in different orders, so the
        # cross-device comparison uses bf16-level tolerances instead of exact.
        torch.testing.assert_close(actual.cpu(), expected)

        # fc1_c contract: rows grouped by local expert, ordered by source rank,
        # then the source's token-major route order.  Every rank can rebuild all
        # source ranks' inputs because they derive from deterministic seeds.
        expected_rows = []
        expected_metadata = []
        for local_expert in range(experts_per_rank):
            global_expert = begin + local_expert
            for src_rank in range(world_size):
                src_generator = torch.Generator().manual_seed(2000 + src_rank)
                src_token_count = 3 + src_rank
                src_activation = torch.randn(src_token_count, hidden, generator=src_generator)
                for token in range(src_token_count):
                    for slot, slot_expert in enumerate(
                        (
                            (src_rank + token) % experts,
                            (src_rank + token + 2) % experts,
                        )
                    ):
                        if slot_expert == global_expert:
                            expected_rows.append(src_activation[token] @ global_fc1[global_expert])
                            expected_metadata.append([local_expert, src_rank, token, slot])
        if expected_rows:
            expected_fc1_c = torch.stack(expected_rows).to(torch.bfloat16)
        else:
            expected_fc1_c = torch.empty((0, 2 * intermediate), dtype=torch.bfloat16)
        assert fc1_c.shape == expected_fc1_c.shape
        assert route_metadata.dtype == torch.int32
        assert route_metadata.cpu().tolist() == expected_metadata

        # Backward: gradients re-dispatch along the same routes.  Expected
        # values come from autograd replicas over every rank's reconstructed
        # inputs: token/router grads are local to this rank's tokens, while
        # weight grads accumulate contributions from all source ranks.
        grad_output = torch.randn(token_count, hidden, generator=torch.Generator().manual_seed(3000 + rank))
        grad_activation, grad_fc1, grad_fc2, grad_topk = op.backward(
            grad_output.to(device),
            activation.to(device),
            global_fc1[begin:end].contiguous().to(device),
            global_fc2[begin:end].contiguous().to(device),
            topk_idx.to(device),
            topk_weights.to(device),
            fc1_c,
            route_metadata,
        )

        expected_g1_global = torch.zeros_like(global_fc1)
        expected_g2_global = torch.zeros_like(global_fc2)
        for src_rank in range(world_size):
            src_token_count = 3 + src_rank
            src_activation = torch.randn(src_token_count, hidden, generator=torch.Generator().manual_seed(2000 + src_rank))
            src_topk_idx = torch.tensor(
                [[(src_rank + token) % experts, (src_rank + token + 2) % experts] for token in range(src_token_count)],
                dtype=torch.int64,
            )
            src_topk_weights = torch.tensor([[0.625, 0.375]]).expand(src_token_count, -1).contiguous()
            src_grad_output = torch.randn(src_token_count, hidden, generator=torch.Generator().manual_seed(3000 + src_rank))
            src_ga, src_g1, src_g2, src_gw = _autograd_replica_grads(
                src_activation,
                global_fc1,
                global_fc2,
                src_topk_idx,
                src_topk_weights,
                src_grad_output,
                apply_topk_in_fc1=True,
            )
            expected_g1_global += src_g1
            expected_g2_global += src_g2
            if src_rank == rank:
                expected_ga, expected_gw = src_ga, src_gw

        # GPU/CPU GEMM rounding and rare bf16-stash boundary flips justify the
        # loose tolerance.
        torch.testing.assert_close(grad_activation.cpu(), expected_ga, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(grad_topk.cpu(), expected_gw, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(grad_fc1.cpu(), expected_g1_global[begin:end], rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(grad_fc2.cpu(), expected_g2_global[begin:end], rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(fc1_c.cpu(), expected_fc1_c)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_nccl_available() or torch.cuda.device_count() < 4,
    reason="requires NCCL and at least 4 GPUs",
)
def test_four_rank_expert_parallel(tmp_path):
    """One GPU per rank; NCCL runs the token all-to-all device-to-device."""

    init_file = tmp_path / "moe_ep_nccl_init"
    mp.spawn(_distributed_worker, args=(4, str(init_file)), nprocs=4, join=True)
