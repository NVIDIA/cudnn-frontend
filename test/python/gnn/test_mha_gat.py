# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import pytest
import cudnn
import torch
from cudnn.gnn import CscGraph, mha_gat
from gnn._mha_test_utils import _require_gnn_mha, _reference_gat, _tolerances, graph_data

__all__ = ["graph_data"]


@pytest.mark.parametrize("graph_data", [torch.int32, torch.int64], indirect=True)
@pytest.mark.parametrize(
    "dtype",
    [pytest.param(torch.float32, marks=pytest.mark.L0), pytest.param(torch.float16, marks=pytest.mark.L1), pytest.param(torch.bfloat16, marks=pytest.mark.L1)],
)
@pytest.mark.parametrize("concat_heads", [False, True])
@pytest.mark.parametrize("with_edge", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_gnn_mha_gat_forward_backward(graph_data, dtype, concat_heads, with_edge, deterministic):
    _require_gnn_mha("gat")
    torch.manual_seed(1234)
    graph = graph_data.with_reverse_csc()
    assert graph.has_reverse_csc
    torch.testing.assert_close(graph.csc_rev_offsets, torch.tensor([0, 1, 2, 4, 5], device="cuda", dtype=graph.offsets.dtype))
    torch.testing.assert_close(graph.map_rev_to_coo, torch.tensor([3, 4, 0, 1, 2], device="cuda", dtype=graph.offsets.dtype))
    num_heads = 2
    dim_node = 8
    dim_edge = 4
    node = torch.randn((4, dim_node), device="cuda", dtype=dtype, requires_grad=True)
    edge = torch.randn((5, dim_edge), device="cuda", dtype=dtype, requires_grad=True) if with_edge else None
    weight_dim = 2 * dim_node + (dim_edge if with_edge else 0)
    weights = torch.randn((weight_dim,), device="cuda", dtype=dtype, requires_grad=True)
    dropout_mask = torch.tensor([[0.0, 1.25, 1.25, 1.25, 0.0], [1.25, 0.0, 1.25, 1.25, 1.25]], device="cuda", dtype=torch.float32)
    actual_output, actual_attention = mha_gat(
        graph,
        node,
        weights,
        edge_features=edge,
        dropout_mask=dropout_mask,
        num_heads=num_heads,
        concat_heads=concat_heads,
        activation="leaky_relu",
        activation_alpha=0.2,
        return_attention_weights=True,
        deterministic=deterministic,
    )
    reference_node = node.detach().clone().requires_grad_()
    reference_edge = None if edge is None else edge.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    expected_output, expected_attention = _reference_gat(
        graph, reference_node, reference_node, reference_edge, reference_weights, dropout_mask, num_heads, concat_heads, "leaky_relu", 0.2
    )
    output_tolerance, gradient_tolerance = _tolerances(dtype)
    torch.testing.assert_close(actual_output, expected_output, atol=output_tolerance, rtol=output_tolerance)
    torch.testing.assert_close(actual_attention, expected_attention, atol=output_tolerance, rtol=output_tolerance)
    grad_output = torch.randn_like(actual_output)
    grad_attention = torch.randn_like(actual_attention)
    torch.autograd.backward((actual_output, actual_attention), (grad_output, grad_attention))
    torch.autograd.backward((expected_output, expected_attention), (grad_output, grad_attention))
    torch.testing.assert_close(node.grad, reference_node.grad, atol=gradient_tolerance, rtol=gradient_tolerance)
    torch.testing.assert_close(weights.grad, reference_weights.grad, atol=gradient_tolerance, rtol=gradient_tolerance)
    if edge is not None:
        torch.testing.assert_close(edge.grad, reference_edge.grad, atol=gradient_tolerance, rtol=gradient_tolerance)


@pytest.mark.L0
@pytest.mark.parametrize("deterministic", [False, True])
def test_gnn_mha_gat_bipartite_forward_backward(deterministic):
    _require_gnn_mha("gat")
    torch.manual_seed(5678)
    graph = CscGraph(
        torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
        torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=torch.int32),
        num_src_nodes=4,
        map_csc_to_coo=torch.tensor([3, 0, 4, 1, 2], device="cuda", dtype=torch.int32),
    ).with_reverse_csc()
    src_features = torch.randn((4, 8), device="cuda", requires_grad=True)
    dst_features = torch.randn((2, 8), device="cuda", requires_grad=True)
    edge_features = torch.randn((5, 4), device="cuda", requires_grad=True)
    attn_weights = torch.randn((20,), device="cuda", requires_grad=True)
    actual_output, actual_attention = mha_gat(
        graph, (src_features, dst_features), attn_weights, edge_features=edge_features, num_heads=2, return_attention_weights=True, deterministic=deterministic
    )
    reference_src = src_features.detach().clone().requires_grad_()
    reference_dst = dst_features.detach().clone().requires_grad_()
    reference_edge = edge_features.detach().clone().requires_grad_()
    reference_weights = attn_weights.detach().clone().requires_grad_()
    expected_output, expected_attention = _reference_gat(
        graph, reference_src, reference_dst, reference_edge, reference_weights, None, 2, True, "leaky_relu", 0.2
    )
    torch.testing.assert_close(actual_output, expected_output, atol=0.0005, rtol=0.0005)
    torch.testing.assert_close(actual_attention, expected_attention, atol=0.0005, rtol=0.0005)
    grad_output = torch.randn_like(actual_output)
    grad_attention = torch.randn_like(actual_attention)
    torch.autograd.backward((actual_output, actual_attention), (grad_output, grad_attention))
    torch.autograd.backward((expected_output, expected_attention), (grad_output, grad_attention))
    torch.testing.assert_close(src_features.grad, reference_src.grad, atol=0.006, rtol=0.006)
    torch.testing.assert_close(dst_features.grad, reference_dst.grad, atol=0.006, rtol=0.006)
    torch.testing.assert_close(edge_features.grad, reference_edge.grad, atol=0.006, rtol=0.006)
    torch.testing.assert_close(attn_weights.grad, reference_weights.grad, atol=0.006, rtol=0.006)


@pytest.mark.L0
def test_gnn_mha_gat_default_returns_only_output():
    _require_gnn_mha("gat")
    graph = CscGraph(torch.tensor([0, 2], device="cuda", dtype=torch.int32), torch.tensor([0, 1], device="cuda", dtype=torch.int32), num_src_nodes=2)
    node = torch.randn((2, 8), device="cuda", requires_grad=True)
    weights = torch.randn((16,), device="cuda", requires_grad=True)
    output = mha_gat(graph, node, weights, num_heads=2)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 8)
    output.sum().backward()
    assert node.grad is not None
    assert weights.grad is not None


@pytest.mark.L0
def test_gnn_mha_gat_rejects_invalid_inputs():
    graph = CscGraph(torch.tensor([0, 1], device="cuda", dtype=torch.int32), torch.tensor([0], device="cuda", dtype=torch.int32), num_src_nodes=1)
    node = torch.randn((1, 8), device="cuda")
    weights = torch.randn((16,), device="cuda")
    with pytest.raises(TypeError, match="features must be a Tensor or a .* tuple"):
        mha_gat(graph, (node,), weights)
    with pytest.raises(ValueError, match="dst_features must have shape"):
        mha_gat(graph, (node, torch.randn((2, 8), device="cuda")), weights)
    with pytest.raises(ValueError, match="divisible by num_heads"):
        mha_gat(graph, node, weights, num_heads=3)
    with pytest.raises(ValueError, match="attn_weights must have shape"):
        mha_gat(graph, node, weights[:-1])
    with pytest.raises(ValueError, match="Unsupported activation"):
        mha_gat(graph, node, weights, activation="gelu")
    with pytest.raises(TypeError, match="dropout_mask must have dtype float32"):
        mha_gat(graph, node, weights, dropout_mask=torch.ones((1, 1), device="cuda", dtype=torch.float16))
    with pytest.raises(ValueError, match="deterministic backward requires"):
        mha_gat(graph, node, weights, deterministic=True)
    with pytest.raises(ValueError, match="must be provided together"):
        CscGraph(graph.offsets, graph.indices, graph.num_src_nodes, csc_rev_offsets=torch.tensor([0, 1], device="cuda", dtype=torch.int32))
    reverse_graph = CscGraph(
        graph.offsets,
        graph.indices,
        graph.num_src_nodes,
        csc_rev_offsets=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        map_rev_to_coo=torch.tensor([0], device="cuda", dtype=torch.int32),
    )
    assert reverse_graph.has_reverse_csc


@pytest.mark.L0
def test_gnn_mha_gat_torch_compile():
    _require_gnn_mha("gat")
    graph = CscGraph(torch.tensor([0, 2], device="cuda", dtype=torch.int32), torch.tensor([0, 1], device="cuda", dtype=torch.int32), num_src_nodes=2)

    def fn(node, weights):
        return mha_gat(graph, node, weights, num_heads=2)

    node = torch.randn((2, 8), device="cuda")
    weights = torch.randn((16,), device="cuda")
    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(node, weights), fn(node, weights))


@pytest.mark.L1
@pytest.mark.parametrize("activation", ["linear", "relu", "sigmoid", "tanh", "elu", "scalar", "leaky_relu"])
def test_gnn_mha_gat_activations(activation):
    _require_gnn_mha("gat")
    graph = CscGraph(
        torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32), torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=torch.int32), num_src_nodes=4
    )
    node = torch.randn((4, 8), device="cuda", requires_grad=True)
    edge = torch.randn((5, 4), device="cuda", requires_grad=True)
    weights = torch.randn((20,), device="cuda", requires_grad=True)
    actual = mha_gat(graph, node, weights, edge_features=edge, num_heads=2, activation=activation, activation_alpha=0.3)
    reference_node = node.detach().clone().requires_grad_()
    reference_edge = edge.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    expected, _ = _reference_gat(graph, reference_node, reference_node, reference_edge, reference_weights, None, 2, True, activation, 0.3)
    torch.testing.assert_close(actual, expected, atol=0.0005, rtol=0.0005)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(node.grad, reference_node.grad, atol=0.006, rtol=0.006)
    torch.testing.assert_close(edge.grad, reference_edge.grad, atol=0.006, rtol=0.006)
    torch.testing.assert_close(weights.grad, reference_weights.grad, atol=0.006, rtol=0.006)


@pytest.mark.L0
@pytest.mark.parametrize("mapped", [False, True])
def test_gnn_mha_gat_attention_edge_order(mapped):
    _require_gnn_mha("gat")
    offsets = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    indices = torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=torch.int32)
    edge_map = torch.tensor([3, 0, 4, 1, 2], device="cuda", dtype=torch.int32) if mapped else None
    graph = CscGraph(offsets, indices, num_src_nodes=4, map_csc_to_coo=edge_map)
    node = torch.randn((4, 8), device="cuda")
    edge = torch.randn((5, 4), device="cuda")
    weights = torch.randn((20,), device="cuda")
    dropout_mask = torch.tensor([[0.0, 1.25, 1.25, 1.25, 0.0], [1.25, 0.0, 1.25, 1.25, 1.25]], device="cuda")
    actual_output, actual_attention = mha_gat(graph, node, weights, edge_features=edge, dropout_mask=dropout_mask, num_heads=2, return_attention_weights=True)
    expected_output, expected_attention = _reference_gat(graph, node, node, edge, weights, dropout_mask, 2, True, "leaky_relu", 0.2)
    torch.testing.assert_close(actual_output, expected_output, atol=0.0005, rtol=0.0005)
    torch.testing.assert_close(actual_attention, expected_attention, atol=0.0005, rtol=0.0005)


@pytest.mark.L1
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("grad_target", ["node", "edge", "weights"])
def test_gnn_mha_gat_selective_gradients(deterministic, grad_target):
    _require_gnn_mha("gat")
    graph = CscGraph(
        torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32), torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=torch.int32), num_src_nodes=4
    ).with_reverse_csc()
    node = torch.randn((4, 8), device="cuda", requires_grad=grad_target == "node")
    edge = torch.randn((5, 4), device="cuda", requires_grad=grad_target == "edge")
    weights = torch.randn((20,), device="cuda", requires_grad=grad_target == "weights")
    mha_gat(graph, node, weights, edge_features=edge, num_heads=2, deterministic=deterministic).sum().backward()
    assert (node.grad is not None) == (grad_target == "node")
    assert (edge.grad is not None) == (grad_target == "edge")
    assert (weights.grad is not None) == (grad_target == "weights")


@pytest.mark.L0
@pytest.mark.parametrize("deterministic", [False, True])
def test_gnn_mha_gat_empty_destination_set(deterministic):
    _require_gnn_mha("gat")
    graph = CscGraph(
        torch.tensor([0], device="cuda", dtype=torch.int32), torch.empty((0,), device="cuda", dtype=torch.int32), num_src_nodes=3
    ).with_reverse_csc()
    node = torch.randn((3, 8), device="cuda", requires_grad=True)
    weights = torch.randn((16,), device="cuda", requires_grad=True)
    output, attention = mha_gat(graph, node, weights, num_heads=2, return_attention_weights=True, deterministic=deterministic)
    assert output.shape == (0, 8)
    assert attention.shape == (2, 0)
    (output.sum() + attention.sum()).backward()
    torch.testing.assert_close(node.grad, torch.zeros_like(node))
    torch.testing.assert_close(weights.grad, torch.zeros_like(weights))


@pytest.mark.L0
@pytest.mark.parametrize("deterministic", [False, True])
def test_gnn_mha_gat_zero_degree_destination(deterministic):
    _require_gnn_mha("gat")
    graph = CscGraph(
        torch.tensor([0, 0, 2], device="cuda", dtype=torch.int32), torch.tensor([0, 2], device="cuda", dtype=torch.int32), num_src_nodes=3
    ).with_reverse_csc()
    node = torch.randn((3, 8), device="cuda", requires_grad=True)
    weights = torch.randn((16,), device="cuda", requires_grad=True)
    actual = mha_gat(graph, node, weights, num_heads=2, deterministic=deterministic)
    reference_node = node.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    expected, _ = _reference_gat(graph, reference_node, reference_node, None, reference_weights, None, 2, True, "leaky_relu", 0.2)
    torch.testing.assert_close(actual, expected, atol=0.0005, rtol=0.0005)
    torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]))
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(node.grad, reference_node.grad, atol=0.006, rtol=0.006)
    torch.testing.assert_close(weights.grad, reference_weights.grad, atol=0.006, rtol=0.006)


@pytest.mark.L0
def test_gnn_mha_gat_uses_current_stream():
    _require_gnn_mha("gat")
    graph = CscGraph(torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32), torch.tensor([0, 1, 1, 2], device="cuda", dtype=torch.int32), num_src_nodes=3)
    node = torch.randn((3, 8), device="cuda")
    weights = torch.randn((16,), device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = mha_gat(graph, node, weights, num_heads=2)
    stream.synchronize()
    expected, _ = _reference_gat(graph, node, node, None, weights, None, 2, True, "leaky_relu", 0.2)
    torch.testing.assert_close(actual, expected, atol=0.0005, rtol=0.0005)


@pytest.mark.L0
def test_gnn_mha_gat_backward_initializes_cuda_context_on_new_thread():
    _require_gnn_mha("gat")
    offsets = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    node = torch.randn((2, 8), device="cuda")
    weights = torch.randn((16,), device="cuda")
    output = torch.empty((1, 8), device="cuda")
    sm_scores = torch.empty((2, 2, 2), device="cuda", dtype=torch.float32)
    act_scores = None
    common_forward = (
        torch.cuda.current_stream().cuda_stream,
        offsets.data_ptr(),
        indices.data_ptr(),
        0,
        2,
        1,
        2,
        4,
        node.data_ptr(),
        node.data_ptr(),
        0,
        weights.data_ptr(),
        0,
        output.data_ptr(),
        sm_scores.data_ptr(),
    )
    cudnn.gnn_mha_gat_forward(*common_forward, 8, 0, 6, 0.2, 2, True, 0)
    torch.cuda.synchronize()
    grad_output = torch.randn_like(output)
    grad_node = torch.empty_like(node)
    grad_dst = torch.empty((1, node.shape[1]), device="cuda", dtype=node.dtype)
    grad_weights = torch.empty_like(weights)
    grad_sm_scores = torch.empty_like(sm_scores)
    errors = []

    def run_backward():
        try:
            common_backward = (
                0,
                offsets.data_ptr(),
                indices.data_ptr(),
                0,
                2,
                1,
                2,
                4,
                grad_output.data_ptr(),
                node.data_ptr(),
                node.data_ptr(),
                0,
                weights.data_ptr(),
                sm_scores.data_ptr(),
            )
            cudnn.gnn_mha_gat_backward(
                *common_backward,
                0,
                0,
                grad_node.data_ptr(),
                grad_dst.data_ptr(),
                0,
                grad_weights.data_ptr(),
                grad_sm_scores.data_ptr(),
                8,
                0,
                6,
                0.2,
                2,
                True,
                0,
                0,
                0,
                0,
                0,
            )
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=run_backward)
    worker.start()
    worker.join()
    if errors:
        raise errors[0]
    torch.cuda.synchronize()
    assert torch.isfinite(grad_node).all()
    assert torch.isfinite(grad_dst).all()
    assert torch.isfinite(grad_weights).all()


@pytest.mark.L0
def test_gnn_mha_gat_deterministic_gradients_are_repeatable():
    _require_gnn_mha("gat")
    graph = CscGraph(
        torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
        torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=torch.int32),
        num_src_nodes=4,
        map_csc_to_coo=torch.tensor([3, 0, 4, 1, 2], device="cuda", dtype=torch.int32),
    ).with_reverse_csc()
    node_value = torch.randn((4, 8), device="cuda")
    edge_value = torch.randn((5, 4), device="cuda")
    weight_value = torch.randn((20,), device="cuda")
    dropout_mask = torch.tensor([[0.0, 1.25, 1.25, 1.25, 0.0], [1.25, 0.0, 1.25, 1.25, 1.25]], device="cuda")

    def gradients():
        node = node_value.clone().requires_grad_()
        edge = edge_value.clone().requires_grad_()
        weights = weight_value.clone().requires_grad_()
        output, attention = mha_gat(
            graph, node, weights, edge_features=edge, dropout_mask=dropout_mask, num_heads=2, return_attention_weights=True, deterministic=True
        )
        torch.autograd.backward((output, attention), (torch.ones_like(output), torch.ones_like(attention)))
        return (node.grad, edge.grad, weights.grad)

    first = gradients()
    second = gradients()
    for first_grad, second_grad in zip(first, second):
        assert torch.equal(first_grad, second_grad)


@pytest.mark.L1
@pytest.mark.parametrize("num_heads", [1, 3, 5])
@pytest.mark.parametrize("dim_head", [3, 16])
def test_gnn_mha_gat_head_dimensions(num_heads, dim_head):
    _require_gnn_mha("gat")
    graph = CscGraph(
        torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32), torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=torch.int32), num_src_nodes=4
    )
    dim_node = num_heads * dim_head
    node = torch.randn((4, dim_node), device="cuda", requires_grad=True)
    weights = torch.randn((2 * dim_node,), device="cuda", requires_grad=True)
    actual = mha_gat(graph, node, weights, num_heads=num_heads, concat_heads=False)
    reference_node = node.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    expected, _ = _reference_gat(graph, reference_node, reference_node, None, reference_weights, None, num_heads, False, "leaky_relu", 0.2)
    assert actual.shape == (graph.num_dst_nodes, dim_head)
    torch.testing.assert_close(actual, expected, atol=0.0005, rtol=0.0005)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(node.grad, reference_node.grad, atol=0.006, rtol=0.006)
    torch.testing.assert_close(weights.grad, reference_weights.grad, atol=0.006, rtol=0.006)


@pytest.mark.L1
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize(
    ("input_dtype", "high_precision_dgrad", "high_precision_wgrad"),
    [
        (torch.float16, False, False),
        (torch.float16, False, True),
        (torch.float16, True, True),
        (torch.bfloat16, False, False),
        (torch.bfloat16, False, True),
        (torch.bfloat16, True, True),
        (torch.float32, True, True),
    ],
)
def test_gnn_mha_gat_gradient_precision(deterministic, input_dtype, high_precision_dgrad, high_precision_wgrad):
    _require_gnn_mha("gat")
    feature_grad_dtype = torch.float32 if high_precision_dgrad else input_dtype
    weight_grad_dtype = torch.float32 if high_precision_wgrad else input_dtype
    graph = CscGraph(
        torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32), torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=torch.int32), num_src_nodes=4
    ).with_reverse_csc()
    src_features = torch.randn((4, 8), device="cuda", dtype=input_dtype, requires_grad=True)
    dst_features = torch.randn((2, 8), device="cuda", dtype=input_dtype, requires_grad=True)
    edge_features = torch.randn((5, 4), device="cuda", dtype=input_dtype, requires_grad=True)
    attn_weights = torch.randn((20,), device="cuda", dtype=input_dtype, requires_grad=True)
    src_features.grad_dtype = feature_grad_dtype
    dst_features.grad_dtype = feature_grad_dtype
    edge_features.grad_dtype = feature_grad_dtype
    attn_weights.grad_dtype = weight_grad_dtype
    output = mha_gat(
        graph,
        (src_features, dst_features),
        attn_weights,
        edge_features=edge_features,
        num_heads=2,
        deterministic=deterministic,
        high_precision_dgrad=high_precision_dgrad,
        high_precision_wgrad=high_precision_wgrad,
    )
    output.sum().backward()
    for feature in (src_features, dst_features, edge_features):
        assert feature.grad.dtype == feature_grad_dtype
        assert torch.isfinite(feature.grad).all()
    assert attn_weights.grad.dtype == weight_grad_dtype
    assert torch.isfinite(attn_weights.grad).all()


@pytest.mark.L0
def test_gnn_mha_gat_rejects_invalid_precision_options():
    graph = CscGraph(torch.tensor([0, 1], device="cuda", dtype=torch.int32), torch.tensor([0], device="cuda", dtype=torch.int32), num_src_nodes=1)
    node = torch.randn((1, 8), device="cuda", dtype=torch.float16)
    weights = torch.randn((16,), device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="high_precision_dgrad=True requires high_precision_wgrad=True"):
        mha_gat(graph, node, weights, high_precision_dgrad=True)
    with pytest.raises(TypeError, match="high_precision_dgrad must be a bool"):
        mha_gat(graph, node, weights, high_precision_dgrad=1)
    with pytest.raises(TypeError, match="high_precision_wgrad must be a bool"):
        mha_gat(graph, node, weights, high_precision_wgrad="true")
