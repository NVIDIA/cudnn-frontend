# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

import cudnn
from cudnn.gnn import CscGraph, gat, gat_v2


def _require_gnn_mha(variant: str) -> None:
    if not hasattr(cudnn, f"gnn_mha_{variant}_forward") or not hasattr(cudnn, f"gnn_mha_{variant}_backward"):
        pytest.skip(f"cudnn-frontend was built without cudnnGnnMha{variant.title().replace('_', '')} support")


@pytest.fixture
def graph_data(request):
    index_dtype = request.param
    offsets = torch.tensor([0, 2, 5], device="cuda", dtype=index_dtype)
    indices = torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=index_dtype)
    edge_map = torch.tensor([3, 0, 4, 1, 2], device="cuda", dtype=index_dtype)
    return CscGraph(offsets, indices, num_src_nodes=4, map_csc_to_coo=edge_map)


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 5e-4, 6e-3
    return 2e-2, 6e-2


def _activation(x: torch.Tensor, name: str, alpha: float) -> torch.Tensor:
    if name == "linear":
        return x
    if name == "relu":
        return F.relu(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name == "tanh":
        return torch.tanh(x)
    if name == "elu":
        return F.elu(x)
    if name == "scalar":
        return x * alpha
    if name == "leaky_relu":
        return F.leaky_relu(x, negative_slope=alpha)
    raise AssertionError(name)


def _reference(
    variant: str,
    graph: CscGraph,
    node_features: torch.Tensor,
    edge_features: Optional[torch.Tensor],
    attn_weights: torch.Tensor,
    dropout_mask: Optional[torch.Tensor],
    num_heads: int,
    concat_heads: bool,
    activation: str,
    activation_alpha: float,
):
    offsets = graph.offsets.cpu().tolist()
    dim_node = node_features.shape[1]
    dim_head = dim_node // num_heads
    dim_edge = 0 if edge_features is None else edge_features.shape[1]
    dim_edge_head = dim_edge // num_heads
    map_csc_to_coo = graph.map_csc_to_coo
    attention_rows = [None] * num_heads
    head_outputs = []

    for head in range(num_heads):
        per_dst = []
        per_edge = torch.zeros(graph.num_edges, device=node_features.device, dtype=torch.float32)
        node_slice = slice(head * dim_head, (head + 1) * dim_head)
        if variant == "gat":
            w_src = attn_weights[node_slice]
            w_dst = attn_weights[dim_node + node_slice.start : dim_node + node_slice.stop]
            w_edge = attn_weights[2 * dim_node + head * dim_edge_head : 2 * dim_node + (head + 1) * dim_edge_head]
        else:
            weight = attn_weights[node_slice]

        for dst in range(graph.num_dst_nodes):
            begin, end = offsets[dst], offsets[dst + 1]
            source_rows = graph.indices[begin:end].long()
            edge_positions = torch.arange(begin, end, device=node_features.device)
            edge_rows = edge_positions if map_csc_to_coo is None else map_csc_to_coo[begin:end].long()
            source = node_features[source_rows, node_slice]
            destination = node_features[dst, node_slice]
            if variant == "gat":
                logits = source @ w_src + destination @ w_dst
                if edge_features is not None:
                    edge_slice = slice(head * dim_edge_head, (head + 1) * dim_edge_head)
                    logits = logits + edge_features[edge_rows, edge_slice] @ w_edge
                logits = _activation(logits, activation, activation_alpha)
            else:
                activated = source + destination
                if edge_features is not None:
                    activated = activated + edge_features[edge_rows, node_slice]
                logits = _activation(activated, activation, activation_alpha) @ weight
            coefficients = torch.softmax(logits.float(), dim=0)
            post_dropout = coefficients if dropout_mask is None else coefficients * dropout_mask[head, edge_rows]
            per_edge = per_edge.index_copy(0, edge_rows, post_dropout)
            per_dst.append((post_dropout.to(source.dtype).unsqueeze(1) * source).sum(dim=0))
        attention_rows[head] = per_edge
        head_outputs.append(torch.stack(per_dst))

    attention = torch.stack(attention_rows)
    output = torch.cat(head_outputs, dim=1) if concat_heads else torch.stack(head_outputs).mean(dim=0)
    return output, attention


@pytest.mark.parametrize("graph_data", [torch.int32, torch.int64], indirect=True)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.L0),
        pytest.param(torch.float16, marks=pytest.mark.L1),
        pytest.param(torch.bfloat16, marks=pytest.mark.L1),
    ],
)
@pytest.mark.parametrize("variant", ["gat", "gat_v2"])
@pytest.mark.parametrize("concat_heads", [False, True])
@pytest.mark.parametrize("with_edge", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_gnn_mha_forward_backward(graph_data, dtype, variant, concat_heads, with_edge, deterministic):
    _require_gnn_mha(variant)
    torch.manual_seed(1234)
    graph = graph_data.with_reverse_csc()
    assert graph.has_reverse_csc
    torch.testing.assert_close(graph.csc_rev_offsets, torch.tensor([0, 1, 2, 4, 5], device="cuda", dtype=graph.offsets.dtype))
    torch.testing.assert_close(graph.map_rev_to_coo, torch.tensor([3, 4, 0, 1, 2], device="cuda", dtype=graph.offsets.dtype))
    num_heads = 2
    dim_node = 8
    dim_edge = dim_node if variant == "gat_v2" else 4

    node = torch.randn((4, dim_node), device="cuda", dtype=dtype, requires_grad=True)
    edge = torch.randn((5, dim_edge), device="cuda", dtype=dtype, requires_grad=True) if with_edge else None
    weight_dim = dim_node if variant == "gat_v2" else 2 * dim_node + (dim_edge if with_edge else 0)
    weights = torch.randn((weight_dim,), device="cuda", dtype=dtype, requires_grad=True)
    dropout_mask = torch.tensor([[0.0, 1.25, 1.25, 1.25, 0.0], [1.25, 0.0, 1.25, 1.25, 1.25]], device="cuda", dtype=torch.float32)

    op = gat if variant == "gat" else gat_v2
    actual_output, actual_attention = op(
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
    expected_output, expected_attention = _reference(
        variant,
        graph,
        reference_node,
        reference_edge,
        reference_weights,
        dropout_mask,
        num_heads,
        concat_heads,
        "leaky_relu",
        0.2,
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
@pytest.mark.parametrize("op,weight_size", [(gat, 16), (gat_v2, 8)])
def test_gnn_mha_default_returns_only_output(op, weight_size):
    variant = "gat" if op is gat else "gat_v2"
    _require_gnn_mha(variant)
    graph = CscGraph(
        torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        num_src_nodes=2,
    )
    node = torch.randn((2, 8), device="cuda", requires_grad=True)
    weights = torch.randn((weight_size,), device="cuda", requires_grad=True)
    output = op(graph, node, weights, num_heads=2)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 8)
    output.sum().backward()
    assert node.grad is not None
    assert weights.grad is not None


@pytest.mark.L0
def test_gnn_mha_rejects_invalid_inputs():
    graph = CscGraph(
        torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        torch.tensor([0], device="cuda", dtype=torch.int32),
        num_src_nodes=1,
    )
    node = torch.randn((1, 8), device="cuda")
    weights = torch.randn((16,), device="cuda")
    with pytest.raises(ValueError, match="divisible by num_heads"):
        gat(graph, node, weights, num_heads=3)
    with pytest.raises(ValueError, match="attn_weights must have shape"):
        gat(graph, node, weights[:-1])
    with pytest.raises(ValueError, match="Unsupported activation"):
        gat(graph, node, weights, activation="gelu")
    with pytest.raises(TypeError, match="dropout_mask must have dtype float32"):
        gat(graph, node, weights, dropout_mask=torch.ones((1, 1), device="cuda", dtype=torch.float16))
    with pytest.raises(ValueError, match="deterministic backward requires"):
        gat(graph, node, weights, deterministic=True)
    with pytest.raises(ValueError, match="must be provided together"):
        CscGraph(
            graph.offsets,
            graph.indices,
            graph.num_src_nodes,
            csc_rev_offsets=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        )

    reverse_graph = CscGraph(
        graph.offsets,
        graph.indices,
        graph.num_src_nodes,
        csc_rev_offsets=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        map_rev_to_coo=torch.tensor([0], device="cuda", dtype=torch.int32),
    )
    assert reverse_graph.has_reverse_csc


@pytest.mark.L0
@pytest.mark.parametrize("op,weight_size", [(gat, 16), (gat_v2, 8)])
def test_gnn_mha_torch_compile(op, weight_size):
    variant = "gat" if op is gat else "gat_v2"
    _require_gnn_mha(variant)
    graph = CscGraph(
        torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        num_src_nodes=2,
    )

    def fn(node, weights):
        return op(graph, node, weights, num_heads=2)

    node = torch.randn((2, 8), device="cuda")
    weights = torch.randn((weight_size,), device="cuda")
    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(node, weights), fn(node, weights))


@pytest.mark.L1
@pytest.mark.parametrize("variant", ["gat", "gat_v2"])
@pytest.mark.parametrize("activation", ["linear", "relu", "sigmoid", "tanh", "elu", "scalar", "leaky_relu"])
def test_gnn_mha_activations(variant, activation):
    _require_gnn_mha(variant)
    graph = CscGraph(
        torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
        torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=torch.int32),
        num_src_nodes=4,
    )
    node = torch.randn((4, 8), device="cuda", requires_grad=True)
    edge = torch.randn((5, 8 if variant == "gat_v2" else 4), device="cuda", requires_grad=True)
    weight_size = 8 if variant == "gat_v2" else 20
    weights = torch.randn((weight_size,), device="cuda", requires_grad=True)
    op = gat if variant == "gat" else gat_v2

    actual = op(graph, node, weights, edge_features=edge, num_heads=2, activation=activation, activation_alpha=0.3)
    reference_node = node.detach().clone().requires_grad_()
    reference_edge = edge.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    expected, _ = _reference(
        variant,
        graph,
        reference_node,
        reference_edge,
        reference_weights,
        None,
        2,
        True,
        activation,
        0.3,
    )
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(node.grad, reference_node.grad, atol=6e-3, rtol=6e-3)
    torch.testing.assert_close(edge.grad, reference_edge.grad, atol=6e-3, rtol=6e-3)
    torch.testing.assert_close(weights.grad, reference_weights.grad, atol=6e-3, rtol=6e-3)


@pytest.mark.L0
@pytest.mark.parametrize("variant", ["gat", "gat_v2"])
@pytest.mark.parametrize("mapped", [False, True])
def test_gnn_mha_attention_edge_order(variant, mapped):
    _require_gnn_mha(variant)
    offsets = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    indices = torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=torch.int32)
    edge_map = torch.tensor([3, 0, 4, 1, 2], device="cuda", dtype=torch.int32) if mapped else None
    graph = CscGraph(offsets, indices, num_src_nodes=4, map_csc_to_coo=edge_map)
    node = torch.randn((4, 8), device="cuda")
    edge = torch.randn((5, 8 if variant == "gat_v2" else 4), device="cuda")
    weights = torch.randn((8 if variant == "gat_v2" else 20,), device="cuda")
    dropout_mask = torch.tensor([[0.0, 1.25, 1.25, 1.25, 0.0], [1.25, 0.0, 1.25, 1.25, 1.25]], device="cuda")
    op = gat if variant == "gat" else gat_v2

    actual_output, actual_attention = op(
        graph,
        node,
        weights,
        edge_features=edge,
        dropout_mask=dropout_mask,
        num_heads=2,
        return_attention_weights=True,
    )
    expected_output, expected_attention = _reference(variant, graph, node, edge, weights, dropout_mask, 2, True, "leaky_relu", 0.2)
    torch.testing.assert_close(actual_output, expected_output, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(actual_attention, expected_attention, atol=5e-4, rtol=5e-4)


@pytest.mark.L1
@pytest.mark.parametrize("variant", ["gat", "gat_v2"])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("grad_target", ["node", "edge", "weights"])
def test_gnn_mha_selective_gradients(variant, deterministic, grad_target):
    _require_gnn_mha(variant)
    graph = CscGraph(
        torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
        torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=torch.int32),
        num_src_nodes=4,
    ).with_reverse_csc()
    node = torch.randn((4, 8), device="cuda", requires_grad=grad_target == "node")
    edge = torch.randn((5, 8 if variant == "gat_v2" else 4), device="cuda", requires_grad=grad_target == "edge")
    weights = torch.randn((8 if variant == "gat_v2" else 20,), device="cuda", requires_grad=grad_target == "weights")
    op = gat if variant == "gat" else gat_v2

    op(graph, node, weights, edge_features=edge, num_heads=2, deterministic=deterministic).sum().backward()
    assert (node.grad is not None) == (grad_target == "node")
    assert (edge.grad is not None) == (grad_target == "edge")
    assert (weights.grad is not None) == (grad_target == "weights")


@pytest.mark.L0
@pytest.mark.parametrize("variant", ["gat", "gat_v2"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_gnn_mha_empty_destination_set(variant, deterministic):
    _require_gnn_mha(variant)
    graph = CscGraph(
        torch.tensor([0], device="cuda", dtype=torch.int32),
        torch.empty((0,), device="cuda", dtype=torch.int32),
        num_src_nodes=3,
    ).with_reverse_csc()
    node = torch.randn((3, 8), device="cuda", requires_grad=True)
    weights = torch.randn((8 if variant == "gat_v2" else 16,), device="cuda", requires_grad=True)
    op = gat if variant == "gat" else gat_v2

    output, attention = op(
        graph,
        node,
        weights,
        num_heads=2,
        return_attention_weights=True,
        deterministic=deterministic,
    )
    assert output.shape == (0, 8)
    assert attention.shape == (2, 0)
    (output.sum() + attention.sum()).backward()
    torch.testing.assert_close(node.grad, torch.zeros_like(node))
    torch.testing.assert_close(weights.grad, torch.zeros_like(weights))


@pytest.mark.L0
@pytest.mark.parametrize("variant", ["gat", "gat_v2"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_gnn_mha_zero_degree_destination(variant, deterministic):
    _require_gnn_mha(variant)
    graph = CscGraph(
        torch.tensor([0, 0, 2], device="cuda", dtype=torch.int32),
        torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        num_src_nodes=3,
    ).with_reverse_csc()
    node = torch.randn((3, 8), device="cuda", requires_grad=True)
    weights = torch.randn((8 if variant == "gat_v2" else 16,), device="cuda", requires_grad=True)
    op = gat if variant == "gat" else gat_v2
    actual = op(graph, node, weights, num_heads=2, deterministic=deterministic)

    reference_node = node.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    expected, _ = _reference(variant, graph, reference_node, None, reference_weights, None, 2, True, "leaky_relu", 0.2)
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]))

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(node.grad, reference_node.grad, atol=6e-3, rtol=6e-3)
    torch.testing.assert_close(weights.grad, reference_weights.grad, atol=6e-3, rtol=6e-3)


@pytest.mark.L0
@pytest.mark.parametrize("variant", ["gat", "gat_v2"])
def test_gnn_mha_uses_current_stream(variant):
    _require_gnn_mha(variant)
    graph = CscGraph(
        torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32),
        torch.tensor([0, 1, 1, 2], device="cuda", dtype=torch.int32),
        num_src_nodes=3,
    )
    node = torch.randn((3, 8), device="cuda")
    weights = torch.randn((8 if variant == "gat_v2" else 16,), device="cuda")
    op = gat if variant == "gat" else gat_v2
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = op(graph, node, weights, num_heads=2)
    stream.synchronize()
    expected, _ = _reference(variant, graph, node, None, weights, None, 2, True, "leaky_relu", 0.2)
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)


@pytest.mark.L0
@pytest.mark.parametrize("variant", ["gat", "gat_v2"])
def test_gnn_mha_backward_initializes_cuda_context_on_new_thread(variant):
    _require_gnn_mha(variant)
    offsets = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    node = torch.randn((2, 8), device="cuda")
    weights = torch.randn((8 if variant == "gat_v2" else 16,), device="cuda")
    output = torch.empty((1, 8), device="cuda")
    sm_scores = torch.empty((2, 2, 2), device="cuda", dtype=torch.float32)
    act_scores = torch.empty((2, 8), device="cuda") if variant == "gat_v2" else None
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
        0,
        weights.data_ptr(),
        0,
        output.data_ptr(),
        sm_scores.data_ptr(),
    )
    if variant == "gat":
        cudnn.gnn_mha_gat_forward(*common_forward, 8, 0, 6, 0.2, 2, True, 0)
    else:
        cudnn.gnn_mha_gat_v2_forward(*common_forward, act_scores.data_ptr(), 8, 6, 0.2, 2, True, 0)
    torch.cuda.synchronize()

    grad_output = torch.randn_like(output)
    grad_node = torch.empty_like(node)
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
                0,
                weights.data_ptr(),
                sm_scores.data_ptr(),
            )
            if variant == "gat":
                cudnn.gnn_mha_gat_backward(
                    *common_backward,
                    0,
                    0,
                    grad_node.data_ptr(),
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
            else:
                cudnn.gnn_mha_gat_v2_backward(
                    *common_backward,
                    act_scores.data_ptr(),
                    0,
                    0,
                    grad_node.data_ptr(),
                    0,
                    grad_weights.data_ptr(),
                    grad_sm_scores.data_ptr(),
                    8,
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
    assert torch.isfinite(grad_weights).all()


@pytest.mark.L0
@pytest.mark.parametrize("variant", ["gat", "gat_v2"])
def test_gnn_mha_deterministic_gradients_are_repeatable(variant):
    _require_gnn_mha(variant)
    graph = CscGraph(
        torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
        torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=torch.int32),
        num_src_nodes=4,
        map_csc_to_coo=torch.tensor([3, 0, 4, 1, 2], device="cuda", dtype=torch.int32),
    ).with_reverse_csc()
    node_value = torch.randn((4, 8), device="cuda")
    edge_value = torch.randn((5, 8 if variant == "gat_v2" else 4), device="cuda")
    weight_value = torch.randn((8 if variant == "gat_v2" else 20,), device="cuda")
    dropout_mask = torch.tensor([[0.0, 1.25, 1.25, 1.25, 0.0], [1.25, 0.0, 1.25, 1.25, 1.25]], device="cuda")
    op = gat if variant == "gat" else gat_v2

    def gradients():
        node = node_value.clone().requires_grad_()
        edge = edge_value.clone().requires_grad_()
        weights = weight_value.clone().requires_grad_()
        output, attention = op(
            graph,
            node,
            weights,
            edge_features=edge,
            dropout_mask=dropout_mask,
            num_heads=2,
            return_attention_weights=True,
            deterministic=True,
        )
        torch.autograd.backward((output, attention), (torch.ones_like(output), torch.ones_like(attention)))
        return node.grad, edge.grad, weights.grad

    first = gradients()
    second = gradients()
    for first_grad, second_grad in zip(first, second):
        assert torch.equal(first_grad, second_grad)


@pytest.mark.L1
@pytest.mark.parametrize("variant", ["gat", "gat_v2"])
@pytest.mark.parametrize("num_heads", [1, 3, 5])
@pytest.mark.parametrize("dim_head", [3, 16])
def test_gnn_mha_head_dimensions(variant, num_heads, dim_head):
    _require_gnn_mha(variant)
    graph = CscGraph(
        torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
        torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=torch.int32),
        num_src_nodes=4,
    )
    dim_node = num_heads * dim_head
    node = torch.randn((4, dim_node), device="cuda", requires_grad=True)
    weights = torch.randn((dim_node if variant == "gat_v2" else 2 * dim_node,), device="cuda", requires_grad=True)
    op = gat if variant == "gat" else gat_v2

    actual = op(graph, node, weights, num_heads=num_heads, concat_heads=False)
    reference_node = node.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    expected, _ = _reference(
        variant,
        graph,
        reference_node,
        None,
        reference_weights,
        None,
        num_heads,
        False,
        "leaky_relu",
        0.2,
    )
    assert actual.shape == (graph.num_dst_nodes, dim_head)
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(node.grad, reference_node.grad, atol=6e-3, rtol=6e-3)
    torch.testing.assert_close(weights.grad, reference_weights.grad, atol=6e-3, rtol=6e-3)
