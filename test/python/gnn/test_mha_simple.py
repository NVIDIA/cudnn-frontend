# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest
import torch

import cudnn
from cudnn.gnn import CscGraph, mha_simple
from gnn._mha_test_utils import _reference_mha_simple, _require_gnn_mha, _tolerances


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, marks=pytest.mark.L0),
        pytest.param(torch.float16, marks=pytest.mark.L1),
        pytest.param(torch.bfloat16, marks=pytest.mark.L1),
    ],
)
@pytest.mark.parametrize(
    "use_edge,use_bias,use_weight,use_dropout,concat_heads,norm_by_dim,return_attention,deterministic",
    [
        (False, False, False, False, True, True, False, False),
        (True, True, True, True, True, True, True, False),
        (True, False, True, False, False, False, True, True),
    ],
)
def test_gnn_mha_simple_forward_backward(
    small_reverse_csc_graph,
    dtype,
    use_edge,
    use_bias,
    use_weight,
    use_dropout,
    concat_heads,
    norm_by_dim,
    return_attention,
    deterministic,
):
    _require_gnn_mha("simple")
    torch.manual_seed(1234)
    graph = small_reverse_csc_graph
    num_heads = 2
    dim_node = 8
    key = torch.randn((graph.num_src_nodes, dim_node), device="cuda", dtype=dtype, requires_grad=True)
    query = torch.randn((graph.num_dst_nodes, dim_node), device="cuda", dtype=dtype, requires_grad=True)
    value = torch.randn((graph.num_src_nodes, dim_node), device="cuda", dtype=dtype, requires_grad=True)
    edge = torch.randn((graph.num_edges, dim_node), device="cuda", dtype=dtype, requires_grad=True) if use_edge else None
    score_bias = torch.randn((num_heads, graph.num_edges), device="cuda", requires_grad=True) if use_bias else None
    score_weight = torch.randn((num_heads, graph.num_edges), device="cuda", requires_grad=True) if use_weight else None
    dropout_mask = torch.tensor([[0.0, 1.25, 1.25, 1.25, 0.0], [1.25, 0.0, 1.25, 1.25, 1.25]], device="cuda") if use_dropout else None

    actual = mha_simple(
        graph,
        key,
        query,
        value,
        edge_features=edge,
        score_bias=score_bias,
        score_weight=score_weight,
        dropout_mask=dropout_mask,
        num_heads=num_heads,
        concat_heads=concat_heads,
        norm_by_dim=norm_by_dim,
        return_attention_weights=return_attention,
        deterministic=deterministic,
    )

    reference_key = key.detach().clone().requires_grad_()
    reference_query = query.detach().clone().requires_grad_()
    reference_value = value.detach().clone().requires_grad_()
    reference_edge = None if edge is None else edge.detach().clone().requires_grad_()
    reference_bias = None if score_bias is None else score_bias.detach().clone().requires_grad_()
    reference_weight = None if score_weight is None else score_weight.detach().clone().requires_grad_()
    expected_output, expected_attention = _reference_mha_simple(
        graph,
        reference_key,
        reference_query,
        reference_value,
        reference_edge,
        reference_bias,
        reference_weight,
        dropout_mask,
        num_heads,
        concat_heads,
        norm_by_dim,
    )

    output_tolerance, gradient_tolerance = _tolerances(dtype)
    if return_attention:
        actual_output, actual_attention = actual
        torch.testing.assert_close(actual_attention, expected_attention, atol=output_tolerance, rtol=output_tolerance)
        grad_attention = torch.randn_like(actual_attention)
        torch.autograd.backward((actual_output, actual_attention), (torch.ones_like(actual_output), grad_attention))
        torch.autograd.backward((expected_output, expected_attention), (torch.ones_like(expected_output), grad_attention))
    else:
        actual_output = actual
        actual_output.sum().backward()
        expected_output.sum().backward()
    torch.testing.assert_close(actual_output, expected_output, atol=output_tolerance, rtol=output_tolerance)

    for actual_tensor, reference_tensor in (
        (key, reference_key),
        (query, reference_query),
        (value, reference_value),
        (edge, reference_edge),
        (score_bias, reference_bias),
        (score_weight, reference_weight),
    ):
        if actual_tensor is not None:
            torch.testing.assert_close(actual_tensor.grad, reference_tensor.grad, atol=gradient_tolerance, rtol=gradient_tolerance)


@pytest.mark.L0
def test_gnn_mha_simple_returns_pre_dropout_attention():
    _require_gnn_mha("simple")
    graph = CscGraph(
        torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        num_src_nodes=2,
    )
    key = torch.zeros((2, 4), device="cuda", requires_grad=True)
    query = torch.ones((1, 4), device="cuda", requires_grad=True)
    value = torch.randn((2, 4), device="cuda", requires_grad=True)
    dropout_mask = torch.tensor([[0.0, 2.0]], device="cuda")
    output, attention = mha_simple(
        graph,
        key,
        query,
        value,
        dropout_mask=dropout_mask,
        return_attention_weights=True,
    )
    torch.testing.assert_close(attention, torch.tensor([[0.5, 0.5]], device="cuda"))
    torch.testing.assert_close(output, value[1:2])

    # Direct gradients of returned pre-dropout attention are not masked out.
    attention.backward(torch.tensor([[1.0, -1.0]], device="cuda"))
    assert torch.count_nonzero(key.grad).item() > 0
    assert torch.count_nonzero(query.grad).item() == 0
    torch.testing.assert_close(value.grad, torch.zeros_like(value))


@pytest.mark.L0
@pytest.mark.parametrize("mapped", [False, True])
def test_gnn_mha_simple_edge_order(mapped):
    _require_gnn_mha("simple")
    torch.manual_seed(2468)
    offsets = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    indices = torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=torch.int32)
    edge_map = torch.tensor([3, 0, 4, 1, 2], device="cuda", dtype=torch.int32) if mapped else None
    graph = CscGraph(offsets, indices, num_src_nodes=4, map_csc_to_coo=edge_map)
    key = torch.randn((4, 8), device="cuda", requires_grad=True)
    query = torch.randn((2, 8), device="cuda", requires_grad=True)
    value = torch.randn((4, 8), device="cuda", requires_grad=True)
    edge = torch.randn((5, 8), device="cuda", requires_grad=True)
    score_bias = torch.randn((2, 5), device="cuda", requires_grad=True)
    score_weight = torch.randn((2, 5), device="cuda", requires_grad=True)
    dropout_mask = torch.tensor([[0.0, 1.25, 1.25, 1.25, 0.0], [1.25, 0.0, 1.25, 1.25, 1.25]], device="cuda")

    actual_output, actual_attention = mha_simple(
        graph,
        key,
        query,
        value,
        edge_features=edge,
        score_bias=score_bias,
        score_weight=score_weight,
        dropout_mask=dropout_mask,
        num_heads=2,
        return_attention_weights=True,
    )
    reference_key = key.detach().clone().requires_grad_()
    reference_query = query.detach().clone().requires_grad_()
    reference_value = value.detach().clone().requires_grad_()
    reference_edge = edge.detach().clone().requires_grad_()
    reference_bias = score_bias.detach().clone().requires_grad_()
    reference_weight = score_weight.detach().clone().requires_grad_()
    expected_output, expected_attention = _reference_mha_simple(
        graph,
        reference_key,
        reference_query,
        reference_value,
        reference_edge,
        reference_bias,
        reference_weight,
        dropout_mask,
        2,
        True,
        True,
    )
    torch.testing.assert_close(actual_output, expected_output, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(actual_attention, expected_attention, atol=5e-4, rtol=5e-4)
    grad_output = torch.randn_like(actual_output)
    grad_attention = torch.randn_like(actual_attention)
    torch.autograd.backward((actual_output, actual_attention), (grad_output, grad_attention))
    torch.autograd.backward((expected_output, expected_attention), (grad_output, grad_attention))
    for actual_tensor, reference_tensor in (
        (key, reference_key),
        (query, reference_query),
        (value, reference_value),
        (edge, reference_edge),
        (score_bias, reference_bias),
        (score_weight, reference_weight),
    ):
        torch.testing.assert_close(actual_tensor.grad, reference_tensor.grad, atol=6e-3, rtol=6e-3)


@pytest.mark.L0
@pytest.mark.parametrize("deterministic", [False, True])
def test_gnn_mha_simple_noncontiguous_inputs(deterministic):
    _require_gnn_mha("simple")
    graph = CscGraph(
        torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
        torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=torch.int32),
        num_src_nodes=4,
        map_csc_to_coo=torch.tensor([3, 0, 4, 1, 2], device="cuda", dtype=torch.int32),
    ).with_reverse_csc()
    key = torch.randn((8, 4), device="cuda").T.detach().requires_grad_()
    query = torch.randn((8, 2), device="cuda").T.detach().requires_grad_()
    value = torch.randn((8, 4), device="cuda").T.detach().requires_grad_()
    edge = torch.randn((8, 5), device="cuda").T.detach().requires_grad_()
    assert not key.is_contiguous() and not query.is_contiguous() and not value.is_contiguous() and not edge.is_contiguous()

    actual = mha_simple(graph, key, query, value, edge_features=edge, num_heads=2, deterministic=deterministic)
    reference_key = key.detach().clone().requires_grad_()
    reference_query = query.detach().clone().requires_grad_()
    reference_value = value.detach().clone().requires_grad_()
    reference_edge = edge.detach().clone().requires_grad_()
    expected, _ = _reference_mha_simple(graph, reference_key, reference_query, reference_value, reference_edge, None, None, None, 2, True, True)
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for actual_tensor, reference_tensor in (
        (key, reference_key),
        (query, reference_query),
        (value, reference_value),
        (edge, reference_edge),
    ):
        torch.testing.assert_close(actual_tensor.grad, reference_tensor.grad, atol=6e-3, rtol=6e-3)


@pytest.mark.L0
def test_gnn_mha_simple_rejects_invalid_inputs():
    graph = CscGraph(
        torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        torch.tensor([0], device="cuda", dtype=torch.int32),
        num_src_nodes=1,
    )
    key = torch.randn((1, 8), device="cuda")
    query = torch.randn((1, 8), device="cuda")
    value = torch.randn((1, 8), device="cuda")
    with pytest.raises(ValueError, match="query_features must have shape"):
        mha_simple(graph, key, torch.randn((2, 8), device="cuda"), value)
    with pytest.raises(ValueError, match="value_features must have shape"):
        mha_simple(graph, key, query, torch.randn((2, 8), device="cuda"))
    with pytest.raises(ValueError, match="divisible by num_heads"):
        mha_simple(graph, key, query, value, num_heads=3)
    with pytest.raises(TypeError, match="score_bias must have dtype float32"):
        mha_simple(graph, key, query, value, score_bias=torch.ones((1, 1), device="cuda", dtype=torch.float16))
    with pytest.raises(ValueError, match="dropout_mask must not require gradients"):
        mha_simple(graph, key, query, value, dropout_mask=torch.ones((1, 1), device="cuda", requires_grad=True))
    with pytest.raises(ValueError, match="deterministic backward requires"):
        mha_simple(graph, key, query, value, deterministic=True)


@pytest.mark.L0
def test_gnn_mha_simple_default_returns_only_output():
    _require_gnn_mha("simple")
    graph = CscGraph(
        torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        num_src_nodes=2,
    )
    key = torch.randn((2, 8), device="cuda", requires_grad=True)
    query = torch.randn((1, 8), device="cuda", requires_grad=True)
    value = torch.randn((2, 8), device="cuda", requires_grad=True)
    output = mha_simple(graph, key, query, value, num_heads=2)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 8)


@pytest.mark.L0
@pytest.mark.parametrize("deterministic", [False, True])
def test_gnn_mha_simple_empty_destination_set(deterministic):
    _require_gnn_mha("simple")
    graph = CscGraph(
        torch.tensor([0], device="cuda", dtype=torch.int32),
        torch.empty((0,), device="cuda", dtype=torch.int32),
        num_src_nodes=3,
    ).with_reverse_csc()
    key = torch.randn((3, 8), device="cuda", requires_grad=True)
    query = torch.empty((0, 8), device="cuda", requires_grad=True)
    value = torch.randn((3, 8), device="cuda", requires_grad=True)
    score_bias = torch.empty((2, 0), device="cuda", requires_grad=True)
    score_weight = torch.empty((2, 0), device="cuda", requires_grad=True)
    output, attention = mha_simple(
        graph,
        key,
        query,
        value,
        score_bias=score_bias,
        score_weight=score_weight,
        num_heads=2,
        return_attention_weights=True,
        deterministic=deterministic,
    )
    assert output.shape == (0, 8)
    assert attention.shape == (2, 0)
    (output.sum() + attention.sum()).backward()
    for tensor in (key, query, value, score_bias, score_weight):
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))


@pytest.mark.L0
@pytest.mark.parametrize("deterministic", [False, True])
def test_gnn_mha_simple_zero_degree_destination(deterministic):
    _require_gnn_mha("simple")
    graph = CscGraph(
        torch.tensor([0, 0, 2], device="cuda", dtype=torch.int32),
        torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        num_src_nodes=3,
    ).with_reverse_csc()
    key = torch.randn((3, 8), device="cuda", requires_grad=True)
    query = torch.randn((2, 8), device="cuda", requires_grad=True)
    value = torch.randn((3, 8), device="cuda", requires_grad=True)
    actual = mha_simple(graph, key, query, value, num_heads=2, deterministic=deterministic)
    reference_key = key.detach().clone().requires_grad_()
    reference_query = query.detach().clone().requires_grad_()
    reference_value = value.detach().clone().requires_grad_()
    expected, _ = _reference_mha_simple(
        graph,
        reference_key,
        reference_query,
        reference_value,
        None,
        None,
        None,
        None,
        2,
        True,
        True,
    )
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]))
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for actual_tensor, reference_tensor in ((key, reference_key), (query, reference_query), (value, reference_value)):
        torch.testing.assert_close(actual_tensor.grad, reference_tensor.grad, atol=6e-3, rtol=6e-3)


@pytest.mark.L0
@pytest.mark.parametrize("grad_target", ["key", "query", "value", "edge", "bias", "weight"])
def test_gnn_mha_simple_selective_gradients(small_reverse_csc_graph, grad_target):
    _require_gnn_mha("simple")
    graph = small_reverse_csc_graph
    key = torch.randn((graph.num_src_nodes, 8), device="cuda", requires_grad=grad_target == "key")
    query = torch.randn((graph.num_dst_nodes, 8), device="cuda", requires_grad=grad_target == "query")
    value = torch.randn((graph.num_src_nodes, 8), device="cuda", requires_grad=grad_target == "value")
    edge = torch.randn((graph.num_edges, 8), device="cuda", requires_grad=grad_target == "edge")
    score_bias = torch.randn((2, graph.num_edges), device="cuda", requires_grad=grad_target == "bias")
    score_weight = torch.randn((2, graph.num_edges), device="cuda", requires_grad=grad_target == "weight")
    mha_simple(
        graph,
        key,
        query,
        value,
        edge_features=edge,
        score_bias=score_bias,
        score_weight=score_weight,
        num_heads=2,
        deterministic=True,
    ).sum().backward()
    for name, tensor in (
        ("key", key),
        ("query", query),
        ("value", value),
        ("edge", edge),
        ("bias", score_bias),
        ("weight", score_weight),
    ):
        assert (tensor.grad is not None) == (name == grad_target)


@pytest.mark.L0
def test_gnn_mha_simple_uses_current_stream():
    _require_gnn_mha("simple")
    graph = CscGraph(
        torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32),
        torch.tensor([0, 1, 1, 2], device="cuda", dtype=torch.int32),
        num_src_nodes=3,
    )
    key = torch.randn((3, 8), device="cuda")
    query = torch.randn((2, 8), device="cuda")
    value = torch.randn((3, 8), device="cuda")
    expected = mha_simple(graph, key, query, value, num_heads=2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = mha_simple(graph, key, query, value, num_heads=2)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, expected)


@pytest.mark.L0
def test_gnn_mha_simple_backward_initializes_cuda_context_on_new_thread():
    _require_gnn_mha("simple")
    offsets = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    key = torch.randn((2, 8), device="cuda")
    query = torch.randn((1, 8), device="cuda")
    value = torch.randn((2, 8), device="cuda")
    output = torch.empty((1, 8), device="cuda")
    sm_scores = torch.empty((2, 2, 2), device="cuda", dtype=torch.float32)
    cudnn.gnn_mha_simple_forward(
        torch.cuda.current_stream().cuda_stream,
        offsets.data_ptr(),
        indices.data_ptr(),
        0,
        2,
        1,
        2,
        4,  # CUDNN_DATA_INT32
        key.data_ptr(),
        query.data_ptr(),
        value.data_ptr(),
        0,
        0,
        0,
        0,
        output.data_ptr(),
        sm_scores.data_ptr(),
        8,
        True,
        2,
        True,
        0,  # CUDNN_DATA_FLOAT
    )
    torch.cuda.synchronize()
    grad_output = torch.randn_like(output)
    grad_key = torch.empty_like(key)
    grad_query = torch.empty_like(query)
    grad_value = torch.empty_like(value)
    grad_sm_scores = torch.empty_like(sm_scores)
    errors = []

    def run_backward():
        try:
            # Make no CUDA calls on this worker before entering the binding.
            cudnn.gnn_mha_simple_backward(
                0,
                offsets.data_ptr(),
                indices.data_ptr(),
                0,
                2,
                1,
                2,
                4,  # CUDNN_DATA_INT32
                grad_output.data_ptr(),
                key.data_ptr(),
                query.data_ptr(),
                value.data_ptr(),
                0,
                0,
                sm_scores.data_ptr(),
                0,
                0,
                grad_key.data_ptr(),
                grad_query.data_ptr(),
                grad_value.data_ptr(),
                0,
                0,
                grad_sm_scores.data_ptr(),
                0,
                0,
                8,
                True,
                2,
                True,
                0,  # CUDNN_DATA_FLOAT
                0,
                0,
                0,  # CUDNN_DATA_FLOAT
            )
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=run_backward)
    worker.start()
    worker.join()
    if errors:
        raise errors[0]
    torch.cuda.synchronize()
    assert torch.isfinite(grad_key).all()
    assert torch.isfinite(grad_query).all()
    assert torch.isfinite(grad_value).all()


@pytest.mark.L0
def test_gnn_mha_simple_torch_compile():
    _require_gnn_mha("simple")
    graph = CscGraph(
        torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        num_src_nodes=2,
    )

    def fn(key, query, value):
        return mha_simple(graph, key, query, value, num_heads=2)

    key = torch.randn((2, 8), device="cuda")
    query = torch.randn((1, 8), device="cuda")
    value = torch.randn((2, 8), device="cuda")
    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(key, query, value), fn(key, query, value))


@pytest.mark.L0
def test_gnn_mha_simple_deterministic_gradients_are_repeatable():
    _require_gnn_mha("simple")
    graph = CscGraph(
        torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
        torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=torch.int32),
        num_src_nodes=4,
        map_csc_to_coo=torch.tensor([3, 0, 4, 1, 2], device="cuda", dtype=torch.int32),
    ).with_reverse_csc()
    key_value = torch.randn((4, 8), device="cuda")
    query_value = torch.randn((2, 8), device="cuda")
    value_value = torch.randn((4, 8), device="cuda")
    edge_value = torch.randn((5, 8), device="cuda")
    bias_value = torch.randn((2, 5), device="cuda")
    weight_value = torch.randn((2, 5), device="cuda")
    dropout_mask = torch.tensor([[0.0, 1.25, 1.25, 1.25, 0.0], [1.25, 0.0, 1.25, 1.25, 1.25]], device="cuda")

    def gradients():
        key = key_value.clone().requires_grad_()
        query = query_value.clone().requires_grad_()
        value = value_value.clone().requires_grad_()
        edge = edge_value.clone().requires_grad_()
        score_bias = bias_value.clone().requires_grad_()
        score_weight = weight_value.clone().requires_grad_()
        output, attention = mha_simple(
            graph,
            key,
            query,
            value,
            edge_features=edge,
            score_bias=score_bias,
            score_weight=score_weight,
            dropout_mask=dropout_mask,
            num_heads=2,
            return_attention_weights=True,
            deterministic=True,
        )
        torch.autograd.backward((output, attention), (torch.ones_like(output), torch.ones_like(attention)))
        return key.grad, query.grad, value.grad, edge.grad, score_bias.grad, score_weight.grad

    first = gradients()
    second = gradients()
    for first_grad, second_grad in zip(first, second):
        assert torch.equal(first_grad, second_grad)


@pytest.mark.L1
@pytest.mark.parametrize("num_heads", [1, 3, 5])
@pytest.mark.parametrize("dim_head", [3, 16])
def test_gnn_mha_simple_head_dimensions(num_heads, dim_head):
    _require_gnn_mha("simple")
    graph = CscGraph(
        torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
        torch.tensor([0, 2, 1, 2, 3], device="cuda", dtype=torch.int32),
        num_src_nodes=4,
    )
    dim_node = num_heads * dim_head
    key = torch.randn((4, dim_node), device="cuda", requires_grad=True)
    query = torch.randn((2, dim_node), device="cuda", requires_grad=True)
    value = torch.randn((4, dim_node), device="cuda", requires_grad=True)
    actual = mha_simple(graph, key, query, value, num_heads=num_heads, concat_heads=False)
    reference_key = key.detach().clone().requires_grad_()
    reference_query = query.detach().clone().requires_grad_()
    reference_value = value.detach().clone().requires_grad_()
    expected, _ = _reference_mha_simple(
        graph,
        reference_key,
        reference_query,
        reference_value,
        None,
        None,
        None,
        None,
        num_heads,
        False,
        True,
    )
    assert actual.shape == (graph.num_dst_nodes, dim_head)
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for actual_tensor, reference_tensor in ((key, reference_key), (query, reference_query), (value, reference_value)):
        torch.testing.assert_close(actual_tensor.grad, reference_tensor.grad, atol=6e-3, rtol=6e-3)


@pytest.mark.L1
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("high_precision_dgrad", [False, True])
def test_gnn_mha_simple_feature_gradient_dtype(small_reverse_csc_graph, dtype, high_precision_dgrad):
    _require_gnn_mha("simple")
    graph = small_reverse_csc_graph
    key = torch.randn((graph.num_src_nodes, 8), device="cuda", dtype=dtype, requires_grad=True)
    query = torch.randn((graph.num_dst_nodes, 8), device="cuda", dtype=dtype, requires_grad=True)
    value = torch.randn((graph.num_src_nodes, 8), device="cuda", dtype=dtype, requires_grad=True)
    expected_grad_dtype = torch.float32 if high_precision_dgrad else dtype
    if high_precision_dgrad and dtype != torch.float32:
        key.grad_dtype = expected_grad_dtype
        query.grad_dtype = expected_grad_dtype
        value.grad_dtype = expected_grad_dtype
    output = mha_simple(
        graph,
        key,
        query,
        value,
        num_heads=2,
        deterministic=True,
        high_precision_dgrad=high_precision_dgrad,
    )
    output.sum().backward()
    assert key.grad.dtype == expected_grad_dtype
    assert query.grad.dtype == expected_grad_dtype
    assert value.grad.dtype == expected_grad_dtype


@pytest.mark.L2
def test_gnn_mha_simple_medium_graph(medium_reverse_csc_graph):
    _require_gnn_mha("simple")
    graph = medium_reverse_csc_graph
    key = torch.randn((graph.num_src_nodes, 32), device="cuda", requires_grad=True)
    query = torch.randn((graph.num_dst_nodes, 32), device="cuda", requires_grad=True)
    value = torch.randn((graph.num_src_nodes, 32), device="cuda", requires_grad=True)
    output = mha_simple(graph, key, query, value, num_heads=4, deterministic=True)
    assert output.shape == (graph.num_dst_nodes, 32)
    output.square().mean().backward()
    assert key.grad is not None and query.grad is not None and value.grad is not None
