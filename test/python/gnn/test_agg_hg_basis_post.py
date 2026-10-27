# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from typing import Optional

import pytest
import torch

import cudnn
from cudnn.gnn import CscGraph, agg_hg_basis_post


def _require_gnn_agg_hg_basis_post() -> None:
    if not hasattr(cudnn, "gnn_agg_hg_basis_post_forward") or not hasattr(cudnn, "gnn_agg_hg_basis_post_backward"):
        pytest.skip("cudnn-frontend was built without cudnnGnnAggHgBasisPost support")
    if cudnn.backend_version() < 92900:
        pytest.skip("GNN AggHgBasisPost requires cuDNN 9.29.0 or newer")


def _reference(
    graph: CscGraph,
    node_features: torch.Tensor,
    edge_types: torch.Tensor,
    weights_comb: Optional[torch.Tensor],
    num_edge_types: int,
    concat_own: bool,
    norm_by_edge_type: bool,
) -> torch.Tensor:
    offsets = graph.offsets.cpu().tolist()
    indices = graph.indices.cpu().tolist()
    types = edge_types.cpu().tolist()
    rows = []
    for dst in range(graph.num_dst_nodes):
        per_type = []
        for edge_type in range(num_edge_types):
            sources = [indices[e] for e in range(offsets[dst], offsets[dst + 1]) if types[e] == edge_type]
            aggregate = node_features[sources].sum(dim=0)
            if norm_by_edge_type and sources:
                aggregate = aggregate / len(sources)
            per_type.append(aggregate)
        aggregates = torch.stack(per_type)
        bases = aggregates if weights_comb is None else weights_comb.T @ aggregates
        parts = [bases.flatten()]
        if concat_own:
            parts.append(node_features[dst])
        rows.append(torch.cat(parts))
    if rows:
        return torch.stack(rows)
    num_bases = num_edge_types if weights_comb is None else weights_comb.shape[1]
    output = node_features.new_empty((0, node_features.shape[1] * (num_bases + int(concat_own))))
    output = output + node_features.sum() * 0
    return output if weights_comb is None else output + weights_comb.sum() * 0


def _check_forward_backward(graph, features, edge_types, weights, num_edge_types, concat_own, norm_by_edge_type):
    actual = agg_hg_basis_post(
        graph,
        node_features=features,
        edge_types=edge_types,
        weights_comb=weights,
        num_edge_types=num_edge_types,
        concat_own=concat_own,
        norm_by_edge_type=norm_by_edge_type,
    )
    reference_features = features.detach().cpu().requires_grad_(features.requires_grad)
    reference_weights = None if weights is None else weights.detach().cpu().requires_grad_(weights.requires_grad)
    expected = _reference(graph, reference_features, edge_types, reference_weights, num_edge_types, concat_own, norm_by_edge_type)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)
    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    expected.backward(grad_output.cpu())
    for tensor, reference_tensor in ((features, reference_features), (weights, reference_weights)):
        if tensor is not None:
            if tensor.requires_grad:
                torch.testing.assert_close(tensor.grad.cpu(), reference_tensor.grad, atol=5e-4, rtol=5e-4)
            else:
                assert tensor.grad is None


@pytest.mark.parametrize(
    "use_weights,needs_node_grad,needs_weights_grad",
    [
        pytest.param(True, True, True, id="both-gradients", marks=pytest.mark.L0),
        pytest.param(False, True, False, id="without-weights", marks=pytest.mark.L0),
        pytest.param(True, True, False, id="node-gradient-only", marks=pytest.mark.L1),
        pytest.param(True, False, True, id="weight-gradient-only", marks=pytest.mark.L1),
    ],
)
@pytest.mark.parametrize("concat_own", [False, True])
@pytest.mark.parametrize("norm_by_edge_type", [False, True])
def test_agg_hg_basis_post_forward_backward(small_csc_graph, use_weights, needs_node_grad, needs_weights_grad, concat_own, norm_by_edge_type):
    _require_gnn_agg_hg_basis_post()
    torch.manual_seed(1234)
    graph = small_csc_graph
    features = torch.randn((graph.num_src_nodes, 8), device="cuda", requires_grad=needs_node_grad)
    # The shared graph is mapped, but edge types must remain in direct CSC order.
    edge_types = torch.tensor([2, 0, 1, 1, 0], device="cuda", dtype=torch.int32)
    weights = torch.randn((4, 2), device="cuda", requires_grad=needs_weights_grad) if use_weights else None
    _check_forward_backward(graph, features, edge_types, weights, 4, concat_own, norm_by_edge_type)


@pytest.mark.L2
def test_agg_hg_basis_post_medium_graph(medium_csc_graph):
    _require_gnn_agg_hg_basis_post()
    torch.manual_seed(2345)
    graph = medium_csc_graph
    features = torch.randn((graph.num_src_nodes, 16), device="cuda", requires_grad=True)
    edge_types = torch.arange(graph.num_edges, device="cuda", dtype=torch.int32) % 3
    weights = torch.randn((4, 3), device="cuda", requires_grad=True)
    _check_forward_backward(graph, features, edge_types, weights, 4, True, True)


@pytest.mark.L0
def test_agg_hg_basis_post_usage(small_csc_graph):
    _require_gnn_agg_hg_basis_post()
    graph = small_csc_graph
    features = torch.ones((graph.num_src_nodes, 2), device="cuda")
    edge_types = torch.tensor([0, 1, 0, 1, 0], device="cuda", dtype=torch.int32)
    output = agg_hg_basis_post(graph, node_features=features, edge_types=edge_types, num_edge_types=2)
    torch.testing.assert_close(output, torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 1.0, 1.0]], device="cuda"))


@pytest.mark.L0
def test_agg_hg_basis_post_noncontiguous_inputs(small_csc_graph):
    _require_gnn_agg_hg_basis_post()
    torch.manual_seed(3456)
    graph = small_csc_graph
    features = torch.randn((8, graph.num_src_nodes), device="cuda").T.detach().requires_grad_()
    weights = torch.randn((2, 3), device="cuda").T.detach().requires_grad_()
    edge_types = torch.tensor([2, 99, 0, 99, 1, 99, 2, 99, 0, 99], device="cuda", dtype=torch.int32)[::2]
    assert not features.is_contiguous() and not weights.is_contiguous() and not edge_types.is_contiguous()
    _check_forward_backward(graph, features, edge_types, weights, 3, True, True)


@pytest.mark.L0
@pytest.mark.parametrize("use_weights", [False, True])
@pytest.mark.parametrize("concat_own", [False, True])
def test_agg_hg_basis_post_empty_destination_set(use_weights, concat_own):
    graph = CscGraph(torch.tensor([0], device="cuda", dtype=torch.int32), torch.empty(0, device="cuda", dtype=torch.int32), 3)
    features = torch.randn((3, 8), device="cuda", requires_grad=True)
    edge_types = torch.empty(0, device="cuda", dtype=torch.int32)
    weights = torch.randn((3, 2), device="cuda", requires_grad=True) if use_weights else None
    _check_forward_backward(graph, features, edge_types, weights, 3, concat_own, True)
    torch.testing.assert_close(features.grad, torch.zeros_like(features))
    if weights is not None:
        torch.testing.assert_close(weights.grad, torch.zeros_like(weights))


@pytest.mark.L0
@pytest.mark.parametrize("norm_by_edge_type", [False, True])
def test_agg_hg_basis_post_zero_degree_destination(norm_by_edge_type):
    _require_gnn_agg_hg_basis_post()
    graph = CscGraph(torch.tensor([0, 0, 2], device="cuda", dtype=torch.int32), torch.tensor([0, 2], device="cuda", dtype=torch.int32), 3)
    features = torch.randn((3, 8), device="cuda", requires_grad=True)
    edge_types = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    weights = torch.randn((3, 2), device="cuda", requires_grad=True)
    _check_forward_backward(graph, features, edge_types, weights, 3, True, norm_by_edge_type)


@pytest.mark.L0
def test_agg_hg_basis_post_more_destinations_than_sources():
    _require_gnn_agg_hg_basis_post()
    graph = CscGraph(torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int32), torch.tensor([0, 1, 0], device="cuda", dtype=torch.int32), 2)
    features = torch.randn((2, 8), device="cuda", requires_grad=True)
    edge_types = torch.tensor([0, 1, 1], device="cuda", dtype=torch.int32)
    _check_forward_backward(graph, features, edge_types, None, 2, False, True)
    with pytest.raises(ValueError, match="num_dst_nodes <= num_src_nodes"):
        agg_hg_basis_post(graph, node_features=features, edge_types=edge_types, num_edge_types=2, concat_own=True)


@pytest.mark.L0
def test_agg_hg_basis_post_rejects_invalid_inputs(small_csc_graph):
    graph = small_csc_graph
    features = torch.randn((graph.num_src_nodes, 8), device="cuda")
    edge_types = torch.zeros(graph.num_edges, device="cuda", dtype=torch.int32)
    args = dict(node_features=features, edge_types=edge_types, num_edge_types=3)
    for name, value, error, message in (
        ("node_features", features.half(), TypeError, "node_features must have dtype float32"),
        ("node_features", features[:1], ValueError, "node_features must have shape"),
        ("node_features", features[:, :0], ValueError, "positive feature dimension"),
        ("node_features", features.cpu(), ValueError, "node_features must be on"),
        ("edge_types", edge_types.long(), TypeError, "edge_types must have dtype int32"),
        ("edge_types", edge_types[:1], ValueError, "edge_types must have shape"),
        ("edge_types", edge_types.cpu(), ValueError, "edge_types must be on"),
        ("num_edge_types", 0, ValueError, "num_edge_types must be positive"),
        ("num_edge_types", True, TypeError, "num_edge_types must be an int"),
        ("concat_own", 1, TypeError, "concat_own must be a bool"),
        ("norm_by_edge_type", "true", TypeError, "norm_by_edge_type must be a bool"),
        ("weights_comb", torch.ones((2, 3), device="cuda"), ValueError, "weights_comb must have shape"),
        ("weights_comb", torch.ones((3, 0), device="cuda"), ValueError, "weights_comb must have shape"),
        ("weights_comb", torch.ones((3, 2), device="cuda", dtype=torch.float16), TypeError, "weights_comb must have dtype float32"),
    ):
        with pytest.raises(error, match=message):
            agg_hg_basis_post(graph, **(args | {name: value}))
    empty_edges = CscGraph(torch.tensor([0, 0], device="cuda", dtype=torch.int32), edge_types[:0], 4)
    with pytest.raises(ValueError, match="nonempty destination set with zero edges"):
        agg_hg_basis_post(empty_edges, **(args | {"edge_types": edge_types[:0]}))


@pytest.mark.L0
def test_agg_hg_basis_post_uses_current_stream(small_csc_graph):
    _require_gnn_agg_hg_basis_post()
    graph = small_csc_graph
    features = torch.randn((graph.num_src_nodes, 8), device="cuda", requires_grad=True)
    edge_types = torch.tensor([0, 1, 2, 0, 1], device="cuda", dtype=torch.int32)
    weights = torch.randn((3, 2), device="cuda", requires_grad=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _check_forward_backward(graph, features, edge_types, weights, 3, True, True)
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.L0
def test_agg_hg_basis_post_backward_initializes_cuda_context_on_new_thread():
    _require_gnn_agg_hg_basis_post()
    graph = CscGraph(torch.tensor([0, 2], device="cuda", dtype=torch.int32), torch.tensor([0, 1], device="cuda", dtype=torch.int32), 2)
    features = torch.randn((2, 8), device="cuda")
    edge_types = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    weights = torch.randn((2, 3), device="cuda")
    output = agg_hg_basis_post(graph, node_features=features, edge_types=edge_types, num_edge_types=2, weights_comb=weights)
    grad_output = torch.randn_like(output)
    grad_node = torch.empty_like(features)
    grad_weights = torch.empty_like(weights)
    torch.cuda.synchronize()
    errors = []

    def run_backward():
        try:
            cudnn.gnn_agg_hg_basis_post_backward(
                stream=0,
                csc_offsets=graph.offsets.data_ptr(),
                csc_indices=graph.indices.data_ptr(),
                n_src_nodes=2,
                n_dst_nodes=1,
                n_indices=2,
                idx_type=4,
                edge_types=edge_types.data_ptr(),
                num_edge_types=2,
                grad_output=grad_output.data_ptr(),
                node_features=features.data_ptr(),
                weights_comb=weights.data_ptr(),
                grad_node_features=grad_node.data_ptr(),
                grad_weights_comb=grad_weights.data_ptr(),
                node_feat_dim=8,
                num_bases=3,
                concat_own=False,
                norm_by_edge_type=False,
            )
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=run_backward)
    worker.start()
    worker.join()
    if errors:
        raise errors[0]
    torch.cuda.synchronize()
    ref_features = features.cpu().requires_grad_()
    ref_weights = weights.cpu().requires_grad_()
    _reference(graph, ref_features, edge_types, ref_weights, 2, False, False).backward(grad_output.cpu())
    torch.testing.assert_close(grad_node.cpu(), ref_features.grad, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(grad_weights.cpu(), ref_weights.grad, atol=5e-4, rtol=5e-4)


@pytest.mark.L0
def test_agg_hg_basis_post_torch_compile(small_csc_graph):
    _require_gnn_agg_hg_basis_post()
    graph = small_csc_graph
    edge_types = torch.tensor([0, 1, 2, 0, 1], device="cuda", dtype=torch.int32)

    def fn(features, weights):
        return agg_hg_basis_post(graph, node_features=features, edge_types=edge_types, num_edge_types=3, weights_comb=weights, norm_by_edge_type=True)

    features = torch.randn((graph.num_src_nodes, 8), device="cuda", requires_grad=True)
    weights = torch.randn((3, 2), device="cuda", requires_grad=True)
    compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)
    actual = compiled(features, weights)
    reference_features = features.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    expected = fn(reference_features, reference_weights)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(features.grad, reference_features.grad, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(weights.grad, reference_weights.grad, atol=5e-4, rtol=5e-4)


@pytest.mark.L1
@pytest.mark.parametrize("node_feat_dim", [1, 3, 32, 65])
def test_agg_hg_basis_post_feature_dimensions(small_csc_graph, node_feat_dim):
    _require_gnn_agg_hg_basis_post()
    graph = small_csc_graph
    features = torch.randn((graph.num_src_nodes, node_feat_dim), device="cuda", requires_grad=True)
    edge_types = torch.tensor([0, 1, 2, 0, 1], device="cuda", dtype=torch.int32)
    weights = torch.randn((3, 5), device="cuda", requires_grad=True)
    _check_forward_backward(graph, features, edge_types, weights, 3, True, True)


@pytest.mark.L1
def test_agg_hg_basis_post_opcheck(small_csc_graph):
    _require_gnn_agg_hg_basis_post()
    graph = small_csc_graph
    features = torch.randn((graph.num_src_nodes, 8), device="cuda", requires_grad=True)
    weights = torch.randn((3, 2), device="cuda", requires_grad=True)
    edge_types = torch.tensor([0, 1, 2, 0, 1], device="cuda", dtype=torch.int32)
    torch.library.opcheck(
        torch.ops.cudnn.gnn_agg_hg_basis_post_fwd.default,
        (graph.offsets, graph.indices, features, edge_types, weights, graph.num_src_nodes, 3, True, True),
    )
