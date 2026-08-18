# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import pytest
import torch

import cudnn
from cudnn.gnn import (
    CscGraph,
    agg_simple,
    agg_simple_e2n,
    agg_simple_n2n,
    agg_simple_n2n_e2n,
)


def _require_gnn_agg_simple() -> None:
    if not hasattr(cudnn, "gnn_agg_simple_forward"):
        pytest.skip("cudnn-frontend was built without cudnnGnnAggSimple declarations")
    if not cudnn.is_gnn_agg_simple_available():
        pytest.skip("the loaded cuDNN library does not export cudnnGnnAggSimple forward and backward")


@pytest.fixture
def graph_data(request):
    index_dtype = request.param
    offsets = torch.tensor([0, 2, 4, 6], device="cuda", dtype=index_dtype)
    indices = torch.tensor([0, 1, 1, 2, 2, 3], device="cuda", dtype=index_dtype)
    edge_map = torch.tensor([4, 0, 5, 1, 3, 2], device="cuda", dtype=index_dtype)
    return CscGraph(offsets, indices, num_src_nodes=4, map_csc_to_coo=edge_map)


def _reduce(values: torch.Tensor, aggr: str) -> torch.Tensor:
    if aggr == "sum":
        return values.sum(dim=0)
    if aggr == "mean":
        return values.mean(dim=0)
    if aggr == "max":
        return values.max(dim=0).values
    if aggr == "min":
        return values.min(dim=0).values
    raise AssertionError(f"unexpected aggregation {aggr}")


def _reference(
    graph: CscGraph,
    node_features: Optional[torch.Tensor],
    edge_features: Optional[torch.Tensor],
    concat_features: Optional[torch.Tensor],
    aggr: str,
) -> torch.Tensor:
    offsets = graph.offsets.cpu().tolist()
    rows = []
    for dst in range(graph.num_dst_nodes):
        begin, end = offsets[dst], offsets[dst + 1]
        parts = []
        if node_features is not None:
            source_rows = graph.indices[begin:end].long()
            parts.append(_reduce(node_features[source_rows], aggr))
        if edge_features is not None:
            if graph.map_csc_to_coo is None:
                edge_rows = torch.arange(begin, end, device=edge_features.device)
            else:
                edge_rows = graph.map_csc_to_coo[begin:end].long()
            parts.append(_reduce(edge_features[edge_rows], aggr))
        if concat_features is not None:
            parts.append(concat_features[dst])
        rows.append(torch.cat(parts))
    if rows:
        return torch.stack(rows)
    dtype_source = node_features if node_features is not None else edge_features
    assert dtype_source is not None
    concat_dim = 0 if concat_features is None else concat_features.shape[1]
    return torch.empty((0, dtype_source.shape[1] + concat_dim), device=dtype_source.device, dtype=dtype_source.dtype)


@pytest.mark.L0
@pytest.mark.parametrize("graph_data", [torch.int32, torch.int64], indirect=True)
@pytest.mark.parametrize("aggr", ["sum", "mean", "max", "min"])
@pytest.mark.parametrize("mode", ["node", "edge", "node_edge_concat"])
def test_agg_simple_forward_backward(graph_data, aggr, mode):
    _require_gnn_agg_simple()
    graph = graph_data
    torch.manual_seed(1234)

    node_features = None
    edge_features = None
    concat_features = None
    if mode in ("node", "node_edge_concat"):
        node_features = torch.randn((4, 5), device="cuda", dtype=torch.float32, requires_grad=True)
    if mode in ("edge", "node_edge_concat"):
        edge_features = torch.randn((6, 3), device="cuda", dtype=torch.float32, requires_grad=True)
    if mode == "node_edge_concat":
        concat_features = torch.randn((3, 2), device="cuda", dtype=torch.float32, requires_grad=True)

    actual = agg_simple(
        graph,
        node_features=node_features,
        edge_features=edge_features,
        concat_features=concat_features,
        aggr=aggr,
    )

    reference_inputs = []
    reference_node = None if node_features is None else node_features.detach().clone().requires_grad_()
    reference_edge = None if edge_features is None else edge_features.detach().clone().requires_grad_()
    reference_concat = None if concat_features is None else concat_features.detach().clone().requires_grad_()
    for tensor in (reference_node, reference_edge, reference_concat):
        if tensor is not None:
            reference_inputs.append(tensor)
    expected = _reference(graph, reference_node, reference_edge, reference_concat, aggr)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    actual_inputs = [tensor for tensor in (node_features, edge_features, concat_features) if tensor is not None]
    for actual_input, reference_input in zip(actual_inputs, reference_inputs):
        torch.testing.assert_close(actual_input.grad, reference_input.grad, atol=1e-4, rtol=1e-4)


@pytest.mark.L0
def test_agg_simple_concat_gradient():
    """Concat gradients are correct while the frontend initializes backend outputs."""
    _require_gnn_agg_simple()
    graph = CscGraph(
        torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32),
        torch.tensor([0, 1, 1, 2], device="cuda", dtype=torch.int32),
        num_src_nodes=3,
    )
    for aggr in ("sum", "mean", "max", "min"):
        node_features = torch.randn((3, 5), device="cuda", requires_grad=True)
        concat_features = torch.randn((2, 3), device="cuda", requires_grad=True)
        output = agg_simple(
            graph,
            node_features=node_features,
            concat_features=concat_features,
            aggr=aggr,
        )
        grad_output = torch.arange(output.numel(), device="cuda", dtype=output.dtype).reshape_as(output) + 1

        output.backward(grad_output)

        torch.testing.assert_close(concat_features.grad, grad_output[:, node_features.shape[1] :])


@pytest.mark.L1
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_agg_simple_low_precision(dtype):
    _require_gnn_agg_simple()
    graph = CscGraph(
        torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32),
        torch.tensor([0, 1, 1, 2], device="cuda", dtype=torch.int32),
        num_src_nodes=3,
    )
    features = torch.randn((3, 16), device="cuda", dtype=dtype, requires_grad=True)
    actual = agg_simple_n2n(features, graph, aggr="mean")
    expected_features = features.detach().clone().requires_grad_()
    expected = _reference(graph, expected_features, None, None, "mean")
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(features.grad, expected_features.grad, atol=2e-2, rtol=2e-2)
