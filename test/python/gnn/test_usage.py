# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


@pytest.mark.L0
def test_agg_simple_usage():
    _require_gnn_agg_simple()
    graph = CscGraph(
        torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32),
        torch.tensor([0, 1, 1, 2], device="cuda", dtype=torch.int32),
        num_src_nodes=3,
    )
    features = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4).requires_grad_()

    output = agg_simple(graph, node_features=features, aggr="mean")

    expected = torch.stack(((features[0] + features[1]) / 2, (features[1] + features[2]) / 2))
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    torch.testing.assert_close(
        features.grad,
        torch.tensor([[0.5] * 4, [1.0] * 4, [0.5] * 4], device="cuda"),
    )


@pytest.mark.L0
def test_agg_simple_compatibility_wrappers():
    _require_gnn_agg_simple()
    graph = CscGraph(
        torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32),
        torch.tensor([0, 1, 1, 2], device="cuda", dtype=torch.int32),
        num_src_nodes=3,
    )
    node = torch.randn((3, 4), device="cuda")
    edge = torch.randn((4, 2), device="cuda")
    torch.testing.assert_close(agg_simple_n2n(node, graph), agg_simple(graph, node_features=node))
    torch.testing.assert_close(agg_simple_e2n(edge, graph), agg_simple(graph, edge_features=edge))
    torch.testing.assert_close(agg_simple_n2n_e2n(node, edge, graph), agg_simple(graph, node_features=node, edge_features=edge))


@pytest.mark.L0
def test_agg_simple_empty_destination_set():
    graph = CscGraph(
        torch.tensor([0], device="cuda", dtype=torch.int32),
        torch.empty((0,), device="cuda", dtype=torch.int32),
        num_src_nodes=3,
    )
    features = torch.randn((3, 4), device="cuda", requires_grad=True)
    output = agg_simple_n2n(features, graph)
    assert output.shape == (0, 4)
    output.sum().backward()
    torch.testing.assert_close(features.grad, torch.zeros_like(features))


@pytest.mark.L0
def test_agg_simple_rejects_invalid_inputs():
    offsets = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    graph = CscGraph(offsets, indices, num_src_nodes=1)
    features = torch.randn((1, 2), device="cuda")

    with pytest.raises(ValueError, match="Unsupported aggregation"):
        agg_simple_n2n(features, graph, aggr="product")
    with pytest.raises(TypeError, match="indices dtype"):
        agg_simple_n2n(features, CscGraph(offsets, indices.to(torch.int64), num_src_nodes=1))
    with pytest.raises(ValueError, match="zero edges"):
        agg_simple_n2n(
            features,
            CscGraph(torch.tensor([0, 0], device="cuda", dtype=torch.int32), torch.empty(0, device="cuda", dtype=torch.int32), 1),
        )
