# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
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
def test_agg_simple_uses_current_stream():
    _require_gnn_agg_simple()
    graph = CscGraph(
        torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32),
        torch.tensor([0, 1, 1, 2], device="cuda", dtype=torch.int32),
        num_src_nodes=3,
    )
    features = torch.randn((3, 8), device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = agg_simple_n2n(features, graph)
    stream.synchronize()
    torch.testing.assert_close(actual, _reference(graph, features, None, None, "sum"))


@pytest.mark.L0
def test_agg_simple_backward_initializes_cuda_context_on_new_thread():
    _require_gnn_agg_simple()
    offsets = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
    indices = torch.tensor([0, 1, 1, 2], device="cuda", dtype=torch.int32)
    grad_output = torch.arange(6, device="cuda", dtype=torch.float32).reshape(2, 3)
    grad_edge = torch.empty((4, 3), device="cuda", dtype=torch.float32)
    backward_args = (
        0,
        offsets.data_ptr(),
        indices.data_ptr(),
        0,
        3,
        2,
        4,
        4,  # CUDNN_DATA_INT32
        grad_output.data_ptr(),
        0,
        0,
        grad_edge.data_ptr(),
        0,
        0,
        3,
        0,
        0,  # CUDNN_DATA_FLOAT
        0,  # CUDNN_GNN_AGG_SUM
    )
    errors = []

    def run_backward():
        try:
            # Make no CUDA calls on this worker before entering the binding.
            cudnn.gnn_agg_simple_backward(*backward_args)
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=run_backward)
    worker.start()
    worker.join()

    if errors:
        raise errors[0]
    torch.testing.assert_close(grad_edge, grad_output.repeat_interleave(2, dim=0))


@pytest.mark.L0
def test_agg_simple_torch_compile():
    _require_gnn_agg_simple()
    graph = CscGraph(
        torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32),
        torch.tensor([0, 1, 1, 2], device="cuda", dtype=torch.int32),
        num_src_nodes=3,
    )

    def fn(features):
        return agg_simple_n2n(features, graph, aggr="sum")

    features = torch.randn((3, 4), device="cuda")
    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(features), fn(features))
