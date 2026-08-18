# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN backend implementation for GNN simple aggregation."""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from ..api import CscGraph

_AGGREGATION_TO_INT: Dict[str, int] = {
    "sum": 0,
    "mean": 1,
    "max": 2,
    "min": 3,
}

_TORCH_DTYPE_TO_CUDNN: Dict[torch.dtype, int] = {
    torch.float32: 0,  # CUDNN_DATA_FLOAT
    torch.float16: 2,  # CUDNN_DATA_HALF
    torch.bfloat16: 9,  # CUDNN_DATA_BFLOAT16
}

_TORCH_INDEX_DTYPE_TO_CUDNN: Dict[torch.dtype, int] = {
    torch.int32: 4,  # CUDNN_DATA_INT32
    torch.int64: 10,  # CUDNN_DATA_INT64
}


def _validate_aggregation(aggr: str) -> None:
    if aggr not in _AGGREGATION_TO_INT:
        supported = ", ".join(_AGGREGATION_TO_INT)
        raise ValueError(f"Unsupported aggregation '{aggr}'. Supported: {supported}.")


def _validate_graph(
    offsets: Tensor,
    indices: Tensor,
    map_csc_to_coo: Optional[Tensor],
    num_src_nodes: int,
) -> Tuple[int, int]:
    if offsets.ndim != 1:
        raise ValueError(f"offsets must be rank 1, got shape {tuple(offsets.shape)}")
    if indices.ndim != 1:
        raise ValueError(f"indices must be rank 1, got shape {tuple(indices.shape)}")
    if offsets.numel() == 0:
        raise ValueError("offsets must contain at least one element")
    if offsets.dtype not in _TORCH_INDEX_DTYPE_TO_CUDNN:
        raise TypeError(f"offsets must have dtype int32 or int64, got {offsets.dtype}")
    if indices.dtype != offsets.dtype:
        raise TypeError(f"indices dtype {indices.dtype} must match offsets dtype {offsets.dtype}")
    if not offsets.is_cuda or not indices.is_cuda:
        raise ValueError(f"offsets and indices must be CUDA tensors, got {offsets.device} and {indices.device}")
    if offsets.device != indices.device:
        raise ValueError(f"offsets and indices must be on the same device, got {offsets.device} and {indices.device}")
    if not isinstance(num_src_nodes, int):
        raise TypeError(f"num_src_nodes must be an int, got {type(num_src_nodes).__name__}")
    if num_src_nodes <= 0:
        raise ValueError(f"num_src_nodes must be positive, got {num_src_nodes}")

    num_dst_nodes = offsets.numel() - 1
    num_edges = indices.numel()
    if num_dst_nodes == 0 and num_edges != 0:
        raise ValueError("a graph with zero destination nodes cannot contain edges")
    if num_dst_nodes > 0 and num_edges == 0:
        raise ValueError("cuDNN AggSimple does not yet support a nonempty destination set with zero edges")

    if map_csc_to_coo is not None:
        if map_csc_to_coo.ndim != 1 or map_csc_to_coo.numel() != num_edges:
            raise ValueError(f"map_csc_to_coo must have shape ({num_edges},), got {tuple(map_csc_to_coo.shape)}")
        if map_csc_to_coo.dtype != offsets.dtype:
            raise TypeError(f"map_csc_to_coo dtype {map_csc_to_coo.dtype} must match offsets dtype {offsets.dtype}")
        if map_csc_to_coo.device != offsets.device:
            raise ValueError(f"map_csc_to_coo must be on {offsets.device}, got {map_csc_to_coo.device}")

    return num_dst_nodes, num_edges


def _validate_features(
    offsets: Tensor,
    num_src_nodes: int,
    num_dst_nodes: int,
    num_edges: int,
    node_features: Optional[Tensor],
    edge_features: Optional[Tensor],
    concat_features: Optional[Tensor],
) -> Tuple[torch.dtype, int, int, int]:
    if node_features is None and edge_features is None:
        raise ValueError("at least one of node_features or edge_features must be provided")

    feature_tensors = [tensor for tensor in (node_features, edge_features, concat_features) if tensor is not None]
    dtype = feature_tensors[0].dtype
    if dtype not in _TORCH_DTYPE_TO_CUDNN:
        raise TypeError(f"feature tensors must have dtype float32, float16, or bfloat16, got {dtype}")

    for tensor in feature_tensors:
        if tensor.ndim != 2:
            raise ValueError(f"feature tensors must be rank 2, got shape {tuple(tensor.shape)}")
        if not tensor.is_cuda:
            raise ValueError(f"feature tensors must be CUDA tensors, got {tensor.device}")
        if tensor.device != offsets.device:
            raise ValueError(f"all graph and feature tensors must be on {offsets.device}, got {tensor.device}")
        if tensor.dtype != dtype:
            raise TypeError(f"all feature tensors must have dtype {dtype}, got {tensor.dtype}")

    node_feat_dim = 0
    if node_features is not None:
        if node_features.shape[0] != num_src_nodes:
            raise ValueError(f"node_features must have {num_src_nodes} rows, got {node_features.shape[0]}")
        node_feat_dim = node_features.shape[1]

    edge_feat_dim = 0
    if edge_features is not None:
        if edge_features.shape[0] != num_edges:
            raise ValueError(f"edge_features must have {num_edges} rows, got {edge_features.shape[0]}")
        edge_feat_dim = edge_features.shape[1]

    concat_feat_dim = 0
    if concat_features is not None:
        if concat_features.shape[0] != num_dst_nodes:
            raise ValueError(f"concat_features must have {num_dst_nodes} rows, got {concat_features.shape[0]}")
        concat_feat_dim = concat_features.shape[1]

    if node_feat_dim == 0 and edge_feat_dim == 0:
        raise ValueError("node_features and edge_features cannot both have zero feature dimension")

    return dtype, node_feat_dim, edge_feat_dim, concat_feat_dim


def _validate_inputs(
    offsets: Tensor,
    indices: Tensor,
    map_csc_to_coo: Optional[Tensor],
    node_features: Optional[Tensor],
    edge_features: Optional[Tensor],
    concat_features: Optional[Tensor],
    num_src_nodes: int,
    aggr: str,
) -> Tuple[int, int, torch.dtype, int, int, int]:
    _validate_aggregation(aggr)
    num_dst_nodes, num_edges = _validate_graph(offsets, indices, map_csc_to_coo, num_src_nodes)
    dtype, node_feat_dim, edge_feat_dim, concat_feat_dim = _validate_features(
        offsets, num_src_nodes, num_dst_nodes, num_edges, node_features, edge_features, concat_features
    )
    return num_dst_nodes, num_edges, dtype, node_feat_dim, edge_feat_dim, concat_feat_dim


def _pointer(tensor: Optional[Tensor]) -> int:
    return 0 if tensor is None else tensor.data_ptr()


def _require_backend() -> None:
    import cudnn

    if not hasattr(cudnn, "gnn_agg_simple_forward") or not cudnn.is_gnn_agg_simple_available():
        raise RuntimeError(
            "cudnnGnnAggSimpleForward/Backward are unavailable. Build cudnn-frontend against headers containing the GNN API "
            "and load a cuDNN library exporting both symbols."
        )


def forward(
    offsets: Tensor,
    indices: Tensor,
    map_csc_to_coo: Optional[Tensor],
    node_features: Optional[Tensor],
    edge_features: Optional[Tensor],
    concat_features: Optional[Tensor],
    num_src_nodes: int,
    aggr: str,
) -> Tuple[Tensor, Tensor]:
    num_dst_nodes, num_edges, dtype, node_feat_dim, edge_feat_dim, concat_feat_dim = _validate_inputs(
        offsets, indices, map_csc_to_coo, node_features, edge_features, concat_features, num_src_nodes, aggr
    )

    offsets = offsets.contiguous()
    indices = indices.contiguous()
    map_csc_to_coo = None if map_csc_to_coo is None else map_csc_to_coo.contiguous()
    node_features = None if node_features is None else node_features.contiguous()
    edge_features = None if edge_features is None else edge_features.contiguous()
    concat_features = None if concat_features is None else concat_features.contiguous()

    output = torch.empty((num_dst_nodes, node_feat_dim + edge_feat_dim + concat_feat_dim), device=offsets.device, dtype=dtype)
    if aggr in ("max", "min"):
        out_positions = torch.empty((num_dst_nodes, node_feat_dim + edge_feat_dim), device=offsets.device, dtype=offsets.dtype)
    else:
        out_positions = torch.empty((0,), device=offsets.device, dtype=offsets.dtype)

    if num_dst_nodes == 0:
        return output, out_positions

    _require_backend()
    import cudnn

    cudnn.gnn_agg_simple_forward(
        torch.cuda.current_stream(offsets.device).cuda_stream,
        offsets.data_ptr(),
        indices.data_ptr(),
        _pointer(map_csc_to_coo),
        num_src_nodes,
        num_dst_nodes,
        num_edges,
        _TORCH_INDEX_DTYPE_TO_CUDNN[offsets.dtype],
        _pointer(node_features),
        _pointer(edge_features),
        _pointer(concat_features),
        output.data_ptr(),
        out_positions.data_ptr() if out_positions.numel() else 0,
        node_feat_dim,
        edge_feat_dim,
        concat_feat_dim,
        _TORCH_DTYPE_TO_CUDNN[dtype],
        _AGGREGATION_TO_INT[aggr],
    )
    return output, out_positions


def backward(
    grad_output: Tensor,
    offsets: Tensor,
    indices: Tensor,
    map_csc_to_coo: Optional[Tensor],
    out_positions: Tensor,
    num_src_nodes: int,
    node_feat_dim: int,
    edge_feat_dim: int,
    concat_feat_dim: int,
    aggr: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    _validate_aggregation(aggr)
    num_dst_nodes, num_edges = _validate_graph(offsets, indices, map_csc_to_coo, num_src_nodes)
    expected_shape = (num_dst_nodes, node_feat_dim + edge_feat_dim + concat_feat_dim)
    if tuple(grad_output.shape) != expected_shape:
        raise ValueError(f"grad_output must have shape {expected_shape}, got {tuple(grad_output.shape)}")
    if grad_output.dtype not in _TORCH_DTYPE_TO_CUDNN or not grad_output.is_cuda:
        raise TypeError(f"grad_output must be a CUDA float32, float16, or bfloat16 tensor, got {grad_output.dtype}")
    if grad_output.device != offsets.device:
        raise ValueError(f"grad_output must be on {offsets.device}, got {grad_output.device}")

    if aggr in ("max", "min"):
        expected_positions_shape = (num_dst_nodes, node_feat_dim + edge_feat_dim)
        if tuple(out_positions.shape) != expected_positions_shape or out_positions.dtype != offsets.dtype:
            raise ValueError(
                f"out_positions must have shape {expected_positions_shape} and dtype {offsets.dtype}, "
                f"got {tuple(out_positions.shape)} and {out_positions.dtype}"
            )

    offsets = offsets.contiguous()
    indices = indices.contiguous()
    map_csc_to_coo = None if map_csc_to_coo is None else map_csc_to_coo.contiguous()
    grad_output = grad_output.contiguous()
    out_positions = out_positions.contiguous()

    if num_dst_nodes == 0:
        return (
            torch.zeros((num_src_nodes, node_feat_dim), device=offsets.device, dtype=grad_output.dtype),
            torch.empty((num_edges, edge_feat_dim), device=offsets.device, dtype=grad_output.dtype),
            torch.empty((num_dst_nodes, concat_feat_dim), device=offsets.device, dtype=grad_output.dtype),
        )

    # TODO: Use torch.empty once cuDNN AggSimple backward initializes every
    # gradient output buffer before accumulating into it.
    grad_node = torch.zeros((num_src_nodes, node_feat_dim), device=offsets.device, dtype=grad_output.dtype)
    grad_edge = torch.zeros((num_edges, edge_feat_dim), device=offsets.device, dtype=grad_output.dtype)
    grad_concat = torch.zeros((num_dst_nodes, concat_feat_dim), device=offsets.device, dtype=grad_output.dtype)

    _require_backend()
    import cudnn

    cudnn.gnn_agg_simple_backward(
        torch.cuda.current_stream(offsets.device).cuda_stream,
        offsets.data_ptr(),
        indices.data_ptr(),
        _pointer(map_csc_to_coo),
        num_src_nodes,
        num_dst_nodes,
        num_edges,
        _TORCH_INDEX_DTYPE_TO_CUDNN[offsets.dtype],
        grad_output.data_ptr(),
        out_positions.data_ptr() if out_positions.numel() else 0,
        grad_node.data_ptr() if node_feat_dim else 0,
        grad_edge.data_ptr() if edge_feat_dim else 0,
        grad_concat.data_ptr() if concat_feat_dim else 0,
        node_feat_dim,
        edge_feat_dim,
        concat_feat_dim,
        _TORCH_DTYPE_TO_CUDNN[grad_output.dtype],
        _AGGREGATION_TO_INT[aggr],
    )
    return grad_node, grad_edge, grad_concat
