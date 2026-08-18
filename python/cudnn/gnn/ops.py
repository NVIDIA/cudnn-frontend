# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch custom-op registration and autograd for GNN simple aggregation."""

from typing import Optional, Tuple

import torch
from torch import Tensor

from .backend.cudnn import backward, forward, _validate_aggregation, _validate_graph, _validate_inputs

_lib = torch.library.Library("cudnn", "FRAGMENT")
_lib.define(
    "gnn_agg_simple_fwd(Tensor offsets, Tensor indices, Tensor? map_csc_to_coo, Tensor? node_features, "
    "Tensor? edge_features, Tensor? concat_features, int num_src_nodes, str aggr) -> (Tensor, Tensor)"
)
_lib.define(
    "gnn_agg_simple_bwd(Tensor grad_output, Tensor offsets, Tensor indices, Tensor? map_csc_to_coo, Tensor out_positions, "
    "int num_src_nodes, int node_feat_dim, int edge_feat_dim, int concat_feat_dim, str aggr) -> (Tensor, Tensor, Tensor)"
)
_lib.impl("gnn_agg_simple_fwd", forward, "CUDA")
_lib.impl("gnn_agg_simple_bwd", backward, "CUDA")


@torch.library.register_fake("cudnn::gnn_agg_simple_fwd")
def _fwd_fake(
    offsets: Tensor,
    indices: Tensor,
    map_csc_to_coo: Optional[Tensor],
    node_features: Optional[Tensor],
    edge_features: Optional[Tensor],
    concat_features: Optional[Tensor],
    num_src_nodes: int,
    aggr: str,
) -> Tuple[Tensor, Tensor]:
    num_dst_nodes, _, dtype, node_feat_dim, edge_feat_dim, concat_feat_dim = _validate_inputs(
        offsets, indices, map_csc_to_coo, node_features, edge_features, concat_features, num_src_nodes, aggr
    )
    output = torch.empty((num_dst_nodes, node_feat_dim + edge_feat_dim + concat_feat_dim), device=offsets.device, dtype=dtype)
    positions_shape = (num_dst_nodes, node_feat_dim + edge_feat_dim) if aggr in ("max", "min") else (0,)
    return output, torch.empty(positions_shape, device=offsets.device, dtype=offsets.dtype)


@torch.library.register_fake("cudnn::gnn_agg_simple_bwd")
def _bwd_fake(
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
    del map_csc_to_coo, out_positions, aggr
    num_dst_nodes = offsets.numel() - 1
    num_edges = indices.numel()
    return (
        torch.empty((num_src_nodes, node_feat_dim), device=grad_output.device, dtype=grad_output.dtype),
        torch.empty((num_edges, edge_feat_dim), device=grad_output.device, dtype=grad_output.dtype),
        torch.empty((num_dst_nodes, concat_feat_dim), device=grad_output.device, dtype=grad_output.dtype),
    )


def _setup_context(ctx, inputs, output) -> None:
    offsets, indices, map_csc_to_coo, node_features, edge_features, concat_features, num_src_nodes, aggr = inputs
    _, out_positions = output
    ctx.save_for_backward(offsets, indices, map_csc_to_coo, out_positions)
    ctx.mark_non_differentiable(out_positions)
    ctx.num_src_nodes = num_src_nodes
    ctx.node_feat_dim = 0 if node_features is None else node_features.shape[1]
    ctx.edge_feat_dim = 0 if edge_features is None else edge_features.shape[1]
    ctx.concat_feat_dim = 0 if concat_features is None else concat_features.shape[1]
    ctx.has_node_features = node_features is not None
    ctx.has_edge_features = edge_features is not None
    ctx.has_concat_features = concat_features is not None
    ctx.aggr = aggr


@torch.compiler.allow_in_graph
def _autograd_backward(ctx, grad_output: Tensor, grad_out_positions: Optional[Tensor]):
    del grad_out_positions
    offsets, indices, map_csc_to_coo, out_positions = ctx.saved_tensors
    grad_node, grad_edge, grad_concat = torch.ops.cudnn.gnn_agg_simple_bwd(
        grad_output,
        offsets,
        indices,
        map_csc_to_coo,
        out_positions,
        ctx.num_src_nodes,
        ctx.node_feat_dim,
        ctx.edge_feat_dim,
        ctx.concat_feat_dim,
        ctx.aggr,
    )
    return (
        None,
        None,
        None,
        grad_node if ctx.has_node_features else None,
        grad_edge if ctx.has_edge_features else None,
        grad_concat if ctx.has_concat_features else None,
        None,
        None,
    )


torch.library.register_autograd("cudnn::gnn_agg_simple_fwd", _autograd_backward, setup_context=_setup_context)
