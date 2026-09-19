# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import pytest
import torch
import torch.nn.functional as F
import cudnn
from cudnn.gnn import CscGraph


def _require_gnn_mha(variant: str) -> None:
    if not hasattr(cudnn, f"gnn_mha_{variant}_forward") or not hasattr(cudnn, f"gnn_mha_{variant}_backward"):
        pytest.skip(f"cudnn-frontend was built without cudnnGnnMha{variant.title().replace('_', '')} support")


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return (0.0005, 0.006)
    return (0.02, 0.06)


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


def _reference_gat(
    graph: CscGraph,
    src_features: torch.Tensor,
    dst_features: torch.Tensor,
    edge_features: Optional[torch.Tensor],
    attn_weights: torch.Tensor,
    dropout_mask: Optional[torch.Tensor],
    num_heads: int,
    concat_heads: bool,
    activation: str,
    activation_alpha: float,
):
    offsets = graph.offsets.cpu().tolist()
    dim_node = src_features.shape[1]
    dim_head = dim_node // num_heads
    dim_edge = 0 if edge_features is None else edge_features.shape[1]
    dim_edge_head = dim_edge // num_heads
    map_csc_to_coo = graph.map_csc_to_coo
    attention_rows = [None] * num_heads
    head_outputs = []
    for head in range(num_heads):
        per_dst = []
        per_edge = torch.zeros(graph.num_edges, device=src_features.device, dtype=torch.float32)
        node_slice = slice(head * dim_head, (head + 1) * dim_head)
        w_src = attn_weights[node_slice]
        w_dst = attn_weights[dim_node + node_slice.start : dim_node + node_slice.stop]
        w_edge = attn_weights[2 * dim_node + head * dim_edge_head : 2 * dim_node + (head + 1) * dim_edge_head]
        for dst in range(graph.num_dst_nodes):
            begin, end = (offsets[dst], offsets[dst + 1])
            source_rows = graph.indices[begin:end].long()
            edge_positions = torch.arange(begin, end, device=src_features.device)
            edge_rows = edge_positions if map_csc_to_coo is None else map_csc_to_coo[begin:end].long()
            source = src_features[source_rows, node_slice]
            destination = dst_features[dst, node_slice]
            logits = source @ w_src + destination @ w_dst
            if edge_features is not None:
                edge_slice = slice(head * dim_edge_head, (head + 1) * dim_edge_head)
                logits = logits + edge_features[edge_rows, edge_slice] @ w_edge
            logits = _activation(logits, activation, activation_alpha)
            coefficients = torch.softmax(logits.float(), dim=0)
            post_dropout = coefficients if dropout_mask is None else coefficients * dropout_mask[head, edge_rows]
            per_edge = per_edge.index_copy(0, edge_rows, post_dropout)
            per_dst.append((post_dropout.to(source.dtype).unsqueeze(1) * source).sum(dim=0))
        attention_rows[head] = per_edge
        head_outputs.append(torch.stack(per_dst))
    attention = torch.stack(attention_rows)
    output = torch.cat(head_outputs, dim=1) if concat_heads else torch.stack(head_outputs).mean(dim=0)
    return (output, attention)


def _reference_gat_v2(
    graph: CscGraph,
    src_features: torch.Tensor,
    dst_features: torch.Tensor,
    edge_features: Optional[torch.Tensor],
    attn_weights: torch.Tensor,
    dropout_mask: Optional[torch.Tensor],
    num_heads: int,
    concat_heads: bool,
    activation: str,
    activation_alpha: float,
):
    offsets = graph.offsets.cpu().tolist()
    dim_node = src_features.shape[1]
    dim_head = dim_node // num_heads
    dim_edge = 0 if edge_features is None else edge_features.shape[1]
    dim_edge_head = dim_edge // num_heads
    map_csc_to_coo = graph.map_csc_to_coo
    attention_rows = [None] * num_heads
    head_outputs = []
    for head in range(num_heads):
        per_dst = []
        per_edge = torch.zeros(graph.num_edges, device=src_features.device, dtype=torch.float32)
        node_slice = slice(head * dim_head, (head + 1) * dim_head)
        weight = attn_weights[node_slice]
        for dst in range(graph.num_dst_nodes):
            begin, end = (offsets[dst], offsets[dst + 1])
            source_rows = graph.indices[begin:end].long()
            edge_positions = torch.arange(begin, end, device=src_features.device)
            edge_rows = edge_positions if map_csc_to_coo is None else map_csc_to_coo[begin:end].long()
            source = src_features[source_rows, node_slice]
            destination = dst_features[dst, node_slice]
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
    return (output, attention)
