# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python API and Torch custom-op implementation for GNN heterogeneous basis aggregation."""

from typing import Optional, Tuple

import torch
from torch import Tensor

from ._dtypes import TORCH_INDEX_DTYPE_TO_CUDNN
from ._utils import require_backend_symbols, tensor_pointer, validate_csc_graph
from .graph import CscGraph

# Torch custom-op implementation


def _validate_graph(offsets: Tensor, indices: Tensor, num_src_nodes: int) -> Tuple[int, int]:
    num_dst_nodes, num_edges = validate_csc_graph(offsets, indices, None, num_src_nodes)
    if num_dst_nodes > 0 and num_edges == 0:
        raise ValueError("cuDNN AggHgBasisPost does not support a nonempty destination set with zero edges")
    int_max = torch.iinfo(torch.int32).max
    if num_dst_nodes > int_max:
        raise ValueError("num_dst_nodes exceeds the backend INT32 limit")
    if offsets.dtype == torch.int32 and max(num_src_nodes, num_dst_nodes, num_edges) > int_max:
        raise ValueError("graph counts exceed the INT32 index range")
    return num_dst_nodes, num_edges


def _validate_options(num_edge_types: int, concat_own: bool, norm_by_edge_type: bool) -> None:
    if not isinstance(num_edge_types, int) or isinstance(num_edge_types, bool):
        raise TypeError("num_edge_types must be an int")
    if not 0 < num_edge_types <= torch.iinfo(torch.int32).max:
        raise ValueError("num_edge_types must be positive and fit in INT32")
    for name, flag in (("concat_own", concat_own), ("norm_by_edge_type", norm_by_edge_type)):
        if not isinstance(flag, bool):
            raise TypeError(f"{name} must be a bool")


def _validate_features(
    offsets: Tensor,
    node_features: Tensor,
    edge_types: Tensor,
    weights_comb: Optional[Tensor],
    num_src_nodes: int,
    num_dst_nodes: int,
    num_edges: int,
    num_edge_types: int,
    concat_own: bool,
    norm_by_edge_type: bool,
) -> Tuple[int, int]:
    _validate_options(num_edge_types, concat_own, norm_by_edge_type)
    if node_features.ndim != 2 or node_features.shape[0] != num_src_nodes:
        raise ValueError(f"node_features must have shape ({num_src_nodes}, node_feat_dim), got {tuple(node_features.shape)}")
    node_feat_dim = node_features.shape[1]
    if node_feat_dim <= 0:
        raise ValueError("node_features must have a positive feature dimension")
    if node_features.dtype != torch.float32:
        raise TypeError(f"node_features must have dtype float32, got {node_features.dtype}")
    if node_features.device != offsets.device:
        raise ValueError(f"node_features must be on {offsets.device}, got {node_features.device}")
    if tuple(edge_types.shape) != (num_edges,):
        raise ValueError(f"edge_types must have shape ({num_edges},), got {tuple(edge_types.shape)}")
    if edge_types.dtype != torch.int32:
        raise TypeError(f"edge_types must have dtype int32, got {edge_types.dtype}")
    if edge_types.device != offsets.device:
        raise ValueError(f"edge_types must be on {offsets.device}, got {edge_types.device}")
    num_bases = num_edge_types
    if weights_comb is not None:
        if weights_comb.ndim != 2 or weights_comb.shape[0] != num_edge_types or weights_comb.shape[1] <= 0:
            raise ValueError(f"weights_comb must have shape ({num_edge_types}, num_bases) with positive num_bases")
        if weights_comb.dtype != torch.float32:
            raise TypeError(f"weights_comb must have dtype float32, got {weights_comb.dtype}")
        if weights_comb.device != offsets.device:
            raise ValueError(f"weights_comb must be on {offsets.device}, got {weights_comb.device}")
        num_bases = weights_comb.shape[1]
    if concat_own and num_dst_nodes > num_src_nodes:
        raise ValueError("concat_own=True requires num_dst_nodes <= num_src_nodes")
    if node_feat_dim * (num_bases + int(concat_own)) >= torch.iinfo(torch.int32).max:
        raise ValueError("output feature dimension must be smaller than the backend INT32 limit")
    return node_feat_dim, num_bases


def _validate_inputs(
    offsets: Tensor,
    indices: Tensor,
    node_features: Tensor,
    edge_types: Tensor,
    weights_comb: Optional[Tensor],
    num_src_nodes: int,
    num_edge_types: int,
    concat_own: bool,
    norm_by_edge_type: bool,
) -> Tuple[int, int, int]:
    num_dst_nodes, num_edges = _validate_graph(offsets, indices, num_src_nodes)
    node_feat_dim, num_bases = _validate_features(
        offsets, node_features, edge_types, weights_comb, num_src_nodes, num_dst_nodes, num_edges, num_edge_types, concat_own, norm_by_edge_type
    )
    return num_dst_nodes, node_feat_dim, num_bases


def _require_backend() -> None:
    require_backend_symbols("gnn_agg_hg_basis_post_forward", "gnn_agg_hg_basis_post_backward")


def forward(
    offsets: Tensor,
    indices: Tensor,
    node_features: Tensor,
    edge_types: Tensor,
    weights_comb: Optional[Tensor],
    num_src_nodes: int,
    num_edge_types: int,
    concat_own: bool,
    norm_by_edge_type: bool,
) -> Tensor:
    num_dst_nodes, node_feat_dim, num_bases = _validate_inputs(
        offsets, indices, node_features, edge_types, weights_comb, num_src_nodes, num_edge_types, concat_own, norm_by_edge_type
    )
    output = torch.empty((num_dst_nodes, node_feat_dim * (num_bases + int(concat_own))), device=offsets.device, dtype=node_features.dtype)
    if num_dst_nodes == 0:
        return output

    _require_backend()
    import cudnn

    offsets = offsets.contiguous()
    indices = indices.contiguous()
    node_features = node_features.contiguous()
    edge_types = edge_types.contiguous()
    weights_comb = None if weights_comb is None else weights_comb.contiguous()
    with torch.cuda.device(offsets.device):
        cudnn.gnn_agg_hg_basis_post_forward(
            torch.cuda.current_stream(offsets.device).cuda_stream,
            offsets.data_ptr(),
            indices.data_ptr(),
            num_src_nodes,
            num_dst_nodes,
            indices.numel(),
            TORCH_INDEX_DTYPE_TO_CUDNN[offsets.dtype],
            edge_types.data_ptr(),
            num_edge_types,
            node_features.data_ptr(),
            tensor_pointer(weights_comb),
            output.data_ptr(),
            node_feat_dim,
            num_bases,
            concat_own,
            norm_by_edge_type,
        )
    return output


def backward(
    grad_output: Tensor,
    offsets: Tensor,
    indices: Tensor,
    node_features: Tensor,
    edge_types: Tensor,
    weights_comb: Optional[Tensor],
    num_src_nodes: int,
    num_edge_types: int,
    concat_own: bool,
    norm_by_edge_type: bool,
    needs_node_grad: bool,
    needs_weights_grad: bool,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    num_dst_nodes, node_feat_dim, num_bases = _validate_inputs(
        offsets, indices, node_features, edge_types, weights_comb, num_src_nodes, num_edge_types, concat_own, norm_by_edge_type
    )
    expected_shape = (num_dst_nodes, node_feat_dim * (num_bases + int(concat_own)))
    if tuple(grad_output.shape) != expected_shape:
        raise ValueError(f"grad_output must have shape {expected_shape}, got {tuple(grad_output.shape)}")
    if grad_output.dtype != torch.float32:
        raise TypeError(f"grad_output must have dtype float32, got {grad_output.dtype}")
    if grad_output.device != offsets.device:
        raise ValueError(f"grad_output must be on {offsets.device}, got {grad_output.device}")
    if needs_weights_grad and weights_comb is None:
        raise ValueError("weight gradients require weights_comb")
    grad_node = torch.empty_like(node_features, memory_format=torch.contiguous_format) if needs_node_grad else None
    grad_weights = torch.empty_like(weights_comb, memory_format=torch.contiguous_format) if needs_weights_grad else None
    if num_dst_nodes == 0:
        if grad_node is not None:
            grad_node.zero_()
        if grad_weights is not None:
            grad_weights.zero_()
        return grad_node, grad_weights
    if not needs_node_grad and not needs_weights_grad:
        return None, None

    _require_backend()
    import cudnn

    offsets = offsets.contiguous()
    indices = indices.contiguous()
    node_features = node_features.contiguous()
    edge_types = edge_types.contiguous()
    weights_comb = None if weights_comb is None else weights_comb.contiguous()
    grad_output = grad_output.contiguous()
    with torch.cuda.device(offsets.device):
        cudnn.gnn_agg_hg_basis_post_backward(
            torch.cuda.current_stream(offsets.device).cuda_stream,
            offsets.data_ptr(),
            indices.data_ptr(),
            num_src_nodes,
            num_dst_nodes,
            indices.numel(),
            TORCH_INDEX_DTYPE_TO_CUDNN[offsets.dtype],
            edge_types.data_ptr(),
            num_edge_types,
            grad_output.data_ptr(),
            node_features.data_ptr(),
            tensor_pointer(weights_comb),
            tensor_pointer(grad_node),
            tensor_pointer(grad_weights),
            node_feat_dim,
            num_bases,
            concat_own,
            norm_by_edge_type,
        )
    return grad_node, grad_weights


_lib = torch.library.Library("cudnn", "FRAGMENT")
_lib.define(
    "gnn_agg_hg_basis_post_fwd(Tensor offsets, Tensor indices, Tensor node_features, Tensor edge_types, "
    "Tensor? weights_comb, int num_src_nodes, int num_edge_types, bool concat_own, bool norm_by_edge_type) -> Tensor"
)
_lib.define(
    "gnn_agg_hg_basis_post_bwd(Tensor grad_output, Tensor offsets, Tensor indices, Tensor node_features, "
    "Tensor edge_types, Tensor? weights_comb, int num_src_nodes, int num_edge_types, bool concat_own, "
    "bool norm_by_edge_type, bool needs_node_grad, bool needs_weights_grad) -> (Tensor?, Tensor?)"
)
_lib.impl("gnn_agg_hg_basis_post_fwd", forward, "CUDA")
_lib.impl("gnn_agg_hg_basis_post_bwd", backward, "CUDA")


@torch.library.register_fake("cudnn::gnn_agg_hg_basis_post_fwd")
def _fwd_fake(offsets, indices, node_features, edge_types, weights_comb, num_src_nodes, num_edge_types, concat_own, norm_by_edge_type):
    num_dst_nodes, node_feat_dim, num_bases = _validate_inputs(
        offsets, indices, node_features, edge_types, weights_comb, num_src_nodes, num_edge_types, concat_own, norm_by_edge_type
    )
    return torch.empty((num_dst_nodes, node_feat_dim * (num_bases + int(concat_own))), device=offsets.device, dtype=node_features.dtype)


@torch.library.register_fake("cudnn::gnn_agg_hg_basis_post_bwd")
def _bwd_fake(
    grad_output,
    offsets,
    indices,
    node_features,
    edge_types,
    weights_comb,
    num_src_nodes,
    num_edge_types,
    concat_own,
    norm_by_edge_type,
    needs_node_grad,
    needs_weights_grad,
):
    del grad_output, offsets, indices, edge_types, num_src_nodes, num_edge_types, concat_own, norm_by_edge_type
    return (
        torch.empty_like(node_features, memory_format=torch.contiguous_format) if needs_node_grad else None,
        torch.empty_like(weights_comb, memory_format=torch.contiguous_format) if needs_weights_grad else None,
    )


def _setup_context(ctx, inputs, output) -> None:
    del output
    offsets, indices, node_features, edge_types, weights_comb, num_src_nodes, num_edge_types, concat_own, norm_by_edge_type = inputs
    ctx.save_for_backward(offsets, indices, node_features, edge_types, weights_comb)
    ctx.num_src_nodes = num_src_nodes
    ctx.num_edge_types = num_edge_types
    ctx.concat_own = concat_own
    ctx.norm_by_edge_type = norm_by_edge_type


@torch.compiler.allow_in_graph
def _autograd_backward(ctx, grad_output: Tensor):
    offsets, indices, node_features, edge_types, weights_comb = ctx.saved_tensors
    grad_node, grad_weights = torch.ops.cudnn.gnn_agg_hg_basis_post_bwd(
        grad_output,
        offsets,
        indices,
        node_features,
        edge_types,
        weights_comb,
        ctx.num_src_nodes,
        ctx.num_edge_types,
        ctx.concat_own,
        ctx.norm_by_edge_type,
        ctx.needs_input_grad[2],
        ctx.needs_input_grad[4],
    )
    return None, None, grad_node, None, grad_weights, None, None, None, None


torch.library.register_autograd("cudnn::gnn_agg_hg_basis_post_fwd", _autograd_backward, setup_context=_setup_context)
# Public API


def agg_hg_basis_post(
    graph: CscGraph,
    *,
    node_features: Tensor,
    edge_types: Tensor,
    num_edge_types: int,
    weights_comb: Optional[Tensor] = None,
    concat_own: bool = False,
    norm_by_edge_type: bool = False,
) -> Tensor:
    """Aggregate incoming features by edge type and combine them into basis outputs.

    Feature transformations may be applied after this aggregation, as in a
    basis-regularized relational graph convolution.

    Args:
        graph (CscGraph): Input graph in compressed sparse column format.
        node_features (torch.Tensor): FP32 source-node features shaped
            ``(num_src_nodes, node_feat_dim)``.
        edge_types (torch.Tensor): INT32 edge types shaped ``(num_edges,)`` in
            direct CSC order. Values must be in ``[0, num_edge_types)``.
            ``graph.map_csc_to_coo`` does not remap these entries.
        num_edge_types (int): Positive number of edge types, including any
            types with no edges in the graph.
        weights_comb (torch.Tensor, optional): FP32 combination weights shaped
            ``(num_edge_types, num_bases)``. Without weights, each edge type
            produces a separate output group. Default: None.
        concat_own (bool): Append ``node_features[:num_dst_nodes]`` to the
            output. Requires ``num_dst_nodes <= num_src_nodes``. Default: False.
        norm_by_edge_type (bool): Divide each per-type sum by the destination's
            incoming degree for that edge type. Default: False.

    Returns:
        torch.Tensor: FP32 output shaped ``(num_dst_nodes, node_feat_dim *
        (effective_bases + int(concat_own)))``, where ``effective_bases`` is
        ``num_bases`` with weights, otherwise ``num_edge_types``. Basis groups
        precede the optional destination features.

    Note:
        All tensors must be CUDA tensors on the graph device. The loaded
        backend must support cuDNN 9.29 GNN APIs and SM 8.0 or newer.
        Backward uses atomic accumulation and does not guarantee repeatable
        gradients. Higher-order gradients are not supported.
    """
    _validate_options(num_edge_types, concat_own, norm_by_edge_type)
    return torch.ops.cudnn.gnn_agg_hg_basis_post_fwd(
        graph.offsets,
        graph.indices,
        node_features,
        edge_types,
        weights_comb,
        graph.num_src_nodes,
        num_edge_types,
        concat_own,
        norm_by_edge_type,
    )
