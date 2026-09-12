# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch custom operations for cuDNN GAT and GATv2 attention."""

from numbers import Real
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from ._dtypes import TORCH_DTYPE_TO_CUDNN, TORCH_INDEX_DTYPE_TO_CUDNN
from ._utils import require_backend_symbols, tensor_pointer, validate_csc_graph
from .graph import CscGraph

_ACTIVATION_TO_INT: Dict[str, int] = {
    "linear": 0,
    "relu": 1,
    "sigmoid": 2,
    "tanh": 3,
    "elu": 4,
    "scalar": 5,
    "leaky_relu": 6,
}


def _validate_inputs(
    variant: str,
    offsets: Tensor,
    indices: Tensor,
    map_csc_to_coo: Optional[Tensor],
    node_features: Tensor,
    edge_features: Optional[Tensor],
    attn_weights: Tensor,
    dropout_mask: Optional[Tensor],
    num_src_nodes: int,
    num_heads: int,
    concat_heads: bool,
    activation: str,
    activation_alpha: float,
    csc_rev_offsets: Optional[Tensor],
    map_rev_to_coo: Optional[Tensor],
) -> Tuple[int, int, int, int]:
    num_dst_nodes, num_edges = validate_csc_graph(offsets, indices, map_csc_to_coo, num_src_nodes, csc_rev_offsets, map_rev_to_coo)
    if num_dst_nodes > num_src_nodes:
        raise ValueError(f"GAT requires num_dst_nodes <= num_src_nodes, got {num_dst_nodes} and {num_src_nodes}")
    if num_dst_nodes > 0 and num_edges == 0:
        raise ValueError("cuDNN GAT does not support a nonempty destination set with zero edges")

    if not isinstance(num_heads, int) or isinstance(num_heads, bool):
        raise TypeError(f"num_heads must be an int, got {type(num_heads).__name__}")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if not isinstance(concat_heads, bool):
        raise TypeError(f"concat_heads must be a bool, got {type(concat_heads).__name__}")
    if activation not in _ACTIVATION_TO_INT:
        supported = ", ".join(_ACTIVATION_TO_INT)
        raise ValueError(f"Unsupported activation '{activation}'. Supported: {supported}.")
    if not isinstance(activation_alpha, Real):
        raise TypeError(f"activation_alpha must be a real number, got {type(activation_alpha).__name__}")

    feature_tensors = [node_features, attn_weights]
    if edge_features is not None:
        feature_tensors.append(edge_features)
    dtype = node_features.dtype
    if dtype not in TORCH_DTYPE_TO_CUDNN:
        raise TypeError(f"features and weights must have dtype float32, float16, or bfloat16, got {dtype}")
    for tensor in feature_tensors:
        if not tensor.is_cuda:
            raise ValueError(f"features and weights must be CUDA tensors, got {tensor.device}")
        if tensor.device != offsets.device:
            raise ValueError(f"all graph, feature, and weight tensors must be on {offsets.device}, got {tensor.device}")
        if tensor.dtype != dtype:
            raise TypeError(f"all features and weights must have dtype {dtype}, got {tensor.dtype}")

    if node_features.ndim != 2 or node_features.shape[0] != num_src_nodes:
        raise ValueError(f"node_features must have shape ({num_src_nodes}, dim_node), got {tuple(node_features.shape)}")
    dim_node = node_features.shape[1]
    if dim_node == 0 or dim_node % num_heads != 0:
        raise ValueError(f"node feature dimension {dim_node} must be positive and divisible by num_heads ({num_heads})")

    dim_edge = 0
    if edge_features is not None:
        if edge_features.ndim != 2 or edge_features.shape[0] != num_edges:
            raise ValueError(f"edge_features must have shape ({num_edges}, dim_edge), got {tuple(edge_features.shape)}")
        dim_edge = edge_features.shape[1]
        if dim_edge == 0:
            raise ValueError("edge_features cannot have zero feature dimension")
        if variant == "gat" and dim_edge % num_heads != 0:
            raise ValueError(f"edge feature dimension {dim_edge} must be divisible by num_heads ({num_heads})")
        if variant == "gat_v2" and dim_edge != dim_node:
            raise ValueError(f"GATv2 edge feature dimension must equal node feature dimension {dim_node}, got {dim_edge}")

    expected_weights = 2 * dim_node + dim_edge if variant == "gat" else dim_node
    if attn_weights.ndim != 1 or attn_weights.numel() != expected_weights:
        raise ValueError(f"attn_weights must have shape ({expected_weights},), got {tuple(attn_weights.shape)}")

    if dropout_mask is not None:
        expected_mask_shape = (num_heads, num_edges)
        if tuple(dropout_mask.shape) != expected_mask_shape:
            raise ValueError(f"dropout_mask must have shape {expected_mask_shape}, got {tuple(dropout_mask.shape)}")
        if not dropout_mask.is_cuda or dropout_mask.device != offsets.device:
            raise ValueError(f"dropout_mask must be a CUDA tensor on {offsets.device}, got {dropout_mask.device}")
        if dropout_mask.dtype != torch.float32:
            raise TypeError(f"dropout_mask must have dtype float32, got {dropout_mask.dtype}")
        if dropout_mask.requires_grad:
            raise ValueError("dropout_mask must not require gradients")

    return num_dst_nodes, num_edges, dim_node, dim_edge


def _require_backend(variant: str) -> None:
    require_backend_symbols(f"gnn_mha_{variant}_forward", f"gnn_mha_{variant}_backward")


def _forward(
    variant: str,
    offsets: Tensor,
    indices: Tensor,
    map_csc_to_coo: Optional[Tensor],
    node_features: Tensor,
    edge_features: Optional[Tensor],
    attn_weights: Tensor,
    dropout_mask: Optional[Tensor],
    num_src_nodes: int,
    num_heads: int,
    concat_heads: bool,
    activation: str,
    activation_alpha: float,
    return_attention_weights: bool,
    csc_rev_offsets: Optional[Tensor],
    map_rev_to_coo: Optional[Tensor],
    deterministic: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    num_dst_nodes, num_edges, dim_node, dim_edge = _validate_inputs(
        variant,
        offsets,
        indices,
        map_csc_to_coo,
        node_features,
        edge_features,
        attn_weights,
        dropout_mask,
        num_src_nodes,
        num_heads,
        concat_heads,
        activation,
        activation_alpha,
        csc_rev_offsets,
        map_rev_to_coo,
    )
    if not isinstance(return_attention_weights, bool):
        raise TypeError(f"return_attention_weights must be a bool, got {type(return_attention_weights).__name__}")
    if not isinstance(deterministic, bool):
        raise TypeError(f"deterministic must be a bool, got {type(deterministic).__name__}")
    if deterministic and csc_rev_offsets is None:
        raise ValueError("deterministic backward requires graph.csc_rev_offsets and graph.map_rev_to_coo")
    offsets = offsets.contiguous()
    indices = indices.contiguous()
    map_csc_to_coo = None if map_csc_to_coo is None else map_csc_to_coo.contiguous()
    node_features = node_features.contiguous()
    edge_features = None if edge_features is None else edge_features.contiguous()
    attn_weights = attn_weights.contiguous()
    dropout_mask = None if dropout_mask is None else dropout_mask.contiguous()

    dim_head = dim_node // num_heads
    output_dim = dim_node if concat_heads else dim_head
    output = torch.empty((num_dst_nodes, output_dim), device=offsets.device, dtype=node_features.dtype)
    sm_scores = torch.empty((2, num_heads, num_edges), device=offsets.device, dtype=torch.float32)
    act_scores = (
        torch.empty((num_edges, dim_node), device=offsets.device, dtype=node_features.dtype)
        if variant == "gat_v2"
        else torch.empty((0,), device=offsets.device, dtype=node_features.dtype)
    )
    attention = torch.empty((0,), device=offsets.device, dtype=torch.float32)

    if num_dst_nodes == 0:
        if return_attention_weights:
            attention = torch.empty((num_heads, 0), device=offsets.device, dtype=torch.float32)
        return output, attention, sm_scores, act_scores

    _require_backend(variant)
    import cudnn

    common_args = (
        torch.cuda.current_stream(offsets.device).cuda_stream,
        offsets.data_ptr(),
        indices.data_ptr(),
        tensor_pointer(map_csc_to_coo),
        num_src_nodes,
        num_dst_nodes,
        num_edges,
        TORCH_INDEX_DTYPE_TO_CUDNN[offsets.dtype],
        node_features.data_ptr(),
        tensor_pointer(edge_features),
        attn_weights.data_ptr(),
        tensor_pointer(dropout_mask),
        output.data_ptr(),
        sm_scores.data_ptr(),
    )
    with torch.cuda.device(offsets.device):
        if variant == "gat":
            cudnn.gnn_mha_gat_forward(
                *common_args,
                dim_node,
                dim_edge,
                _ACTIVATION_TO_INT[activation],
                float(activation_alpha),
                num_heads,
                concat_heads,
                TORCH_DTYPE_TO_CUDNN[node_features.dtype],
            )
        else:
            cudnn.gnn_mha_gat_v2_forward(
                *common_args,
                act_scores.data_ptr(),
                dim_node,
                _ACTIVATION_TO_INT[activation],
                float(activation_alpha),
                num_heads,
                concat_heads,
                TORCH_DTYPE_TO_CUDNN[node_features.dtype],
            )
        if return_attention_weights:
            attention = sm_scores[1].clone()
            if dropout_mask is not None:
                attention.mul_(dropout_mask)
    return output, attention, sm_scores, act_scores


def _backward(
    variant: str,
    grad_output: Optional[Tensor],
    grad_attention: Optional[Tensor],
    offsets: Tensor,
    indices: Tensor,
    map_csc_to_coo: Optional[Tensor],
    node_features: Tensor,
    edge_features: Optional[Tensor],
    attn_weights: Tensor,
    sm_scores: Tensor,
    act_scores: Tensor,
    dropout_mask: Optional[Tensor],
    num_src_nodes: int,
    num_heads: int,
    concat_heads: bool,
    activation: str,
    activation_alpha: float,
    csc_rev_offsets: Optional[Tensor],
    map_rev_to_coo: Optional[Tensor],
    deterministic: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    num_dst_nodes, num_edges, dim_node, dim_edge = _validate_inputs(
        variant,
        offsets,
        indices,
        map_csc_to_coo,
        node_features,
        edge_features,
        attn_weights,
        dropout_mask,
        num_src_nodes,
        num_heads,
        concat_heads,
        activation,
        activation_alpha,
        csc_rev_offsets,
        map_rev_to_coo,
    )
    if deterministic and csc_rev_offsets is None:
        raise ValueError("deterministic backward requires graph.csc_rev_offsets and graph.map_rev_to_coo")
    output_dim = dim_node if concat_heads else dim_node // num_heads
    if grad_output is None:
        grad_output = torch.zeros((num_dst_nodes, output_dim), device=offsets.device, dtype=node_features.dtype)
    elif tuple(grad_output.shape) != (num_dst_nodes, output_dim) or grad_output.dtype != node_features.dtype:
        raise ValueError(f"grad_output must have shape {(num_dst_nodes, output_dim)} and dtype {node_features.dtype}")
    elif not grad_output.is_cuda or grad_output.device != offsets.device:
        raise ValueError(f"grad_output must be a CUDA tensor on {offsets.device}, got {grad_output.device}")
    if grad_attention is not None:
        if tuple(grad_attention.shape) != (num_heads, num_edges) or grad_attention.dtype != torch.float32:
            raise ValueError(f"grad_attention must have shape {(num_heads, num_edges)} and dtype float32")
        if not grad_attention.is_cuda or grad_attention.device != offsets.device:
            raise ValueError(f"grad_attention must be a CUDA tensor on {offsets.device}, got {grad_attention.device}")
        grad_attention = grad_attention.contiguous()

    grad_node = torch.empty_like(node_features)
    grad_edge = torch.empty_like(edge_features) if edge_features is not None else torch.empty((0,), device=offsets.device, dtype=node_features.dtype)
    grad_weights = torch.empty_like(attn_weights)
    grad_sm_scores = torch.empty_like(sm_scores)
    workspace_weight_dim = 2 * dim_node + dim_edge if variant == "gat" else dim_node
    workspace_features = torch.empty((num_edges, dim_node), device=offsets.device, dtype=node_features.dtype) if deterministic else None
    workspace_weights = torch.empty((num_dst_nodes, workspace_weight_dim), device=offsets.device, dtype=attn_weights.dtype) if deterministic else None
    if num_dst_nodes == 0:
        grad_node.zero_()
        grad_edge.zero_()
        grad_weights.zero_()
        return grad_node, grad_edge, grad_weights

    _require_backend(variant)
    import cudnn

    offsets = offsets.contiguous()
    indices = indices.contiguous()
    map_csc_to_coo = None if map_csc_to_coo is None else map_csc_to_coo.contiguous()
    csc_rev_offsets = None if csc_rev_offsets is None else csc_rev_offsets.contiguous()
    map_rev_to_coo = None if map_rev_to_coo is None else map_rev_to_coo.contiguous()
    grad_output = grad_output.contiguous()
    node_features = node_features.contiguous()
    edge_features = None if edge_features is None else edge_features.contiguous()
    attn_weights = attn_weights.contiguous()
    sm_scores = sm_scores.contiguous()
    act_scores = act_scores.contiguous()
    dropout_mask = None if dropout_mask is None else dropout_mask.contiguous()
    common_args = (
        torch.cuda.current_stream(offsets.device).cuda_stream,
        offsets.data_ptr(),
        indices.data_ptr(),
        tensor_pointer(map_csc_to_coo),
        num_src_nodes,
        num_dst_nodes,
        num_edges,
        TORCH_INDEX_DTYPE_TO_CUDNN[offsets.dtype],
        grad_output.data_ptr(),
        node_features.data_ptr(),
        tensor_pointer(edge_features),
        attn_weights.data_ptr(),
        sm_scores.data_ptr(),
    )
    with torch.cuda.device(offsets.device):
        if variant == "gat":
            cudnn.gnn_mha_gat_backward(
                *common_args,
                tensor_pointer(dropout_mask),
                tensor_pointer(grad_attention),
                grad_node.data_ptr(),
                tensor_pointer(grad_edge if edge_features is not None else None),
                grad_weights.data_ptr(),
                grad_sm_scores.data_ptr(),
                dim_node,
                dim_edge,
                _ACTIVATION_TO_INT[activation],
                float(activation_alpha),
                num_heads,
                concat_heads,
                TORCH_DTYPE_TO_CUDNN[node_features.dtype],
                tensor_pointer(csc_rev_offsets),
                tensor_pointer(map_rev_to_coo),
                tensor_pointer(workspace_features),
                tensor_pointer(workspace_weights),
            )
        else:
            cudnn.gnn_mha_gat_v2_backward(
                *common_args,
                act_scores.data_ptr(),
                tensor_pointer(dropout_mask),
                tensor_pointer(grad_attention),
                grad_node.data_ptr(),
                tensor_pointer(grad_edge if edge_features is not None else None),
                grad_weights.data_ptr(),
                grad_sm_scores.data_ptr(),
                dim_node,
                _ACTIVATION_TO_INT[activation],
                float(activation_alpha),
                num_heads,
                concat_heads,
                TORCH_DTYPE_TO_CUDNN[node_features.dtype],
                tensor_pointer(csc_rev_offsets),
                tensor_pointer(map_rev_to_coo),
                tensor_pointer(workspace_features),
                tensor_pointer(workspace_weights),
            )
    return grad_node, grad_edge, grad_weights


_lib = torch.library.Library("cudnn", "FRAGMENT")
for _variant in ("gat", "gat_v2"):
    _lib.define(
        f"gnn_mha_{_variant}_fwd(Tensor offsets, Tensor indices, Tensor? map_csc_to_coo, Tensor node_features, "
        "Tensor? edge_features, Tensor attn_weights, Tensor? dropout_mask, int num_src_nodes, int num_heads, "
        "bool concat_heads, str activation, float activation_alpha, bool return_attention_weights, "
        "Tensor? csc_rev_offsets, Tensor? map_rev_to_coo, bool deterministic) "
        "-> (Tensor, Tensor, Tensor, Tensor)"
    )
    _lib.define(
        f"gnn_mha_{_variant}_bwd(Tensor? grad_output, Tensor? grad_attention, Tensor offsets, Tensor indices, "
        "Tensor? map_csc_to_coo, Tensor node_features, Tensor? edge_features, Tensor attn_weights, Tensor sm_scores, "
        "Tensor act_scores, Tensor? dropout_mask, int num_src_nodes, int num_heads, bool concat_heads, "
        "str activation, float activation_alpha, Tensor? csc_rev_offsets, Tensor? map_rev_to_coo, "
        "bool deterministic) -> (Tensor, Tensor, Tensor)"
    )


def _gat_forward(*args):
    return _forward("gat", *args)


def _gat_v2_forward(*args):
    return _forward("gat_v2", *args)


def _gat_backward(*args):
    return _backward("gat", *args)


def _gat_v2_backward(*args):
    return _backward("gat_v2", *args)


_lib.impl("gnn_mha_gat_fwd", _gat_forward, "CUDA")
_lib.impl("gnn_mha_gat_bwd", _gat_backward, "CUDA")
_lib.impl("gnn_mha_gat_v2_fwd", _gat_v2_forward, "CUDA")
_lib.impl("gnn_mha_gat_v2_bwd", _gat_v2_backward, "CUDA")


def _fake_forward(variant: str, *args) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    offsets, indices, _, node_features, _, _, _, _, num_heads, concat_heads, _, _, return_attention_weights = args[:13]
    num_dst_nodes = offsets.numel() - 1
    num_edges = indices.numel()
    dim_node = node_features.shape[1]
    output_dim = dim_node if concat_heads else dim_node // num_heads
    output = torch.empty((num_dst_nodes, output_dim), device=offsets.device, dtype=node_features.dtype)
    attention_shape = (num_heads, num_edges) if return_attention_weights else (0,)
    attention = torch.empty(attention_shape, device=offsets.device, dtype=torch.float32)
    sm_scores = torch.empty((2, num_heads, num_edges), device=offsets.device, dtype=torch.float32)
    act_shape = (num_edges, dim_node) if variant == "gat_v2" else (0,)
    act_scores = torch.empty(act_shape, device=offsets.device, dtype=node_features.dtype)
    return output, attention, sm_scores, act_scores


@torch.library.register_fake("cudnn::gnn_mha_gat_fwd")
def _gat_fake(*args):
    return _fake_forward("gat", *args)


@torch.library.register_fake("cudnn::gnn_mha_gat_v2_fwd")
def _gat_v2_fake(*args):
    return _fake_forward("gat_v2", *args)


def _fake_backward(grad_output, grad_attention, offsets, indices, map_csc_to_coo, node_features, edge_features, attn_weights, *args):
    del grad_output, grad_attention, offsets, indices, map_csc_to_coo, args
    grad_edge = torch.empty_like(edge_features) if edge_features is not None else torch.empty((0,), device=node_features.device, dtype=node_features.dtype)
    return torch.empty_like(node_features), grad_edge, torch.empty_like(attn_weights)


torch.library.register_fake("cudnn::gnn_mha_gat_bwd")(_fake_backward)
torch.library.register_fake("cudnn::gnn_mha_gat_v2_bwd")(_fake_backward)


def _setup_context(variant: str, ctx, inputs, output) -> None:
    (
        offsets,
        indices,
        map_csc_to_coo,
        node_features,
        edge_features,
        attn_weights,
        dropout_mask,
        num_src_nodes,
        num_heads,
        concat_heads,
        activation,
        activation_alpha,
        return_attention_weights,
        csc_rev_offsets,
        map_rev_to_coo,
        deterministic,
    ) = inputs
    _, attention, sm_scores, act_scores = output
    ctx.save_for_backward(
        offsets,
        indices,
        map_csc_to_coo,
        node_features,
        edge_features,
        attn_weights,
        sm_scores,
        act_scores,
        dropout_mask,
        csc_rev_offsets,
        map_rev_to_coo,
    )
    if return_attention_weights:
        ctx.mark_non_differentiable(sm_scores, act_scores)
    else:
        ctx.mark_non_differentiable(attention, sm_scores, act_scores)
    ctx.variant = variant
    ctx.has_edge_features = edge_features is not None
    ctx.return_attention_weights = return_attention_weights
    ctx.deterministic = deterministic
    ctx.num_src_nodes = num_src_nodes
    ctx.num_heads = num_heads
    ctx.concat_heads = concat_heads
    ctx.activation = activation
    ctx.activation_alpha = activation_alpha


def _setup_gat(ctx, inputs, output):
    _setup_context("gat", ctx, inputs, output)


def _setup_gat_v2(ctx, inputs, output):
    _setup_context("gat_v2", ctx, inputs, output)


@torch.compiler.allow_in_graph
def _autograd_backward(ctx, grad_output, grad_attention, grad_sm_scores, grad_act_scores):
    del grad_sm_scores, grad_act_scores
    if not ctx.return_attention_weights:
        grad_attention = None
    (
        offsets,
        indices,
        map_csc_to_coo,
        node_features,
        edge_features,
        attn_weights,
        sm_scores,
        act_scores,
        dropout_mask,
        csc_rev_offsets,
        map_rev_to_coo,
    ) = ctx.saved_tensors
    grad_node, grad_edge, grad_weights = getattr(torch.ops.cudnn, f"gnn_mha_{ctx.variant}_bwd")(
        grad_output,
        grad_attention,
        offsets,
        indices,
        map_csc_to_coo,
        node_features,
        edge_features,
        attn_weights,
        sm_scores,
        act_scores,
        dropout_mask,
        ctx.num_src_nodes,
        ctx.num_heads,
        ctx.concat_heads,
        ctx.activation,
        ctx.activation_alpha,
        csc_rev_offsets,
        map_rev_to_coo,
        ctx.deterministic,
    )
    return (
        None,
        None,
        None,
        grad_node,
        grad_edge if ctx.has_edge_features else None,
        grad_weights,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd("cudnn::gnn_mha_gat_fwd", _autograd_backward, setup_context=_setup_gat)
torch.library.register_autograd("cudnn::gnn_mha_gat_v2_fwd", _autograd_backward, setup_context=_setup_gat_v2)


def _mha(
    variant: str,
    graph: CscGraph,
    node_features: Tensor,
    attn_weights: Tensor,
    *,
    edge_features: Optional[Tensor] = None,
    dropout_mask: Optional[Tensor] = None,
    num_heads: int = 1,
    concat_heads: bool = True,
    activation: str = "leaky_relu",
    activation_alpha: float = 0.2,
    return_attention_weights: bool = False,
    deterministic: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    output, attention, _, _ = getattr(torch.ops.cudnn, f"gnn_mha_{variant}_fwd")(
        graph.offsets,
        graph.indices,
        graph.map_csc_to_coo,
        node_features,
        edge_features,
        attn_weights,
        dropout_mask,
        graph.num_src_nodes,
        num_heads,
        concat_heads,
        activation,
        activation_alpha,
        return_attention_weights,
        graph.csc_rev_offsets,
        graph.map_rev_to_coo,
        deterministic,
    )
    return (output, attention) if return_attention_weights else output


def gat(
    graph: CscGraph,
    node_features: Tensor,
    attn_weights: Tensor,
    *,
    edge_features: Optional[Tensor] = None,
    dropout_mask: Optional[Tensor] = None,
    num_heads: int = 1,
    concat_heads: bool = True,
    activation: str = "leaky_relu",
    activation_alpha: float = 0.2,
    return_attention_weights: bool = False,
    deterministic: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Apply GAT multi-head attention to a homogeneous CSC graph."""
    return _mha(
        "gat",
        graph,
        node_features,
        attn_weights,
        edge_features=edge_features,
        dropout_mask=dropout_mask,
        num_heads=num_heads,
        concat_heads=concat_heads,
        activation=activation,
        activation_alpha=activation_alpha,
        return_attention_weights=return_attention_weights,
        deterministic=deterministic,
    )


def gat_v2(
    graph: CscGraph,
    node_features: Tensor,
    attn_weights: Tensor,
    *,
    edge_features: Optional[Tensor] = None,
    dropout_mask: Optional[Tensor] = None,
    num_heads: int = 1,
    concat_heads: bool = True,
    activation: str = "leaky_relu",
    activation_alpha: float = 0.2,
    return_attention_weights: bool = False,
    deterministic: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Apply GATv2 multi-head attention to a homogeneous CSC graph."""
    return _mha(
        "gat_v2",
        graph,
        node_features,
        attn_weights,
        edge_features=edge_features,
        dropout_mask=dropout_mask,
        num_heads=num_heads,
        concat_heads=concat_heads,
        activation=activation,
        activation_alpha=activation_alpha,
        return_attention_weights=return_attention_weights,
        deterministic=deterministic,
    )
