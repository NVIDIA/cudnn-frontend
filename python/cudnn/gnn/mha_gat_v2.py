# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch custom operation and public API for cuDNN GATv2 attention."""

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
    offsets: Tensor,
    indices: Tensor,
    map_csc_to_coo: Optional[Tensor],
    src_features: Tensor,
    dst_features: Tensor,
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
    num_dst_nodes, num_edges = validate_csc_graph(
        offsets,
        indices,
        map_csc_to_coo,
        num_src_nodes,
        csc_rev_offsets,
        map_rev_to_coo,
    )
    if num_dst_nodes > 0 and num_edges == 0:
        raise ValueError("cuDNN GATv2 does not support a nonempty destination set with zero edges")

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

    feature_tensors = [src_features, dst_features, attn_weights]
    if edge_features is not None:
        feature_tensors.append(edge_features)
    dtype = src_features.dtype
    if dtype not in TORCH_DTYPE_TO_CUDNN:
        raise TypeError(f"features and weights must have dtype float32, float16, or bfloat16, got {dtype}")
    for tensor in feature_tensors:
        if not tensor.is_cuda:
            raise ValueError(f"features and weights must be CUDA tensors, got {tensor.device}")
        if tensor.device != offsets.device:
            raise ValueError(f"all graph, feature, and weight tensors must be on {offsets.device}, got {tensor.device}")
        if tensor.dtype != dtype:
            raise TypeError(f"all features and weights must have dtype {dtype}, got {tensor.dtype}")

    if src_features.ndim != 2 or src_features.shape[0] != num_src_nodes:
        raise ValueError(f"src_features must have shape ({num_src_nodes}, dim_node), got {tuple(src_features.shape)}")
    dim_node = src_features.shape[1]
    if dst_features.ndim != 2 or tuple(dst_features.shape) != (num_dst_nodes, dim_node):
        raise ValueError(f"dst_features must have shape ({num_dst_nodes}, {dim_node}), got {tuple(dst_features.shape)}")
    if dim_node == 0 or dim_node % num_heads != 0:
        raise ValueError(f"node feature dimension {dim_node} must be positive and divisible by num_heads ({num_heads})")

    dim_edge = 0
    if edge_features is not None:
        if edge_features.ndim != 2 or edge_features.shape[0] != num_edges:
            raise ValueError(f"edge_features must have shape ({num_edges}, dim_edge), got {tuple(edge_features.shape)}")
        dim_edge = edge_features.shape[1]
        if dim_edge == 0:
            raise ValueError("edge_features cannot have zero feature dimension")

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

    if edge_features is not None and dim_edge != dim_node:
        raise ValueError(f"GATv2 edge feature dimension must equal node feature dimension {dim_node}, got {dim_edge}")
    if attn_weights.ndim != 1 or attn_weights.numel() != dim_node:
        raise ValueError(f"attn_weights must have shape ({dim_node},), got {tuple(attn_weights.shape)}")
    return num_dst_nodes, num_edges, dim_node, dim_edge


def _validate_execution_options(
    return_attention_weights: bool,
    deterministic: bool,
    csc_rev_offsets: Optional[Tensor],
) -> None:
    if not isinstance(return_attention_weights, bool):
        raise TypeError(f"return_attention_weights must be a bool, got {type(return_attention_weights).__name__}")
    if not isinstance(deterministic, bool):
        raise TypeError(f"deterministic must be a bool, got {type(deterministic).__name__}")
    if deterministic and csc_rev_offsets is None:
        raise ValueError("deterministic backward requires graph.csc_rev_offsets and graph.map_rev_to_coo")


def _validate_gradient_dtype(grad_dtype: torch.dtype, input_dtype: torch.dtype) -> torch.dtype:
    if not isinstance(grad_dtype, torch.dtype):
        raise TypeError(f"grad_dtype must be a torch.dtype, got {type(grad_dtype).__name__}")
    if grad_dtype not in (input_dtype, torch.float32):
        raise ValueError(f"grad_dtype must be {input_dtype} or torch.float32, got {grad_dtype}")
    return grad_dtype


def _resolve_gradient_dtype(input_dtype: torch.dtype, high_precision_grad: bool) -> torch.dtype:
    if not isinstance(high_precision_grad, bool):
        raise TypeError(f"high_precision_grad must be a bool, got {type(high_precision_grad).__name__}")
    return torch.float32 if high_precision_grad else input_dtype


def _validate_backward_gradients(
    grad_output: Optional[Tensor],
    grad_attention: Optional[Tensor],
    offsets: Tensor,
    src_features: Tensor,
    num_dst_nodes: int,
    num_edges: int,
    dim_node: int,
    num_heads: int,
    concat_heads: bool,
) -> Tuple[Tensor, Optional[Tensor]]:
    output_dim = dim_node if concat_heads else dim_node // num_heads
    if grad_output is None:
        grad_output = torch.zeros((num_dst_nodes, output_dim), device=offsets.device, dtype=src_features.dtype)
    elif tuple(grad_output.shape) != (num_dst_nodes, output_dim) or grad_output.dtype != src_features.dtype:
        raise ValueError(f"grad_output must have shape {(num_dst_nodes, output_dim)} and dtype {src_features.dtype}")
    elif not grad_output.is_cuda or grad_output.device != offsets.device:
        raise ValueError(f"grad_output must be a CUDA tensor on {offsets.device}, got {grad_output.device}")

    if grad_attention is not None:
        if tuple(grad_attention.shape) != (num_heads, num_edges) or grad_attention.dtype != torch.float32:
            raise ValueError(f"grad_attention must have shape {(num_heads, num_edges)} and dtype float32")
        if not grad_attention.is_cuda or grad_attention.device != offsets.device:
            raise ValueError(f"grad_attention must be a CUDA tensor on {offsets.device}, got {grad_attention.device}")
        grad_attention = grad_attention.contiguous()
    return grad_output, grad_attention


def _allocate_forward_outputs(
    offsets: Tensor,
    src_features: Tensor,
    num_dst_nodes: int,
    num_edges: int,
    dim_node: int,
    num_heads: int,
    concat_heads: bool,
    return_attention_weights: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    output_dim = dim_node if concat_heads else dim_node // num_heads
    output = torch.empty((num_dst_nodes, output_dim), device=offsets.device, dtype=src_features.dtype)
    attention_shape = (num_heads, num_edges) if return_attention_weights and num_dst_nodes == 0 else (0,)
    attention = torch.empty(attention_shape, device=offsets.device, dtype=torch.float32)
    sm_scores = torch.empty((2, num_heads, num_edges), device=offsets.device, dtype=torch.float32)
    return output, attention, sm_scores


def _extract_attention(sm_scores: Tensor, dropout_mask: Optional[Tensor]) -> Tensor:
    attention = sm_scores[1].clone()
    if dropout_mask is not None:
        attention.mul_(dropout_mask)
    return attention


def _normalize_features(
    graph: CscGraph,
    features: Union[Tensor, Tuple[Tensor, Tensor]],
) -> Tuple[Tensor, Tensor]:
    if isinstance(features, Tensor):
        return features, features[: graph.num_dst_nodes]
    if isinstance(features, tuple) and len(features) == 2 and all(isinstance(feature, Tensor) for feature in features):
        return features
    raise TypeError("features must be a Tensor or a (src_features, dst_features) tuple")


def _fake_forward_outputs(
    offsets: Tensor,
    indices: Tensor,
    src_features: Tensor,
    num_heads: int,
    concat_heads: bool,
    return_attention_weights: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    num_dst_nodes = offsets.numel() - 1
    num_edges = indices.numel()
    dim_node = src_features.shape[1]
    output_dim = dim_node if concat_heads else dim_node // num_heads
    output = torch.empty((num_dst_nodes, output_dim), device=offsets.device, dtype=src_features.dtype)
    attention_shape = (num_heads, num_edges) if return_attention_weights else (0,)
    attention = torch.empty(attention_shape, device=offsets.device, dtype=torch.float32)
    sm_scores = torch.empty((2, num_heads, num_edges), device=offsets.device, dtype=torch.float32)
    return output, attention, sm_scores


def _fake_backward_outputs(
    src_features: Tensor,
    dst_features: Tensor,
    edge_features: Optional[Tensor],
    attn_weights: Tensor,
    grad_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    grad_edge = (
        torch.empty_like(edge_features, dtype=grad_dtype, memory_format=torch.contiguous_format)
        if edge_features is not None
        else torch.empty((0,), device=src_features.device, dtype=grad_dtype)
    )
    return (
        torch.empty_like(src_features, dtype=grad_dtype, memory_format=torch.contiguous_format),
        torch.empty_like(dst_features, dtype=grad_dtype, memory_format=torch.contiguous_format),
        grad_edge,
        torch.empty_like(attn_weights, dtype=grad_dtype, memory_format=torch.contiguous_format),
    )


def _autograd_input_gradients(ctx, grad_src: Tensor, grad_dst: Tensor, grad_edge: Tensor, grad_weights: Tensor):
    return (
        None,
        None,
        None,
        grad_src,
        grad_dst,
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
        None,
    )


def _require_backend() -> None:
    require_backend_symbols("gnn_mha_gat_v2_forward", "gnn_mha_gat_v2_backward")


def _forward(
    offsets: Tensor,
    indices: Tensor,
    map_csc_to_coo: Optional[Tensor],
    src_features: Tensor,
    dst_features: Tensor,
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
    grad_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    num_dst_nodes, num_edges, dim_node, _ = _validate_inputs(
        offsets,
        indices,
        map_csc_to_coo,
        src_features,
        dst_features,
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
    _validate_execution_options(return_attention_weights, deterministic, csc_rev_offsets)
    _validate_gradient_dtype(grad_dtype, src_features.dtype)

    offsets = offsets.contiguous()
    indices = indices.contiguous()
    map_csc_to_coo = None if map_csc_to_coo is None else map_csc_to_coo.contiguous()
    src_features = src_features.contiguous()
    dst_features = dst_features.contiguous()
    edge_features = None if edge_features is None else edge_features.contiguous()
    attn_weights = attn_weights.contiguous()
    dropout_mask = None if dropout_mask is None else dropout_mask.contiguous()

    output, attention, sm_scores = _allocate_forward_outputs(
        offsets,
        src_features,
        num_dst_nodes,
        num_edges,
        dim_node,
        num_heads,
        concat_heads,
        return_attention_weights,
    )
    act_scores = torch.empty((num_edges, dim_node), device=offsets.device, dtype=src_features.dtype)
    if num_dst_nodes == 0:
        return output, attention, sm_scores, act_scores

    _require_backend()
    import cudnn

    with torch.cuda.device(offsets.device):
        cudnn.gnn_mha_gat_v2_forward(
            torch.cuda.current_stream(offsets.device).cuda_stream,
            offsets.data_ptr(),
            indices.data_ptr(),
            tensor_pointer(map_csc_to_coo),
            num_src_nodes,
            num_dst_nodes,
            num_edges,
            TORCH_INDEX_DTYPE_TO_CUDNN[offsets.dtype],
            src_features.data_ptr(),
            dst_features.data_ptr(),
            tensor_pointer(edge_features),
            attn_weights.data_ptr(),
            tensor_pointer(dropout_mask),
            output.data_ptr(),
            sm_scores.data_ptr(),
            act_scores.data_ptr(),
            dim_node,
            _ACTIVATION_TO_INT[activation],
            float(activation_alpha),
            num_heads,
            concat_heads,
            TORCH_DTYPE_TO_CUDNN[src_features.dtype],
        )
        if return_attention_weights:
            attention = _extract_attention(sm_scores, dropout_mask)
    return output, attention, sm_scores, act_scores


def _backward(
    grad_output: Optional[Tensor],
    grad_attention: Optional[Tensor],
    offsets: Tensor,
    indices: Tensor,
    map_csc_to_coo: Optional[Tensor],
    src_features: Tensor,
    dst_features: Tensor,
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
    grad_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    num_dst_nodes, num_edges, dim_node, _ = _validate_inputs(
        offsets,
        indices,
        map_csc_to_coo,
        src_features,
        dst_features,
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
    _validate_execution_options(False, deterministic, csc_rev_offsets)
    grad_output, grad_attention = _validate_backward_gradients(
        grad_output,
        grad_attention,
        offsets,
        src_features,
        num_dst_nodes,
        num_edges,
        dim_node,
        num_heads,
        concat_heads,
    )
    grad_dtype = _validate_gradient_dtype(grad_dtype, src_features.dtype)

    grad_src = torch.empty_like(src_features, dtype=grad_dtype, memory_format=torch.contiguous_format)
    grad_dst = torch.empty_like(dst_features, dtype=grad_dtype, memory_format=torch.contiguous_format)
    grad_edge = (
        torch.empty_like(edge_features, dtype=grad_dtype, memory_format=torch.contiguous_format)
        if edge_features is not None
        else torch.empty((0,), device=offsets.device, dtype=grad_dtype)
    )
    grad_weights = torch.empty_like(attn_weights, dtype=grad_dtype, memory_format=torch.contiguous_format)
    grad_sm_scores = torch.empty_like(sm_scores)
    workspace_features = torch.empty((num_edges, dim_node), device=offsets.device, dtype=grad_dtype) if deterministic else None
    workspace_weights = torch.empty((num_dst_nodes, dim_node), device=offsets.device, dtype=grad_dtype) if deterministic else None
    if num_dst_nodes == 0:
        grad_src.zero_()
        grad_dst.zero_()
        grad_edge.zero_()
        grad_weights.zero_()
        return grad_src, grad_dst, grad_edge, grad_weights

    _require_backend()
    import cudnn

    offsets = offsets.contiguous()
    indices = indices.contiguous()
    map_csc_to_coo = None if map_csc_to_coo is None else map_csc_to_coo.contiguous()
    csc_rev_offsets = None if csc_rev_offsets is None else csc_rev_offsets.contiguous()
    map_rev_to_coo = None if map_rev_to_coo is None else map_rev_to_coo.contiguous()
    grad_output = grad_output.contiguous()
    src_features = src_features.contiguous()
    dst_features = dst_features.contiguous()
    edge_features = None if edge_features is None else edge_features.contiguous()
    attn_weights = attn_weights.contiguous()
    sm_scores = sm_scores.contiguous()
    act_scores = act_scores.contiguous()
    dropout_mask = None if dropout_mask is None else dropout_mask.contiguous()

    with torch.cuda.device(offsets.device):
        cudnn.gnn_mha_gat_v2_backward(
            torch.cuda.current_stream(offsets.device).cuda_stream,
            offsets.data_ptr(),
            indices.data_ptr(),
            tensor_pointer(map_csc_to_coo),
            num_src_nodes,
            num_dst_nodes,
            num_edges,
            TORCH_INDEX_DTYPE_TO_CUDNN[offsets.dtype],
            grad_output.data_ptr(),
            src_features.data_ptr(),
            dst_features.data_ptr(),
            tensor_pointer(edge_features),
            attn_weights.data_ptr(),
            sm_scores.data_ptr(),
            act_scores.data_ptr(),
            tensor_pointer(dropout_mask),
            tensor_pointer(grad_attention),
            grad_src.data_ptr(),
            grad_dst.data_ptr(),
            tensor_pointer(grad_edge if edge_features is not None else None),
            grad_weights.data_ptr(),
            grad_sm_scores.data_ptr(),
            dim_node,
            _ACTIVATION_TO_INT[activation],
            float(activation_alpha),
            num_heads,
            concat_heads,
            TORCH_DTYPE_TO_CUDNN[src_features.dtype],
            tensor_pointer(csc_rev_offsets),
            tensor_pointer(map_rev_to_coo),
            tensor_pointer(workspace_features),
            tensor_pointer(workspace_weights),
            TORCH_DTYPE_TO_CUDNN[grad_dtype],
        )
    return grad_src, grad_dst, grad_edge, grad_weights


_lib = torch.library.Library("cudnn", "FRAGMENT")
_lib.define(
    "gnn_mha_gat_v2_fwd(Tensor offsets, Tensor indices, Tensor? map_csc_to_coo, Tensor src_features, Tensor dst_features, "
    "Tensor? edge_features, Tensor attn_weights, Tensor? dropout_mask, int num_src_nodes, int num_heads, "
    "bool concat_heads, str activation, float activation_alpha, bool return_attention_weights, "
    "Tensor? csc_rev_offsets, Tensor? map_rev_to_coo, bool deterministic, ScalarType grad_dtype) -> (Tensor, Tensor, Tensor, Tensor)"
)
_lib.define(
    "gnn_mha_gat_v2_bwd(Tensor? grad_output, Tensor? grad_attention, Tensor offsets, Tensor indices, "
    "Tensor? map_csc_to_coo, Tensor src_features, Tensor dst_features, Tensor? edge_features, Tensor attn_weights, "
    "Tensor sm_scores, Tensor act_scores, Tensor? dropout_mask, int num_src_nodes, int num_heads, bool concat_heads, "
    "str activation, float activation_alpha, Tensor? csc_rev_offsets, Tensor? map_rev_to_coo, "
    "bool deterministic, ScalarType grad_dtype) -> (Tensor, Tensor, Tensor, Tensor)"
)
_lib.impl("gnn_mha_gat_v2_fwd", _forward, "CUDA")
_lib.impl("gnn_mha_gat_v2_bwd", _backward, "CUDA")


@torch.library.register_fake("cudnn::gnn_mha_gat_v2_fwd")
def _fake_forward(
    offsets,
    indices,
    map_csc_to_coo,
    src_features,
    dst_features,
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
    grad_dtype,
):
    del map_csc_to_coo, dst_features, edge_features, attn_weights, dropout_mask
    del num_src_nodes, activation, activation_alpha, csc_rev_offsets, map_rev_to_coo, deterministic
    del grad_dtype
    output, attention, sm_scores = _fake_forward_outputs(
        offsets,
        indices,
        src_features,
        num_heads,
        concat_heads,
        return_attention_weights,
    )
    act_scores = torch.empty((indices.numel(), src_features.shape[1]), device=offsets.device, dtype=src_features.dtype)
    return output, attention, sm_scores, act_scores


@torch.library.register_fake("cudnn::gnn_mha_gat_v2_bwd")
def _fake_backward(
    grad_output,
    grad_attention,
    offsets,
    indices,
    map_csc_to_coo,
    src_features,
    dst_features,
    edge_features,
    attn_weights,
    sm_scores,
    act_scores,
    dropout_mask,
    num_src_nodes,
    num_heads,
    concat_heads,
    activation,
    activation_alpha,
    csc_rev_offsets,
    map_rev_to_coo,
    deterministic,
    grad_dtype,
):
    del grad_output, grad_attention, offsets, indices, map_csc_to_coo, sm_scores, act_scores, dropout_mask
    del num_src_nodes, num_heads, concat_heads, activation, activation_alpha
    del csc_rev_offsets, map_rev_to_coo, deterministic
    return _fake_backward_outputs(src_features, dst_features, edge_features, attn_weights, grad_dtype)


def _setup_context(ctx, inputs, output) -> None:
    (
        offsets,
        indices,
        map_csc_to_coo,
        src_features,
        dst_features,
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
        grad_dtype,
    ) = inputs
    _, attention, sm_scores, act_scores = output
    ctx.save_for_backward(
        offsets,
        indices,
        map_csc_to_coo,
        src_features,
        dst_features,
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
    ctx.has_edge_features = edge_features is not None
    ctx.return_attention_weights = return_attention_weights
    ctx.deterministic = deterministic
    ctx.num_src_nodes = num_src_nodes
    ctx.num_heads = num_heads
    ctx.concat_heads = concat_heads
    ctx.activation = activation
    ctx.activation_alpha = activation_alpha
    ctx.grad_dtype = grad_dtype


@torch.compiler.allow_in_graph
def _autograd_backward(ctx, grad_output, grad_attention, grad_sm_scores, grad_act_scores):
    del grad_sm_scores, grad_act_scores
    if not ctx.return_attention_weights:
        grad_attention = None
    (
        offsets,
        indices,
        map_csc_to_coo,
        src_features,
        dst_features,
        edge_features,
        attn_weights,
        sm_scores,
        act_scores,
        dropout_mask,
        csc_rev_offsets,
        map_rev_to_coo,
    ) = ctx.saved_tensors
    grad_src, grad_dst, grad_edge, grad_weights = torch.ops.cudnn.gnn_mha_gat_v2_bwd(
        grad_output,
        grad_attention,
        offsets,
        indices,
        map_csc_to_coo,
        src_features,
        dst_features,
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
        ctx.grad_dtype,
    )
    return _autograd_input_gradients(ctx, grad_src, grad_dst, grad_edge, grad_weights)


torch.library.register_autograd("cudnn::gnn_mha_gat_v2_fwd", _autograd_backward, setup_context=_setup_context)


def mha_gat_v2(
    graph: CscGraph,
    features: Union[Tensor, Tuple[Tensor, Tensor]],
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
    high_precision_grad: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Apply GATv2 multi-head attention to a homogeneous or bipartite CSC graph.

    Pass one feature tensor for a homogeneous graph or separate source and
    destination tensors for a bipartite graph. Deterministic backward requires
    reverse-CSC metadata created by :meth:`CscGraph.with_reverse_csc`.

    Args:
        graph (CscGraph): Input graph in compressed sparse column format.
        features (torch.Tensor or tuple[torch.Tensor, torch.Tensor]): Source
            features shaped (num_src_nodes, dim_node), or a
            (src_features, dst_features) pair with destination features shaped
            (num_dst_nodes, dim_node).
        attn_weights (torch.Tensor): Flat attention weights shaped (dim_node,).
        edge_features (torch.Tensor, optional): Edge features shaped
            (num_edges, dim_node).
        dropout_mask (torch.Tensor, optional): FP32 inverted-dropout factors
            shaped (num_heads, num_edges). Entries use mapped edge order when
            graph.map_csc_to_coo is present.
        num_heads (int): Number of attention heads. dim_node must be divisible
            by this value. Default: 1.
        concat_heads (bool): Concatenate head outputs when True; otherwise
            average them. Default: True.
        activation (str): Logit activation. Supported values are "linear",
            "relu", "sigmoid", "tanh", "elu", "scalar", and "leaky_relu".
            Default: "leaky_relu".
        activation_alpha (float): Negative slope for "leaky_relu" or scale for
            "scalar". Default: 0.2.
        return_attention_weights (bool): Return post-softmax, post-dropout
            attention coefficients with the output. Default: False.
        deterministic (bool): Use deterministic backward reductions. The graph
            must contain reverse-CSC metadata. Default: False.
        high_precision_grad (bool): Compute all feature and attention-weight
            gradients in FP32 for FP16 or BF16 inputs. Default: False.

    Returns:
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]: The output tensor,
        or (output, attention_weights) when return_attention_weights=True.
        Attention weights have shape (num_heads, num_edges) and FP32 dtype.

    Note:
        PyTorch separately controls leaf-gradient accumulation. Set each input
        tensor's grad_dtype to torch.float32 to retain a high-precision
        gradient.
    """
    src_features, dst_features = _normalize_features(graph, features)
    grad_dtype = _resolve_gradient_dtype(src_features.dtype, high_precision_grad)
    output, attention, _, _ = torch.ops.cudnn.gnn_mha_gat_v2_fwd(
        graph.offsets,
        graph.indices,
        graph.map_csc_to_coo,
        src_features,
        dst_features,
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
        grad_dtype,
    )
    return (output, attention) if return_attention_weights else output
