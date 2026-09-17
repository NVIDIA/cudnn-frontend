# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch custom operator for cuDNN layer normalization."""

from enum import IntEnum
from typing import Optional, Tuple

import cudnn
import torch

from ._common import (
    GraphCache,
    TORCH_DTYPE_TO_CUDNN,
    epsilon_tensor,
    get_handle,
    require_canonical_4d,
    require_cuda,
    require_dtype,
    require_same_device,
)

_fprop_cache = GraphCache()
_bprop_cache = GraphCache()


class _UIDs(IntEnum):
    X = 1
    SCALE = 2
    BIAS = 3
    EPSILON = 4
    Y = 100
    MEAN = 101
    INV_VAR = 102
    DY = 200
    DX = 201
    DSCALE = 202
    DBIAS = 203


def _tensor_key(tensor: torch.Tensor) -> tuple:
    return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype


def _build_fprop_graph(handle, x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor):
    io_dtype = TORCH_DTYPE_TO_CUDNN[x.dtype]
    graph = cudnn.pygraph(
        handle=handle,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x_t = graph.tensor(name="X", dim=list(x.shape), stride=list(x.stride()), data_type=io_dtype, uid=_UIDs.X)
    scale_t = graph.tensor(
        name="scale",
        dim=list(scale.shape),
        stride=list(scale.stride()),
        data_type=cudnn.data_type.FLOAT,
        uid=_UIDs.SCALE,
    )
    bias_t = graph.tensor(
        name="bias",
        dim=list(bias.shape),
        stride=list(bias.stride()),
        data_type=cudnn.data_type.FLOAT,
        uid=_UIDs.BIAS,
    )
    epsilon_t = graph.tensor(
        name="epsilon",
        dim=[1, 1, 1, 1],
        stride=[1, 1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
        uid=_UIDs.EPSILON,
        is_pass_by_value=True,
    )
    y_t, mean_t, inv_var_t = graph.layernorm(
        name="layernorm",
        norm_forward_phase=cudnn.norm_forward_phase.TRAINING,
        input=x_t,
        scale=scale_t,
        bias=bias_t,
        epsilon=epsilon_t,
    )
    y_t.set_uid(_UIDs.Y).set_output(True).set_data_type(io_dtype)
    mean_t.set_uid(_UIDs.MEAN).set_output(True).set_data_type(cudnn.data_type.FLOAT)
    inv_var_t.set_uid(_UIDs.INV_VAR).set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    return graph, graph.get_workspace_size()


def _build_bprop_graph(
    handle,
    dy: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    mean: torch.Tensor,
    inv_var: torch.Tensor,
):
    io_dtype = TORCH_DTYPE_TO_CUDNN[x.dtype]
    graph = cudnn.pygraph(
        handle=handle,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    dy_t = graph.tensor(name="DY", dim=list(dy.shape), stride=list(dy.stride()), data_type=io_dtype, uid=_UIDs.DY)
    x_t = graph.tensor(name="X", dim=list(x.shape), stride=list(x.stride()), data_type=io_dtype, uid=_UIDs.X)
    scale_t = graph.tensor(
        name="scale",
        dim=list(scale.shape),
        stride=list(scale.stride()),
        data_type=cudnn.data_type.FLOAT,
        uid=_UIDs.SCALE,
    )
    mean_t = graph.tensor(
        name="mean",
        dim=list(mean.shape),
        stride=list(mean.stride()),
        data_type=cudnn.data_type.FLOAT,
        uid=_UIDs.MEAN,
    )
    inv_var_t = graph.tensor(
        name="inv_var",
        dim=list(inv_var.shape),
        stride=list(inv_var.stride()),
        data_type=cudnn.data_type.FLOAT,
        uid=_UIDs.INV_VAR,
    )
    dx_t, dscale_t, dbias_t = graph.layernorm_backward(
        name="layernorm_bwd",
        grad=dy_t,
        input=x_t,
        scale=scale_t,
        mean=mean_t,
        inv_variance=inv_var_t,
    )
    dx_t.set_uid(_UIDs.DX).set_output(True).set_data_type(io_dtype)
    dscale_t.set_uid(_UIDs.DSCALE).set_output(True).set_data_type(cudnn.data_type.FLOAT)
    dbias_t.set_uid(_UIDs.DBIAS).set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    return graph, graph.get_workspace_size()


def _validate_fprop(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> Tuple[int, int]:
    require_cuda("cudnn::layernorm", x=x, scale=scale, bias=bias)
    require_same_device("cudnn::layernorm", x, scale=scale, bias=bias)
    if x.dtype not in TORCH_DTYPE_TO_CUDNN:
        raise TypeError(f"cudnn::layernorm: unsupported input dtype {x.dtype}")
    if x.ndim != 4 or x.shape[2:] != (1, 1):
        raise ValueError(f"cudnn::layernorm: expected x with shape (rows, hidden_size, 1, 1), got {tuple(x.shape)}")
    rows, hidden_size = x.shape[:2]
    require_canonical_4d("x", x, rows, hidden_size)
    require_canonical_4d("scale", scale, 1, hidden_size)
    require_canonical_4d("bias", bias, 1, hidden_size)
    require_dtype("scale", scale, torch.float32)
    require_dtype("bias", bias, torch.float32)
    return rows, hidden_size


def _validate_bprop(
    dy: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    mean: torch.Tensor,
    inv_var: torch.Tensor,
) -> Tuple[int, int]:
    rows, hidden_size = _validate_fprop(x, scale, scale)
    require_cuda("cudnn::layernorm_bwd", dy=dy, x=x, scale=scale, mean=mean, inv_var=inv_var)
    require_same_device("cudnn::layernorm_bwd", x, dy=dy, scale=scale, mean=mean, inv_var=inv_var)
    require_canonical_4d("dy", dy, rows, hidden_size)
    require_dtype("dy", dy, x.dtype)
    for name, tensor in (("mean", mean), ("inv_var", inv_var)):
        if tuple(tensor.shape) != (rows, 1, 1, 1) or not tensor.is_contiguous():
            raise ValueError(f"{name}: expected contiguous shape {(rows, 1, 1, 1)}, got {tuple(tensor.shape)}")
        require_dtype(name, tensor, torch.float32)
    return rows, hidden_size


_lib = torch.library.Library("cudnn", "FRAGMENT")
_lib.define("layernorm(Tensor x, Tensor scale, Tensor bias, float eps) -> (Tensor, Tensor, Tensor)")
_lib.define("layernorm_bwd(Tensor dy, Tensor x, Tensor scale, Tensor mean, Tensor inv_var) -> (Tensor, Tensor, Tensor)")


def _layernorm_impl(
    x: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, _hidden_size = _validate_fprop(x, scale, bias)
    handle = get_handle(x.device)
    key = ("layernorm_fprop", _tensor_key(x), _tensor_key(scale), _tensor_key(bias), x.device)
    graph, workspace_size = _fprop_cache.get_or_build(key, lambda: _build_fprop_graph(handle, x, scale, bias))

    y = torch.empty_like(x)
    mean = torch.empty(rows, 1, 1, 1, dtype=torch.float32, device=x.device)
    inv_var = torch.empty_like(mean)
    workspace = torch.empty(max(workspace_size, 1), dtype=torch.uint8, device=x.device)
    graph.execute(
        {
            int(_UIDs.X): x,
            int(_UIDs.SCALE): scale,
            int(_UIDs.BIAS): bias,
            int(_UIDs.EPSILON): epsilon_tensor(eps),
            int(_UIDs.Y): y,
            int(_UIDs.MEAN): mean,
            int(_UIDs.INV_VAR): inv_var,
        },
        workspace,
        handle=handle,
    )
    return y, mean, inv_var


_lib.impl("layernorm", _layernorm_impl, "CUDA")


@torch.library.register_fake("cudnn::layernorm")
def _layernorm_fake(
    x: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stats_shape = (x.shape[0], 1, 1, 1)
    return (
        torch.empty_like(x),
        torch.empty(stats_shape, dtype=torch.float32, device=x.device),
        torch.empty(stats_shape, dtype=torch.float32, device=x.device),
    )


def _layernorm_bwd_impl(
    dy: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    mean: torch.Tensor,
    inv_var: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _rows, hidden_size = _validate_bprop(dy, x, scale, mean, inv_var)
    handle = get_handle(x.device)
    key = (
        "layernorm_bprop",
        _tensor_key(dy),
        _tensor_key(x),
        _tensor_key(scale),
        _tensor_key(mean),
        _tensor_key(inv_var),
        x.device,
    )
    graph, workspace_size = _bprop_cache.get_or_build(
        key,
        lambda: _build_bprop_graph(handle, dy, x, scale, mean, inv_var),
    )

    dx = torch.empty_like(x)
    dscale = torch.empty(1, hidden_size, 1, 1, dtype=torch.float32, device=x.device)
    dbias = torch.empty_like(dscale)
    workspace = torch.empty(max(workspace_size, 1), dtype=torch.uint8, device=x.device)
    graph.execute(
        {
            int(_UIDs.DY): dy,
            int(_UIDs.X): x,
            int(_UIDs.SCALE): scale,
            int(_UIDs.MEAN): mean,
            int(_UIDs.INV_VAR): inv_var,
            int(_UIDs.DX): dx,
            int(_UIDs.DSCALE): dscale,
            int(_UIDs.DBIAS): dbias,
        },
        workspace,
        handle=handle,
    )
    return dx, dscale, dbias


_lib.impl("layernorm_bwd", _layernorm_bwd_impl, "CUDA")


@torch.library.register_fake("cudnn::layernorm_bwd")
def _layernorm_bwd_fake(
    dy: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    mean: torch.Tensor,
    inv_var: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parameter_shape = (1, x.shape[1], 1, 1)
    return (
        torch.empty_like(x),
        torch.empty(parameter_shape, dtype=torch.float32, device=x.device),
        torch.empty(parameter_shape, dtype=torch.float32, device=x.device),
    )


def _layernorm_setup_context(ctx, inputs, output):
    x, scale, _bias, _eps = inputs
    _y, mean, inv_var = output
    ctx.save_for_backward(x, scale, mean, inv_var)
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(mean, inv_var)


def _layernorm_backward(ctx, dy, _d_mean, _d_inv_var):
    x, scale, mean, inv_var = ctx.saved_tensors
    dx, dscale, dbias = torch.ops.cudnn.layernorm_bwd(dy.contiguous(), x, scale, mean, inv_var)
    return dx, dscale, dbias, None


torch.library.register_autograd("cudnn::layernorm", _layernorm_backward, setup_context=_layernorm_setup_context)


def layer_norm(
    input: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply cuDNN-accelerated LayerNorm over the final input dimension."""
    normalized_shape = tuple(normalized_shape)
    if len(normalized_shape) != 1:
        raise NotImplementedError(f"only a one-dimensional normalized_shape is supported, got {normalized_shape}")
    if input.ndim == 0:
        raise ValueError("input must have at least one dimension")
    if not input.is_cuda:
        raise ValueError(f"input must be a CUDA tensor, got {input.device}")
    if input.dtype not in TORCH_DTYPE_TO_CUDNN:
        raise TypeError(f"unsupported input dtype: {input.dtype}")
    hidden_size = normalized_shape[0]
    if hidden_size <= 0:
        raise ValueError(f"normalized_shape must contain a positive dimension, got {normalized_shape}")
    if input.shape[-1] != hidden_size:
        raise ValueError(f"input's final dimension must be {hidden_size}, got {input.shape[-1]}")

    allowed_parameter_dtypes = {input.dtype}
    if input.dtype in (torch.float16, torch.bfloat16):
        allowed_parameter_dtypes.add(torch.float32)
    for name, parameter in (("weight", weight), ("bias", bias)):
        if parameter is None:
            continue
        if tuple(parameter.shape) != normalized_shape or parameter.device != input.device:
            raise ValueError(f"{name} must match normalized_shape and be on the input device")
        if parameter.dtype not in allowed_parameter_dtypes:
            raise TypeError(f"{name} dtype must be one of {allowed_parameter_dtypes}, got {parameter.dtype}")
    if weight is not None and bias is not None and weight.dtype != bias.dtype:
        raise TypeError(f"weight and bias must have the same dtype, got {weight.dtype} and {bias.dtype}")

    original_shape = input.shape
    rows = input.numel() // hidden_size
    if rows == 0:
        raise ValueError("input must contain at least one normalization row")
    x_4d = input.reshape(rows, hidden_size, 1, 1).contiguous()
    scale_4d = (
        weight.float().reshape(1, hidden_size, 1, 1).contiguous()
        if weight is not None
        else torch.ones(1, hidden_size, 1, 1, dtype=torch.float32, device=input.device)
    )
    bias_4d = (
        bias.float().reshape(1, hidden_size, 1, 1).contiguous()
        if bias is not None
        else torch.zeros(1, hidden_size, 1, 1, dtype=torch.float32, device=input.device)
    )
    y_4d, _mean, _inv_var = torch.ops.cudnn.layernorm(x_4d, scale_4d, bias_4d, eps)
    return y_4d.reshape(original_shape)


__all__ = ["layer_norm"]
