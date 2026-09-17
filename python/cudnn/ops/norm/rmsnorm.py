# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch custom operator for cuDNN RMS normalization."""

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
    INV_VAR = 101
    DY = 200
    DX = 201
    DSCALE = 202
    DBIAS = 203


def _tensor_key(tensor: torch.Tensor) -> tuple:
    return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype


def _build_fprop_graph(handle, x: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor]):
    io_dtype = TORCH_DTYPE_TO_CUDNN[x.dtype]
    graph = cudnn.pygraph(
        handle=handle,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x_t = graph.tensor(name="X", dim=list(x.shape), stride=list(x.stride()), data_type=io_dtype, uid=_UIDs.X)
    scale_t = graph.tensor(name="scale", dim=list(scale.shape), stride=list(scale.stride()), data_type=io_dtype, uid=_UIDs.SCALE)
    epsilon_t = graph.tensor(
        name="epsilon",
        dim=[1, 1, 1, 1],
        stride=[1, 1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
        uid=_UIDs.EPSILON,
        is_pass_by_value=True,
    )
    bias_t = None
    if bias is not None:
        bias_t = graph.tensor(name="bias", dim=list(bias.shape), stride=list(bias.stride()), data_type=io_dtype, uid=_UIDs.BIAS)

    y_t, inv_var_t = graph.rmsnorm(
        name="rmsnorm",
        norm_forward_phase=cudnn.norm_forward_phase.TRAINING,
        input=x_t,
        scale=scale_t,
        bias=bias_t,
        epsilon=epsilon_t,
    )
    y_t.set_uid(_UIDs.Y).set_output(True).set_data_type(io_dtype)
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
    inv_var: torch.Tensor,
    has_dbias: bool,
):
    io_dtype = TORCH_DTYPE_TO_CUDNN[x.dtype]
    graph = cudnn.pygraph(
        handle=handle,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    dy_t = graph.tensor(name="DY", dim=list(dy.shape), stride=list(dy.stride()), data_type=io_dtype, uid=_UIDs.DY)
    x_t = graph.tensor(name="X", dim=list(x.shape), stride=list(x.stride()), data_type=io_dtype, uid=_UIDs.X)
    scale_t = graph.tensor(name="scale", dim=list(scale.shape), stride=list(scale.stride()), data_type=io_dtype, uid=_UIDs.SCALE)
    inv_var_t = graph.tensor(
        name="inv_var",
        dim=list(inv_var.shape),
        stride=list(inv_var.stride()),
        data_type=cudnn.data_type.FLOAT,
        uid=_UIDs.INV_VAR,
    )
    dx_t, dscale_t, dbias_t = graph.rmsnorm_backward(
        name="rmsnorm_bwd",
        grad=dy_t,
        input=x_t,
        scale=scale_t,
        inv_variance=inv_var_t,
        has_dbias=has_dbias,
    )
    dx_t.set_uid(_UIDs.DX).set_output(True).set_data_type(io_dtype)
    dscale_t.set_uid(_UIDs.DSCALE).set_output(True).set_data_type(io_dtype)
    if has_dbias:
        dbias_t.set_uid(_UIDs.DBIAS).set_output(True).set_data_type(io_dtype)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    return graph, graph.get_workspace_size()


def _validate_fprop(
    x: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Tuple[int, int]:
    tensors = {"x": x, "scale": scale}
    if bias is not None:
        tensors["bias"] = bias
    require_cuda("cudnn::rmsnorm", **tensors)
    require_same_device("cudnn::rmsnorm", x, **{name: tensor for name, tensor in tensors.items() if name != "x"})
    if x.dtype not in TORCH_DTYPE_TO_CUDNN:
        raise TypeError(f"cudnn::rmsnorm: unsupported input dtype {x.dtype}")
    if x.ndim != 4 or x.shape[2:] != (1, 1):
        raise ValueError(f"cudnn::rmsnorm: expected x with shape (rows, hidden_size, 1, 1), got {tuple(x.shape)}")
    rows, hidden_size = x.shape[:2]
    require_canonical_4d("x", x, rows, hidden_size)
    require_canonical_4d("scale", scale, 1, hidden_size)
    require_dtype("scale", scale, x.dtype)
    if bias is not None:
        require_canonical_4d("bias", bias, 1, hidden_size)
        require_dtype("bias", bias, x.dtype)
    return rows, hidden_size


def _validate_bprop(
    dy: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    inv_var: torch.Tensor,
) -> Tuple[int, int]:
    rows, hidden_size = _validate_fprop(x, scale, None)
    require_cuda("cudnn::rmsnorm_bwd", dy=dy, x=x, scale=scale, inv_var=inv_var)
    require_same_device("cudnn::rmsnorm_bwd", x, dy=dy, scale=scale, inv_var=inv_var)
    require_canonical_4d("dy", dy, rows, hidden_size)
    require_dtype("dy", dy, x.dtype)
    if tuple(inv_var.shape) != (rows, 1, 1, 1) or not inv_var.is_contiguous():
        raise ValueError(f"inv_var: expected contiguous shape {(rows, 1, 1, 1)}, got {tuple(inv_var.shape)}")
    require_dtype("inv_var", inv_var, torch.float32)
    return rows, hidden_size


_lib = torch.library.Library("cudnn", "FRAGMENT")
_lib.define("rmsnorm(Tensor x, Tensor scale, float eps, Tensor? bias=None) -> (Tensor, Tensor)")
_lib.define("rmsnorm_bwd(Tensor dy, Tensor x, Tensor scale, Tensor inv_var, bool has_dbias=False) -> (Tensor, Tensor, Tensor)")


def _rmsnorm_impl(
    x: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rows, _hidden_size = _validate_fprop(x, scale, bias)
    handle = get_handle(x.device)
    key = (
        "rmsnorm_fprop",
        _tensor_key(x),
        _tensor_key(scale),
        _tensor_key(bias) if bias is not None else None,
        x.device,
    )
    graph, workspace_size = _fprop_cache.get_or_build(key, lambda: _build_fprop_graph(handle, x, scale, bias))

    y = torch.empty_like(x)
    inv_var = torch.empty(rows, 1, 1, 1, dtype=torch.float32, device=x.device)
    workspace = torch.empty(max(workspace_size, 1), dtype=torch.uint8, device=x.device)
    variant = {
        int(_UIDs.X): x,
        int(_UIDs.SCALE): scale,
        int(_UIDs.EPSILON): epsilon_tensor(eps),
        int(_UIDs.Y): y,
        int(_UIDs.INV_VAR): inv_var,
    }
    if bias is not None:
        variant[int(_UIDs.BIAS)] = bias
    graph.execute(variant, workspace, handle=handle)
    return y, inv_var


_lib.impl("rmsnorm", _rmsnorm_impl, "CUDA")


@torch.library.register_fake("cudnn::rmsnorm")
def _rmsnorm_fake(
    x: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty(x.shape[0], 1, 1, 1, dtype=torch.float32, device=x.device)


def _rmsnorm_bwd_impl(
    dy: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    inv_var: torch.Tensor,
    has_dbias: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_bprop(dy, x, scale, inv_var)
    handle = get_handle(x.device)
    key = (
        "rmsnorm_bprop",
        _tensor_key(dy),
        _tensor_key(x),
        _tensor_key(scale),
        _tensor_key(inv_var),
        has_dbias,
        x.device,
    )
    graph, workspace_size = _bprop_cache.get_or_build(
        key,
        lambda: _build_bprop_graph(handle, dy, x, scale, inv_var, has_dbias),
    )

    dx = torch.empty_like(x)
    dscale = torch.empty_like(scale)
    dbias = torch.empty_like(scale) if has_dbias else torch.empty(0, dtype=scale.dtype, device=x.device)
    workspace = torch.empty(max(workspace_size, 1), dtype=torch.uint8, device=x.device)
    variant = {
        int(_UIDs.DY): dy,
        int(_UIDs.X): x,
        int(_UIDs.SCALE): scale,
        int(_UIDs.INV_VAR): inv_var,
        int(_UIDs.DX): dx,
        int(_UIDs.DSCALE): dscale,
    }
    if has_dbias:
        variant[int(_UIDs.DBIAS)] = dbias
    graph.execute(variant, workspace, handle=handle)
    return dx, dscale, dbias


_lib.impl("rmsnorm_bwd", _rmsnorm_bwd_impl, "CUDA")


@torch.library.register_fake("cudnn::rmsnorm_bwd")
def _rmsnorm_bwd_fake(
    dy: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    inv_var: torch.Tensor,
    has_dbias: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dbias = torch.empty_like(scale) if has_dbias else torch.empty(0, dtype=scale.dtype, device=x.device)
    return torch.empty_like(x), torch.empty_like(scale), dbias


def _rmsnorm_setup_context(ctx, inputs, output):
    x, scale, _eps, bias = inputs
    _y, inv_var = output
    ctx.save_for_backward(x, scale, inv_var)
    ctx.has_bias = bias is not None
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(inv_var)


def _rmsnorm_backward(ctx, dy, _d_inv_var):
    x, scale, inv_var = ctx.saved_tensors
    dx, dscale, dbias = torch.ops.cudnn.rmsnorm_bwd(dy.contiguous(), x, scale, inv_var, ctx.has_bias)
    return dx, dscale, None, dbias if ctx.has_bias else None


torch.library.register_autograd("cudnn::rmsnorm", _rmsnorm_backward, setup_context=_rmsnorm_setup_context)


def rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply experimental cuDNN RMSNorm with an optional additive bias."""
    if input.ndim == 0:
        raise ValueError("input must have at least one dimension")
    if not input.is_cuda:
        raise ValueError(f"input must be a CUDA tensor, got {input.device}")
    if input.dtype not in TORCH_DTYPE_TO_CUDNN:
        raise TypeError(f"unsupported input dtype: {input.dtype}")
    hidden_size = input.shape[-1]
    if hidden_size <= 0:
        raise ValueError(f"input's final dimension must be positive, got {hidden_size}")
    if tuple(weight.shape) != (hidden_size,):
        raise ValueError(f"weight must have shape ({hidden_size},), got {tuple(weight.shape)}")
    if weight.device != input.device or weight.dtype != input.dtype:
        raise ValueError("weight must have the same device and dtype as input")
    if bias is not None and (bias.shape != weight.shape or bias.device != input.device or bias.dtype != input.dtype):
        raise ValueError("bias must have the same shape, device, and dtype as weight")

    original_shape = input.shape
    rows = input.numel() // hidden_size
    if rows == 0:
        raise ValueError("input must contain at least one normalization row")
    x_4d = input.reshape(rows, hidden_size, 1, 1).contiguous()
    scale_4d = weight.reshape(1, hidden_size, 1, 1).contiguous()
    bias_4d = bias.reshape(1, hidden_size, 1, 1).contiguous() if bias is not None else None
    y_4d, _inv_var = torch.ops.cudnn.rmsnorm(x_4d, scale_4d, eps, bias_4d)
    return y_4d.reshape(original_shape)


__all__ = ["rms_norm"]
