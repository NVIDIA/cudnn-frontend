"""PyTorch custom operator for cuDNN layer normalization."""

import logging
from enum import IntEnum
from typing import Dict, Optional, Tuple

import cudnn
import torch

_logger = logging.getLogger(__name__)

_cudnn_handles = {}
_fprop_cache: Dict[tuple, tuple] = {}
_bprop_cache: Dict[tuple, tuple] = {}

_TORCH_DTYPE_TO_CUDNN = {
    torch.float16: cudnn.data_type.HALF,
    torch.bfloat16: cudnn.data_type.BFLOAT16,
    torch.float32: cudnn.data_type.FLOAT,
}


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


def _get_handle(device: torch.device):
    if device not in _cudnn_handles:
        _cudnn_handles[device] = cudnn.create_handle()
    cudnn.set_stream(handle=_cudnn_handles[device], stream=torch.cuda.current_stream(device).cuda_stream)
    return _cudnn_handles[device]


def _get_uid_order(graph, present_uids):
    if hasattr(graph, "_get_variant_pack_uids_sorted"):
        return graph._get_variant_pack_uids_sorted()
    return sorted(present_uids)


def _execute(graph, uid_order, uid_to_tensor, workspace, handle):
    if hasattr(graph, "_execute_with_ptrs"):
        graph._execute_with_ptrs([uid_to_tensor[uid].data_ptr() for uid in uid_order], workspace.data_ptr(), int(handle))
    else:
        graph.execute(uid_to_tensor, workspace, handle=handle)


def _make_cache_key(kind: str, rows: int, hidden_size: int, dtype: torch.dtype, device: torch.device):
    return kind, rows, hidden_size, dtype, device


def _build_fprop_graph(handle, rows: int, hidden_size: int, io_dtype: torch.dtype):
    cudnn_io_dtype = _TORCH_DTYPE_TO_CUDNN[io_dtype]
    graph = cudnn.pygraph(
        handle=handle,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x_t = graph.tensor(
        name="X",
        dim=[rows, hidden_size, 1, 1],
        stride=[hidden_size, 1, hidden_size, hidden_size],
        data_type=cudnn_io_dtype,
        uid=_UIDs.X,
    )
    scale_t = graph.tensor(
        name="scale",
        dim=[1, hidden_size, 1, 1],
        stride=[hidden_size, 1, hidden_size, hidden_size],
        data_type=cudnn.data_type.FLOAT,
        uid=_UIDs.SCALE,
    )
    bias_t = graph.tensor(
        name="bias",
        dim=[1, hidden_size, 1, 1],
        stride=[hidden_size, 1, hidden_size, hidden_size],
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
    y_t.set_uid(_UIDs.Y).set_output(True).set_data_type(cudnn_io_dtype)
    mean_t.set_uid(_UIDs.MEAN).set_output(True).set_data_type(cudnn.data_type.FLOAT)
    inv_var_t.set_uid(_UIDs.INV_VAR).set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    return graph, graph.get_workspace_size()


def _build_bprop_graph(handle, rows: int, hidden_size: int, io_dtype: torch.dtype):
    cudnn_io_dtype = _TORCH_DTYPE_TO_CUDNN[io_dtype]
    graph = cudnn.pygraph(
        handle=handle,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    dy_t = graph.tensor(
        name="DY",
        dim=[rows, hidden_size, 1, 1],
        stride=[hidden_size, 1, hidden_size, hidden_size],
        data_type=cudnn_io_dtype,
        uid=_UIDs.DY,
    )
    x_t = graph.tensor(
        name="X",
        dim=[rows, hidden_size, 1, 1],
        stride=[hidden_size, 1, hidden_size, hidden_size],
        data_type=cudnn_io_dtype,
        uid=_UIDs.X,
    )
    scale_t = graph.tensor(
        name="scale",
        dim=[1, hidden_size, 1, 1],
        stride=[hidden_size, 1, hidden_size, hidden_size],
        data_type=cudnn.data_type.FLOAT,
        uid=_UIDs.SCALE,
    )
    mean_t = graph.tensor(
        name="mean",
        dim=[rows, 1, 1, 1],
        stride=[1, 1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
        uid=_UIDs.MEAN,
    )
    inv_var_t = graph.tensor(
        name="inv_var",
        dim=[rows, 1, 1, 1],
        stride=[1, 1, 1, 1],
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
    dx_t.set_uid(_UIDs.DX).set_output(True).set_data_type(cudnn_io_dtype)
    dscale_t.set_uid(_UIDs.DSCALE).set_output(True).set_data_type(cudnn.data_type.FLOAT)
    dbias_t.set_uid(_UIDs.DBIAS).set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    return graph, graph.get_workspace_size()


_lib = torch.library.Library("cudnn", "FRAGMENT")
_lib.define("layernorm(Tensor x, Tensor scale, Tensor bias, float eps) -> (Tensor, Tensor, Tensor)")
_lib.define("layernorm_bwd(Tensor dy, Tensor x, Tensor scale, Tensor mean, Tensor inv_var) -> (Tensor, Tensor, Tensor)")


def _layernorm_impl(
    x: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    handle = _get_handle(x.device)
    rows, hidden_size = x.shape[:2]
    cache_key = _make_cache_key("fprop", rows, hidden_size, x.dtype, x.device)
    if cache_key not in _fprop_cache:
        graph, workspace_size = _build_fprop_graph(handle, rows, hidden_size, x.dtype)
        _fprop_cache[cache_key] = (
            graph,
            workspace_size,
            _get_uid_order(graph, [_UIDs.X, _UIDs.SCALE, _UIDs.BIAS, _UIDs.EPSILON, _UIDs.Y, _UIDs.MEAN, _UIDs.INV_VAR]),
        )
    graph, workspace_size, uid_order = _fprop_cache[cache_key]

    y = torch.empty_like(x)
    mean = torch.empty(rows, 1, 1, 1, dtype=torch.float32, device=x.device)
    inv_var = torch.empty_like(mean)
    epsilon = torch.tensor(eps, dtype=torch.float32).reshape(1, 1, 1, 1)
    workspace = torch.empty(max(workspace_size, 1), dtype=torch.uint8, device=x.device)
    uid_to_tensor = {
        int(_UIDs.X): x,
        int(_UIDs.SCALE): scale,
        int(_UIDs.BIAS): bias,
        int(_UIDs.EPSILON): epsilon,
        int(_UIDs.Y): y,
        int(_UIDs.MEAN): mean,
        int(_UIDs.INV_VAR): inv_var,
    }
    _execute(graph, uid_order, uid_to_tensor, workspace, handle)
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
    return torch.empty_like(x), torch.empty(stats_shape, dtype=torch.float32, device=x.device), torch.empty(stats_shape, dtype=torch.float32, device=x.device)


def _layernorm_bwd_impl(
    dy: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    mean: torch.Tensor,
    inv_var: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    handle = _get_handle(dy.device)
    rows, hidden_size = x.shape[:2]
    cache_key = _make_cache_key("bprop", rows, hidden_size, x.dtype, x.device)
    if cache_key not in _bprop_cache:
        graph, workspace_size = _build_bprop_graph(handle, rows, hidden_size, x.dtype)
        _bprop_cache[cache_key] = (
            graph,
            workspace_size,
            _get_uid_order(graph, [_UIDs.DY, _UIDs.X, _UIDs.SCALE, _UIDs.MEAN, _UIDs.INV_VAR, _UIDs.DX, _UIDs.DSCALE, _UIDs.DBIAS]),
        )
    graph, workspace_size, uid_order = _bprop_cache[cache_key]

    dx = torch.empty_like(x)
    dscale = torch.empty(1, hidden_size, 1, 1, dtype=torch.float32, device=x.device)
    dbias = torch.empty_like(dscale)
    workspace = torch.empty(max(workspace_size, 1), dtype=torch.uint8, device=x.device)
    uid_to_tensor = {
        int(_UIDs.DY): dy,
        int(_UIDs.X): x,
        int(_UIDs.SCALE): scale,
        int(_UIDs.MEAN): mean,
        int(_UIDs.INV_VAR): inv_var,
        int(_UIDs.DX): dx,
        int(_UIDs.DSCALE): dscale,
        int(_UIDs.DBIAS): dbias,
    }
    _execute(graph, uid_order, uid_to_tensor, workspace, handle)
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
    if input.dtype not in _TORCH_DTYPE_TO_CUDNN:
        raise TypeError(f"unsupported input dtype: {input.dtype}")
    hidden_size = normalized_shape[0]
    if input.shape[-1] != hidden_size:
        raise ValueError(f"input's final dimension must be {hidden_size}, got {input.shape[-1]}")
    if weight is not None and (weight.shape != normalized_shape or weight.device != input.device):
        raise ValueError("weight must match normalized_shape and be on the input device")
    if bias is not None and (bias.shape != normalized_shape or bias.device != input.device):
        raise ValueError("bias must match normalized_shape and be on the input device")

    original_shape = input.shape
    rows = input.numel() // hidden_size
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
