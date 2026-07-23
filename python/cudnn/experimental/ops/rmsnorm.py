"""PyTorch custom operator for cuDNN RMS normalization."""

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
    INV_VAR = 101
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


def _make_cache_key(kind: str, x: torch.Tensor, scale: torch.Tensor, has_bias: bool):
    return (
        kind,
        tuple(x.shape),
        tuple(x.stride()),
        x.dtype,
        tuple(scale.shape),
        tuple(scale.stride()),
        scale.dtype,
        has_bias,
        x.device,
    )


def _build_fprop_graph(handle, x: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor], epsilon: torch.Tensor):
    io_dtype = _TORCH_DTYPE_TO_CUDNN[x.dtype]
    graph = cudnn.pygraph(
        handle=handle,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    x_t = graph.tensor(name="X", dim=list(x.shape), stride=list(x.stride()), data_type=io_dtype, uid=_UIDs.X)
    scale_t = graph.tensor(name="scale", dim=list(scale.shape), stride=list(scale.stride()), data_type=io_dtype, uid=_UIDs.SCALE)
    epsilon_t = graph.tensor(
        name="epsilon",
        dim=list(epsilon.shape),
        stride=list(epsilon.stride()),
        data_type=cudnn.data_type.FLOAT,
        uid=_UIDs.EPSILON,
        is_pass_by_value=True,
    )
    bias_t = None
    if bias is not None:
        bias_t = graph.tensor(name="bias", dim=list(bias.shape), stride=list(bias.stride()), data_type=io_dtype, uid=_UIDs.BIAS)

    y_t, inv_var_t = graph.rmsnorm(
        name="RMS",
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


def _build_bprop_graph(handle, dy: torch.Tensor, x: torch.Tensor, scale: torch.Tensor, inv_var: torch.Tensor, has_dbias: bool):
    io_dtype = _TORCH_DTYPE_TO_CUDNN[x.dtype]
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
        name="DRMS",
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


_lib = torch.library.Library("cudnn", "FRAGMENT")
_lib.define("rmsnorm(Tensor x, Tensor scale, Tensor epsilon, Tensor? bias=None) -> (Tensor, Tensor)")
_lib.define("rmsnorm_bwd(Tensor dy, Tensor x, Tensor scale, Tensor inv_var, bool has_dbias=False) -> (Tensor, Tensor, Tensor)")


def _rmsnorm_impl(
    x: torch.Tensor,
    scale: torch.Tensor,
    epsilon: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    handle = _get_handle(x.device)
    has_bias = bias is not None
    cache_key = _make_cache_key("fprop", x, scale, has_bias)
    if cache_key not in _fprop_cache:
        graph, workspace_size = _build_fprop_graph(handle, x, scale, bias, epsilon)
        present_uids = [_UIDs.X, _UIDs.SCALE, _UIDs.EPSILON, _UIDs.Y, _UIDs.INV_VAR]
        if has_bias:
            present_uids.append(_UIDs.BIAS)
        _fprop_cache[cache_key] = (graph, workspace_size, _get_uid_order(graph, present_uids))
    graph, workspace_size, uid_order = _fprop_cache[cache_key]

    y = torch.empty_like(x)
    inv_var = torch.empty(x.shape[0], 1, 1, 1, dtype=torch.float32, device=x.device)
    workspace = torch.empty(max(workspace_size, 1), dtype=torch.uint8, device=x.device)
    uid_to_tensor = {
        int(_UIDs.X): x,
        int(_UIDs.SCALE): scale,
        int(_UIDs.EPSILON): epsilon,
        int(_UIDs.Y): y,
        int(_UIDs.INV_VAR): inv_var,
    }
    if bias is not None:
        uid_to_tensor[int(_UIDs.BIAS)] = bias
    _execute(graph, uid_order, uid_to_tensor, workspace, handle)
    return y, inv_var


_lib.impl("rmsnorm", _rmsnorm_impl, "CUDA")


@torch.library.register_fake("cudnn::rmsnorm")
def _rmsnorm_fake(
    x: torch.Tensor,
    scale: torch.Tensor,
    epsilon: torch.Tensor,
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
    handle = _get_handle(dy.device)
    cache_key = _make_cache_key("bprop", x, scale, has_dbias)
    if cache_key not in _bprop_cache:
        graph, workspace_size = _build_bprop_graph(handle, dy, x, scale, inv_var, has_dbias)
        present_uids = [_UIDs.DY, _UIDs.X, _UIDs.SCALE, _UIDs.INV_VAR, _UIDs.DX, _UIDs.DSCALE]
        if has_dbias:
            present_uids.append(_UIDs.DBIAS)
        _bprop_cache[cache_key] = (graph, workspace_size, _get_uid_order(graph, present_uids))
    graph, workspace_size, uid_order = _bprop_cache[cache_key]

    dx = torch.empty_like(x)
    dscale = torch.empty_like(scale)
    dbias = torch.empty_like(scale) if has_dbias else torch.empty(0, dtype=scale.dtype, device=x.device)
    workspace = torch.empty(max(workspace_size, 1), dtype=torch.uint8, device=x.device)
    uid_to_tensor = {
        int(_UIDs.DY): dy,
        int(_UIDs.X): x,
        int(_UIDs.SCALE): scale,
        int(_UIDs.INV_VAR): inv_var,
        int(_UIDs.DX): dx,
        int(_UIDs.DSCALE): dscale,
    }
    if has_dbias:
        uid_to_tensor[int(_UIDs.DBIAS)] = dbias
    _execute(graph, uid_order, uid_to_tensor, workspace, handle)
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
    x, scale, _epsilon, bias = inputs
    _y, inv_var = output
    ctx.save_for_backward(x, scale, inv_var)
    ctx.has_bias = bias is not None


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
    """Apply cuDNN-accelerated RMSNorm over the final input dimension."""
    if input.ndim == 0:
        raise ValueError("input must have at least one dimension")
    if input.dtype not in _TORCH_DTYPE_TO_CUDNN:
        raise TypeError(f"unsupported input dtype: {input.dtype}")
    hidden_size = input.shape[-1]
    if weight.shape != (hidden_size,):
        raise ValueError(f"weight must have shape ({hidden_size},), got {tuple(weight.shape)}")
    if weight.device != input.device or weight.dtype != input.dtype:
        raise ValueError("weight must have the same device and dtype as input")
    if bias is not None and (bias.shape != weight.shape or bias.device != input.device or bias.dtype != input.dtype):
        raise ValueError("bias must have the same shape, device, and dtype as weight")

    original_shape = input.shape
    rows = input.numel() // hidden_size
    x_4d = input.reshape(rows, hidden_size, 1, 1).contiguous()
    scale_4d = weight.reshape(1, hidden_size, 1, 1).contiguous()
    bias_4d = bias.reshape(1, hidden_size, 1, 1).contiguous() if bias is not None else None
    epsilon = torch.tensor(eps, dtype=torch.float32).reshape(1, 1, 1, 1)
    y_4d, _inv_var = torch.ops.cudnn.rmsnorm(x_4d, scale_4d, epsilon, bias_4d)
    return y_4d.reshape(original_shape)
