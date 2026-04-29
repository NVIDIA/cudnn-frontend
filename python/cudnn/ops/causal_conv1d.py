from typing import List, Optional

import torch
from torch import Tensor

_TORCH_DTYPE_TO_CUDNN = {
    torch.float32: 0,  # CUDNN_DATA_FLOAT
    torch.float16: 2,  # CUDNN_DATA_HALF
    torch.bfloat16: 9,  # CUDNN_DATA_BFLOAT16
}

_ACTIVATION_TO_INT = {
    "identity": 0,  # CUDNN_CAUSAL_CONV1D_ACTIVATION_IDENTITY
    "silu": 1,  # CUDNN_CAUSAL_CONV1D_ACTIVATION_SILU
}


def _dtype_to_int(dtype: torch.dtype) -> int:
    if dtype not in _TORCH_DTYPE_TO_CUDNN:
        raise ValueError(f"Unsupported dtype {dtype}. Supported: float32, float16, bfloat16.")
    return _TORCH_DTYPE_TO_CUDNN[dtype]


def _activation_to_int(activation: str) -> int:
    if activation not in _ACTIVATION_TO_INT:
        raise ValueError(f"Unsupported activation '{activation}'. Supported: 'identity', 'silu'.")
    return _ACTIVATION_TO_INT[activation]


# ---------------------------------------------------------------------------
# Forward primitive
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "cudnn::causal_conv1d_fwd_primitive",
    mutates_args=(),
    device_types="cuda",
)
def _fwd_primitive(x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> Tensor:
    if x.dim() != 3 or weight.dim() != 2 or bias.dim() != 1:
        raise ValueError(f"Expected x(3D), weight(2D), bias(1D); got {x.shape}, {weight.shape}, {bias.shape}")

    if not (x.is_cuda and weight.is_cuda and bias.is_cuda):
        raise ValueError(f"All tensors must be on CUDA: x.device={x.device}, " f"weight.device={weight.device}, bias.device={bias.device}")
    if not (x.device == weight.device == bias.device):
        raise ValueError(f"All tensors must be on the same device: x.device={x.device}, " f"weight.device={weight.device}, bias.device={bias.device}")

    if not (x.dtype == weight.dtype == bias.dtype):
        raise TypeError(f"Dtype mismatch: x.dtype={x.dtype}, weight.dtype={weight.dtype}, " f"bias.dtype={bias.dtype} (all must match)")

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch, dim, seq_len = x.shape
    kernel_size = weight.shape[1]

    if weight.shape[0] != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim} but weight has shape {weight.shape} " f"(expected weight.shape[0]={dim})")

    if bias.shape[0] != dim:
        raise ValueError(f"Bias mismatch: x has dim={dim} but bias has shape {bias.shape} " f"(expected bias.shape[0]={dim})")

    y = torch.empty_like(x)

    import cudnn

    cudnn.causal_conv1d_forward(
        torch.cuda.current_stream().cuda_stream,
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        y.data_ptr(),
        batch,
        dim,
        seq_len,
        kernel_size,
        _dtype_to_int(x.dtype),
        _activation_to_int(activation),
    )
    return y


@torch.library.register_fake("cudnn::causal_conv1d_fwd_primitive")
def _fwd_fake(x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> Tensor:
    return torch.empty_like(x)


# ---------------------------------------------------------------------------
# Backward primitive
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "cudnn::causal_conv1d_bwd_primitive",
    mutates_args=(),
    device_types="cuda",
)
def _bwd_primitive(grad_out: Tensor, x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> List[Tensor]:
    if x.dim() != 3 or weight.dim() != 2 or bias.dim() != 1:
        raise ValueError(f"Expected x(3D), weight(2D), bias(1D); got {x.shape}, {weight.shape}, {bias.shape}")
    if grad_out.shape != x.shape:
        raise ValueError(f"Shape mismatch: dy has shape {grad_out.shape} but x has shape {x.shape} " f"(expected dy.shape == x.shape)")
    if not grad_out.is_cuda:
        raise ValueError(f"grad_out must be on CUDA: grad_out.device={grad_out.device}")
    if grad_out.device != x.device:
        raise ValueError(f"Device mismatch: grad_out.device={grad_out.device}, x.device={x.device}")
    if grad_out.dtype != x.dtype:
        raise ValueError(f"Dtype mismatch: grad_out.dtype={grad_out.dtype}, x.dtype={x.dtype}")

    if not (x.is_cuda and weight.is_cuda and bias.is_cuda):
        raise ValueError(f"All tensors must be on CUDA: x.device={x.device}, " f"weight.device={weight.device}, bias.device={bias.device}")
    if not (x.device == weight.device == bias.device):
        raise ValueError(f"All tensors must be on the same device: x.device={x.device}, " f"weight.device={weight.device}, bias.device={bias.device}")

    if not (x.dtype == weight.dtype == bias.dtype):
        raise TypeError(f"Dtype mismatch: x.dtype={x.dtype}, weight.dtype={weight.dtype}, " f"bias.dtype={bias.dtype} (all must match)")

    batch, dim, seq_len = x.shape

    if weight.shape[0] != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim} but weight has shape {weight.shape} " f"(expected weight.shape[0]={dim})")

    if bias.shape[0] != dim:
        raise ValueError(f"Bias mismatch: x has dim={dim} but bias has shape {bias.shape} " f"(expected bias.shape[0]={dim})")

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    grad_out = grad_out.contiguous()

    kernel_size = weight.shape[1]

    dx = torch.empty_like(x)
    dweight = torch.zeros(weight.shape, device=x.device, dtype=torch.float32)
    dbias = torch.zeros(bias.shape, device=x.device, dtype=torch.float32)

    import cudnn

    cudnn.causal_conv1d_backward(
        torch.cuda.current_stream().cuda_stream,
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        grad_out.data_ptr(),
        dx.data_ptr(),
        dweight.data_ptr(),
        dbias.data_ptr(),
        batch,
        dim,
        seq_len,
        kernel_size,
        _dtype_to_int(x.dtype),
        _dtype_to_int(torch.float32),
        _activation_to_int(activation),
    )
    return [dx, dweight.to(x.dtype), dbias.to(x.dtype)]


@torch.library.register_fake("cudnn::causal_conv1d_bwd_primitive")
def _bwd_fake(grad_out: Tensor, x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> List[Tensor]:
    return [torch.empty_like(x), torch.empty_like(weight), torch.empty_like(bias)]


# ---------------------------------------------------------------------------
# Autograd glue
# ---------------------------------------------------------------------------


def _setup_context(ctx, inputs, output):
    x, weight, bias, activation = inputs
    ctx.save_for_backward(x, weight, bias)
    ctx.activation = activation


@torch.compiler.allow_in_graph
def _autograd_bwd(ctx, grad_out):
    x, weight, bias = ctx.saved_tensors
    dx, dw, db = torch.ops.cudnn.causal_conv1d_bwd_primitive(grad_out, x, weight, bias, ctx.activation)
    return dx, dw, db, None


torch.library.register_autograd(
    "cudnn::causal_conv1d_fwd_primitive",
    _autograd_bwd,
    setup_context=_setup_context,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def causal_conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    activation: str = "identity",
) -> Tensor:
    r"""Depthwise causal 1D convolution with optional activation.

    Computes a depthwise 1D convolution with causal (left-only) padding
    and optional fused activation::

        y = activation(conv1d_causal(x, weight) + bias)

    Causal padding: ``(kernel_size - 1)`` on the left, ``0`` on the right.
    Each channel is convolved independently with its own 1D filter.

    Supports ``torch.compile`` and ``torch.autograd`` — backward is handled
    automatically when inputs require gradients.

    Args:
        x (torch.Tensor): Input tensor of shape ``(batch, dim, seq_len)``.
            Must be BF16, FP16, or FP32. Must be contiguous and on CUDA.
        weight (torch.Tensor): Filter tensor of shape ``(dim, kernel_size)``.
            Same dtype as *x*.
        bias (torch.Tensor | None): Optional bias of shape ``(dim,)``.
            Same dtype as *x*. Defaults to zeros if ``None``.
        activation (str): ``"identity"`` (default) or ``"silu"``.

    Returns:
        torch.Tensor: Output of shape ``(batch, dim, seq_len)``, same dtype as *x*.
    """
    if activation not in _ACTIVATION_TO_INT:
        raise ValueError(f"Unsupported activation '{activation}'. Supported: 'identity', 'silu'.")
    if bias is None:
        bias = torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype)
    return torch.ops.cudnn.causal_conv1d_fwd_primitive(x, weight, bias, activation)
