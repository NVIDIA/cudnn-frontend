# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple

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
            Must be BF16, FP16, or FP32 and on CUDA.
        weight (torch.Tensor): Filter tensor of shape ``(dim, kernel_size)``.
            Same dtype as *x*. ``kernel_size`` must be between 2 and 256,
            inclusive.
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


# ===========================================================================
# NWH variant — x is (batch, seq_len, dim)
# ===========================================================================


# ---------------------------------------------------------------------------
# NWH Forward primitive
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "cudnn::causal_conv1d_nwh_fwd_primitive",
    mutates_args=(),
    device_types="cuda",
)
def _nwh_fwd_primitive(x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> Tensor:
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

    batch, seq_len, dim = x.shape
    kernel_size = weight.shape[0]

    if weight.shape[1] != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim} but weight has shape {weight.shape} " f"(expected weight.shape[1]={dim})")

    if bias.shape[0] != dim:
        raise ValueError(f"Bias mismatch: x has dim={dim} but bias has shape {bias.shape} " f"(expected bias.shape[0]={dim})")

    y = torch.empty_like(x)

    import cudnn

    cudnn.causal_conv1d_nwh_forward(
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


@torch.library.register_fake("cudnn::causal_conv1d_nwh_fwd_primitive")
def _nwh_fwd_fake(x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> Tensor:
    return torch.empty_like(x)


# ---------------------------------------------------------------------------
# NWH Backward primitive
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "cudnn::causal_conv1d_nwh_bwd_primitive",
    mutates_args=(),
    device_types="cuda",
)
def _nwh_bwd_primitive(grad_out: Tensor, x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> List[Tensor]:
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

    batch, seq_len, dim = x.shape

    if weight.shape[1] != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim} but weight has shape {weight.shape} " f"(expected weight.shape[1]={dim})")

    if bias.shape[0] != dim:
        raise ValueError(f"Bias mismatch: x has dim={dim} but bias has shape {bias.shape} " f"(expected bias.shape[0]={dim})")

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    grad_out = grad_out.contiguous()

    kernel_size = weight.shape[0]

    dx = torch.empty_like(x)
    dweight = torch.zeros(weight.shape, device=x.device, dtype=torch.float32)
    dbias = torch.zeros(bias.shape, device=x.device, dtype=torch.float32)

    import cudnn

    cudnn.causal_conv1d_nwh_backward(
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


@torch.library.register_fake("cudnn::causal_conv1d_nwh_bwd_primitive")
def _nwh_bwd_fake(grad_out: Tensor, x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> List[Tensor]:
    return [torch.empty_like(x), torch.empty_like(weight), torch.empty_like(bias)]


# ---------------------------------------------------------------------------
# NWH Autograd glue
# ---------------------------------------------------------------------------


def _nwh_setup_context(ctx, inputs, output):
    x, weight, bias, activation = inputs
    ctx.save_for_backward(x, weight, bias)
    ctx.activation = activation


@torch.compiler.allow_in_graph
def _nwh_autograd_bwd(ctx, grad_out):
    x, weight, bias = ctx.saved_tensors
    dx, dw, db = torch.ops.cudnn.causal_conv1d_nwh_bwd_primitive(grad_out, x, weight, bias, ctx.activation)
    return dx, dw, db, None


torch.library.register_autograd(
    "cudnn::causal_conv1d_nwh_fwd_primitive",
    _nwh_autograd_bwd,
    setup_context=_nwh_setup_context,
)


# ---------------------------------------------------------------------------
# NWH Public API
# ---------------------------------------------------------------------------


def causal_conv1d_nwh(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    activation: str = "identity",
) -> Tensor:
    r"""Depthwise causal 1D convolution (NWH layout).

    Same operation as :func:`causal_conv1d` but with NWH tensor layout::

        y = activation(conv1d_causal(x, weight) + bias)

    Supports ``torch.compile`` and ``torch.autograd`` — backward is handled
    automatically when inputs require gradients.

    Args:
        x (torch.Tensor): Input tensor of shape ``(batch, seq_len, dim)``.
        weight (torch.Tensor): Filter tensor of shape ``(kernel_size, dim)``.
            ``kernel_size`` must be between 2 and 128, inclusive.
        bias (torch.Tensor | None): Optional bias of shape ``(dim,)``.
        activation (str): ``"identity"`` (default) or ``"silu"``.

    Returns:
        torch.Tensor: Output of shape ``(batch, seq_len, dim)``.
    """
    if activation not in _ACTIVATION_TO_INT:
        raise ValueError(f"Unsupported activation '{activation}'. Supported: 'identity', 'silu'.")
    if bias is None:
        bias = torch.zeros(weight.shape[1], device=x.device, dtype=x.dtype)
    return torch.ops.cudnn.causal_conv1d_nwh_fwd_primitive(x, weight, bias, activation)


# ===========================================================================
# Back-to-back (B2B) causal conv1d — fused projection + gating + mixer
# ===========================================================================


# ---------------------------------------------------------------------------
# B2B Forward primitive
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "cudnn::b2b_causal_conv1d_fwd_primitive",
    mutates_args=(),
    device_types="cuda",
)
def _b2b_fwd_primitive(x: Tensor, weights_proj: Tensor, weights_mixer: Tensor, skip_bias: Tensor) -> Tuple[Tensor, Tensor]:
    if x.dim() != 3:
        raise ValueError(f"Expected x(3D) with shape (batch, 3*dim, seq_len); got {x.shape}")
    batch = x.shape[0]
    if x.shape[1] % 3 != 0:
        raise ValueError(f"Expected x.shape[1] divisible by 3; got {x.shape[1]}")
    dim = x.shape[1] // 3
    seq_len = x.shape[2]

    if weights_proj.dim() != 2 or weights_mixer.dim() != 2 or skip_bias.dim() != 1:
        raise ValueError(
            f"Expected weights_proj(2D: 3*dim,K_proj), weights_mixer(2D: dim,K_mixer), skip_bias(1D: dim); "
            f"got {weights_proj.shape}, {weights_mixer.shape}, {skip_bias.shape}"
        )

    if not (x.is_cuda and weights_proj.is_cuda and weights_mixer.is_cuda and skip_bias.is_cuda):
        raise ValueError("All tensors must be on CUDA")
    if not (x.device == weights_proj.device == weights_mixer.device == skip_bias.device):
        raise ValueError("All tensors must be on the same device")

    if not (x.dtype == weights_proj.dtype == weights_mixer.dtype == skip_bias.dtype):
        raise TypeError(
            f"Dtype mismatch: x.dtype={x.dtype}, weights_proj.dtype={weights_proj.dtype}, "
            f"weights_mixer.dtype={weights_mixer.dtype}, skip_bias.dtype={skip_bias.dtype}"
        )

    if weights_proj.shape[0] != 3 * dim:
        raise ValueError(f"Channel mismatch: x has 3*dim={3*dim} but weights_proj has shape {weights_proj.shape}")
    if weights_mixer.shape[0] != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim} but weights_mixer has shape {weights_mixer.shape}")
    if skip_bias.shape[0] != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim} but skip_bias has shape {skip_bias.shape}")

    x = x.contiguous()
    weights_proj = weights_proj.contiguous()
    weights_mixer = weights_mixer.contiguous()
    skip_bias = skip_bias.contiguous()

    kernel_size_proj = weights_proj.shape[1]
    kernel_size_mixer = weights_mixer.shape[1]

    y = torch.empty(batch, dim, seq_len, device=x.device, dtype=x.dtype)
    y_gated = torch.empty(batch, dim, seq_len, device=x.device, dtype=x.dtype)

    import cudnn

    cudnn.b2b_causal_conv1d_forward(
        torch.cuda.current_stream().cuda_stream,
        x.data_ptr(),
        weights_proj.data_ptr(),
        weights_mixer.data_ptr(),
        skip_bias.data_ptr(),
        y.data_ptr(),
        y_gated.data_ptr(),
        batch,
        dim,
        seq_len,
        kernel_size_proj,
        kernel_size_mixer,
        _dtype_to_int(x.dtype),
    )
    return y, y_gated


@torch.library.register_fake("cudnn::b2b_causal_conv1d_fwd_primitive")
def _b2b_fwd_fake(x: Tensor, weights_proj: Tensor, weights_mixer: Tensor, skip_bias: Tensor) -> Tuple[Tensor, Tensor]:
    batch = x.shape[0]
    dim = x.shape[1] // 3
    seq_len = x.shape[2]
    y = torch.empty(batch, dim, seq_len, device=x.device, dtype=x.dtype)
    y_gated = torch.empty(batch, dim, seq_len, device=x.device, dtype=x.dtype)
    return y, y_gated


# ---------------------------------------------------------------------------
# B2B Backward primitive
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "cudnn::b2b_causal_conv1d_bwd_primitive",
    mutates_args=(),
    device_types="cuda",
)
def _b2b_bwd_primitive(
    grad_y: Tensor,
    x: Tensor,
    weights_proj: Tensor,
    weights_mixer: Tensor,
    skip_bias: Tensor,
    y: Tensor,
) -> List[Tensor]:
    if x.dim() != 3:
        raise ValueError(f"Expected x(3D) with shape (batch, 3*dim, seq_len); got {x.shape}")
    if x.shape[1] % 3 != 0:
        raise ValueError(f"Expected x.shape[1] divisible by 3; got {x.shape[1]}")
    batch = x.shape[0]
    dim = x.shape[1] // 3
    seq_len = x.shape[2]

    if weights_proj.dim() != 2 or weights_mixer.dim() != 2 or skip_bias.dim() != 1:
        raise ValueError(
            f"Expected weights_proj(2D: 3*dim,K_proj), weights_mixer(2D: dim,K_mixer), skip_bias(1D: dim); "
            f"got {weights_proj.shape}, {weights_mixer.shape}, {skip_bias.shape}"
        )
    if y.shape != (batch, dim, seq_len):
        raise ValueError(f"Shape mismatch: expected y shape {(batch, dim, seq_len)}; got {y.shape}")
    if grad_y.shape != y.shape:
        raise ValueError(f"Shape mismatch: grad_y {grad_y.shape} vs y {y.shape}")
    if not (x.is_cuda and weights_proj.is_cuda and weights_mixer.is_cuda and skip_bias.is_cuda and y.is_cuda and grad_y.is_cuda):
        raise ValueError("All tensors must be on CUDA")
    if not (x.device == weights_proj.device == weights_mixer.device == skip_bias.device == y.device == grad_y.device):
        raise ValueError("All tensors must be on the same device")

    if not (x.dtype == weights_proj.dtype == weights_mixer.dtype == skip_bias.dtype == y.dtype == grad_y.dtype):
        raise TypeError(
            f"Dtype mismatch: x.dtype={x.dtype}, weights_proj.dtype={weights_proj.dtype}, "
            f"weights_mixer.dtype={weights_mixer.dtype}, skip_bias.dtype={skip_bias.dtype}, "
            f"y.dtype={y.dtype}, grad_y.dtype={grad_y.dtype}"
        )

    if weights_proj.shape[0] != 3 * dim:
        raise ValueError(f"Channel mismatch: x has 3*dim={3*dim} but weights_proj has shape {weights_proj.shape}")
    if weights_mixer.shape[0] != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim} but weights_mixer has shape {weights_mixer.shape}")
    if skip_bias.shape[0] != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim} but skip_bias has shape {skip_bias.shape}")

    x = x.contiguous()
    weights_proj = weights_proj.contiguous()
    weights_mixer = weights_mixer.contiguous()
    skip_bias = skip_bias.contiguous()
    y = y.contiguous()
    grad_y = grad_y.contiguous()

    kernel_size_proj = weights_proj.shape[1]
    kernel_size_mixer = weights_mixer.shape[1]

    dx = torch.empty_like(x)
    dweights_proj = torch.zeros(weights_proj.shape, device=x.device, dtype=torch.float32)
    dweights_mixer = torch.zeros(weights_mixer.shape, device=x.device, dtype=torch.float32)
    dskip_bias = torch.zeros(skip_bias.shape, device=x.device, dtype=torch.float32)

    import cudnn

    cudnn.b2b_causal_conv1d_backward(
        torch.cuda.current_stream().cuda_stream,
        x.data_ptr(),
        weights_proj.data_ptr(),
        weights_mixer.data_ptr(),
        skip_bias.data_ptr(),
        y.data_ptr(),
        grad_y.data_ptr(),
        dx.data_ptr(),
        dweights_proj.data_ptr(),
        dweights_mixer.data_ptr(),
        dskip_bias.data_ptr(),
        batch,
        dim,
        seq_len,
        kernel_size_proj,
        kernel_size_mixer,
        _dtype_to_int(x.dtype),
        _dtype_to_int(torch.float32),
    )
    return [
        dx,
        dweights_proj.to(x.dtype),
        dweights_mixer.to(x.dtype),
        dskip_bias.to(x.dtype),
    ]


@torch.library.register_fake("cudnn::b2b_causal_conv1d_bwd_primitive")
def _b2b_bwd_fake(
    grad_y: Tensor,
    x: Tensor,
    weights_proj: Tensor,
    weights_mixer: Tensor,
    skip_bias: Tensor,
    y: Tensor,
) -> List[Tensor]:
    return [
        torch.empty_like(x),
        torch.empty_like(weights_proj),
        torch.empty_like(weights_mixer),
        torch.empty_like(skip_bias),
    ]


# ---------------------------------------------------------------------------
# B2B Autograd glue
# ---------------------------------------------------------------------------


def _b2b_setup_context(ctx, inputs, output):
    x, weights_proj, weights_mixer, skip_bias = inputs
    y = output[0]
    ctx.save_for_backward(x, weights_proj, weights_mixer, skip_bias, y)


@torch.compiler.allow_in_graph
def _b2b_autograd_bwd(ctx, grad_y, grad_y_gated):
    # PyTorch may pass a ZeroTensor for the discarded intermediate output.
    if grad_y is not None and not torch._is_zerotensor(grad_y) and torch.count_nonzero(grad_y).item() != 0:
        raise RuntimeError("Gradient for the intermediate B2B output y is not supported; use cudnn.ops.b2b_causal_conv1d")
    x, weights_proj, weights_mixer, skip_bias, y = ctx.saved_tensors
    dx, dwp, dwm, dsb = torch.ops.cudnn.b2b_causal_conv1d_bwd_primitive(grad_y_gated, x, weights_proj, weights_mixer, skip_bias, y)
    return dx, dwp, dwm, dsb


torch.library.register_autograd(
    "cudnn::b2b_causal_conv1d_fwd_primitive",
    _b2b_autograd_bwd,
    setup_context=_b2b_setup_context,
)


# ---------------------------------------------------------------------------
# B2B Public API
# ---------------------------------------------------------------------------


def b2b_causal_conv1d(
    x: Tensor,
    weights_proj: Tensor,
    weights_mixer: Tensor,
    skip_bias: Tensor,
) -> Tensor:
    r"""Fused back-to-back causal conv1d: projection conv, gating, mixer conv, and post-gating.

    Computes the fused Hyena-SE block::

        proj       = causal_conv1d(x, weights_proj)            # (batch, 3*dim, seq_len)
        gated      = proj[:, 1::3, :] * proj[:, 2::3, :]       # Q * K gate
        y          = causal_conv1d(gated, weights_mixer) + skip_bias[:, None] * gated
        y_gated    = y * proj[:, 0::3, :]                      # final * V

    Supports ``torch.compile`` and ``torch.autograd`` — backward is handled
    automatically when inputs require gradients.

    Args:
        x (torch.Tensor): Input tensor of shape ``(batch, 3*dim, seq_len)``.
        weights_proj (torch.Tensor): Projection filter ``(3*dim, kernel_size_proj)``.
            ``kernel_size_proj`` must be between 2 and 32, inclusive.
        weights_mixer (torch.Tensor): Mixer filter ``(dim, kernel_size_mixer)``.
            ``kernel_size_mixer`` must be between 2 and 256, inclusive.
        skip_bias (torch.Tensor): Skip-connection bias ``(dim,)``.

    Returns:
        torch.Tensor: ``y_gated`` of shape ``(batch, dim, seq_len)`` — the
        post-gated final output of the fused Hyena-SE block.
    """
    _, y_gated = torch.ops.cudnn.b2b_causal_conv1d_fwd_primitive(x, weights_proj, weights_mixer, skip_bias)
    return y_gated
