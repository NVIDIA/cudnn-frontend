# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-facing causal-convolution decode update implementation."""

from typing import Optional

import torch
from torch import Tensor


def _normalize_activation(activation: Optional[str]) -> str:
    if activation is None or activation == "identity":
        return "identity"
    if activation in ("silu", "swish"):
        return "silu"
    raise ValueError("activation must be None, 'identity', 'silu', or 'swish', " f"got {activation!r}")


def _validate_semantic_contract(
    x: Tensor,
    conv_state: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    cache_seqlens: Optional[Tensor],
    conv_state_indices: Optional[Tensor],
) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must have shape [N, D], got {tuple(x.shape)}")
    if conv_state.ndim != 3:
        raise ValueError(f"conv_state must have shape [S, D, L], got {tuple(conv_state.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [D, W], got {tuple(weight.shape)}")

    n_rows, n_channels = x.shape
    n_slots = conv_state.shape[0]
    if n_rows <= 0:
        raise ValueError(f"x row count N must be positive, got N={n_rows}")
    if n_channels <= 0:
        raise ValueError(f"x channel count D must be positive, got D={n_channels}")
    if n_slots <= 0:
        raise ValueError(f"conv_state slot count S must be positive, got S={n_slots}")
    width = weight.shape[1]
    state_len = conv_state.shape[2]
    if width < 1:
        raise ValueError(f"weight width W must be positive, got W={width}")
    if weight.shape[0] != n_channels:
        raise ValueError(f"weight must have shape [D, W] with D={n_channels}, got {tuple(weight.shape)}")
    if conv_state.shape[1] != n_channels:
        raise ValueError(f"conv_state must have shape [S, D, L] with D={n_channels}, got {tuple(conv_state.shape)}")
    if state_len < width - 1:
        raise ValueError(f"conv_state length L must satisfy L >= W - 1, got L={state_len}, W={width}")
    if bias is not None and tuple(bias.shape) != (n_channels,):
        raise ValueError(f"bias must have shape {(n_channels,)}, got {tuple(bias.shape)}")
    if conv_state_indices is None:
        if n_slots < n_rows:
            raise ValueError(f"conv_state needs at least N slots when conv_state_indices is omitted, got S={n_slots}, N={n_rows}")
    elif tuple(conv_state_indices.shape) != (n_rows,):
        raise ValueError(f"conv_state_indices must have shape {(n_rows,)}, got {tuple(conv_state_indices.shape)}")
    if cache_seqlens is not None and tuple(cache_seqlens.shape) != (n_rows,):
        raise ValueError(f"cache_seqlens must have shape {(n_rows,)}, got {tuple(cache_seqlens.shape)}")

    tensors = (("x", x), ("conv_state", conv_state), ("weight", weight), ("bias", bias))
    for name, tensor in tensors:
        if tensor is None:
            continue
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype torch.bfloat16, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.device != x.device:
            raise ValueError(f"{name} must be on {x.device}, got {tensor.device}")
    for name, tensor in (("cache_seqlens", cache_seqlens), ("conv_state_indices", conv_state_indices)):
        if tensor is None:
            continue
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.device != x.device:
            raise ValueError(f"{name} must be on {x.device}, got {tensor.device}")


def _require_native_subset(
    x: Tensor,
    conv_state: Tensor,
    weight: Tensor,
    cache_seqlens: Optional[Tensor],
) -> None:
    """Decline semantic configurations the current native kernel cannot run."""

    width = weight.shape[1]
    state_len = conv_state.shape[2]
    if width != 4 or state_len not in (3, 4) or cache_seqlens is not None:
        circular = "present" if cache_seqlens is not None else "omitted"
        raise NotImplementedError(
            "the current native causal_conv1d_update kernel supports only one-token "
            "x[N, D], weight[D, 4], conv_state[S, D, L] with L in {3, 4}, "
            "and cache_seqlens=None; "
            f"got x{tuple(x.shape)}, weight{tuple(weight.shape)}, "
            f"conv_state{tuple(conv_state.shape)}, cache_seqlens={circular}"
        )


@torch.library.custom_op(
    "cudnn::_causal_conv1d_update",
    mutates_args=("conv_state",),
    device_types="cuda",
)
def _causal_conv1d_update_primitive(
    x: Tensor,
    conv_state: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    activation: str,
    cache_seqlens: Optional[Tensor],
    conv_state_indices: Optional[Tensor],
) -> Tensor:
    from cudnn.causal_conv1d_update_sm100 import _causal_conv1d_update

    activation = _normalize_activation(activation)
    _validate_semantic_contract(x, conv_state, weight, bias, cache_seqlens, conv_state_indices)
    _require_native_subset(x, conv_state, weight, cache_seqlens)
    return _causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation,
        conv_state_indices=conv_state_indices,
    )


@torch.library.register_fake("cudnn::_causal_conv1d_update")
def _causal_conv1d_update_fake(
    x: Tensor,
    conv_state: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    activation: str,
    cache_seqlens: Optional[Tensor],
    conv_state_indices: Optional[Tensor],
) -> Tensor:
    _normalize_activation(activation)
    _validate_semantic_contract(x, conv_state, weight, bias, cache_seqlens, conv_state_indices)
    _require_native_subset(x, conv_state, weight, cache_seqlens)
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def causal_conv1d_update(
    x: Tensor,
    conv_state: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    activation: Optional[str] = None,
    *,
    cache_seqlens: Optional[Tensor] = None,
    conv_state_indices: Optional[Tensor] = None,
) -> Tensor:
    r"""Advance a mutable causal-convolution state for one decode step.

    ``conv_state`` is mutated in place. The returned Tensor is newly allocated
    and has the same shape and dtype as ``x``. ``activation=None`` and
    ``"identity"`` select the identity epilogue; ``"silu"`` and ``"swish"``
    select the same compile-time SiLU specialization.

    Args:
        x: Contiguous CUDA BF16 tensor with shape ``[N, D]``.
        conv_state: Contiguous CUDA BF16 tensor with shape ``[S, D, L]``, where
            ``L >= W - 1``.
        weight: Contiguous CUDA BF16 tensor with shape ``[D, W]``.
        bias: Optional contiguous CUDA BF16 tensor with shape ``[D]``.
        activation: ``None``, ``"identity"``, ``"silu"``, or ``"swish"``.
        cache_seqlens: Optional contiguous CUDA int32 tensor with shape ``[N]``.
            When present, ``conv_state`` is interpreted as a circular buffer.
        conv_state_indices: Optional contiguous CUDA int32 tensor with shape
            ``[N]`` selecting unique state slots.

    Returns:
        A contiguous CUDA BF16 Tensor with shape ``[N, D]``.
    """

    for name, tensor in (("x", x), ("conv_state", conv_state), ("weight", weight)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError(f"bias must be a torch.Tensor or None, got {type(bias).__name__}")
    if cache_seqlens is not None and not isinstance(cache_seqlens, torch.Tensor):
        raise TypeError(f"cache_seqlens must be a torch.Tensor or None, got {type(cache_seqlens).__name__}")
    if conv_state_indices is not None and not isinstance(conv_state_indices, torch.Tensor):
        raise TypeError("conv_state_indices must be a torch.Tensor or None, " f"got {type(conv_state_indices).__name__}")
    if not x.is_cuda and x.device.type != "meta":
        raise ValueError(f"x must be a CUDA tensor, got {x.device}")

    normalized_activation = _normalize_activation(activation)
    grad_tensors = (x, conv_state, weight, bias)
    if torch.is_grad_enabled() and any(tensor is not None and tensor.requires_grad for tensor in grad_tensors):
        raise RuntimeError("causal_conv1d_update is inference-only; call it under torch.no_grad()")

    return torch.ops.cudnn._causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        normalized_activation,
        cache_seqlens,
        conv_state_indices,
    )


__all__ = ["causal_conv1d_update"]
