# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN acceleration for FLA 0.5.2's decode short-convolution entry point.

This adapter was implemented independently against FLA's public call contract.
It does not include or translate FLA's Triton kernel implementation.  The
supported path only normalizes FLA's decode layouts into zero-copy 2D views and
calls :func:`cudnn.ops.causal_conv1d_update`; every other configuration calls the
original FLA function unchanged.
"""

from __future__ import annotations

import functools
from importlib import metadata

import torch

import cudnn
from cudnn._causal_conv1d_arch import (
    is_supported_causal_conv1d_update_compute_capability,
)

_SUPPORTED_FLA_VERSION = "0.5.2"
_DECLINE_ERRORS = (NotImplementedError, cudnn.cudnnGraphNotSupportedError, ImportError)
_LAST = {"path": None}


def last_path() -> str | None:
    """Return the route taken by the most recent shimmed call."""

    return _LAST["path"]


def _installed_fla_version() -> str | None:
    try:
        return metadata.version("flash-linear-attention")
    except metadata.PackageNotFoundError:
        return None


def _supports_installed_fla() -> bool:
    return _installed_fla_version() == _SUPPORTED_FLA_VERSION


def _is_cuda_tensor(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda


def _device_capability(device: torch.device) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and compiler.is_compiling():
        return True
    dynamo = getattr(torch, "_dynamo", None)
    return bool(dynamo is not None and dynamo.is_compiling())


def _call_native(x: torch.Tensor, cache: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Resolve lazily so importing cudnn.fla does not import the optional
    # CuTeDSL stack or compile a kernel.
    from cudnn.ops import causal_conv1d_update

    return causal_conv1d_update(x, cache, weight, activation="silu")


def _native_input_view(x: torch.Tensor, weight: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    """Validate the native contract and return a zero-copy row-strided ``[N, D]`` view."""

    if type(x) is not torch.Tensor or type(weight) is not torch.Tensor or type(cache) is not torch.Tensor:
        raise ValueError("tensor-subclass")
    if _is_compiling():
        raise ValueError("compile")
    if not (x.dtype == weight.dtype == cache.dtype == torch.bfloat16):
        raise ValueError("non-bf16")
    if not (_is_cuda_tensor(x) and _is_cuda_tensor(weight) and _is_cuda_tensor(cache)):
        raise ValueError("non-cuda")
    if not (x.device == weight.device == cache.device):
        raise ValueError("device")
    if not is_supported_causal_conv1d_update_compute_capability(_device_capability(x.device)):
        raise ValueError("unsupported-arch")
    if not (weight.is_contiguous() and cache.is_contiguous()):
        raise ValueError("noncontiguous")

    if x.ndim == 2:
        n_rows, n_channels = x.shape
    elif x.ndim == 3 and x.shape[1] == 1:
        n_rows, _, n_channels = x.shape
    elif x.ndim == 3 and x.shape[0] == 1:
        _, n_rows, n_channels = x.shape
    else:
        raise ValueError("shape")
    if n_rows <= 0 or n_channels <= 0:
        raise ValueError("shape")

    # Each admitted 3D layout only adds a singleton dimension, so view() is a
    # zero-copy normalization.  Preserve X's leading dimension: fused QKV
    # projections commonly expose each slice as (3 * D, 1), which the native
    # kernel addresses directly without an adapter-side copy.
    try:
        native_x = x.view(n_rows, n_channels)
    except RuntimeError as error:
        raise ValueError("noncontiguous") from error
    row_stride, channel_stride = native_x.stride()
    if channel_stride != 1 or row_stride < n_channels:
        raise ValueError("noncontiguous")
    if row_stride > n_channels and row_stride % 8 != 0:
        raise ValueError("alignment")

    if tuple(weight.shape) != (n_channels, 4) or tuple(cache.shape) != (n_rows, n_channels, 4):
        raise ValueError("shape")
    if x.data_ptr() % 16 or weight.data_ptr() % 16 or cache.data_ptr() % 16:
        raise ValueError("alignment")
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (x, weight, cache)):
        raise ValueError("autograd")

    # The singleton layout and explicit stride checks above make this a true
    # view; no adapter-side allocation, gather, or copy is permitted here.
    return native_x


def make_causal_conv1d_update(real_fn):
    """Wrap FLA's public decode update with a narrow cuDNN native path."""

    @functools.wraps(real_fn)
    def causal_conv1d_update(
        x,
        cache,
        residual=None,
        weight=None,
        bias=None,
        activation=None,
    ):
        def fallback(reason):
            _LAST["path"] = f"fallback:{reason}"
            return real_fn(
                x,
                cache,
                residual=residual,
                weight=weight,
                bias=bias,
                activation=activation,
            )

        if residual is not None:
            return fallback("residual")
        if weight is None:
            return fallback("weight")
        if bias is not None:
            return fallback("bias")
        if activation not in ("silu", "swish"):
            return fallback("activation")

        try:
            native_x = _native_input_view(x, weight, cache)
        except (TypeError, ValueError) as error:
            return fallback(str(error) or type(error).__name__)

        try:
            native_y = _call_native(native_x, cache, weight)
        except _DECLINE_ERRORS as error:
            return fallback(type(error).__name__)
        except Exception as error:
            # Binding/allocation/launch failures are not evidence that the
            # configuration is unsupported.  Surface them to the caller.
            _LAST["path"] = f"error:{type(error).__name__}"
            raise

        _LAST["path"] = "native"
        return native_y.view(x.shape), cache

    return causal_conv1d_update


__all__ = ["last_path", "make_causal_conv1d_update"]
