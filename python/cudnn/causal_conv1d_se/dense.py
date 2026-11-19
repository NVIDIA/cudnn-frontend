# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native channel-last SE autograd adapter for SM103 BF16 width-four SiLU."""

from functools import cache
from hashlib import sha256
from pathlib import Path
import warnings

import torch
from torch import Tensor

from .packed import _require_dsl


@cache
def _source_digest():
    here = Path(__file__).resolve().parent
    return sha256(
        b"".join(
            (here / name).read_bytes()
            for name in (
                "dense.py",
                "kernel.py",
                "backward_kernel.py",
                "persistent_backward.py",
                "packed_kernel.py",
            )
        )
    ).hexdigest()


def _persistent_shape(shape, capability):
    batch, tokens, channels = shape
    return capability == (10, 3) and channels == 2048 and batch * tokens * channels >= 100_000_000 and 1 < tokens and batch * tokens <= 2**31 - 257


def _compiled(kind, tensors):
    _require_dsl()
    import cutlass
    from cutlass.cute.runtime import make_fake_tensor
    from cuda.bindings import driver as cuda
    from cudnn.frost.compiled_cache import compile_cached

    if kind == "forward":
        from .kernel import launch_fwd as entry

        extra = (4, True, True, 8, 8, False)
    elif kind == "backward":
        from .backward_kernel import launch_bwd as entry

        extra = (4, True, True, 32)
    else:
        from .persistent_backward import launch_supported_backward as entry

        extra = (4, True, True, True, True)
    dtypes = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32, torch.int32: cutlass.Int32}
    args, signature = [], []
    for t in tensors:
        if t is None:
            args.append(None)
            signature.append(None)
        else:
            args.append(make_fake_tensor(dtypes[t.dtype], tuple(t.shape), tuple(t.stride())))
            signature.append((str(t.dtype), tuple(t.shape), tuple(t.stride())))
    key = repr((_source_digest(), kind, signature))
    with torch.cuda.device(tensors[0].device):
        return compile_cached(entry, *args, cuda.CUstream(0), *extra, cache_key=key, symbol="causal_conv1d_se_dense_" + kind, options="--enable-tvm-ffi")


def _launch(kind, tensors):
    from cuda.bindings import driver as cuda

    with torch.cuda.device(tensors[0].device):
        fn = _compiled(kind, tensors)
        fn(*tensors, cuda.CUstream(torch.cuda.current_stream(tensors[0].device).cuda_stream))


@torch.library.custom_op("cudnn::causal_conv1d_dense_se_fwd", mutates_args=(), device_types="cuda")
def _forward(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    _require_dsl()
    if torch.cuda.get_device_capability(x.device) != (10, 3):
        raise NotImplementedError("dense SE causal_conv1d is qualified on SM103")
    if any(t.data_ptr() % 16 for t in (x, weight, bias)):
        raise ValueError("dense SE tensors require 16-byte aligned storage")
    y = torch.empty_like(x)
    _launch("forward", (x, weight, bias, y))
    return y


@_forward.register_fake
def _forward_fake(x, weight, bias):
    return torch.empty_like(x)


@torch.library.custom_op("cudnn::causal_conv1d_dense_se_bwd", mutates_args=(), device_types="cuda")
def _backward(x: Tensor, weight: Tensor, bias: Tensor, dy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    with torch.cuda.device(x.device):
        capability = torch.cuda.get_device_capability(x.device)
        multiprocessors = torch.cuda.get_device_properties(x.device).multi_processor_count
    persistent = _persistent_shape(x.shape, capability)
    if not persistent and torch.are_deterministic_algorithms_enabled():
        message = "dense SE tiled backward uses atomic parameter reductions; deterministic mode requires the large SM103 persistent route or packed SE"
        if not torch.is_deterministic_algorithms_warn_only_enabled():
            raise RuntimeError(message)
        warnings.warn(message, UserWarning, stacklevel=2)
    # Autograd may provide broadcast gradients; materialize them at this allocating API boundary.
    dy = dy.contiguous()
    dx = torch.empty_like(x)
    dz = torch.empty_like(x, dtype=torch.float32)
    if persistent:
        batch, tokens, channels = x.shape
        cu = torch.arange(batch + 1, dtype=torch.int32, device=x.device) * tokens
        dw, db = torch.empty_like(weight), torch.empty_like(bias)
        partials = torch.empty((max(1, (2 * multiprocessors) // ((channels + 255) // 256)), 5, channels), device=x.device, dtype=torch.float32)
        args = (
            x.view(-1, channels),
            weight.T,
            bias,
            cu,
            None,
            dy.view(-1, channels),
            None,
            dz.view(-1, channels),
            dx.view(-1, channels),
            dw.T,
            db,
            None,
            partials,
        )
        _launch("persistent", args)
    else:
        dw = torch.empty_like(weight, dtype=torch.float32)
        db = torch.empty_like(bias, dtype=torch.float32)
        _launch("backward", (x, weight, bias, dy, dz, dx, dw, db))
        dw, db = dw.to(weight.dtype), db.to(bias.dtype)
    return dx, dw, db


@_backward.register_fake
def _backward_fake(x, weight, bias, dy):
    return torch.empty_like(x), torch.empty_like(weight), torch.empty_like(bias)


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _autograd(ctx, dy):
    return _backward(*ctx.saved_tensors, dy)


_forward.register_autograd(_autograd, setup_context=_setup_context)


def dense_causal_conv1d(x, weight, bias, activation):
    if not x.is_cuda or x.dtype != torch.bfloat16 or weight.dtype != x.dtype:
        raise NotImplementedError("dense SE causal_conv1d requires BF16 CUDA activations and weights")
    if activation != "silu" or weight.shape[1] != 4 or x.shape[1] % 8 or x.shape[1] == 0:
        raise NotImplementedError("dense SE causal_conv1d requires width four, SiLU, and channels divisible by eight")
    if x.shape[0] <= 0 or x.shape[2] <= 0:
        raise ValueError("dense SE causal_conv1d requires positive batch and token counts; use packed offsets for empty inputs")
    # The existing SM103 K4 forward schedules put batch in grid.z and token
    # tiles in grid.y. Reject unsupported launch geometry before compiling.
    elements = x.shape[0] * x.shape[1] * x.shape[2]
    forward_tile = 8 if elements < 32_000_000 else 512 if elements < 100_000_000 else 256
    if x.shape[0] > 65535 or (x.shape[2] + forward_tile - 1) // forward_tile > 65535 or x.shape[1] > 2**31 - 1024:
        raise NotImplementedError("dense SE causal_conv1d exceeds its CUDA grid limits; use packed offsets")
    if weight.device != x.device or (bias is not None and (bias.device != x.device or bias.dtype != x.dtype)):
        raise ValueError("dense SE weight and bias must match the input dtype and device")
    x_btd = x.transpose(1, 2)
    if not x_btd.is_contiguous():
        raise ValueError("dense SE causal_conv1d requires native channel-last physical storage")
    # The existing dense SE kernels consume contiguous [K, C] filters.
    # Prepare once for forward and retain that same tensor for backward.
    weight_kc = weight.T.contiguous()
    bias = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device) if bias is None else bias.contiguous()
    return _forward(x_btd, weight_kc, bias).transpose(1, 2)
