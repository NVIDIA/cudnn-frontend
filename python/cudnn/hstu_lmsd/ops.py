# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allocation-and-dispatch helpers for the explicit HSTU LMSD operation."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from typing import Optional

from cuda.bindings import driver as cuda
import torch

from cudnn.api_base import TupleDict

from .api import HSTULMSDBwdSm100, HSTULMSDFwdSm100
from .cutedsl.cute_dsl_ln_mul_dropout_bwd import TARGET_TILES

_CACHE_CAPACITY = 128
_FWD_CACHE: OrderedDict = OrderedDict()
_BWD_CACHE: OrderedDict = OrderedDict()


def _tensor_signature(tensor: torch.Tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _cache_get(cache: OrderedDict, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_put(cache: OrderedDict, key, value) -> None:
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > _CACHE_CAPACITY:
        cache.popitem(last=False)


def _as_torch_stream(stream, device: torch.device) -> torch.cuda.Stream:
    if isinstance(stream, torch.cuda.Stream):
        if stream.device != device:
            raise ValueError(f"stream must be on {device}, got {stream.device}")
        return stream
    if int(stream) == 0:
        return torch.cuda.default_stream(device)
    return torch.cuda.ExternalStream(int(stream), device=device)


def _allocation_context(stream, device: torch.device):
    if stream is None:
        return nullcontext()
    return torch.cuda.stream(_as_torch_stream(stream, device))


def hstu_lmsd_forward(
    x_tensor: torch.Tensor,
    u_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    bias_tensor: torch.Tensor,
    eps: float = 1e-6,
    dropout_ratio: float = 0.1,
    seed: int = 0,
    stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
) -> TupleDict:
    """Run explicit LMSD forward and return all backward-save tensors.

    This is an operation API, not an autograd registration. The returned mask
    packs three keep decisions per element: bit 0 for the LMSD result, bit 1
    for the copied x segment, and bit 2 for the SiLU(u) segment.
    """
    if x_tensor.ndim != 2:
        raise ValueError("x_tensor must be rank 2")
    n, d = x_tensor.shape
    with torch.cuda.device(x_tensor.device), _allocation_context(
        stream, x_tensor.device
    ):
        y_tensor = torch.empty(
            (n, 3 * d), dtype=x_tensor.dtype, device=x_tensor.device
        )
        mean_tensor = torch.empty((n,), dtype=torch.float32, device=x_tensor.device)
        rstd_tensor = torch.empty((n,), dtype=torch.float32, device=x_tensor.device)
        mask_tensor = torch.empty((n, d), dtype=torch.int8, device=x_tensor.device)

    key = (
        _tensor_signature(x_tensor),
        _tensor_signature(u_tensor),
        _tensor_signature(weight_tensor),
        _tensor_signature(bias_tensor),
        _tensor_signature(y_tensor),
        _tensor_signature(mean_tensor),
        _tensor_signature(rstd_tensor),
        _tensor_signature(mask_tensor),
        float(eps),
        float(dropout_ratio),
    )
    api = _cache_get(_FWD_CACHE, key)
    if api is None:
        api = HSTULMSDFwdSm100(
            sample_x=x_tensor,
            sample_u=u_tensor,
            sample_weight=weight_tensor,
            sample_bias=bias_tensor,
            sample_y=y_tensor,
            sample_mean=mean_tensor,
            sample_rstd=rstd_tensor,
            sample_mask=mask_tensor,
            eps=eps,
            dropout_ratio=dropout_ratio,
        )
        api.check_support()
        api.compile()
        _cache_put(_FWD_CACHE, key, api)
    api.execute(
        x_tensor=x_tensor,
        u_tensor=u_tensor,
        weight_tensor=weight_tensor,
        bias_tensor=bias_tensor,
        y_tensor=y_tensor,
        mean_tensor=mean_tensor,
        rstd_tensor=rstd_tensor,
        mask_tensor=mask_tensor,
        seed=seed,
        current_stream=stream,
    )
    return TupleDict(
        y_tensor=y_tensor,
        mean_tensor=mean_tensor,
        rstd_tensor=rstd_tensor,
        mask_tensor=mask_tensor,
    )


def hstu_lmsd_backward(
    dy_tensor: torch.Tensor,
    x_tensor: torch.Tensor,
    u_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    bias_tensor: torch.Tensor,
    mean_tensor: torch.Tensor,
    rstd_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    dropout_ratio: float = 0.1,
    stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
    dx_tensor: Optional[torch.Tensor] = None,
    du_tensor: Optional[torch.Tensor] = None,
    dweight_tensor: Optional[torch.Tensor] = None,
    dbias_tensor: Optional[torch.Tensor] = None,
) -> TupleDict:
    """Run explicit LMSD backward without recomputing the forward output.

    Missing gradient outputs are allocated here. The lower-level class API
    accepts caller-owned workspaces and performs no allocation in execute().
    """
    if x_tensor.ndim != 2:
        raise ValueError("x_tensor must be rank 2")
    n, d = x_tensor.shape
    with torch.cuda.device(x_tensor.device), _allocation_context(
        stream, x_tensor.device
    ):
        if dx_tensor is None:
            dx_tensor = torch.empty((n, d), dtype=x_tensor.dtype, device=x_tensor.device)
        if du_tensor is None:
            du_tensor = torch.empty((n, d), dtype=x_tensor.dtype, device=x_tensor.device)
        if dweight_tensor is None:
            dweight_tensor = torch.empty((d,), dtype=weight_tensor.dtype, device=x_tensor.device)
        if dbias_tensor is None:
            dbias_tensor = torch.empty((d,), dtype=bias_tensor.dtype, device=x_tensor.device)
        dweight_workspace = torch.empty(
            (TARGET_TILES, d), dtype=torch.float32, device=x_tensor.device
        )
        dbias_workspace = torch.empty(
            (TARGET_TILES, d), dtype=torch.float32, device=x_tensor.device
        )

    key = (
        _tensor_signature(dy_tensor),
        _tensor_signature(x_tensor),
        _tensor_signature(u_tensor),
        _tensor_signature(weight_tensor),
        _tensor_signature(bias_tensor),
        _tensor_signature(mean_tensor),
        _tensor_signature(rstd_tensor),
        _tensor_signature(mask_tensor),
        _tensor_signature(dx_tensor),
        _tensor_signature(du_tensor),
        _tensor_signature(dweight_tensor),
        _tensor_signature(dbias_tensor),
        _tensor_signature(dweight_workspace),
        _tensor_signature(dbias_workspace),
        float(dropout_ratio),
    )
    api = _cache_get(_BWD_CACHE, key)
    if api is None:
        api = HSTULMSDBwdSm100(
            sample_dy=dy_tensor,
            sample_x=x_tensor,
            sample_u=u_tensor,
            sample_weight=weight_tensor,
            sample_bias=bias_tensor,
            sample_mean=mean_tensor,
            sample_rstd=rstd_tensor,
            sample_mask=mask_tensor,
            sample_dx=dx_tensor,
            sample_du=du_tensor,
            sample_dweight=dweight_tensor,
            sample_dbias=dbias_tensor,
            sample_dweight_workspace=dweight_workspace,
            sample_dbias_workspace=dbias_workspace,
            dropout_ratio=dropout_ratio,
        )
        api.check_support()
        api.compile()
        _cache_put(_BWD_CACHE, key, api)
    api.execute(
        dy_tensor=dy_tensor,
        x_tensor=x_tensor,
        u_tensor=u_tensor,
        weight_tensor=weight_tensor,
        bias_tensor=bias_tensor,
        mean_tensor=mean_tensor,
        rstd_tensor=rstd_tensor,
        mask_tensor=mask_tensor,
        dx_tensor=dx_tensor,
        du_tensor=du_tensor,
        dweight_tensor=dweight_tensor,
        dbias_tensor=dbias_tensor,
        dweight_workspace=dweight_workspace,
        dbias_workspace=dbias_workspace,
        current_stream=stream,
    )
    return TupleDict(
        dx_tensor=dx_tensor,
        du_tensor=du_tensor,
        dweight_tensor=dweight_tensor,
        dbias_tensor=dbias_tensor,
    )
