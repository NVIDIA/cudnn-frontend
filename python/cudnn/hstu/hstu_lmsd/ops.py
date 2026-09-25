# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allocation-and-dispatch helpers for the explicit HSTU LMSD operation."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

from cuda.bindings import driver as cuda
import torch

from cudnn.api_base import TupleDict

from ._runtime import allocation_context, tensor_signature
from .api import HSTULMSDBwd, HSTULMSDFwd
from ._kernels._common import normalize_dropout_ratio
from ._kernels._config import HSTULMSDBwdConfig

_CACHE_CAPACITY = 128
_FWD_CACHE: OrderedDict = OrderedDict()
_BWD_CACHE: OrderedDict = OrderedDict()


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


def _tensor_signature_or_none(tensor: Optional[torch.Tensor], *, dynamic_rows: bool = False):
    return None if tensor is None else tensor_signature(tensor, dynamic_rows=dynamic_rows)


def hstu_lmsd_forward(
    x_tensor: torch.Tensor,
    u_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    bias_tensor: torch.Tensor,
    eps: float = 1e-6,
    dropout_ratio: float = 0.1,
    seed: int = 0,
    stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
    apply_u_silu: bool = True,
    concat_u: bool = True,
    concat_x: bool = True,
) -> TupleDict:
    """Run explicit LMSD forward and return all backward-save tensors.

    This is an operation API, not an autograd registration. The mandatory LMSD
    segment follows the optional activated-u and x segments. With dropout, the
    returned mask packs their keep decisions into bits 0, 2, and 1,
    respectively; without dropout, mask_tensor is None.
    """
    if x_tensor.ndim != 2:
        raise ValueError("x_tensor must be rank 2")
    dropout_ratio = normalize_dropout_ratio(dropout_ratio)
    apply_u_silu = bool(apply_u_silu)
    concat_u = bool(concat_u)
    concat_x = bool(concat_x)
    has_dropout = dropout_ratio > 0.0
    n, d = x_tensor.shape
    output_segments = 1 + int(concat_u) + int(concat_x)
    with torch.cuda.device(x_tensor.device), allocation_context(stream, x_tensor.device):
        y_tensor = torch.empty((n, output_segments * d), dtype=x_tensor.dtype, device=x_tensor.device)
        mean_tensor = torch.empty((n,), dtype=torch.float32, device=x_tensor.device)
        rstd_tensor = torch.empty((n,), dtype=torch.float32, device=x_tensor.device)
        mask_tensor = torch.empty((n, d), dtype=torch.int8, device=x_tensor.device) if has_dropout else None

    key = (
        tensor_signature(x_tensor, dynamic_rows=True),
        tensor_signature(u_tensor, dynamic_rows=True),
        tensor_signature(weight_tensor),
        tensor_signature(bias_tensor),
        tensor_signature(y_tensor, dynamic_rows=True),
        tensor_signature(mean_tensor, dynamic_rows=True),
        tensor_signature(rstd_tensor, dynamic_rows=True),
        _tensor_signature_or_none(mask_tensor, dynamic_rows=True),
        float(eps),
        float(dropout_ratio),
        apply_u_silu,
        concat_u,
        concat_x,
    )
    api = _cache_get(_FWD_CACHE, key)
    if api is None:
        with torch.cuda.device(x_tensor.device):
            api = HSTULMSDFwd(
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
                apply_u_silu=apply_u_silu,
                concat_u=concat_u,
                concat_x=concat_x,
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
    mask_tensor: Optional[torch.Tensor],
    dropout_ratio: Optional[float] = None,
    stream: Optional[cuda.CUstream | torch.cuda.Stream] = None,
    dx_tensor: Optional[torch.Tensor] = None,
    du_tensor: Optional[torch.Tensor] = None,
    dweight_tensor: Optional[torch.Tensor] = None,
    dbias_tensor: Optional[torch.Tensor] = None,
    apply_u_silu: Optional[bool] = None,
    concat_u: Optional[bool] = None,
    concat_x: Optional[bool] = None,
    compute_dweight: bool = True,
) -> TupleDict:
    """Run explicit LMSD backward without recomputing the forward output.

    ``dropout_ratio``, ``apply_u_silu``, ``concat_u``, and ``concat_x`` must
    all be provided and must match the forward configuration. Missing enabled
    gradient outputs are allocated here. Set
    ``compute_dweight=False`` for a non-trainable weight.
    """
    if x_tensor.ndim != 2:
        raise ValueError("x_tensor must be rank 2")
    n, d = x_tensor.shape
    missing_config = [
        name
        for name, value in (
            ("dropout_ratio", dropout_ratio),
            ("apply_u_silu", apply_u_silu),
            ("concat_u", concat_u),
            ("concat_x", concat_x),
        )
        if value is None
    ]
    if missing_config:
        raise ValueError(f"hstu_lmsd_backward requires explicit forward configuration: {', '.join(missing_config)}")
    dropout_ratio = normalize_dropout_ratio(dropout_ratio)
    apply_u_silu = bool(apply_u_silu)
    concat_u = bool(concat_u)
    concat_x = bool(concat_x)
    compute_dweight = bool(compute_dweight)
    kernel_config = HSTULMSDBwdConfig.from_hidden_size(d)
    multiprocessor_count = torch.cuda.get_device_properties(x_tensor.device).multi_processor_count
    workspace_rows = kernel_config.workspace_rows(multiprocessor_count)
    if not compute_dweight and dweight_tensor is not None:
        raise ValueError("dweight_tensor must be None when compute_dweight is False")
    with torch.cuda.device(x_tensor.device), allocation_context(stream, x_tensor.device):
        if dx_tensor is None:
            dx_tensor = torch.empty((n, d), dtype=x_tensor.dtype, device=x_tensor.device)
        if du_tensor is None:
            du_tensor = torch.empty((n, d), dtype=x_tensor.dtype, device=x_tensor.device)
        if compute_dweight and dweight_tensor is None:
            dweight_tensor = torch.empty((d,), dtype=weight_tensor.dtype, device=x_tensor.device)
        if dbias_tensor is None:
            dbias_tensor = torch.empty((d,), dtype=bias_tensor.dtype, device=x_tensor.device)
        dweight_workspace = torch.empty((workspace_rows, d), dtype=torch.float32, device=x_tensor.device) if compute_dweight else None
        dbias_workspace = torch.empty((workspace_rows, d), dtype=torch.float32, device=x_tensor.device)

    key = (
        tensor_signature(dy_tensor, dynamic_rows=True),
        tensor_signature(x_tensor, dynamic_rows=True),
        tensor_signature(u_tensor, dynamic_rows=True),
        tensor_signature(weight_tensor),
        tensor_signature(bias_tensor),
        tensor_signature(mean_tensor, dynamic_rows=True),
        tensor_signature(rstd_tensor, dynamic_rows=True),
        _tensor_signature_or_none(mask_tensor, dynamic_rows=True),
        tensor_signature(dx_tensor, dynamic_rows=True),
        tensor_signature(du_tensor, dynamic_rows=True),
        _tensor_signature_or_none(dweight_tensor),
        tensor_signature(dbias_tensor),
        _tensor_signature_or_none(dweight_workspace),
        tensor_signature(dbias_workspace),
        float(dropout_ratio),
        apply_u_silu,
        concat_u,
        concat_x,
        compute_dweight,
    )
    api = _cache_get(_BWD_CACHE, key)
    if api is None:
        with torch.cuda.device(x_tensor.device):
            api = HSTULMSDBwd(
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
                apply_u_silu=apply_u_silu,
                concat_u=concat_u,
                concat_x=concat_x,
                compute_dweight=compute_dweight,
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
