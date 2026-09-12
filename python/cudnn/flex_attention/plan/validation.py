# SPDX-License-Identifier: BSD-3-Clause
"""Host-side validation for public FlexAttention inputs."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from cudnn.flex_attention.runtime.arch import get_device_arch

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
SM100_STANDARD_HEAD_DIMS = tuple((head_dim, head_dim_v) for head_dim in range(8, 129, 8) for head_dim_v in range(8, 129, 8))
SUPPORTED_HEAD_DIMS = (*SM100_STANDARD_HEAD_DIMS, (192, 128), (256, 256))
SUPPORTED_HEAD_DIM_RULE = "Dqk and Dv independently in [8, 128] and divisible by 8, " "or (192, 128), or (256, 256)"


def is_supported_head_dims(head_dim: int, head_dim_v: int) -> bool:
    """Return whether a public head shape has a supported kernel family."""

    return (head_dim, head_dim_v) in SUPPORTED_HEAD_DIMS


@dataclass(frozen=True)
class PlanGeometry:
    is_varlen: bool
    arch: int
    batch_size: int
    seqlen_q: int | None
    seqlen_k: int | None
    total_q: int
    total_k: int
    max_seqlen_q: int
    max_seqlen_k: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    head_dim_v: int
    hmask: int
    nfunc: int
    cu_seqlens_q: torch.Tensor | None
    cu_seqlens_k: torch.Tensor | None


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_varlen: bool) -> None:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.dtype not in SUPPORTED_DTYPES:
            raise TypeError(f"{name} must have dtype torch.float16 or torch.bfloat16")
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous in the last dimension")

    expected_rank = 3 if is_varlen else 4
    if any(tensor.ndim != expected_rank for tensor in (q, k, v)):
        raise ValueError(f"q, k, and v must be rank-{expected_rank} tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, and v must have the same dtype")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension")
    if q.shape[-2] <= 0 or k.shape[-2] <= 0:
        raise ValueError("Hq and Hkv must be positive")
    if k.shape[-3] != v.shape[-3] or k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same sequence and head extents")
    if q.shape[-2] % k.shape[-2] != 0:
        raise ValueError("Hq must be divisible by Hkv")
    dims = (q.shape[-1], v.shape[-1])
    if not is_supported_head_dims(*dims):
        raise ValueError(f"supported head dimensions are {SUPPORTED_HEAD_DIM_RULE}; got {dims}")


def _validate_prefix(
    tensor: torch.Tensor,
    *,
    name: str,
) -> None:
    if tensor.ndim != 1 or tensor.numel() < 2:
        raise ValueError(f"{name} must be rank-1 with at least two elements")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA tensor")


def validate_create_mask_plan_inputs(
    mask_func: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
) -> PlanGeometry:
    """Validate public plan inputs and clone all mutable varlen geometry."""

    varlen_args = (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    is_varlen = all(value is not None for value in varlen_args)
    if not is_varlen and any(value is not None for value in varlen_args):
        raise ValueError("cu_seqlens_q, cu_seqlens_k, max_seqlen_q, and max_seqlen_k " "must be provided together")
    _validate_qkv(q, k, v, is_varlen=is_varlen)

    if is_varlen:
        if type(max_seqlen_q) is not int or type(max_seqlen_k) is not int:
            raise TypeError("max_seqlen_q and max_seqlen_k must be Python ints")
        if max_seqlen_q < 0 or max_seqlen_k < 0:
            raise ValueError("max_seqlen_q and max_seqlen_k must be non-negative")
        if cu_seqlens_q.device != q.device or cu_seqlens_k.device != q.device:
            raise ValueError("cu_seqlens_q/k must be on the same device as q")
        if cu_seqlens_q.shape != cu_seqlens_k.shape:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same shape")
        batch_size = cu_seqlens_q.numel() - 1
        total_q, total_k = q.shape[0], k.shape[0]
        _validate_prefix(
            cu_seqlens_q,
            name="cu_seqlens_q",
        )
        _validate_prefix(
            cu_seqlens_k,
            name="cu_seqlens_k",
        )
        cu_q_owned = cu_seqlens_q.detach().clone()
        cu_k_owned = cu_seqlens_k.detach().clone()
        seqlen_q = None
        seqlen_k = None
    else:
        batch_size, seqlen_q = q.shape[:2]
        if batch_size <= 0:
            raise ValueError("fixed attention batch size must be positive")
        if k.shape[0] != batch_size or v.shape[0] != batch_size:
            raise ValueError("fixed q, k, and v must have the same batch size")
        seqlen_k = k.shape[1]
        total_q = batch_size * seqlen_q
        total_k = batch_size * seqlen_k
        max_seqlen_q = seqlen_q
        max_seqlen_k = seqlen_k
        cu_q_owned = None
        cu_k_owned = None

    if not isinstance(mask_func, torch.Tensor):
        raise TypeError("mask_func must be a torch.Tensor")
    if mask_func.ndim != 3:
        raise ValueError("mask_func must have shape [Hmask, nfunc, total_q]")
    if mask_func.dtype != torch.int32:
        raise TypeError("mask_func must have dtype torch.int32")
    if mask_func.device != q.device or not mask_func.is_contiguous():
        raise ValueError("mask_func must be contiguous and on the same CUDA device as q")
    hmask, nfunc, func_total_q = mask_func.shape
    if func_total_q != total_q:
        raise ValueError(f"mask_func last extent must equal total_q ({total_q}); got {func_total_q}")
    if hmask not in (1, q.shape[-2]):
        raise ValueError(f"Hmask must be 1 or Hq ({q.shape[-2]}); got {hmask}")
    if nfunc <= 0 or nfunc % 2 == 0:
        raise ValueError("nfunc must be a positive odd number")

    return PlanGeometry(
        is_varlen=is_varlen,
        arch=get_device_arch(q.device.index),
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        total_q=total_q,
        total_k=total_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_q_heads=q.shape[-2],
        num_kv_heads=k.shape[-2],
        head_dim=q.shape[-1],
        head_dim_v=v.shape[-1],
        hmask=hmask,
        nfunc=nfunc,
        cu_seqlens_q=cu_q_owned,
        cu_seqlens_k=cu_k_owned,
    )


def validate_call_options(*, softmax_scale: float | None, deterministic: bool, return_lse: bool) -> None:
    if softmax_scale is not None:
        if not isinstance(softmax_scale, (float, int)) or not math.isfinite(softmax_scale):
            raise ValueError("softmax_scale must be a finite number or None")
    if type(deterministic) is not bool:
        raise TypeError("deterministic must be a bool")
    if type(return_lse) is not bool:
        raise TypeError("return_lse must be a bool")


__all__ = [
    "PlanGeometry",
    "SM100_STANDARD_HEAD_DIMS",
    "SUPPORTED_DTYPES",
    "SUPPORTED_HEAD_DIM_RULE",
    "SUPPORTED_HEAD_DIMS",
    "is_supported_head_dims",
    "validate_call_options",
    "validate_create_mask_plan_inputs",
]
