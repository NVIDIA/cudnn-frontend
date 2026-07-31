# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neutral public API for SM100 unfused BF16 grouped GEMM."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from cuda.bindings import driver as cuda

from cudnn.api_base import APIBase, TupleDict
from cudnn.gemm.cutedsl.discrete_grouped.discrete_kernel_utils import _require_pointer_tensor


from ._bf16_api import GroupedGemmBf16API
from ..moe_utils import MoEWeightMode

__all__ = ["GroupedGemmBf16API"]


class GroupedGemmSm100(APIBase):
    """Public lifecycle facade with deferred BF16 descriptor initialization."""

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_c: torch.Tensor,
        sample_d: torch.Tensor,
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_b: Optional[torch.Tensor] = None,
        sample_bias: Optional[torch.Tensor] = None,
        sample_prob: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        b_shape: Optional[Tuple[int, ...]] = None,
        b_dtype: Optional[torch.dtype] = None,
        acc_dtype: torch.dtype = torch.float32,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        vector_f32: bool = False,
        m_aligned: int = 256,
        generate_c: bool = False,
        b_major: str = "k",
        use_dynamic_sched: bool = False,
    ) -> None:
        super().__init__()
        self._pending_init_kwargs = dict(locals())
        self._pending_init_kwargs.pop("self")
        self._pending_init_kwargs.pop("__class__", None)
        self._implementation = None

    def check_support(self) -> bool:
        if self._implementation is None:
            self._implementation = GroupedGemmBf16API(**self._pending_init_kwargs)
            self._kernel = self._implementation._kernel
        supported = self._implementation.check_support()
        self._is_supported = self._implementation._is_supported
        if supported:
            self._pending_init_kwargs = None
        return supported

    def compile(self) -> None:
        if self._implementation is None:
            self.check_support()
        if self._is_supported:
            self._implementation._is_supported = True
        self._implementation.compile()
        self._is_supported = self._implementation._is_supported
        self._compiled_kernel = self._implementation._compiled_kernel

    def execute(
        self,
        a_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        d_tensor: torch.Tensor,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        b_tensor: Optional[torch.Tensor] = None,
        b_ptrs: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        prob_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        if self._implementation is None:
            raise RuntimeError("Kernel not compiled; call compile() first")
        self._implementation.execute(
            a_tensor=a_tensor,
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            b_tensor=b_tensor,
            b_ptrs=b_ptrs,
            bias_tensor=bias_tensor,
            prob_tensor=prob_tensor,
            current_stream=current_stream,
        )
        self._is_supported = self._implementation._is_supported
        self._compiled_kernel = self._implementation._compiled_kernel


_cache_of_GroupedGemmSm100Objects: dict[tuple, GroupedGemmSm100] = {}


def _stride_order(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(
        index
        for index, _ in sorted(
            enumerate(tensor.stride()),
            key=lambda item: (item[1], tensor.shape[item[0]]),
        )
    )


def _tensor_signature(tensor: Optional[torch.Tensor], *, dynamic_m: bool = False) -> tuple:
    if tensor is None:
        return (None, None, None, None)
    shape = (None, *tuple(tensor.shape[1:])) if dynamic_m else tuple(tensor.shape)
    return (
        shape,
        _stride_order(tensor),
        tensor.dtype,
        (tensor.device.type, tensor.device.index),
    )


def _validate_output(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tuple(tensor.shape) != shape or tuple(tensor.stride()) != stride or tensor.dtype != dtype or tensor.device != device:
        raise ValueError(
            f"{name} must have shape {shape}, stride {stride}, dtype {dtype}, "
            f"device {device}; got shape {tuple(tensor.shape)}, stride "
            f"{tuple(tensor.stride())}, dtype {tensor.dtype}, device {tensor.device}"
        )


def _normalize_call(
    a_tensor: torch.Tensor,
    padded_offsets: torch.Tensor,
    b_tensor: Optional[torch.Tensor],
    b_ptrs: Optional[torch.Tensor],
    n: Optional[int],
    b_dtype: Optional[torch.dtype],
    prob_tensor: Optional[torch.Tensor],
    c_dtype: torch.dtype,
    d_dtype: torch.dtype,
    cd_major: str,
    m_aligned: int,
) -> tuple[bool, int, int, int]:
    is_dense = b_tensor is not None
    is_discrete = b_ptrs is not None
    if is_dense and is_discrete:
        raise ValueError("Provide either b_tensor or b_ptrs, not both")
    if not is_dense and not is_discrete:
        raise ValueError("Must provide either b_tensor or b_ptrs")

    if a_tensor.dtype != torch.bfloat16:
        raise ValueError(f"a_tensor must have dtype torch.bfloat16, got {a_tensor.dtype}")
    if a_tensor.ndim != 3 or a_tensor.shape[2] != 1:
        raise ValueError(f"a_tensor must have shape (m, k, 1), got {tuple(a_tensor.shape)}")
    if prob_tensor is None:
        raise ValueError("prob_tensor is required")
    if cd_major != "n":
        raise ValueError(f"cd_major must be 'n', got {cd_major}")
    if c_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"c_dtype must be BF16, FP16, or FP32, got {c_dtype}")
    if d_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"d_dtype must be BF16, FP16, or FP32, got {d_dtype}")
    if m_aligned != 256:
        raise ValueError(f"m_aligned must be 256, got {m_aligned}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    major, minor = torch.cuda.get_device_capability(a_tensor.device)
    compute_capability = major * 10 + minor
    if compute_capability < 100:
        raise RuntimeError(f"GroupedGemmSm100 requires SM100+, found SM{compute_capability}")

    tensor_m, k, _ = a_tensor.shape
    if tensor_m % 256 != 0:
        raise ValueError(f"a_tensor M dimension must be 256-aligned, got {tensor_m}")
    if tuple(prob_tensor.shape) != (tensor_m, 1, 1):
        raise ValueError(f"prob_tensor must have shape {(tensor_m, 1, 1)}, got " f"{tuple(prob_tensor.shape)}")
    if prob_tensor.dtype != torch.float32:
        raise ValueError(f"prob_tensor must have dtype torch.float32, got {prob_tensor.dtype}")
    if is_dense:
        if n is not None:
            raise ValueError("Dense mode forbids n")
        if b_dtype is not None:
            raise ValueError("Dense mode forbids b_dtype")
        if b_tensor.dtype != torch.bfloat16:
            raise ValueError(f"b_tensor must have dtype torch.bfloat16, got {b_tensor.dtype}")
        if b_tensor.ndim != 3:
            raise ValueError(f"b_tensor must have shape (n, k, experts), got {tuple(b_tensor.shape)}")
        n, b_k, experts = b_tensor.shape
        if b_k != k:
            raise ValueError(f"b_tensor K dimension ({b_k}) must match a_tensor ({k})")
    else:
        _require_pointer_tensor(b_ptrs, "b_ptrs")
        if b_ptrs.device != a_tensor.device:
            raise ValueError(f"b_ptrs must be on the same device as a_tensor " f"({a_tensor.device}), got {b_ptrs.device}")
        if b_ptrs.data_ptr() % 8 != 0:
            raise ValueError("b_ptrs data pointer must be 8-byte aligned")
        if padded_offsets.ndim == 1 and b_ptrs.numel() != padded_offsets.numel():
            raise ValueError(f"b_ptrs length mismatch: expected {padded_offsets.numel()}, " f"got {b_ptrs.numel()}")
        if n is None or b_dtype is None:
            raise ValueError("Discrete mode requires n and b_dtype")
        if b_dtype != torch.bfloat16:
            raise ValueError(f"b_dtype must be torch.bfloat16 for the BF16 backend, got {b_dtype}")
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        experts = b_ptrs.numel()

    return is_dense, tensor_m, n, experts


def grouped_gemm_wrapper_sm100(
    a_tensor: torch.Tensor,
    padded_offsets: torch.Tensor,
    alpha_tensor: torch.Tensor,
    b_tensor: Optional[torch.Tensor] = None,
    bias_tensor: Optional[torch.Tensor] = None,
    b_ptrs: Optional[torch.Tensor] = None,
    n: Optional[int] = None,
    b_dtype: Optional[torch.dtype] = None,
    b_major: str = "k",
    prob_tensor: Optional[torch.Tensor] = None,
    acc_dtype: torch.dtype = torch.float32,
    c_dtype: torch.dtype = torch.bfloat16,
    d_dtype: torch.dtype = torch.bfloat16,
    c_tensor: Optional[torch.Tensor] = None,
    d_tensor: Optional[torch.Tensor] = None,
    cd_major: str = "n",
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    vector_f32: bool = False,
    m_aligned: int = 256,
    generate_c: bool = False,
    use_dynamic_sched: bool = False,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    is_dense, tensor_m, n_out, expert_cnt = _normalize_call(
        a_tensor,
        padded_offsets,
        b_tensor,
        b_ptrs,
        n,
        b_dtype,
        prob_tensor,
        c_dtype,
        d_dtype,
        cd_major,
        m_aligned,
    )
    expected_shape = (tensor_m, n_out, 1)
    expected_stride = (n_out, 1, tensor_m * n_out)
    if c_tensor is None:
        internal_c = torch.empty_strided(
            expected_shape,
            expected_stride,
            dtype=c_dtype,
            device=a_tensor.device,
        )
    else:
        _validate_output(
            c_tensor,
            name="c_tensor",
            shape=expected_shape,
            stride=expected_stride,
            dtype=c_dtype,
            device=a_tensor.device,
        )
        internal_c = c_tensor
    if d_tensor is None:
        d_tensor = torch.empty_strided(
            expected_shape,
            expected_stride,
            dtype=d_dtype,
            device=a_tensor.device,
        )
    else:
        _validate_output(
            d_tensor,
            name="d_tensor",
            shape=expected_shape,
            stride=expected_stride,
            dtype=d_dtype,
            device=a_tensor.device,
        )

    overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    workspace_bytes = (128 * expert_cnt if not is_dense else 0) + (4 if use_dynamic_sched else 0)
    workspace_signature = (
        (max(workspace_bytes, 1),),
        (a_tensor.device.type, a_tensor.device.index),
    )
    cache_key = (
        "bf16",
        "dense" if is_dense else "discrete",
        *_tensor_signature(a_tensor, dynamic_m=True),
        *_tensor_signature(b_tensor),
        *_tensor_signature(b_ptrs),
        *_tensor_signature(internal_c, dynamic_m=True),
        *_tensor_signature(d_tensor, dynamic_m=True),
        *_tensor_signature(padded_offsets),
        *_tensor_signature(alpha_tensor),
        *_tensor_signature(bias_tensor),
        *_tensor_signature(prob_tensor, dynamic_m=True),
        n_out,
        expert_cnt,
        b_dtype,
        b_major,
        acc_dtype,
        c_dtype,
        d_dtype,
        cd_major,
        tuple(mma_tiler_mn),
        tuple(cluster_shape_mn) if cluster_shape_mn is not None else None,
        vector_f32,
        m_aligned,
        generate_c,
        use_dynamic_sched,
        overlap_margin,
        workspace_signature,
    )

    op = _cache_of_GroupedGemmSm100Objects.get(cache_key)
    if op is None:
        op = GroupedGemmSm100(
            sample_a=a_tensor,
            sample_c=internal_c,
            sample_d=d_tensor,
            sample_padded_offsets=padded_offsets,
            sample_alpha=alpha_tensor,
            sample_b=b_tensor if is_dense else None,
            sample_bias=bias_tensor,
            sample_prob=prob_tensor,
            num_experts=None if is_dense else expert_cnt,
            b_shape=None if is_dense else (n_out, a_tensor.shape[1]),
            b_dtype=None if is_dense else b_dtype,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            vector_f32=vector_f32,
            m_aligned=m_aligned,
            generate_c=generate_c,
            b_major=b_major,
            use_dynamic_sched=use_dynamic_sched,
        )
        assert op.check_support(), "Unsupported configuration"
        op.compile()
        _cache_of_GroupedGemmSm100Objects[cache_key] = op

    op.execute(
        a_tensor=a_tensor,
        c_tensor=internal_c,
        d_tensor=d_tensor,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha_tensor,
        b_tensor=b_tensor if is_dense else None,
        b_ptrs=None if is_dense else b_ptrs,
        bias_tensor=bias_tensor,
        prob_tensor=prob_tensor,
        current_stream=current_stream,
    )
    return TupleDict(
        d_tensor=d_tensor,
        c_tensor=internal_c if generate_c else None,
    )


__all__ = [
    "GroupedGemmSm100",
    "grouped_gemm_wrapper_sm100",
]
