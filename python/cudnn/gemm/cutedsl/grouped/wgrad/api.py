# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified FE API for grouped GEMM wgrad on SM100+."""

from __future__ import annotations

from typing import Any, Optional, Tuple, overload
import os

import torch
from cuda.bindings import driver as cuda

from cudnn.api_base import APIBase, TupleDict, get_device_type
from cudnn.gemm.cutedsl.discrete_grouped.discrete_kernel_utils import _require_pointer_tensor

from ..backend_utils import (
    GroupedGemmBackend,
    _torch_stream_context,
    backend_cache_key,
    select_grouped_gemm_backend,
)
from ..moe_utils import WGradInputOrder

_BLOCK_SCALED_DTYPE_PAIRS = {
    (dtype, dtype)
    for dtype in (
        torch.float4_e2m1fn_x2,
        torch.uint8,
        torch.float8_e5m2,
        torch.float8_e4m3fn,
    )
}

_cache_of_GroupedGemmWgradSm100Objects = {}


from ._bf16_api import GroupedGemmWgradBf16API
from ._blockscaled_api import (
    GroupedGemmWgradBlockScaledAPI,
    _get_rubin_kernel,
    _is_supported_rubin_quantization,
)


class GroupedGemmWgradSm100(APIBase):
    """Stable public facade that selects the WGrad backend during support checking."""

    # BF16 implementation
    @overload
    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_b: torch.Tensor,
        sample_sfa: None,
        sample_sfb: None,
        sample_offsets: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...

    # Block-scaled implementation
    @overload
    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_b: torch.Tensor,
        sample_sfa: torch.Tensor,
        sample_sfb: torch.Tensor,
        sample_offsets: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_b: torch.Tensor,
        sample_sfa: Optional[torch.Tensor],
        sample_sfb: Optional[torch.Tensor],
        sample_offsets: torch.Tensor,
        sample_wgrad: Optional[torch.Tensor] = None,
        sample_wgrad_expert: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        wgrad_shape: Optional[Tuple[int, int]] = None,
        wgrad_dtype: Optional[torch.dtype] = None,
        sample_global_scale_a: Optional[torch.Tensor] = None,
        sample_global_scale_b: Optional[torch.Tensor] = None,
        acc_dtype: torch.dtype = torch.float32,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        accumulate_on_output: bool = False,
        input_order: WGradInputOrder | str = WGradInputOrder.Tensor2D,
    ) -> None:
        super().__init__()
        self._pending_init_kwargs = dict(locals())
        self._pending_init_kwargs.pop("self")
        self._pending_init_kwargs.pop("__class__", None)
        self._implementation = None

    def check_support(self) -> bool:
        if self._implementation is None:
            kwargs = self._pending_init_kwargs
            backend = select_grouped_gemm_backend(
                operation="grouped_gemm_wgrad_sm100",
                a_dtype=kwargs["sample_a"].dtype,
                b_dtype=kwargs["sample_b"].dtype,
                scale_controls=(
                    ("sample_sfa", kwargs["sample_sfa"]),
                    ("sample_sfb", kwargs["sample_sfb"]),
                    ("sample_global_scale_a", kwargs["sample_global_scale_a"]),
                    ("sample_global_scale_b", kwargs["sample_global_scale_b"]),
                    ("sf_vec_size", kwargs["sf_vec_size"] if kwargs["sf_vec_size"] != 16 else None),
                ),
                block_scaled_dtype_pairs=_BLOCK_SCALED_DTYPE_PAIRS,
            )
            self.backend = backend
            if backend is GroupedGemmBackend.BF16:
                self._implementation = GroupedGemmWgradBf16API(**kwargs)
            else:
                self._implementation = GroupedGemmWgradBlockScaledAPI(**kwargs)
            self._kernel = self._implementation._kernel
            self.weight_mode = self._implementation.weight_mode
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

    # BF16 implementation
    @overload
    def execute(
        self,
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        sfa_tensor: None,
        sfb_tensor: None,
        offsets_tensor: torch.Tensor,
        wgrad_tensor: Optional[torch.Tensor] = None,
        wgrad_ptrs: Optional[torch.Tensor] = None,
        *,
        global_scale_a: None = None,
        global_scale_b: None = None,
    ) -> None: ...

    # Block-scaled implementation
    @overload
    def execute(
        self,
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        sfb_tensor: torch.Tensor,
        offsets_tensor: torch.Tensor,
        wgrad_tensor: Optional[torch.Tensor] = None,
        wgrad_ptrs: Optional[torch.Tensor] = None,
        *,
        global_scale_a: Optional[torch.Tensor] = None,
        global_scale_b: Optional[torch.Tensor] = None,
    ) -> None: ...

    def execute(
        self,
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        sfb_tensor: torch.Tensor,
        offsets_tensor: torch.Tensor,
        wgrad_tensor: Optional[torch.Tensor] = None,
        wgrad_ptrs: Optional[torch.Tensor] = None,
        global_scale_a: Optional[torch.Tensor] = None,
        global_scale_b: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        if self._implementation is None:
            raise RuntimeError("Kernel not compiled; call compile() first")
        self._implementation.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            offsets_tensor=offsets_tensor,
            wgrad_tensor=wgrad_tensor,
            wgrad_ptrs=wgrad_ptrs,
            global_scale_a=global_scale_a,
            global_scale_b=global_scale_b,
            current_stream=current_stream,
        )


def _wgrad_tensor_signature(tensor: Optional[torch.Tensor], *, dynamic_dims: tuple[int, ...] = (), exact_stride: bool):
    if tensor is None:
        return None
    shape = tuple(None if index in dynamic_dims else int(value) for index, value in enumerate(tensor.shape))
    stride = tuple(int(value) for value in tensor.stride())
    layout = stride if exact_stride else tuple(index for index, _ in sorted(enumerate(stride), key=lambda item: (item[1], tensor.shape[item[0]])))
    return (shape, layout, tensor.dtype, tensor.device)


def grouped_gemm_wgrad_wrapper_sm100(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    sfa_tensor: torch.Tensor,
    sfb_tensor: torch.Tensor,
    offsets_tensor: torch.Tensor,
    output_mode: str = "dense",
    wgrad_tensor: Optional[torch.Tensor] = None,
    wgrad_ptrs: Optional[torch.Tensor] = None,
    global_scale_a: Optional[torch.Tensor] = None,
    global_scale_b: Optional[torch.Tensor] = None,
    acc_dtype: torch.dtype = torch.float32,
    wgrad_dtype: torch.dtype = torch.bfloat16,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 16,
    accumulate_on_output: bool = False,
    input_order: WGradInputOrder | str = WGradInputOrder.Tensor2D,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Compile and execute grouped GEMM wgrad through the selected backend API."""
    if output_mode not in ("dense", "discrete"):
        raise ValueError(f'output_mode must be "dense" or "discrete", got {output_mode}')
    if a_tensor.ndim != 2 or b_tensor.ndim != 2:
        raise ValueError("a_tensor and b_tensor must both be rank-2")
    hidden, tokens_sum = a_tensor.shape
    tokens_b, intermediate = b_tensor.shape
    if tokens_sum != tokens_b:
        raise ValueError(f"a_tensor and b_tensor token dimensions must match, got {tokens_sum} and {tokens_b}")
    if offsets_tensor.ndim != 1:
        raise ValueError(f"offsets_tensor must be rank-1, got shape {tuple(offsets_tensor.shape)}")
    input_order = WGradInputOrder(input_order)
    expert_cnt = offsets_tensor.numel()
    if output_mode == "dense" and wgrad_ptrs is not None:
        raise ValueError("dense output_mode forbids wgrad_ptrs")
    if wgrad_ptrs is not None:
        _require_pointer_tensor(wgrad_ptrs, "wgrad_ptrs", expert_cnt)
    backend = select_grouped_gemm_backend(
        operation="grouped_gemm_wgrad_sm100",
        a_dtype=a_tensor.dtype,
        b_dtype=b_tensor.dtype,
        scale_controls=(
            ("sfa_tensor", sfa_tensor),
            ("sfb_tensor", sfb_tensor),
            ("global_scale_a", global_scale_a),
            ("global_scale_b", global_scale_b),
            ("sf_vec_size", sf_vec_size if sf_vec_size != 16 else None),
        ),
        block_scaled_dtype_pairs=_BLOCK_SCALED_DTYPE_PAIRS,
    )
    if wgrad_tensor is None and wgrad_ptrs is None:
        allocator = torch.zeros if accumulate_on_output else torch.empty
        with _torch_stream_context(current_stream, a_tensor.device):
            wgrad_tensor = allocator((expert_cnt, hidden, intermediate), dtype=wgrad_dtype, device=a_tensor.device)
    cache_key = backend_cache_key(
        backend,
        get_device_type(),
        output_mode,
        _wgrad_tensor_signature(a_tensor, dynamic_dims=(1,), exact_stride=False),
        _wgrad_tensor_signature(b_tensor, dynamic_dims=(0,), exact_stride=False),
        _wgrad_tensor_signature(sfa_tensor, dynamic_dims=(1,), exact_stride=False),
        _wgrad_tensor_signature(sfb_tensor, dynamic_dims=(1,), exact_stride=False),
        _wgrad_tensor_signature(offsets_tensor, exact_stride=True),
        _wgrad_tensor_signature(wgrad_tensor, exact_stride=True),
        _wgrad_tensor_signature(wgrad_ptrs, exact_stride=True),
        _wgrad_tensor_signature(global_scale_a, exact_stride=True),
        _wgrad_tensor_signature(global_scale_b, exact_stride=True),
        acc_dtype,
        wgrad_dtype,
        tuple(mma_tiler_mn),
        tuple(cluster_shape_mn) if cluster_shape_mn is not None else None,
        sf_vec_size,
        accumulate_on_output,
        input_order,
        int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0")),
    )
    op = _cache_of_GroupedGemmWgradSm100Objects.get(cache_key)
    if op is None:
        common = dict(
            sample_a=a_tensor,
            sample_b=b_tensor,
            sample_sfa=sfa_tensor,
            sample_sfb=sfb_tensor,
            sample_offsets=offsets_tensor,
            sample_global_scale_a=global_scale_a,
            sample_global_scale_b=global_scale_b,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
            accumulate_on_output=accumulate_on_output,
            input_order=input_order,
        )
        if output_mode == "dense":
            common["sample_wgrad"] = wgrad_tensor
        else:
            common.update(
                sample_wgrad_expert=(
                    wgrad_tensor[0] if wgrad_tensor is not None else torch.empty((hidden, intermediate), dtype=wgrad_dtype, device=a_tensor.device)
                ),
                num_experts=expert_cnt,
                wgrad_shape=(hidden, intermediate),
                wgrad_dtype=wgrad_dtype,
            )
        op = GroupedGemmWgradSm100(**common)
        if not op.check_support():
            raise RuntimeError("Unsupported configuration")
        op.compile()
        _cache_of_GroupedGemmWgradSm100Objects[cache_key] = op
    op.execute(
        a_tensor=a_tensor,
        b_tensor=b_tensor,
        sfa_tensor=sfa_tensor,
        sfb_tensor=sfb_tensor,
        offsets_tensor=offsets_tensor,
        wgrad_tensor=wgrad_tensor,
        wgrad_ptrs=wgrad_ptrs,
        global_scale_a=global_scale_a,
        global_scale_b=global_scale_b,
        current_stream=current_stream,
    )
    return TupleDict(wgrad_tensor=wgrad_tensor)
