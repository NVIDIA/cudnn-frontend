# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unified API for Grouped GEMM GLU Forward Kernel (SM100+)

This module provides a single API class that supports both contiguous (dense)
and discrete weight modes for block-scaled grouped GEMM with GLU activation
(SwiGLU / GeGLU / SiTU-GLU) in MoE (Mixture of Experts) workloads.

Dense mode
    All expert weights are packed contiguously in a 3-D tensor (N, K, L).
    Callers supply ``sample_b`` and ``sample_sfb``.

Discrete mode
    Each expert has its own memory allocation.  Callers supply
    ``num_experts``, ``b_shape``, ``b_dtype``, and per-expert pointer arrays
    at execution time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from ..backend_utils import (
    GroupedGemmBackend,
    backend_cache_key,
    select_grouped_gemm_backend,
)
from ..moe_utils import MoEWeightMode
from cuda.bindings import driver as cuda
import logging
import os
from typing import Any, Literal, Tuple, Optional, overload

import cutlass

from cudnn.api_base import APIBase, TupleDict, ceil_div, get_device_type
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import (
    cuda_is_available,
    detect_framework,
    framework_dtype,
    get_compute_capability,
    get_device,
    get_shape,
    get_strides,
)

_JAX_DENSE_B_ERROR = (
    "Dense weight mode (b_tensor) is not expressible as JAX arrays "
    "(the expert-outermost strided B layout has no row-major equivalent); "
    "use discrete mode (b_ptrs) with per-expert weight pointers"
)
_JAX_BIAS_ERROR = (
    "bias_tensor is not expressible as a JAX array (its (n, experts) column-major layout has no row-major equivalent); " "omit bias for JAX inputs"
)
_JAX_BLOCK_SCALED_ERROR = (
    "The block-scaled grouped GEMM GLU backend is not expressible as JAX arrays "
    "(its scale-factor tensors use an MMA-interleaved layout with no row-major equivalent); "
    "only the BF16 backend supports JAX inputs"
)


def _block_scaled_dtype_pairs():
    # Canonical (cutlass) dtype vocabulary; select_grouped_gemm_backend canonicalizes
    # the caller's dtypes so torch/jax/numpy/str dtypes all compare against these.
    return {
        (dtype, dtype)
        for dtype in (
            cutlass.Float4E2M1FN,
            cutlass.Uint8,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        )
    }


from ._bf16_api import GroupedGemmGluBf16API
from ._blockscaled_api import (
    GroupedGemmGluBlockScaledAPI,
    _reject_unsupported_rubin_glu_tune_params,
)


@dataclass(frozen=True)
class GluCall:
    """Immutable normalized input for GLU dispatch, allocation, and caching."""

    a_tensor: torch.Tensor
    sfa_tensor: Optional[torch.Tensor]
    padded_offsets: torch.Tensor
    alpha_tensor: torch.Tensor
    b_tensor: Optional[torch.Tensor] = None
    sfb_tensor: Optional[torch.Tensor] = None
    bias_tensor: Optional[torch.Tensor] = None
    b_ptrs: Optional[torch.Tensor] = None
    sfb_ptrs: Optional[torch.Tensor] = None
    n: Optional[int] = None
    b_dtype: Optional[torch.dtype] = None
    b_major: str = "k"
    norm_const_tensor: Optional[torch.Tensor] = None
    prob_tensor: Optional[torch.Tensor] = None
    acc_dtype: Optional[torch.dtype] = None
    c_dtype: Optional[torch.dtype] = None
    d_dtype: Optional[torch.dtype] = None
    cd_major: str = "n"
    mma_tiler_mn: Tuple[int, int] = (256, 256)
    cluster_shape_mn: Optional[Tuple[int, int]] = None
    sf_vec_size: int = 16
    sf_fp8_dtype_override: Optional[Literal["e5m3"]] = None
    vector_f32: bool = False
    m_aligned: int = 256
    discrete_col_sfd: bool = False
    act_func: str = "swiglu"
    linear_offset: Optional[float] = None
    geglu_alpha: float = 1.702
    glu_clamp_max: float = 7.0
    glu_clamp_min: float = -7.0
    situ_beta1: float = 4.0
    situ_beta2: float = 25.0
    use_dynamic_sched: bool = False
    use_single_group_runtime_offsets: bool = False
    current_stream: Optional[cuda.CUstream] = None
    generate_c: bool = False
    weight_mode: Optional[MoEWeightMode] = None
    b_shape: Optional[Tuple[int, ...]] = None
    num_experts: Optional[int] = None


class GroupedGemmGluSm100(APIBase):
    """Stable public facade that selects the GLU backend during support checking."""

    # BF16 implementation
    @overload
    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_c: torch.Tensor,
        sample_d: torch.Tensor,
        sample_sfa: None,
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_d_col: None,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...

    # Block-scaled implementation
    @overload
    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_c: torch.Tensor,
        sample_d: torch.Tensor,
        sample_sfa: torch.Tensor,
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_d_col: Optional[torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_c: torch.Tensor,
        sample_d: torch.Tensor,
        sample_sfa: Optional[torch.Tensor],
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_d_col: Optional[torch.Tensor],
        sample_b: Optional[torch.Tensor] = None,
        sample_sfb: Optional[torch.Tensor] = None,
        sample_bias: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        b_shape: Optional[Tuple[int, ...]] = None,
        b_dtype: Optional[torch.dtype] = None,
        sample_sfd_row: Optional[torch.Tensor] = None,
        sample_sfd_col: Optional[torch.Tensor] = None,
        sample_amax: Optional[torch.Tensor] = None,
        sample_norm_const: Optional[torch.Tensor] = None,
        sample_prob: Optional[torch.Tensor] = None,
        acc_dtype: Optional[torch.dtype] = None,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        sf_fp8_dtype_override: Optional[Literal["e5m3"]] = None,
        vector_f32: bool = False,
        m_aligned: int = 256,
        discrete_col_sfd: bool = False,
        act_func: str = "swiglu",
        situ_beta1: float = 4.0,
        b_major: str = "k",
        use_dynamic_sched: bool = False,
        use_single_group_runtime_offsets: bool = False,
        generate_c: bool = False,
    ) -> None:
        super().__init__()
        self._pending_init_kwargs = dict(locals())
        self._pending_init_kwargs.pop("self")
        self._pending_init_kwargs.pop("__class__", None)
        framework = detect_framework(sample_a)
        if sample_a is not None and framework not in ("torch", "jax"):
            raise ValueError(f"Unsupported tensor framework '{framework}' for GroupedGemmGluSm100; pass torch tensors or JAX arrays")
        if acc_dtype is None:
            self._pending_init_kwargs["acc_dtype"] = cutlass.Float32
        self._implementation = None

    def check_support(self) -> bool:
        if self._implementation is None:
            kwargs = self._pending_init_kwargs
            defining_b_dtype = kwargs["sample_b"].dtype if kwargs["sample_b"] is not None else kwargs["b_dtype"]
            backend = select_grouped_gemm_backend(
                operation="grouped_gemm_glu_sm100",
                a_dtype=kwargs["sample_a"].dtype,
                b_dtype=defining_b_dtype,
                scale_controls=(
                    ("sample_sfa", kwargs["sample_sfa"]),
                    ("sample_sfb", kwargs["sample_sfb"]),
                    ("sample_d_col", kwargs["sample_d_col"]),
                    ("sample_sfd_row", kwargs["sample_sfd_row"]),
                    ("sample_sfd_col", kwargs["sample_sfd_col"]),
                    ("sample_amax", kwargs["sample_amax"]),
                    ("sample_norm_const", kwargs["sample_norm_const"]),
                    ("sf_vec_size", kwargs["sf_vec_size"] if kwargs["sf_vec_size"] != 16 else None),
                    ("sf_fp8_dtype_override", kwargs["sf_fp8_dtype_override"]),
                    ("discrete_col_sfd", kwargs["discrete_col_sfd"] if kwargs["discrete_col_sfd"] else None),
                ),
                block_scaled_dtype_pairs=_block_scaled_dtype_pairs(),
            )
            self.backend = backend
            if backend is GroupedGemmBackend.BF16:
                self._value_error_if(
                    kwargs["use_single_group_runtime_offsets"],
                    "use_single_group_runtime_offsets is supported only by the block-scaled kernel",
                )
                self._implementation = GroupedGemmGluBf16API(
                    sample_a=kwargs["sample_a"],
                    sample_c=kwargs["sample_c"],
                    sample_d=kwargs["sample_d"],
                    sample_padded_offsets=kwargs["sample_padded_offsets"],
                    sample_alpha=kwargs["sample_alpha"],
                    sample_b=kwargs["sample_b"],
                    sample_bias=kwargs["sample_bias"],
                    sample_prob=kwargs["sample_prob"],
                    num_experts=kwargs["num_experts"],
                    b_shape=kwargs["b_shape"],
                    b_dtype=kwargs["b_dtype"],
                    acc_dtype=kwargs["acc_dtype"],
                    mma_tiler_mn=kwargs["mma_tiler_mn"],
                    cluster_shape_mn=kwargs["cluster_shape_mn"],
                    vector_f32=kwargs["vector_f32"],
                    m_aligned=kwargs["m_aligned"],
                    generate_c=kwargs["generate_c"],
                    act_func=kwargs["act_func"],
                    b_major=kwargs["b_major"],
                    use_dynamic_sched=kwargs["use_dynamic_sched"],
                )
            else:
                if detect_framework(kwargs["sample_a"]) == "jax":
                    raise ValueError(_JAX_BLOCK_SCALED_ERROR)
                block_kwargs = dict(kwargs)
                block_kwargs.pop("generate_c", None)
                # The block-scaled implementation is torch-native: hand it torch dtypes.
                block_kwargs["acc_dtype"] = framework_dtype(block_kwargs["acc_dtype"], "torch")
                if block_kwargs.get("b_dtype") is not None:
                    block_kwargs["b_dtype"] = framework_dtype(block_kwargs["b_dtype"], "torch")
                self._implementation = GroupedGemmGluBlockScaledAPI(**block_kwargs)
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
        c_tensor: torch.Tensor,
        d_tensor: torch.Tensor,
        sfa_tensor: None,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        b_tensor: Optional[torch.Tensor] = None,
        *,
        sfb_tensor: None = None,
        sfb_ptrs: None = None,
        d_col_tensor: None = None,
        sfd_row_tensor: None = None,
        sfd_col_tensor: None = None,
        amax_tensor: None = None,
        norm_const_tensor: None = None,
    ) -> None: ...

    # Block-scaled implementation
    @overload
    def execute(
        self,
        a_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        d_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        b_tensor: Optional[torch.Tensor] = None,
        *,
        sfb_tensor: Optional[torch.Tensor] = None,
        sfb_ptrs: Optional[torch.Tensor] = None,
        d_col_tensor: Optional[torch.Tensor] = None,
        sfd_row_tensor: Optional[torch.Tensor] = None,
        sfd_col_tensor: Optional[torch.Tensor] = None,
        amax_tensor: Optional[torch.Tensor] = None,
        norm_const_tensor: Optional[torch.Tensor] = None,
    ) -> None: ...

    def execute(
        self,
        a_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        d_tensor: torch.Tensor,
        sfa_tensor: Optional[torch.Tensor],
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        b_tensor: Optional[torch.Tensor] = None,
        sfb_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        b_ptrs: Optional[torch.Tensor] = None,
        sfb_ptrs: Optional[torch.Tensor] = None,
        d_col_tensor: Optional[torch.Tensor] = None,
        sfd_row_tensor: Optional[torch.Tensor] = None,
        sfd_col_tensor: Optional[torch.Tensor] = None,
        amax_tensor: Optional[torch.Tensor] = None,
        norm_const_tensor: Optional[torch.Tensor] = None,
        prob_tensor: Optional[torch.Tensor] = None,
        linear_offset: Optional[float] = None,
        geglu_alpha: float = 1.702,
        glu_clamp_max: float = 7.0,
        glu_clamp_min: float = -7.0,
        situ_beta1: float = 4.0,
        situ_beta2: float = 25.0,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        if self._implementation is None:
            raise RuntimeError("Kernel not compiled; call compile() first")
        if self.backend is GroupedGemmBackend.BF16:
            scale_controls = (
                ("sfa_tensor", sfa_tensor),
                ("sfb_tensor", sfb_tensor),
                ("sfb_ptrs", sfb_ptrs),
                ("d_col_tensor", d_col_tensor),
                ("sfd_row_tensor", sfd_row_tensor),
                ("sfd_col_tensor", sfd_col_tensor),
                ("amax_tensor", amax_tensor),
                ("norm_const_tensor", norm_const_tensor),
                ("geglu_alpha", geglu_alpha if geglu_alpha != 1.702 else None),
                (
                    "glu_clamp_max",
                    glu_clamp_max if glu_clamp_max != 7.0 else None,
                ),
                (
                    "glu_clamp_min",
                    glu_clamp_min if glu_clamp_min != -7.0 else None,
                ),
            )
            forbidden = [name for name, value in scale_controls if value is not None]
            if forbidden:
                raise ValueError(f"grouped_gemm_glu_sm100: BF16 forbids scale control " f"{forbidden[0]}")
            if linear_offset is None:
                linear_offset = 1.0 if self._implementation.act_func == "geglu" else 0.0
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
                linear_offset=linear_offset,
                current_stream=current_stream,
            )
        else:
            self._implementation.execute(
                a_tensor=a_tensor,
                c_tensor=c_tensor,
                d_tensor=d_tensor,
                sfa_tensor=sfa_tensor,
                padded_offsets=padded_offsets,
                alpha_tensor=alpha_tensor,
                b_tensor=b_tensor,
                sfb_tensor=sfb_tensor,
                bias_tensor=bias_tensor,
                b_ptrs=b_ptrs,
                sfb_ptrs=sfb_ptrs,
                d_col_tensor=d_col_tensor,
                sfd_row_tensor=sfd_row_tensor,
                sfd_col_tensor=sfd_col_tensor,
                amax_tensor=amax_tensor,
                norm_const_tensor=norm_const_tensor,
                prob_tensor=prob_tensor,
                linear_offset=linear_offset,
                geglu_alpha=geglu_alpha,
                glu_clamp_max=glu_clamp_max,
                glu_clamp_min=glu_clamp_min,
                situ_beta1=situ_beta1,
                situ_beta2=situ_beta2,
                current_stream=current_stream,
            )
        self._is_supported = self._implementation._is_supported
        self._compiled_kernel = self._implementation._compiled_kernel


# --------------------------------------------------------------------------- #
#  Convenience wrapper with caching
# --------------------------------------------------------------------------- #

_logger = logging.getLogger(__name__)
_cache_of_GroupedGemmGluSm100Objects = {}


def _grouped_gemm_glu_block_scaled_call(call: GluCall) -> TupleDict:
    """Convenience wrapper for grouped GEMM GLU forward operation.

    Auto-detects dense vs. discrete mode based on which weight arguments
    are provided.

    Dense mode: provide ``b_tensor`` and ``sfb_tensor``.
    Discrete mode: provide ``b_ptrs``, ``sfb_ptrs``, ``n``, and ``b_dtype``.

    Compiled kernels are cached for reuse when called with the same configuration.

    Args:
        a_tensor: Input A tensor (valid_m, k, 1)
        sfa_tensor: Scale factor A
        padded_offsets: End offset per expert after padding
        alpha_tensor: Per-group scaling
        b_tensor: (Dense) Weight B tensor (n, k, l)
        sfb_tensor: (Dense) Scale factor B
        bias_tensor: Optional bias tensor with shape (n, l) and stride (1, n)
        b_ptrs: (Discrete) 1-D int64 device tensor of per-expert B data pointers
        sfb_ptrs: (Discrete) 1-D int64 device tensor of per-expert SFB data pointers
        n: (Discrete) B weight N dimension (full N before GLU split)
        b_dtype: (Discrete) B weight data type
        b_major: (Discrete) B tensor major dimension ("k" or "n")
        norm_const_tensor: Optional normalization constant
        prob_tensor: Optional probability tensor for gating
        acc_dtype: Accumulator data type
        c_dtype: Intermediate C tensor data type
        d_dtype: Output D tensor data type
        cd_major: CD major dimension (only "n" supported)
        mma_tiler_mn: MMA tiler shape
        cluster_shape_mn: Cluster shape
        sf_vec_size: Scale factor vector size
        sf_fp8_dtype_override: Reinterpret the FP8-format block scale factors as
            E5M3 instead of the encoding implied by ``sfa_tensor.dtype``. ``None``
            (default) infers as usual -- E4M3 for NVFP4, E8M0 for MXFP4/MXFP8 --
            and is the only accepted value on the BF16 backend, which has no
            scale factors. ``"e5m3"`` selects an unsigned 5-exponent-bit,
            3-mantissa-bit format that trades two mantissa bits for one exponent
            bit to widen the scale range; it is Rubin-only, requires the NVFP4
            recipe, and the scale tensors are still passed as
            ``torch.float8_e4m3fn`` because torch has no e5m3 dtype.
        vector_f32: Use vectorized f32
        m_aligned: M alignment (must be 256)
        discrete_col_sfd: Generate discrete col-major scale factor tensor
        act_func: Activation function ("swiglu", "geglu", or block-scaled "situglu")
        linear_offset: Linear offset applied to the up branch in the
            ``act_func == "geglu"`` activation, i.e.
            ``out = (up + linear_offset) * silu(geglu_alpha * gate)``. Ignored
            when ``act_func == "swiglu"``. When ``None`` (default), the offset
            is chosen based on ``act_func`` for backwards compatibility:
            ``1.0`` for ``"geglu"`` and ``0.0`` for ``"swiglu"``. Runtime
            parameter -- a single compiled kernel serves any value, and
            ``linear_offset`` is intentionally not part of the cache key.
        geglu_alpha: Pre-sigmoid scaling factor for the GeGLU activation.
            The fused activation is
            ``out = (clamp(up, glu_clamp_min, glu_clamp_max) + linear_offset)
                    * silu(geglu_alpha * clamp(gate, max=glu_clamp_max))``.
            Defaults to ``1.702`` (GPT-OSS / scaled-GeGLU). Runtime parameter,
            intentionally not part of the cache key. Ignored when
            ``act_func == "swiglu"``.
        glu_clamp_max: Upper clamp limit applied to both the ``gate`` and
            ``up`` halves of the GeGLU pre-activation. Default ``7.0``. Runtime
            parameter, intentionally not part of the cache key. Ignored when
            ``act_func == "swiglu"``.
        glu_clamp_min: Lower clamp limit applied only to the ``up`` half (the
            kernel never lower-clamps the gate). Default ``-7.0``. Runtime
            parameter, intentionally not part of the cache key. Ignored when
            ``act_func == "swiglu"``.
        situ_beta1: Positive finite gate tanh scale for SiTU-GLU. Default
            ``4.0``. This value specializes the compiled kernel and is part of
            the cache key.
        situ_beta2: Positive finite up-branch tanh scale for SiTU-GLU. Default
            ``25.0``. Runtime parameter, intentionally not part of the cache key.
        use_dynamic_sched: Enable dynamic tile scheduling for load balancing
        current_stream: CUDA stream

    Returns:
        TupleDict with keys: c_tensor, d_tensor, d_col_tensor, amax_tensor,
            sfd_row_tensor, sfd_col_tensor
    """
    import torch

    from cudnn.gemm.cutedsl.discrete_grouped.discrete_kernel_utils import _require_pointer_tensor

    a_tensor = call.a_tensor
    sfa_tensor = call.sfa_tensor
    padded_offsets = call.padded_offsets
    alpha_tensor = call.alpha_tensor
    b_tensor = call.b_tensor
    sfb_tensor = call.sfb_tensor
    bias_tensor = call.bias_tensor
    b_ptrs = call.b_ptrs
    sfb_ptrs = call.sfb_ptrs
    n = call.n
    b_major = call.b_major
    norm_const_tensor = call.norm_const_tensor
    prob_tensor = call.prob_tensor
    cd_major = call.cd_major
    # The block-scaled path is torch-native (torch-only allocations and kernels);
    # the normalized call carries canonical (cutlass) dtypes, so map them back.
    acc_dtype = framework_dtype(call.acc_dtype, "torch")
    c_dtype = framework_dtype(call.c_dtype, "torch")
    d_dtype = framework_dtype(call.d_dtype, "torch")
    b_dtype = framework_dtype(call.b_dtype, "torch") if call.b_dtype is not None else None
    mma_tiler_mn = call.mma_tiler_mn
    cluster_shape_mn = call.cluster_shape_mn
    sf_vec_size = call.sf_vec_size
    sf_fp8_dtype_override = call.sf_fp8_dtype_override
    vector_f32 = call.vector_f32
    m_aligned = call.m_aligned
    discrete_col_sfd = call.discrete_col_sfd
    act_func = call.act_func
    linear_offset = call.linear_offset
    geglu_alpha = call.geglu_alpha
    glu_clamp_max = call.glu_clamp_max
    glu_clamp_min = call.glu_clamp_min
    situ_beta1 = call.situ_beta1
    situ_beta2 = call.situ_beta2
    use_dynamic_sched = call.use_dynamic_sched
    use_single_group_runtime_offsets = call.use_single_group_runtime_offsets
    current_stream = call.current_stream

    # Resolve linear_offset default: None means "use the activation-derived legacy
    # default" (1.0 for geglu, 0.0 for swiglu) for backwards compatibility with
    # callers that have not been updated to pass linear_offset explicitly.
    if linear_offset is None:
        linear_offset = 1.0 if act_func == "geglu" else 0.0

    # ---- Auto-detect weight mode ----
    is_dense = b_tensor is not None
    is_discrete = b_ptrs is not None

    if is_dense and is_discrete:
        raise ValueError("Provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs), not both")
    if not is_dense and not is_discrete:
        raise ValueError("Must provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs)")

    valid_m, k_physical, _ = a_tensor.shape

    if is_dense:
        weight_mode = MoEWeightMode.DENSE
        n_full, _, l = b_tensor.shape
        if bias_tensor is not None and tuple(bias_tensor.shape) != (n_full, l):
            raise ValueError(f"bias_tensor must have shape {(n_full, l)}, got {tuple(bias_tensor.shape)}")
    else:
        weight_mode = MoEWeightMode.DISCRETE
        _require_pointer_tensor(b_ptrs, "b_ptrs")
        num_experts = b_ptrs.shape[0]
        _require_pointer_tensor(sfb_ptrs, "sfb_ptrs", num_experts)
        if n is None or b_dtype is None:
            raise ValueError("n and b_dtype are required for discrete mode")
        n_full = n
        k_logical = k_physical * 2 if b_dtype in (torch.float4_e2m1fn_x2, torch.uint8) else k_physical
        b_shape = (n_full, k_logical)
        l = num_experts
        if bias_tensor is not None and tuple(bias_tensor.shape) != (n_full, num_experts):
            raise ValueError(f"bias_tensor must have shape {(n_full, num_experts)}, got {tuple(bias_tensor.shape)}")

    n_out = n_full // 2

    _logger.debug("grouped_gemm_glu_wrapper_sm100: Creating output tensors")

    if cd_major == "n":
        c_tensor_out = torch.empty_strided((valid_m, n_full, 1), (n_full, 1, valid_m * n_full), dtype=c_dtype, device=a_tensor.device)
        d_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_dtype, device=a_tensor.device)
        d_col_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_dtype, device=a_tensor.device)
    else:
        raise ValueError(f"cd_major must be 'n', got {cd_major}")

    sfd_row_tensor = None
    sfd_col_tensor = None
    amax_tensor = None

    if a_tensor.dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ] and sfa_tensor.dtype in [torch.float8_e8m0fnu, torch.float8_e4m3fn]:
        _logger.debug("grouped_gemm_glu_wrapper_sm100: Detected fp8 config, constructing sfd tensors")

        sf_dtype = sfa_tensor.dtype
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        sf_k_row = ceil_div(n_out, sf_vec_size)
        mma_shape_row = (1, ceil_div(valid_m, 128), ceil_div(sf_k_row, 4), 32, 4, 4)
        sfd_row_tensor = torch.empty(mma_shape_row, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

        sf_k_col = ceil_div(valid_m, sf_vec_size)
        mma_shape_col = (1, ceil_div(n_out, 128), ceil_div(sf_k_col, 4), 32, 4, 4)
        sfd_col_tensor = torch.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

    if d_dtype in [torch.bfloat16, torch.float16]:
        _logger.debug("grouped_gemm_glu_wrapper_sm100: Constructing amax_tensor")
        amax_tensor = torch.full((l, 1), float("-inf"), dtype=torch.float32, device=a_tensor.device)

    if valid_m == 0:
        _logger.debug("grouped_gemm_glu_wrapper_sm100: valid_m is zero, skipping kernel execution")
        return TupleDict(
            c_tensor=c_tensor_out,
            d_tensor=d_tensor,
            d_col_tensor=d_col_tensor,
            amax_tensor=amax_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
        )

    # ---- Build cache key ----
    def stride_order(tensor: torch.Tensor) -> Tuple[int, ...]:
        return tuple(i for i, s in sorted(enumerate(tensor.stride()), key=lambda x: x[1]))

    def tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype

    def dynamic_tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        return None, stride_order(tensor), tensor.dtype

    def dynamic_m_tensor_signature(
        tensor: Optional[torch.Tensor], static_shape_suffix: Optional[Tuple[int, ...]], dynamic_stride_dims: Tuple[int, ...] = ()
    ) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        stride_signature = tuple(None if i in dynamic_stride_dims else s for i, s in enumerate(tensor.stride()))
        return static_shape_suffix, stride_signature, tensor.dtype

    use_full_dynamic = is_dense and os.environ.get("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1") != "0"
    situ_beta1_cache_signature = float(situ_beta1) if act_func == "situglu" else None

    device_type = get_device_type()

    if is_dense:
        cache_key = (
            device_type,
            weight_mode,
            act_func,
            situ_beta1_cache_signature,
            use_full_dynamic,
            a_tensor.shape[1:] if not use_full_dynamic else None,
            b_tensor.shape[2] if use_full_dynamic else tuple(b_tensor.shape),
            c_tensor_out.shape[1:] if not use_full_dynamic else None,
            a_tensor.dtype,
            b_tensor.dtype,
            c_tensor_out.dtype,
            stride_order(a_tensor),
            stride_order(b_tensor),
            stride_order(c_tensor_out),
            *(
                dynamic_tensor_signature(sfa_tensor)
                if use_full_dynamic
                else dynamic_m_tensor_signature(sfa_tensor, (sfa_tensor.shape[4], 1) if sfa_tensor is not None else None, dynamic_stride_dims=(5,))
            ),
            *tensor_signature(alpha_tensor),
            *(dynamic_tensor_signature(sfb_tensor) if use_full_dynamic else tensor_signature(sfb_tensor)),
            *(dynamic_tensor_signature(bias_tensor) if use_full_dynamic else tensor_signature(bias_tensor)),
            norm_const_tensor.shape if norm_const_tensor is not None else None,
            norm_const_tensor.stride() if norm_const_tensor is not None else None,
            norm_const_tensor.dtype if norm_const_tensor is not None else None,
            tuple(padded_offsets.shape),
            tuple(padded_offsets.stride()),
            padded_offsets.dtype,
            acc_dtype,
            c_dtype,
            d_dtype,
            cd_major,
            mma_tiler_mn,
            cluster_shape_mn,
            sf_vec_size,
            sf_fp8_dtype_override,
            vector_f32,
            m_aligned,
            discrete_col_sfd,
            use_dynamic_sched,
            use_single_group_runtime_offsets,
            *(dynamic_m_tensor_signature(prob_tensor, (1, 1)) if not use_full_dynamic else dynamic_tensor_signature(prob_tensor)),
        )
    else:
        cache_key = (
            device_type,
            weight_mode,
            act_func,
            situ_beta1_cache_signature,
            a_tensor.shape[1:],
            stride_order(a_tensor),
            a_tensor.dtype,
            b_shape,
            b_dtype,
            c_tensor_out.shape[1:],
            stride_order(c_tensor_out),
            c_tensor_out.dtype,
            *dynamic_m_tensor_signature(sfa_tensor, (sfa_tensor.shape[4], 1) if sfa_tensor is not None else None, dynamic_stride_dims=(5,)),
            *tensor_signature(alpha_tensor),
            *tensor_signature(bias_tensor),
            *tensor_signature(norm_const_tensor),
            *dynamic_m_tensor_signature(prob_tensor, (1, 1)),
            tuple(b_ptrs.shape),
            tuple(b_ptrs.stride()),
            b_ptrs.dtype,
            tuple(sfb_ptrs.shape),
            tuple(sfb_ptrs.stride()),
            sfb_ptrs.dtype,
            tuple(padded_offsets.shape),
            tuple(padded_offsets.stride()),
            padded_offsets.dtype,
            acc_dtype,
            c_dtype,
            d_dtype,
            cd_major,
            mma_tiler_mn,
            cluster_shape_mn,
            sf_vec_size,
            sf_fp8_dtype_override,
            vector_f32,
            m_aligned,
            discrete_col_sfd,
            use_dynamic_sched,
            use_single_group_runtime_offsets,
            b_major,
            num_experts,
        )

    cache_key = backend_cache_key(GroupedGemmBackend.BLOCK_SCALED, *cache_key)

    # ---- Cache lookup or create + compile ----
    if cache_key in _cache_of_GroupedGemmGluSm100Objects:
        _logger.debug("grouped_gemm_glu_wrapper_sm100: Using cached object")
        api = _cache_of_GroupedGemmGluSm100Objects[cache_key]
    else:
        _logger.debug("grouped_gemm_glu_wrapper_sm100: Creating new object")
        if is_dense:
            api = GroupedGemmGluSm100(
                sample_a=a_tensor,
                sample_c=c_tensor_out,
                sample_d=d_tensor,
                sample_sfa=sfa_tensor,
                sample_padded_offsets=padded_offsets,
                sample_alpha=alpha_tensor,
                sample_d_col=d_col_tensor,
                sample_bias=bias_tensor,
                sample_b=b_tensor,
                sample_sfb=sfb_tensor,
                sample_sfd_row=sfd_row_tensor,
                sample_sfd_col=sfd_col_tensor,
                sample_amax=amax_tensor,
                sample_norm_const=norm_const_tensor,
                sample_prob=prob_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                sf_fp8_dtype_override=sf_fp8_dtype_override,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                discrete_col_sfd=discrete_col_sfd,
                act_func=act_func,
                situ_beta1=situ_beta1,
                use_dynamic_sched=use_dynamic_sched,
                use_single_group_runtime_offsets=use_single_group_runtime_offsets,
            )
        else:
            api = GroupedGemmGluSm100(
                sample_a=a_tensor,
                sample_c=c_tensor_out,
                sample_d=d_tensor,
                sample_sfa=sfa_tensor,
                sample_padded_offsets=padded_offsets,
                sample_alpha=alpha_tensor,
                sample_d_col=d_col_tensor,
                sample_bias=bias_tensor,
                num_experts=num_experts,
                b_shape=b_shape,
                b_dtype=b_dtype,
                sample_sfd_row=sfd_row_tensor,
                sample_sfd_col=sfd_col_tensor,
                sample_amax=amax_tensor,
                sample_norm_const=norm_const_tensor,
                sample_prob=prob_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                sf_fp8_dtype_override=sf_fp8_dtype_override,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                discrete_col_sfd=discrete_col_sfd,
                act_func=act_func,
                situ_beta1=situ_beta1,
                b_major=b_major,
                use_dynamic_sched=use_dynamic_sched,
                use_single_group_runtime_offsets=use_single_group_runtime_offsets,
            )

        if not api.check_support():
            raise RuntimeError("Unsupported configuration")
        api.compile()
        _cache_of_GroupedGemmGluSm100Objects[cache_key] = api

    # ---- Execute ----
    if is_dense:
        api.execute(
            a_tensor=a_tensor,
            c_tensor=c_tensor_out,
            d_tensor=d_tensor,
            bias_tensor=bias_tensor,
            sfa_tensor=sfa_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            b_tensor=b_tensor,
            sfb_tensor=sfb_tensor,
            d_col_tensor=d_col_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            prob_tensor=prob_tensor,
            linear_offset=linear_offset,
            geglu_alpha=geglu_alpha,
            glu_clamp_max=glu_clamp_max,
            glu_clamp_min=glu_clamp_min,
            situ_beta1=situ_beta1,
            situ_beta2=situ_beta2,
            current_stream=current_stream,
        )
    else:
        api.execute(
            a_tensor=a_tensor,
            c_tensor=c_tensor_out,
            d_tensor=d_tensor,
            sfa_tensor=sfa_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            b_ptrs=b_ptrs,
            sfb_ptrs=sfb_ptrs,
            bias_tensor=bias_tensor,
            d_col_tensor=d_col_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            prob_tensor=prob_tensor,
            linear_offset=linear_offset,
            geglu_alpha=geglu_alpha,
            glu_clamp_max=glu_clamp_max,
            glu_clamp_min=glu_clamp_min,
            situ_beta1=situ_beta1,
            situ_beta2=situ_beta2,
            current_stream=current_stream,
        )

    return TupleDict(
        c_tensor=c_tensor_out,
        d_tensor=d_tensor,
        d_col_tensor=d_col_tensor,
        amax_tensor=amax_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
    )


def _normalize_glu_call(call: GluCall) -> tuple[GluCall, GroupedGemmBackend]:
    from cudnn.gemm.cutedsl.grouped.unfused._bf16_api import _validate_pointer_tensor

    call = replace(
        call,
        acc_dtype=_convert_to_cutlass_data_type(call.acc_dtype) if call.acc_dtype is not None else cutlass.Float32,
        c_dtype=_convert_to_cutlass_data_type(call.c_dtype) if call.c_dtype is not None else cutlass.BFloat16,
        d_dtype=_convert_to_cutlass_data_type(call.d_dtype) if call.d_dtype is not None else cutlass.BFloat16,
        b_dtype=_convert_to_cutlass_data_type(call.b_dtype) if call.b_dtype is not None else None,
    )

    if call.act_func not in ("swiglu", "geglu", "situglu"):
        raise ValueError(f"act_func must be 'swiglu', 'geglu', or 'situglu', got {call.act_func}")
    if call.act_func == "situglu":
        if not math.isfinite(call.situ_beta1) or call.situ_beta1 <= 0.0:
            raise ValueError(f"situ_beta1 must be finite and positive, got {call.situ_beta1}")
        if not math.isfinite(call.situ_beta2) or call.situ_beta2 <= 0.0:
            raise ValueError(f"situ_beta2 must be finite and positive, got {call.situ_beta2}")
        if get_device_type() == "rubin":
            raise NotImplementedError("Rubin grouped GEMM GLU does not support situglu")

    is_dense = call.b_tensor is not None
    is_discrete = call.b_ptrs is not None
    if is_dense and is_discrete:
        raise ValueError("Provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs), not both")
    if not is_dense and not is_discrete:
        raise ValueError("Must provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs)")
    a_shape = get_shape(call.a_tensor)
    if len(a_shape) != 3 or a_shape[2] != 1:
        raise ValueError(f"a_tensor must have shape (m, k, 1), got {a_shape}")

    valid_m, k, _ = a_shape
    if is_dense:
        b_full_shape = get_shape(call.b_tensor)
        if len(b_full_shape) != 3:
            raise ValueError(f"b_tensor must have shape (n, k, experts), got " f"{b_full_shape}")
        n_full, b_k, num_experts = b_full_shape
        if b_k != k:
            raise ValueError(f"b_tensor K dimension ({b_k}) must match a_tensor ({k})")
        defining_b_dtype = call.b_tensor.dtype
        b_shape = None
        weight_mode = MoEWeightMode.DENSE
        if call.n is not None or call.b_dtype is not None:
            raise ValueError("Dense mode forbids n and b_dtype")
    else:
        num_experts = _validate_pointer_tensor(call.b_ptrs, "b_ptrs")
        if call.n is None or call.b_dtype is None:
            raise ValueError("n and b_dtype are required for discrete mode")
        n_full = call.n
        defining_b_dtype = call.b_dtype
        b_shape = (n_full, k)
        weight_mode = MoEWeightMode.DISCRETE

    backend = select_grouped_gemm_backend(
        operation="grouped_gemm_glu_sm100",
        a_dtype=call.a_tensor.dtype,
        b_dtype=defining_b_dtype,
        scale_controls=(
            ("sfa_tensor", call.sfa_tensor),
            ("sfb_tensor", call.sfb_tensor),
            ("sfb_ptrs", call.sfb_ptrs),
            ("norm_const_tensor", call.norm_const_tensor),
            ("sf_vec_size", call.sf_vec_size if call.sf_vec_size != 16 else None),
            ("sf_fp8_dtype_override", call.sf_fp8_dtype_override),
            (
                "discrete_col_sfd",
                call.discrete_col_sfd if call.discrete_col_sfd else None,
            ),
            (
                "geglu_alpha",
                call.geglu_alpha if call.geglu_alpha != 1.702 else None,
            ),
            (
                "glu_clamp_max",
                call.glu_clamp_max if call.glu_clamp_max != 7.0 else None,
            ),
            (
                "glu_clamp_min",
                call.glu_clamp_min if call.glu_clamp_min != -7.0 else None,
            ),
        ),
        block_scaled_dtype_pairs=_block_scaled_dtype_pairs(),
    )

    linear_offset = call.linear_offset
    if linear_offset is None:
        linear_offset = 1.0 if call.act_func == "geglu" else 0.0

    normalized = replace(
        call,
        linear_offset=linear_offset,
        weight_mode=weight_mode,
        b_shape=b_shape,
        num_experts=num_experts,
    )
    if backend is GroupedGemmBackend.BLOCK_SCALED:
        return normalized, backend

    if call.use_single_group_runtime_offsets:
        raise ValueError("use_single_group_runtime_offsets is supported only by the block-scaled kernel")
    if call.prob_tensor is None:
        raise ValueError("prob_tensor is required for BF16")
    if call.cd_major != "n":
        raise ValueError(f"cd_major must be 'n', got {call.cd_major}")
    if call.act_func not in ("swiglu", "geglu"):
        raise ValueError(f"BF16 act_func must be 'swiglu' or 'geglu'; situglu is block-scaled only, got {call.act_func}")
    if normalized.c_dtype not in (cutlass.BFloat16, cutlass.Float16, cutlass.Float32):
        raise ValueError(f"c_dtype must be BF16, FP16, or FP32, got {normalized.c_dtype}")
    if normalized.d_dtype not in (cutlass.BFloat16, cutlass.Float16, cutlass.Float32):
        raise ValueError(f"d_dtype must be BF16, FP16, or FP32, got {normalized.d_dtype}")
    if call.m_aligned != 256:
        raise ValueError(f"m_aligned must be 256, got {call.m_aligned}")
    if valid_m % 256 != 0:
        raise ValueError(f"a_tensor M dimension must be 256-aligned, got {valid_m}")
    if n_full <= 0 or n_full % 64 != 0:
        raise ValueError(f"N must be positive and divisible by 64, got {n_full}")
    if get_shape(call.prob_tensor) != (valid_m, 1, 1):
        raise ValueError(f"prob_tensor must have shape {(valid_m, 1, 1)}, got " f"{get_shape(call.prob_tensor)}")
    if call.bias_tensor is not None and get_shape(call.bias_tensor) != (
        n_full,
        num_experts,
    ):
        raise ValueError(f"bias_tensor must have shape {(n_full, num_experts)}, got " f"{get_shape(call.bias_tensor)}")
    if is_discrete:
        if get_device(call.b_ptrs) != get_device(call.a_tensor):
            raise ValueError(f"b_ptrs must be on the same device as a_tensor " f"({get_device(call.a_tensor)}), got {get_device(call.b_ptrs)}")
        offsets_shape = get_shape(call.padded_offsets)
        if len(offsets_shape) == 1 and num_experts != offsets_shape[0]:
            raise ValueError(f"b_ptrs length mismatch: expected {offsets_shape[0]}, " f"got {num_experts}")
    if not cuda_is_available():
        raise RuntimeError("CUDA is not available")
    major, minor = get_compute_capability()
    compute_capability = major * 10 + minor
    if compute_capability < 100:
        raise RuntimeError(f"GroupedGemmGluSm100 requires SM100+, found SM{compute_capability}")
    return normalized, backend


def _glu_stride_order(tensor: torch.Tensor) -> Tuple[int, ...]:
    strides = get_strides(tensor)
    shape = get_shape(tensor)
    return tuple(
        index
        for index, _ in sorted(
            enumerate(strides),
            key=lambda item: (item[1], shape[item[0]]),
        )
    )


def _glu_allocate_output(framework: str, shape: tuple, stride: tuple, dtype, device):
    if framework == "torch":
        import torch

        return torch.empty_strided(shape, stride, dtype=framework_dtype(dtype, "torch"), device=device)
    import jax
    import jax.numpy as jnp

    # n-major C-contiguous; the extent-1 batch dim's stride is unobservable.
    # The kernel writes into this buffer on the launch stream; materialize it first.
    return jax.block_until_ready(jnp.empty(shape, dtype=framework_dtype(dtype, "jax"), device=device))


def _glu_operand_meta(tensor: Optional[torch.Tensor]) -> Optional[tuple]:
    """Everything the wrapper's derivation reads off an operand, and nothing else.

    Deliberately not the object's identity: a tensor's address is recycled by CPython
    as soon as it is freed, so an id-keyed memo answers for tensors it never saw.
    Data pointers are excluded because they vary per call and nothing derived here
    depends on them -- their alignment is re-checked inside execute().
    """
    if tensor is None:
        return None
    device = get_device(tensor)
    return (get_shape(tensor), get_strides(tensor), tensor.dtype, device.type, device.index)


# Operand-metadata key -> the BF16 path's derived result. One entry per distinct
# (operand metadata, config), the same growth as _cache_of_GroupedGemmGluSm100Objects.
_glu_wrapper_memo: dict = {}


def _glu_tensor_signature(tensor: Optional[torch.Tensor], *, dynamic_m: bool = False) -> tuple:
    if tensor is None:
        return (None, None, None, None)
    device = get_device(tensor)
    shape = (None, *get_shape(tensor)[1:]) if dynamic_m else get_shape(tensor)
    return (
        shape,
        _glu_stride_order(tensor),
        _convert_to_cutlass_data_type(tensor.dtype),
        (device.type, device.index),
    )


def _grouped_gemm_glu_bf16_call(call: GluCall, memo_key: Optional[tuple] = None) -> TupleDict:
    framework = detect_framework(call.a_tensor)
    valid_m, k, _ = get_shape(call.a_tensor)
    if call.weight_mode == MoEWeightMode.DENSE:
        n_full = get_shape(call.b_tensor)[0]
    else:
        n_full = call.n
    n_out = n_full // 2

    def _allocate_output(shape, stride, dtype):
        return _glu_allocate_output(framework, shape, stride, dtype, call.a_tensor.device)

    c_tensor = _allocate_output((valid_m, n_full, 1), (n_full, 1, valid_m * n_full), call.c_dtype)
    d_tensor = _allocate_output((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), call.d_dtype)

    overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    workspace_bytes = (128 * call.num_experts if call.weight_mode == MoEWeightMode.DISCRETE else 0) + (4 if call.use_dynamic_sched else 0)
    cache_key = backend_cache_key(
        GroupedGemmBackend.BF16,
        call.weight_mode,
        call.act_func,
        _glu_tensor_signature(call.a_tensor, dynamic_m=True),
        _glu_tensor_signature(call.b_tensor),
        call.b_shape,
        call.b_dtype,
        _glu_tensor_signature(c_tensor, dynamic_m=True),
        _glu_tensor_signature(d_tensor, dynamic_m=True),
        _glu_tensor_signature(call.alpha_tensor),
        _glu_tensor_signature(call.bias_tensor),
        _glu_tensor_signature(call.padded_offsets),
        _glu_tensor_signature(call.prob_tensor, dynamic_m=True),
        (_glu_tensor_signature(call.b_ptrs) if call.b_ptrs is not None else None),
        call.acc_dtype,
        call.c_dtype,
        call.d_dtype,
        call.mma_tiler_mn,
        call.cluster_shape_mn,
        call.vector_f32,
        call.m_aligned,
        call.generate_c,
        call.b_major,
        call.use_dynamic_sched,
        workspace_bytes,
        ((get_device(call.a_tensor).type, get_device(call.a_tensor).index)),
        overlap_margin,
    )

    if cache_key in _cache_of_GroupedGemmGluSm100Objects:
        api = _cache_of_GroupedGemmGluSm100Objects[cache_key]
    else:
        api = GroupedGemmGluSm100(
            sample_a=call.a_tensor,
            sample_c=c_tensor,
            sample_d=d_tensor,
            sample_sfa=None,
            sample_padded_offsets=call.padded_offsets,
            sample_alpha=call.alpha_tensor,
            sample_d_col=None,
            sample_b=call.b_tensor,
            sample_sfb=None,
            sample_bias=call.bias_tensor,
            num_experts=(call.num_experts if call.weight_mode == MoEWeightMode.DISCRETE else None),
            b_shape=call.b_shape,
            b_dtype=call.b_dtype,
            sample_sfd_row=None,
            sample_sfd_col=None,
            sample_amax=None,
            sample_norm_const=None,
            sample_prob=call.prob_tensor,
            acc_dtype=call.acc_dtype,
            mma_tiler_mn=call.mma_tiler_mn,
            cluster_shape_mn=call.cluster_shape_mn,
            sf_vec_size=16,
            vector_f32=call.vector_f32,
            m_aligned=call.m_aligned,
            discrete_col_sfd=False,
            act_func=call.act_func,
            b_major=call.b_major,
            use_dynamic_sched=call.use_dynamic_sched,
            generate_c=call.generate_c,
        )
        if not api.check_support():
            raise RuntimeError("Unsupported BF16 configuration")
        api.compile()
        _cache_of_GroupedGemmGluSm100Objects[cache_key] = api

    if memo_key is not None:
        _glu_wrapper_memo[memo_key] = (api, framework, valid_m, n_full, n_out, call.c_dtype, call.d_dtype)

    api.execute(
        a_tensor=call.a_tensor,
        c_tensor=c_tensor,
        d_tensor=d_tensor,
        sfa_tensor=None,
        padded_offsets=call.padded_offsets,
        alpha_tensor=call.alpha_tensor,
        b_tensor=call.b_tensor,
        sfb_tensor=None,
        bias_tensor=call.bias_tensor,
        b_ptrs=call.b_ptrs,
        sfb_ptrs=None,
        d_col_tensor=None,
        sfd_row_tensor=None,
        sfd_col_tensor=None,
        amax_tensor=None,
        norm_const_tensor=None,
        prob_tensor=call.prob_tensor,
        linear_offset=call.linear_offset,
        geglu_alpha=call.geglu_alpha,
        glu_clamp_max=call.glu_clamp_max,
        glu_clamp_min=call.glu_clamp_min,
        current_stream=call.current_stream,
    )
    return TupleDict(
        c_tensor=c_tensor if call.generate_c else None,
        d_tensor=d_tensor,
        d_col_tensor=None,
        amax_tensor=None,
        sfd_row_tensor=None,
        sfd_col_tensor=None,
    )


def grouped_gemm_glu_wrapper_sm100(
    a_tensor: torch.Tensor,
    sfa_tensor: Optional[torch.Tensor],
    padded_offsets: torch.Tensor,
    alpha_tensor: torch.Tensor,
    b_tensor: Optional[torch.Tensor] = None,
    sfb_tensor: Optional[torch.Tensor] = None,
    bias_tensor: Optional[torch.Tensor] = None,
    b_ptrs: Optional[torch.Tensor] = None,
    sfb_ptrs: Optional[torch.Tensor] = None,
    n: Optional[int] = None,
    b_dtype: Optional[torch.dtype] = None,
    b_major: str = "k",
    norm_const_tensor: Optional[torch.Tensor] = None,
    prob_tensor: Optional[torch.Tensor] = None,
    acc_dtype: Optional[torch.dtype] = None,
    c_dtype: Optional[torch.dtype] = None,
    d_dtype: Optional[torch.dtype] = None,
    cd_major: str = "n",
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
    act_func: str = "swiglu",
    linear_offset: Optional[float] = None,
    geglu_alpha: float = 1.702,
    glu_clamp_max: float = 7.0,
    glu_clamp_min: float = -7.0,
    situ_beta1: float = 4.0,
    situ_beta2: float = 25.0,
    use_dynamic_sched: bool = False,
    use_single_group_runtime_offsets: bool = False,
    current_stream: Optional[cuda.CUstream] = None,
    generate_c: bool = False,
    sf_fp8_dtype_override: Optional[Literal["e5m3"]] = None,
) -> TupleDict:
    """Dispatch grouped GEMM GLU once from an immutable normalized call."""
    # Hot-loop memo; see _operand_meta in the unfused API for the rationale. Everything
    # from here to api.execute() is derivation -- dtype resolution, GluCall construction,
    # normalization, and the op cache-key rebuild -- and is a pure function of the
    # operands' metadata plus the scalar config, both of which are in the key below.
    # A hit skips the derivation, never a check: api.execute() still validates every
    # operand, including the data pointers the key deliberately omits. linear_offset is
    # excluded on purpose (it is not part of the op cache key either) and passed through.
    memo_key = (
        type(a_tensor),
        _glu_operand_meta(a_tensor),
        _glu_operand_meta(sfa_tensor),
        _glu_operand_meta(padded_offsets),
        _glu_operand_meta(alpha_tensor),
        _glu_operand_meta(b_tensor),
        _glu_operand_meta(sfb_tensor),
        _glu_operand_meta(bias_tensor),
        _glu_operand_meta(b_ptrs),
        _glu_operand_meta(sfb_ptrs),
        _glu_operand_meta(norm_const_tensor),
        _glu_operand_meta(prob_tensor),
        n,
        b_dtype,
        b_major,
        acc_dtype,
        c_dtype,
        d_dtype,
        cd_major,
        tuple(mma_tiler_mn),
        None if cluster_shape_mn is None else tuple(cluster_shape_mn),
        sf_vec_size,
        sf_fp8_dtype_override,
        vector_f32,
        m_aligned,
        discrete_col_sfd,
        act_func,
        geglu_alpha,
        glu_clamp_max,
        glu_clamp_min,
        situ_beta1,
        situ_beta2,
        use_dynamic_sched,
        use_single_group_runtime_offsets,
        generate_c,
        os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"),
    )
    memo = _glu_wrapper_memo.get(memo_key)
    if memo is not None:
        api, framework, valid_m, n_full, n_out, memo_c_dtype, memo_d_dtype = memo
        c_out = _glu_allocate_output(framework, (valid_m, n_full, 1), (n_full, 1, valid_m * n_full), memo_c_dtype, a_tensor.device)
        d_out = _glu_allocate_output(framework, (valid_m, n_out, 1), (n_out, 1, valid_m * n_out), memo_d_dtype, a_tensor.device)
        api.execute(
            a_tensor=a_tensor,
            c_tensor=c_out,
            d_tensor=d_out,
            sfa_tensor=None,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            b_tensor=b_tensor,
            sfb_tensor=None,
            bias_tensor=bias_tensor,
            b_ptrs=b_ptrs,
            sfb_ptrs=None,
            d_col_tensor=None,
            sfd_row_tensor=None,
            sfd_col_tensor=None,
            amax_tensor=None,
            norm_const_tensor=None,
            prob_tensor=prob_tensor,
            linear_offset=linear_offset,
            geglu_alpha=geglu_alpha,
            glu_clamp_max=glu_clamp_max,
            glu_clamp_min=glu_clamp_min,
            current_stream=current_stream,
        )
        return TupleDict(
            c_tensor=c_out if generate_c else None,
            d_tensor=d_out,
            d_col_tensor=None,
            amax_tensor=None,
            sfd_row_tensor=None,
            sfd_col_tensor=None,
        )

    framework = detect_framework(a_tensor)
    if framework not in ("torch", "jax"):
        raise ValueError(f"Unsupported tensor framework '{framework}' for grouped_gemm_glu_wrapper_sm100; pass torch tensors or JAX arrays")
    if framework == "jax":
        if b_tensor is not None:
            raise ValueError(_JAX_DENSE_B_ERROR)
        if bias_tensor is not None:
            raise ValueError(_JAX_BIAS_ERROR)
    acc_dtype = _convert_to_cutlass_data_type(acc_dtype) if acc_dtype is not None else cutlass.Float32
    c_dtype = _convert_to_cutlass_data_type(c_dtype) if c_dtype is not None else cutlass.BFloat16
    d_dtype = _convert_to_cutlass_data_type(d_dtype) if d_dtype is not None else cutlass.BFloat16
    b_dtype = _convert_to_cutlass_data_type(b_dtype) if b_dtype is not None else None
    _reject_unsupported_rubin_glu_tune_params(
        get_device_type() == "rubin",
        geglu_alpha,
        glu_clamp_max,
        glu_clamp_min,
    )
    call = GluCall(
        a_tensor=a_tensor,
        sfa_tensor=sfa_tensor,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha_tensor,
        b_tensor=b_tensor,
        sfb_tensor=sfb_tensor,
        bias_tensor=bias_tensor,
        b_ptrs=b_ptrs,
        sfb_ptrs=sfb_ptrs,
        n=n,
        b_dtype=b_dtype,
        b_major=b_major,
        norm_const_tensor=norm_const_tensor,
        prob_tensor=prob_tensor,
        acc_dtype=acc_dtype,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        cd_major=cd_major,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        sf_fp8_dtype_override=sf_fp8_dtype_override,
        vector_f32=vector_f32,
        m_aligned=m_aligned,
        discrete_col_sfd=discrete_col_sfd,
        act_func=act_func,
        linear_offset=linear_offset,
        geglu_alpha=geglu_alpha,
        glu_clamp_max=glu_clamp_max,
        glu_clamp_min=glu_clamp_min,
        situ_beta1=situ_beta1,
        situ_beta2=situ_beta2,
        use_dynamic_sched=use_dynamic_sched,
        use_single_group_runtime_offsets=use_single_group_runtime_offsets,
        current_stream=current_stream,
        generate_c=generate_c,
    )
    call, backend = _normalize_glu_call(call)
    if backend is GroupedGemmBackend.BF16:
        return _grouped_gemm_glu_bf16_call(call, memo_key)
    if framework == "jax":
        raise ValueError(_JAX_BLOCK_SCALED_ERROR)
    return _grouped_gemm_glu_block_scaled_call(call)


__all__ = ["GluCall", "GroupedGemmGluSm100", "grouped_gemm_glu_wrapper_sm100"]
