# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unified API for Grouped GEMM dGLU Backward Kernel (SM100+)

This module provides a single API class that supports both contiguous (dense)
and discrete weight modes for block-scaled grouped GEMM with dGLU activation
gradient (dSwiGLU / dGeGLU / dSiTU-GLU) in MoE (Mixture of Experts) workloads.

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
    _torch_stream_context,
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
_JAX_BLOCK_SCALED_ERROR = (
    "The block-scaled grouped GEMM dGLU backend is not expressible as JAX arrays "
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


from ._bf16_api import GroupedGemmDgluBf16API
from ._blockscaled_api import (
    GroupedGemmDgluBlockScaledAPI,
    _reject_unsupported_rubin_glu_tune_params,
)


@dataclass(frozen=True)
class DgluCall:
    """Immutable normalized input for dGLU dispatch, allocation, and caching."""

    a_tensor: torch.Tensor
    c_tensor: torch.Tensor
    sfa_tensor: Optional[torch.Tensor]
    padded_offsets: torch.Tensor
    alpha_tensor: torch.Tensor
    beta_tensor: torch.Tensor
    prob_tensor: Optional[torch.Tensor]
    dprob_tensor: Optional[torch.Tensor]
    b_tensor: Optional[torch.Tensor] = None
    sfb_tensor: Optional[torch.Tensor] = None
    generate_dbias: bool = False
    b_ptrs: Optional[torch.Tensor] = None
    sfb_ptrs: Optional[torch.Tensor] = None
    n: Optional[int] = None
    b_dtype: Optional[torch.dtype] = None
    b_major: str = "k"
    norm_const_tensor: Optional[torch.Tensor] = None
    acc_dtype: Optional[torch.dtype] = None
    d_dtype: Optional[torch.dtype] = None
    cd_major: str = "n"
    mma_tiler_mn: Tuple[int, int] = (256, 256)
    cluster_shape_mn: Optional[Tuple[int, int]] = None
    sf_vec_size: int = 16
    sf_fp8_dtype_override: Optional[Literal["e5m3"]] = None
    vector_f32: bool = False
    m_aligned: int = 256
    discrete_col_sfd: bool = False
    act_func: str = "dswiglu"
    linear_offset: Optional[float] = None
    geglu_alpha: float = 1.702
    glu_clamp_max: float = 7.0
    glu_clamp_min: float = -7.0
    situ_beta1: float = 4.0
    situ_beta2: float = 25.0
    epilogue_op: Optional[str] = None
    use_dynamic_sched: bool = False
    use_single_group_runtime_offsets: bool = False
    current_stream: Optional[cuda.CUstream] = None
    weight_mode: Optional[MoEWeightMode] = None
    b_shape: Optional[Tuple[int, ...]] = None
    num_experts: Optional[int] = None


class GroupedGemmDgluSm100(APIBase):
    """Stable public facade that selects the dGLU backend during support checking."""

    # BF16 implementation
    @overload
    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_c: torch.Tensor,
        sample_d_row: torch.Tensor,
        sample_d_col: None,
        sample_sfa: None,
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_beta: torch.Tensor,
        sample_prob: torch.Tensor,
        sample_dprob: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...

    # Block-scaled implementation
    @overload
    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_c: torch.Tensor,
        sample_d_row: torch.Tensor,
        sample_d_col: Optional[torch.Tensor],
        sample_sfa: torch.Tensor,
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_beta: torch.Tensor,
        sample_prob: Optional[torch.Tensor],
        sample_dprob: Optional[torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_c: torch.Tensor,
        sample_d_row: torch.Tensor,
        sample_d_col: Optional[torch.Tensor],
        sample_sfa: Optional[torch.Tensor],
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_beta: torch.Tensor,
        sample_prob: Optional[torch.Tensor],
        sample_dprob: Optional[torch.Tensor],
        sample_b: Optional[torch.Tensor] = None,
        sample_sfb: Optional[torch.Tensor] = None,
        sample_dbias: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        b_shape: Optional[Tuple[int, ...]] = None,
        b_dtype: Optional[torch.dtype] = None,
        sample_sfd_row: Optional[torch.Tensor] = None,
        sample_sfd_col: Optional[torch.Tensor] = None,
        sample_amax: Optional[torch.Tensor] = None,
        sample_norm_const: Optional[torch.Tensor] = None,
        acc_dtype: Optional[torch.dtype] = None,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        sf_fp8_dtype_override: Optional[Literal["e5m3"]] = None,
        vector_f32: bool = False,
        m_aligned: int = 256,
        discrete_col_sfd: bool = False,
        act_func: str = "dswiglu",
        b_major: str = "k",
        epilogue_op: Optional[str] = None,
        use_dynamic_sched: bool = False,
        use_single_group_runtime_offsets: bool = False,
        linear_offset: Optional[float] = None,
        geglu_alpha: float = 1.702,
        glu_clamp_max: float = 7.0,
        glu_clamp_min: float = -7.0,
        situ_beta1: float = 4.0,
        situ_beta2: float = 25.0,
    ) -> None:
        super().__init__()
        self._pending_init_kwargs = dict(locals())
        self._pending_init_kwargs.pop("self")
        self._pending_init_kwargs.pop("__class__", None)
        framework = detect_framework(sample_a)
        if sample_a is not None and framework not in ("torch", "jax"):
            raise ValueError(f"Unsupported tensor framework '{framework}' for GroupedGemmDgluSm100; pass torch tensors or JAX arrays")
        if acc_dtype is None:
            self._pending_init_kwargs["acc_dtype"] = cutlass.Float32
        self._implementation = None

    def check_support(self) -> bool:
        if self._implementation is None:
            kwargs = self._pending_init_kwargs
            defining_b_dtype = kwargs["sample_b"].dtype if kwargs["sample_b"] is not None else kwargs["b_dtype"]
            backend = select_grouped_gemm_backend(
                operation="grouped_gemm_dglu_sm100",
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
                    ("geglu_alpha", kwargs["geglu_alpha"] if kwargs["geglu_alpha"] != 1.702 else None),
                    ("glu_clamp_max", kwargs["glu_clamp_max"] if kwargs["glu_clamp_max"] != 7.0 else None),
                    ("glu_clamp_min", kwargs["glu_clamp_min"] if kwargs["glu_clamp_min"] != -7.0 else None),
                    ("epilogue_op", kwargs["epilogue_op"] if kwargs["epilogue_op"] not in (None, "none", "identity") else None),
                ),
                block_scaled_dtype_pairs=_block_scaled_dtype_pairs(),
            )
            self.backend = backend
            self.linear_offset = kwargs["linear_offset"]
            if self.linear_offset is None:
                self.linear_offset = 1.0 if kwargs["act_func"] == "dgeglu" else 0.0
            if backend is GroupedGemmBackend.BF16:
                self._value_error_if(
                    kwargs["use_single_group_runtime_offsets"],
                    "use_single_group_runtime_offsets is supported only by the block-scaled kernel",
                )
                self._implementation = GroupedGemmDgluBf16API(
                    sample_a=kwargs["sample_a"],
                    sample_c=kwargs["sample_c"],
                    sample_d_row=kwargs["sample_d_row"],
                    sample_padded_offsets=kwargs["sample_padded_offsets"],
                    sample_alpha=kwargs["sample_alpha"],
                    sample_beta=kwargs["sample_beta"],
                    sample_prob=kwargs["sample_prob"],
                    sample_dprob=kwargs["sample_dprob"],
                    sample_b=kwargs["sample_b"],
                    sample_dbias=kwargs["sample_dbias"],
                    num_experts=kwargs["num_experts"],
                    b_shape=kwargs["b_shape"],
                    b_dtype=kwargs["b_dtype"],
                    acc_dtype=kwargs["acc_dtype"],
                    mma_tiler_mn=kwargs["mma_tiler_mn"],
                    cluster_shape_mn=kwargs["cluster_shape_mn"],
                    vector_f32=kwargs["vector_f32"],
                    m_aligned=kwargs["m_aligned"],
                    act_func=kwargs["act_func"],
                    b_major=kwargs["b_major"],
                    use_dynamic_sched=kwargs["use_dynamic_sched"],
                )
            else:
                if detect_framework(kwargs["sample_a"]) == "jax":
                    raise ValueError(_JAX_BLOCK_SCALED_ERROR)
                block_kwargs = dict(kwargs)
                # The block-scaled implementation is torch-native: hand it torch dtypes.
                block_kwargs["acc_dtype"] = framework_dtype(block_kwargs["acc_dtype"], "torch")
                if block_kwargs.get("b_dtype") is not None:
                    block_kwargs["b_dtype"] = framework_dtype(block_kwargs["b_dtype"], "torch")
                self._implementation = GroupedGemmDgluBlockScaledAPI(**block_kwargs)
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
        d_row_tensor: torch.Tensor,
        d_col_tensor: None,
        sfa_tensor: None,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        beta_tensor: torch.Tensor,
        prob_tensor: torch.Tensor,
        dprob_tensor: torch.Tensor,
        b_tensor: Optional[torch.Tensor] = None,
        *,
        sfb_tensor: None = None,
        sfb_ptrs: None = None,
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
        d_row_tensor: torch.Tensor,
        d_col_tensor: Optional[torch.Tensor],
        sfa_tensor: torch.Tensor,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        beta_tensor: torch.Tensor,
        prob_tensor: Optional[torch.Tensor],
        dprob_tensor: Optional[torch.Tensor],
        b_tensor: Optional[torch.Tensor] = None,
        *,
        sfb_tensor: Optional[torch.Tensor] = None,
        sfb_ptrs: Optional[torch.Tensor] = None,
        sfd_row_tensor: Optional[torch.Tensor] = None,
        sfd_col_tensor: Optional[torch.Tensor] = None,
        amax_tensor: Optional[torch.Tensor] = None,
        norm_const_tensor: Optional[torch.Tensor] = None,
    ) -> None: ...

    def execute(
        self,
        a_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        d_row_tensor: torch.Tensor,
        d_col_tensor: Optional[torch.Tensor],
        sfa_tensor: Optional[torch.Tensor],
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        beta_tensor: torch.Tensor,
        prob_tensor: Optional[torch.Tensor],
        dprob_tensor: Optional[torch.Tensor],
        b_tensor: Optional[torch.Tensor] = None,
        sfb_tensor: Optional[torch.Tensor] = None,
        dbias_tensor: Optional[torch.Tensor] = None,
        b_ptrs: Optional[torch.Tensor] = None,
        sfb_ptrs: Optional[torch.Tensor] = None,
        sfd_row_tensor: Optional[torch.Tensor] = None,
        sfd_col_tensor: Optional[torch.Tensor] = None,
        amax_tensor: Optional[torch.Tensor] = None,
        norm_const_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        if self._implementation is None:
            raise RuntimeError("Kernel not compiled; call compile() first")
        if self.backend is GroupedGemmBackend.BF16:
            controls = (
                ("sfa_tensor", sfa_tensor),
                ("sfb_tensor", sfb_tensor),
                ("sfb_ptrs", sfb_ptrs),
                ("d_col_tensor", d_col_tensor),
                ("sfd_row_tensor", sfd_row_tensor),
                ("sfd_col_tensor", sfd_col_tensor),
                ("amax_tensor", amax_tensor),
                ("norm_const_tensor", norm_const_tensor),
            )
            forbidden = [name for name, value in controls if value is not None]
            if forbidden:
                raise ValueError(f"grouped_gemm_dglu_sm100: BF16 forbids scale control {forbidden[0]}")
            self._implementation.execute(
                a_tensor=a_tensor,
                c_tensor=c_tensor,
                d_row_tensor=d_row_tensor,
                padded_offsets=padded_offsets,
                alpha_tensor=alpha_tensor,
                beta_tensor=beta_tensor,
                prob_tensor=prob_tensor,
                dprob_tensor=dprob_tensor,
                b_tensor=b_tensor,
                b_ptrs=b_ptrs,
                dbias_tensor=dbias_tensor,
                linear_offset=self.linear_offset,
                current_stream=current_stream,
            )
        else:
            self._implementation.execute(
                a_tensor=a_tensor,
                c_tensor=c_tensor,
                d_row_tensor=d_row_tensor,
                d_col_tensor=d_col_tensor,
                sfa_tensor=sfa_tensor,
                padded_offsets=padded_offsets,
                alpha_tensor=alpha_tensor,
                beta_tensor=beta_tensor,
                prob_tensor=prob_tensor,
                dprob_tensor=dprob_tensor,
                b_tensor=b_tensor,
                sfb_tensor=sfb_tensor,
                dbias_tensor=dbias_tensor,
                b_ptrs=b_ptrs,
                sfb_ptrs=sfb_ptrs,
                sfd_row_tensor=sfd_row_tensor,
                sfd_col_tensor=sfd_col_tensor,
                amax_tensor=amax_tensor,
                norm_const_tensor=norm_const_tensor,
                current_stream=current_stream,
            )
        self._is_supported = self._implementation._is_supported
        self._compiled_kernel = self._implementation._compiled_kernel


# --------------------------------------------------------------------------- #
#  Convenience wrapper with caching
# --------------------------------------------------------------------------- #

_logger = logging.getLogger(__name__)
_cache_of_GroupedGemmDgluSm100Objects = {}


def _grouped_gemm_dglu_block_scaled_call(call: DgluCall) -> TupleDict:
    """Convenience wrapper for grouped GEMM dGLU backward operation.

    Auto-detects dense vs. discrete mode based on which weight arguments
    are provided.

    Dense mode: provide ``b_tensor`` and ``sfb_tensor``.
    Discrete mode: provide ``b_ptrs``, ``sfb_ptrs``, ``n``, and ``b_dtype``.

    Compiled kernels are cached for reuse when called with the same configuration.

    Args:
        a_tensor: Input A tensor (valid_m, k, 1) -- gradient input
        c_tensor: Forward activations input (valid_m, n_out, 1)
        sfa_tensor: Scale factor A
        padded_offsets: End offset per expert after padding
        alpha_tensor: Per-group alpha scaling
        beta_tensor: Per-group beta scaling
        prob_tensor: Per-row probability (from forward)
        dprob_tensor: Gradient of probability (output, must be zero-initialized)
        b_tensor: (Dense) Weight B tensor (n, k, l)
        sfb_tensor: (Dense) Scale factor B
        generate_dbias: Optional flag to allocate and return dbias output
        b_ptrs: (Discrete) 1-D int64 device tensor of per-expert B data pointers
        sfb_ptrs: (Discrete) 1-D int64 device tensor of per-expert SFB data pointers
        n: (Discrete) B weight N dimension
        b_dtype: (Discrete) B weight data type
        b_major: (Discrete) B tensor major dimension ("k" or "n")
        norm_const_tensor: Optional normalization constant
        acc_dtype: Accumulator data type
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
        vector_f32: Use vectorized f32 for dSwiGLU and dGeGLU. K3-default
            dSiTU-GLU (``situ_beta1=4.0``) always uses its packed FP32x2
            specialization; non-default dSiTU-GLU uses scalar FP32.
        m_aligned: M alignment (must be 256)
        discrete_col_sfd: Generate discrete col-major scale factor tensor
        act_func: Activation function ("dswiglu", "dgeglu", or block-scaled "dsituglu")
        linear_offset: Linear offset matching the forward GeGLU activation, i.e.
            the same value used by ``grouped_gemm_glu_wrapper_sm100`` so the
            backward gradients are mathematically consistent. Affects
            ``act_func == "dgeglu"``; ignored when ``act_func == "dswiglu"``.
            When ``None`` (default), the offset is chosen based on ``act_func``
            for backwards compatibility: ``1.0`` for ``"dgeglu"`` and ``0.0``
            for ``"dswiglu"``.
        geglu_alpha: Pre-sigmoid scaling factor for the GeGLU activation being
            differentiated. Must match the value used in the forward.
            Default ``1.702``. Ignored when ``act_func == "dswiglu"``.
        glu_clamp_max: Upper clamp limit applied to ``up`` and ``gate`` in the
            forward GeGLU; the same limit drives the gradient mask here.
            Default ``7.0``. Ignored when ``act_func == "dswiglu"``.
        glu_clamp_min: Lower clamp limit applied to ``up`` only in the forward
            GeGLU; the same limit drives the gradient mask here.
            Default ``-7.0``. Ignored when ``act_func == "dswiglu"``.
        situ_beta1: Positive finite gate tanh scale for dSiTU-GLU. Default
            ``4.0``. Compile-time specialization and part of the cache key.
        situ_beta2: Positive finite up-branch tanh scale for dSiTU-GLU. Default
            ``25.0``. Compile-time specialization and part of the cache key.
        epilogue_op: Optional epilogue operation. Valid: None, "none", "identity", "relu", "srelu"
        use_dynamic_sched: Enable dynamic tile scheduling for load balancing
        current_stream: CUDA stream

    Returns:
        TupleDict with keys: d_row_tensor, d_col_tensor, dprob_tensor,
            dbias_tensor, amax_tensor, sfd_row_tensor, sfd_col_tensor
    """
    import torch

    from cudnn.gemm.cutedsl.discrete_grouped.discrete_kernel_utils import _require_pointer_tensor

    a_tensor = call.a_tensor
    c_tensor = call.c_tensor
    sfa_tensor = call.sfa_tensor
    padded_offsets = call.padded_offsets
    alpha_tensor = call.alpha_tensor
    beta_tensor = call.beta_tensor
    prob_tensor = call.prob_tensor
    dprob_tensor = call.dprob_tensor
    b_tensor = call.b_tensor
    sfb_tensor = call.sfb_tensor
    generate_dbias = call.generate_dbias
    b_ptrs = call.b_ptrs
    sfb_ptrs = call.sfb_ptrs
    n = call.n
    b_major = call.b_major
    norm_const_tensor = call.norm_const_tensor
    cd_major = call.cd_major
    # The block-scaled path is torch-native (torch-only allocations and kernels);
    # the normalized call carries canonical (cutlass) dtypes, so map them back.
    acc_dtype = framework_dtype(call.acc_dtype, "torch")
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
    epilogue_op = call.epilogue_op
    use_dynamic_sched = call.use_dynamic_sched
    use_single_group_runtime_offsets = call.use_single_group_runtime_offsets
    current_stream = call.current_stream

    # Resolve linear_offset default: None means "use the activation-derived
    # default" (1.0 for dgeglu, 0.0 for dswiglu).
    if linear_offset is None:
        linear_offset = 1.0 if act_func == "dgeglu" else 0.0
    device_type = get_device_type()
    _reject_unsupported_rubin_glu_tune_params(
        device_type == "rubin",
        geglu_alpha,
        glu_clamp_max,
        glu_clamp_min,
    )
    activation_cache_signature = None
    if act_func == "dgeglu":
        activation_cache_signature = (
            float(linear_offset),
            float(geglu_alpha),
            float(glu_clamp_max),
            float(glu_clamp_min),
        )
    elif act_func == "dsituglu":
        activation_cache_signature = (float(situ_beta1), float(situ_beta2))

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
        n_weight, _, l = b_tensor.shape
    else:
        weight_mode = MoEWeightMode.DISCRETE
        _require_pointer_tensor(b_ptrs, "b_ptrs")
        num_experts = b_ptrs.shape[0]
        _require_pointer_tensor(sfb_ptrs, "sfb_ptrs", num_experts)
        if n is None or b_dtype is None:
            raise ValueError("n and b_dtype are required for discrete mode")
        n_weight = n
        k_logical = k_physical * 2 if b_dtype in (torch.float4_e2m1fn_x2, torch.uint8) else k_physical
        b_shape = (n_weight, k_logical)
        l = num_experts

    n_out = 2 * n_weight

    _logger.debug("grouped_gemm_dglu_wrapper_sm100: Creating output tensors")

    if cd_major == "n":
        d_row_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_dtype, device=a_tensor.device)
        d_col_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_dtype, device=a_tensor.device)
    else:
        raise ValueError(f"cd_major must be 'n', got {cd_major}")

    sfd_row_tensor = None
    sfd_col_tensor = None
    amax_tensor = None
    dbias_tensor = None

    if a_tensor.dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ] and sfa_tensor.dtype in [torch.float8_e8m0fnu, torch.float8_e4m3fn]:
        _logger.debug("grouped_gemm_dglu_wrapper_sm100: Detected fp8 config, constructing sfd tensors")

        sf_dtype = sfa_tensor.dtype
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        sf_k_row = ceil_div(n_out, sf_vec_size)
        mma_shape_row = (1, ceil_div(valid_m, 128), ceil_div(sf_k_row, 4), 32, 4, 4)
        sfd_row_tensor = torch.empty(mma_shape_row, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

        sf_k_col = ceil_div(valid_m, sf_vec_size)
        mma_shape_col = (1, ceil_div(n_out, 128), ceil_div(sf_k_col, 4), 32, 4, 4)
        sfd_col_tensor = torch.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

    if d_dtype in [torch.bfloat16, torch.float16]:
        _logger.debug("grouped_gemm_dglu_wrapper_sm100: Constructing amax_tensor")
        amax_tensor = torch.full((l, 2, 1), float("-inf"), dtype=torch.float32, device=a_tensor.device)
    if generate_dbias:
        dbias_tensor = torch.zeros((l, n_out, 1), dtype=torch.bfloat16, device=a_tensor.device)

    if valid_m == 0:
        _logger.debug("grouped_gemm_dglu_wrapper_sm100: valid_m is zero, skipping kernel execution")
        return TupleDict(
            d_row_tensor=d_row_tensor,
            d_col_tensor=d_col_tensor,
            dprob_tensor=dprob_tensor,
            dbias_tensor=dbias_tensor,
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

    if is_dense:
        cache_key = (
            device_type,
            weight_mode,
            act_func,
            activation_cache_signature,
            epilogue_op,
            use_full_dynamic,
            a_tensor.shape[1:] if not use_full_dynamic else None,
            b_tensor.shape[2] if use_full_dynamic else tuple(b_tensor.shape),
            c_tensor.shape[1:] if not use_full_dynamic else None,
            a_tensor.dtype,
            b_tensor.dtype,
            c_tensor.dtype,
            stride_order(a_tensor),
            stride_order(b_tensor),
            stride_order(c_tensor),
            *(
                dynamic_tensor_signature(sfa_tensor)
                if use_full_dynamic
                else dynamic_m_tensor_signature(sfa_tensor, (sfa_tensor.shape[4], 1) if sfa_tensor is not None else None, dynamic_stride_dims=(5,))
            ),
            *tensor_signature(alpha_tensor),
            *tensor_signature(beta_tensor),
            *(dynamic_m_tensor_signature(prob_tensor, (1, 1)) if not use_full_dynamic else dynamic_tensor_signature(prob_tensor)),
            *(dynamic_m_tensor_signature(dprob_tensor, (1, 1)) if not use_full_dynamic else dynamic_tensor_signature(dprob_tensor)),
            *(dynamic_tensor_signature(dbias_tensor) if use_full_dynamic else tensor_signature(dbias_tensor)),
            *(dynamic_tensor_signature(sfb_tensor) if use_full_dynamic else tensor_signature(sfb_tensor)),
            norm_const_tensor.shape if norm_const_tensor is not None else None,
            norm_const_tensor.stride() if norm_const_tensor is not None else None,
            norm_const_tensor.dtype if norm_const_tensor is not None else None,
            tuple(padded_offsets.shape),
            tuple(padded_offsets.stride()),
            padded_offsets.dtype,
            acc_dtype,
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
        )
    else:
        cache_key = (
            device_type,
            weight_mode,
            act_func,
            activation_cache_signature,
            epilogue_op,
            *dynamic_m_tensor_signature(a_tensor, tuple(a_tensor.shape[1:]), dynamic_stride_dims=(2,)),
            b_shape,
            b_dtype,
            *dynamic_m_tensor_signature(c_tensor, tuple(c_tensor.shape[1:]), dynamic_stride_dims=(2,)),
            *dynamic_m_tensor_signature(sfa_tensor, (sfa_tensor.shape[4], 1) if sfa_tensor is not None else None, dynamic_stride_dims=(5,)),
            *tensor_signature(alpha_tensor),
            *tensor_signature(beta_tensor),
            *dynamic_m_tensor_signature(prob_tensor, (1, 1)),
            *dynamic_m_tensor_signature(dprob_tensor, (1, 1)),
            *tensor_signature(dbias_tensor),
            *tensor_signature(norm_const_tensor),
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

    # ---- Cache lookup or create + compile ----
    if cache_key in _cache_of_GroupedGemmDgluSm100Objects:
        _logger.debug("grouped_gemm_dglu_wrapper_sm100: Using cached object")
        api = _cache_of_GroupedGemmDgluSm100Objects[cache_key]
    else:
        _logger.debug("grouped_gemm_dglu_wrapper_sm100: Creating new object")
        if is_dense:
            api = GroupedGemmDgluSm100(
                sample_a=a_tensor,
                sample_c=c_tensor,
                sample_d_row=d_row_tensor,
                sample_d_col=d_col_tensor,
                sample_sfa=sfa_tensor,
                sample_padded_offsets=padded_offsets,
                sample_alpha=alpha_tensor,
                sample_beta=beta_tensor,
                sample_prob=prob_tensor,
                sample_dprob=dprob_tensor,
                sample_dbias=dbias_tensor,
                sample_b=b_tensor,
                sample_sfb=sfb_tensor,
                sample_sfd_row=sfd_row_tensor,
                sample_sfd_col=sfd_col_tensor,
                sample_amax=amax_tensor,
                sample_norm_const=norm_const_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                sf_fp8_dtype_override=sf_fp8_dtype_override,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                discrete_col_sfd=discrete_col_sfd,
                act_func=act_func,
                epilogue_op=epilogue_op,
                use_dynamic_sched=use_dynamic_sched,
                use_single_group_runtime_offsets=use_single_group_runtime_offsets,
                linear_offset=linear_offset,
                geglu_alpha=geglu_alpha,
                glu_clamp_max=glu_clamp_max,
                glu_clamp_min=glu_clamp_min,
                situ_beta1=situ_beta1,
                situ_beta2=situ_beta2,
            )
        else:
            api = GroupedGemmDgluSm100(
                sample_a=a_tensor,
                sample_c=c_tensor,
                sample_d_row=d_row_tensor,
                sample_d_col=d_col_tensor,
                sample_sfa=sfa_tensor,
                sample_padded_offsets=padded_offsets,
                sample_alpha=alpha_tensor,
                sample_beta=beta_tensor,
                sample_prob=prob_tensor,
                sample_dprob=dprob_tensor,
                sample_dbias=dbias_tensor,
                num_experts=num_experts,
                b_shape=b_shape,
                b_dtype=b_dtype,
                sample_sfd_row=sfd_row_tensor,
                sample_sfd_col=sfd_col_tensor,
                sample_amax=amax_tensor,
                sample_norm_const=norm_const_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                sf_fp8_dtype_override=sf_fp8_dtype_override,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                discrete_col_sfd=discrete_col_sfd,
                act_func=act_func,
                b_major=b_major,
                epilogue_op=epilogue_op,
                use_dynamic_sched=use_dynamic_sched,
                use_single_group_runtime_offsets=use_single_group_runtime_offsets,
                linear_offset=linear_offset,
                geglu_alpha=geglu_alpha,
                glu_clamp_max=glu_clamp_max,
                glu_clamp_min=glu_clamp_min,
                situ_beta1=situ_beta1,
                situ_beta2=situ_beta2,
            )

        if not api.check_support():
            raise RuntimeError("Unsupported configuration")
        api.compile()
        _cache_of_GroupedGemmDgluSm100Objects[cache_key] = api

    # ---- Execute ----
    if is_dense:
        api.execute(
            a_tensor=a_tensor,
            c_tensor=c_tensor,
            d_row_tensor=d_row_tensor,
            d_col_tensor=d_col_tensor,
            sfa_tensor=sfa_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            beta_tensor=beta_tensor,
            prob_tensor=prob_tensor,
            dprob_tensor=dprob_tensor,
            dbias_tensor=dbias_tensor,
            b_tensor=b_tensor,
            sfb_tensor=sfb_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            current_stream=current_stream,
        )
    else:
        api.execute(
            a_tensor=a_tensor,
            c_tensor=c_tensor,
            d_row_tensor=d_row_tensor,
            d_col_tensor=d_col_tensor,
            sfa_tensor=sfa_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            beta_tensor=beta_tensor,
            prob_tensor=prob_tensor,
            dprob_tensor=dprob_tensor,
            dbias_tensor=dbias_tensor,
            b_ptrs=b_ptrs,
            sfb_ptrs=sfb_ptrs,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            current_stream=current_stream,
        )

    return TupleDict(
        d_row_tensor=d_row_tensor,
        d_col_tensor=d_col_tensor,
        dprob_tensor=dprob_tensor,
        dbias_tensor=dbias_tensor,
        amax_tensor=amax_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
    )


def _normalize_dglu_call(
    call: DgluCall,
) -> tuple[DgluCall, GroupedGemmBackend]:
    from cudnn.gemm.cutedsl.grouped.unfused._bf16_api import _validate_pointer_tensor

    call = replace(
        call,
        acc_dtype=_convert_to_cutlass_data_type(call.acc_dtype) if call.acc_dtype is not None else cutlass.Float32,
        d_dtype=_convert_to_cutlass_data_type(call.d_dtype) if call.d_dtype is not None else cutlass.BFloat16,
        b_dtype=_convert_to_cutlass_data_type(call.b_dtype) if call.b_dtype is not None else None,
    )

    if call.act_func not in ("dswiglu", "dgeglu", "dsituglu"):
        raise ValueError(f"act_func must be 'dswiglu', 'dgeglu', or 'dsituglu', got {call.act_func}")
    if call.act_func == "dsituglu":
        if not math.isfinite(call.situ_beta1) or call.situ_beta1 <= 0.0:
            raise ValueError(f"situ_beta1 must be finite and positive, got {call.situ_beta1}")
        if not math.isfinite(call.situ_beta2) or call.situ_beta2 <= 0.0:
            raise ValueError(f"situ_beta2 must be finite and positive, got {call.situ_beta2}")
        if get_device_type() == "rubin":
            raise NotImplementedError("Rubin grouped GEMM dGLU does not support dsituglu")

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
            raise ValueError(f"b_tensor must have shape (n, k, experts), got {b_full_shape}")
        n_weight, b_k, num_experts = b_full_shape
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
        n_weight = call.n
        defining_b_dtype = call.b_dtype
        b_shape = (n_weight, k)
        weight_mode = MoEWeightMode.DISCRETE

    backend = select_grouped_gemm_backend(
        operation="grouped_gemm_dglu_sm100",
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
            ("geglu_alpha", call.geglu_alpha if call.geglu_alpha != 1.702 else None),
            (
                "glu_clamp_max",
                call.glu_clamp_max if call.glu_clamp_max != 7.0 else None,
            ),
            (
                "glu_clamp_min",
                call.glu_clamp_min if call.glu_clamp_min != -7.0 else None,
            ),
            (
                "epilogue_op",
                call.epilogue_op if call.epilogue_op not in (None, "none", "identity") else None,
            ),
        ),
        block_scaled_dtype_pairs=_block_scaled_dtype_pairs(),
    )
    linear_offset = call.linear_offset
    if linear_offset is None:
        linear_offset = 1.0 if call.act_func == "dgeglu" else 0.0
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
    if call.prob_tensor is None or call.dprob_tensor is None:
        raise ValueError("BF16 grouped GEMM dGLU requires prob_tensor and dprob_tensor")

    if call.cd_major != "n":
        raise ValueError(f"cd_major must be 'n', got {call.cd_major}")
    if call.act_func not in ("dswiglu", "dgeglu"):
        raise ValueError(f"BF16 act_func must be 'dswiglu' or 'dgeglu'; dsituglu is block-scaled only, got {call.act_func}")
    if call.d_dtype not in (cutlass.BFloat16, cutlass.Float16, cutlass.Float32):
        raise ValueError(f"d_dtype must be BF16, FP16, or FP32, got {call.d_dtype}")
    if call.m_aligned != 256:
        raise ValueError(f"m_aligned must be 256, got {call.m_aligned}")
    if valid_m % 256 != 0:
        raise ValueError(f"a_tensor M dimension must be 256-aligned, got {valid_m}")
    if n_weight <= 0 or n_weight % 32 != 0:
        raise ValueError(f"N must be positive and divisible by 32, got {n_weight}")
    two_n = 2 * n_weight
    if get_shape(call.c_tensor) != (valid_m, two_n, 1):
        raise ValueError(f"c_tensor must have shape {(valid_m, two_n, 1)}, got {get_shape(call.c_tensor)}")
    if get_shape(call.prob_tensor) != (valid_m, 1, 1):
        raise ValueError(f"prob_tensor must have shape {(valid_m, 1, 1)}, got {get_shape(call.prob_tensor)}")
    if get_shape(call.dprob_tensor) != (valid_m, 1, 1):
        raise ValueError(f"dprob_tensor must have shape {(valid_m, 1, 1)}, got {get_shape(call.dprob_tensor)}")
    if _convert_to_cutlass_data_type(call.dprob_tensor.dtype) is not cutlass.Float32:
        raise ValueError(f"dprob_tensor must have dtype torch.float32, got {call.dprob_tensor.dtype}")
    offsets_shape = get_shape(call.padded_offsets)
    if is_discrete and len(offsets_shape) == 1 and num_experts != offsets_shape[0]:
        raise ValueError(f"b_ptrs length mismatch: expected {offsets_shape[0]}, " f"got {num_experts}")
    if offsets_shape != (num_experts,):
        raise ValueError(f"padded_offsets length mismatch: expected {num_experts}, got {offsets_shape}")
    if get_shape(call.alpha_tensor) != (num_experts,):
        raise ValueError(f"alpha_tensor must have shape {(num_experts,)}, got {get_shape(call.alpha_tensor)}")
    if get_shape(call.beta_tensor) != (num_experts,):
        raise ValueError(f"beta_tensor must have shape {(num_experts,)}, got {get_shape(call.beta_tensor)}")
    if is_discrete:
        if get_device(call.b_ptrs) != get_device(call.a_tensor):
            raise ValueError(f"b_ptrs must be on the same device as a_tensor ({get_device(call.a_tensor)}), " f"got {get_device(call.b_ptrs)}")
    if not cuda_is_available():
        raise RuntimeError("CUDA is not available")
    major, minor = get_compute_capability()
    capability = major * 10 + minor
    if capability < 100:
        raise RuntimeError(f"GroupedGemmDgluSm100 requires SM100+, found SM{capability}")
    return normalized, backend


def _dglu_stride_order(tensor: torch.Tensor) -> Tuple[int, ...]:
    strides = get_strides(tensor)
    shape = get_shape(tensor)
    return tuple(
        index
        for index, _ in sorted(
            enumerate(strides),
            key=lambda item: (item[1], shape[item[0]]),
        )
    )


def _dglu_tensor_signature(tensor: Optional[torch.Tensor], *, dynamic_m: bool = False) -> tuple:
    if tensor is None:
        return (None, None, None, None)
    device = get_device(tensor)
    shape = (None, *get_shape(tensor)[1:]) if dynamic_m else get_shape(tensor)
    return (
        shape,
        _dglu_stride_order(tensor),
        _convert_to_cutlass_data_type(tensor.dtype),
        (device.type, device.index),
    )


def _dglu_allocate_outputs(framework, valid_m, two_n, d_dtype, generate_dbias, num_experts, a_tensor, current_stream):
    """Outputs for the BF16 dGLU path. dbias is zeroed because the kernel accumulates
    into it with atomic adds; d_row is written in full and only needs allocating."""
    if framework == "torch":
        import torch

        # _torch_stream_context is torch-only; other frameworks never enter it.
        with _torch_stream_context(current_stream, a_tensor.device):
            d_row_tensor = torch.empty_strided(
                (valid_m, two_n, 1),
                (two_n, 1, valid_m * two_n),
                dtype=framework_dtype(d_dtype, "torch"),
                device=a_tensor.device,
            )
            dbias_tensor = torch.zeros((num_experts, two_n, 1), dtype=torch.bfloat16, device=a_tensor.device) if generate_dbias else None
        return d_row_tensor, dbias_tensor

    import jax
    import jax.numpy as jnp

    # n-major C-contiguous outputs; the extent-1 batch dim's stride is unobservable.
    # The kernel writes into these buffers on the launch stream; materialize them first.
    d_row_tensor = jax.block_until_ready(jnp.empty((valid_m, two_n, 1), dtype=framework_dtype(d_dtype, "jax"), device=a_tensor.device))
    dbias_tensor = (
        jax.block_until_ready(jnp.zeros((num_experts, two_n, 1), dtype=framework_dtype(cutlass.BFloat16, "jax"), device=a_tensor.device))
        if generate_dbias
        else None
    )
    return d_row_tensor, dbias_tensor


def _dglu_operand_meta(tensor):
    """Everything the wrapper's derivation reads off an operand, and nothing else.

    Deliberately not the object's identity: CPython recycles a freed tensor's address,
    so an id-keyed memo answers for tensors it never saw. Data pointers are excluded --
    they vary per call and nothing derived here depends on them; their alignment is
    re-checked inside execute().
    """
    if tensor is None or not hasattr(tensor, "shape"):
        return tensor
    device = get_device(tensor)
    return (get_shape(tensor), get_strides(tensor), tensor.dtype, device.type, device.index)


# Operand-metadata key -> the BF16 path's derived result. One entry per distinct
# (operand metadata, config), the same growth as _cache_of_GroupedGemmDgluSm100Objects.
_dglu_wrapper_memo: dict = {}


def _grouped_gemm_dglu_bf16_call(call: DgluCall, memo_key: Optional[tuple] = None) -> TupleDict:
    framework = detect_framework(call.a_tensor)
    valid_m = get_shape(call.a_tensor)[0]
    n_weight = get_shape(call.b_tensor)[0] if call.b_tensor is not None else call.n
    two_n = 2 * n_weight
    d_row_tensor, dbias_tensor = _dglu_allocate_outputs(
        framework, valid_m, two_n, call.d_dtype, call.generate_dbias, call.num_experts, call.a_tensor, call.current_stream
    )

    overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    workspace_bytes = (128 * call.num_experts if call.weight_mode == MoEWeightMode.DISCRETE else 0) + (4 if call.use_dynamic_sched else 0)
    cache_key = backend_cache_key(
        GroupedGemmBackend.BF16,
        call.weight_mode,
        call.act_func,
        _dglu_tensor_signature(call.a_tensor, dynamic_m=True),
        _dglu_tensor_signature(call.b_tensor),
        call.b_shape,
        call.b_dtype,
        _dglu_tensor_signature(call.c_tensor, dynamic_m=True),
        _dglu_tensor_signature(d_row_tensor, dynamic_m=True),
        _dglu_tensor_signature(call.padded_offsets),
        _dglu_tensor_signature(call.alpha_tensor),
        _dglu_tensor_signature(call.beta_tensor),
        _dglu_tensor_signature(call.prob_tensor, dynamic_m=True),
        _dglu_tensor_signature(call.dprob_tensor, dynamic_m=True),
        _dglu_tensor_signature(dbias_tensor),
        (_dglu_tensor_signature(call.b_ptrs) if call.b_ptrs is not None else None),
        call.acc_dtype,
        call.d_dtype,
        call.mma_tiler_mn,
        call.cluster_shape_mn,
        call.vector_f32,
        call.m_aligned,
        call.b_major,
        call.use_dynamic_sched,
        workspace_bytes,
        ((get_device(call.a_tensor).type, get_device(call.a_tensor).index)),
        overlap_margin,
    )

    if cache_key in _cache_of_GroupedGemmDgluSm100Objects:
        api = _cache_of_GroupedGemmDgluSm100Objects[cache_key]
    else:
        api = GroupedGemmDgluSm100(
            sample_a=call.a_tensor,
            sample_c=call.c_tensor,
            sample_d_row=d_row_tensor,
            sample_d_col=None,
            sample_sfa=None,
            sample_padded_offsets=call.padded_offsets,
            sample_alpha=call.alpha_tensor,
            sample_beta=call.beta_tensor,
            sample_prob=call.prob_tensor,
            sample_dprob=call.dprob_tensor,
            sample_b=call.b_tensor,
            sample_sfb=None,
            sample_dbias=dbias_tensor,
            num_experts=(call.num_experts if call.weight_mode == MoEWeightMode.DISCRETE else None),
            b_shape=call.b_shape,
            b_dtype=call.b_dtype,
            sample_sfd_row=None,
            sample_sfd_col=None,
            sample_amax=None,
            sample_norm_const=None,
            acc_dtype=call.acc_dtype,
            mma_tiler_mn=call.mma_tiler_mn,
            cluster_shape_mn=call.cluster_shape_mn,
            sf_vec_size=16,
            vector_f32=call.vector_f32,
            m_aligned=call.m_aligned,
            discrete_col_sfd=False,
            act_func=call.act_func,
            b_major=call.b_major,
            epilogue_op=None,
            use_dynamic_sched=call.use_dynamic_sched,
            linear_offset=call.linear_offset,
            geglu_alpha=1.702,
            glu_clamp_max=7.0,
            glu_clamp_min=-7.0,
        )
        if not api.check_support():
            raise RuntimeError("Unsupported BF16 configuration")
        api.compile()
        _cache_of_GroupedGemmDgluSm100Objects[cache_key] = api

    if memo_key is not None:
        _dglu_wrapper_memo[memo_key] = (api, framework, valid_m, two_n, call.d_dtype, call.generate_dbias, call.num_experts)

    api._implementation.execute(
        a_tensor=call.a_tensor,
        c_tensor=call.c_tensor,
        d_row_tensor=d_row_tensor,
        padded_offsets=call.padded_offsets,
        alpha_tensor=call.alpha_tensor,
        beta_tensor=call.beta_tensor,
        prob_tensor=call.prob_tensor,
        dprob_tensor=call.dprob_tensor,
        b_tensor=call.b_tensor,
        b_ptrs=call.b_ptrs,
        dbias_tensor=dbias_tensor,
        linear_offset=call.linear_offset,
        current_stream=call.current_stream,
    )
    return TupleDict(
        d_row_tensor=d_row_tensor,
        d_col_tensor=None,
        dprob_tensor=call.dprob_tensor,
        dbias_tensor=dbias_tensor,
        amax_tensor=None,
        sfd_row_tensor=None,
        sfd_col_tensor=None,
    )


def grouped_gemm_dglu_wrapper_sm100(
    a_tensor: torch.Tensor,
    c_tensor: torch.Tensor,
    sfa_tensor: Optional[torch.Tensor],
    padded_offsets: torch.Tensor,
    alpha_tensor: torch.Tensor,
    beta_tensor: torch.Tensor,
    prob_tensor: Optional[torch.Tensor],
    dprob_tensor: Optional[torch.Tensor],
    b_tensor: Optional[torch.Tensor] = None,
    sfb_tensor: Optional[torch.Tensor] = None,
    generate_dbias: bool = False,
    b_ptrs: Optional[torch.Tensor] = None,
    sfb_ptrs: Optional[torch.Tensor] = None,
    n: Optional[int] = None,
    b_dtype: Optional[torch.dtype] = None,
    b_major: str = "k",
    norm_const_tensor: Optional[torch.Tensor] = None,
    acc_dtype: Optional[torch.dtype] = None,
    d_dtype: Optional[torch.dtype] = None,
    cd_major: str = "n",
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 16,
    sf_fp8_dtype_override: Optional[Literal["e5m3"]] = None,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
    act_func: str = "dswiglu",
    linear_offset: Optional[float] = None,
    geglu_alpha: float = 1.702,
    glu_clamp_max: float = 7.0,
    glu_clamp_min: float = -7.0,
    situ_beta1: float = 4.0,
    situ_beta2: float = 25.0,
    epilogue_op: Optional[str] = None,
    use_dynamic_sched: bool = False,
    use_single_group_runtime_offsets: bool = False,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Dispatch grouped GEMM dGLU once from an immutable normalized call."""
    # Hot-loop memo; same shape as the unfused and GLU wrappers. Everything from here to
    # execute() is derivation -- dtype resolution, DgluCall construction, normalization and
    # the op cache-key rebuild -- and a pure function of the operands' metadata plus the
    # scalar config, both of which are in the key. A hit skips the derivation, never a
    # check: execute() still validates every operand, including the data pointers the key
    # omits. linear_offset is excluded on purpose (not in the op cache key either) and
    # passed through per call.
    memo_key = (
        type(a_tensor),
        _dglu_operand_meta(a_tensor),
        _dglu_operand_meta(c_tensor),
        _dglu_operand_meta(sfa_tensor),
        _dglu_operand_meta(padded_offsets),
        _dglu_operand_meta(alpha_tensor),
        _dglu_operand_meta(beta_tensor),
        _dglu_operand_meta(prob_tensor),
        _dglu_operand_meta(dprob_tensor),
        _dglu_operand_meta(b_tensor),
        _dglu_operand_meta(sfb_tensor),
        _dglu_operand_meta(b_ptrs),
        _dglu_operand_meta(sfb_ptrs),
        _dglu_operand_meta(norm_const_tensor),
        use_single_group_runtime_offsets,
        generate_dbias,
        n,
        b_dtype,
        b_major,
        acc_dtype,
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
        epilogue_op,
        use_dynamic_sched,
        os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"),
    )
    memo = _dglu_wrapper_memo.get(memo_key)
    if memo is not None:
        api, memo_framework, valid_m, two_n, memo_d_dtype, memo_generate_dbias, memo_experts = memo
        # The full path resolves this during normalization; the memo bypasses that, so
        # apply the same default here rather than handing None to the kernel.
        resolved_linear_offset = linear_offset if linear_offset is not None else (1.0 if act_func == "dgeglu" else 0.0)
        d_row_tensor, dbias_tensor = _dglu_allocate_outputs(
            memo_framework, valid_m, two_n, memo_d_dtype, memo_generate_dbias, memo_experts, a_tensor, current_stream
        )
        api._implementation.execute(
            a_tensor=a_tensor,
            c_tensor=c_tensor,
            d_row_tensor=d_row_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            beta_tensor=beta_tensor,
            prob_tensor=prob_tensor,
            dprob_tensor=dprob_tensor,
            b_tensor=b_tensor,
            b_ptrs=b_ptrs,
            dbias_tensor=dbias_tensor,
            linear_offset=resolved_linear_offset,
            current_stream=current_stream,
        )
        return TupleDict(
            d_row_tensor=d_row_tensor,
            d_col_tensor=None,
            dprob_tensor=dprob_tensor,
            dbias_tensor=dbias_tensor,
            amax_tensor=None,
            sfd_row_tensor=None,
            sfd_col_tensor=None,
        )

    framework = detect_framework(a_tensor)
    if framework not in ("torch", "jax"):
        raise ValueError(f"Unsupported tensor framework '{framework}' for grouped_gemm_dglu_wrapper_sm100; pass torch tensors or JAX arrays")
    if framework == "jax" and b_tensor is not None:
        raise ValueError(_JAX_DENSE_B_ERROR)
    acc_dtype = _convert_to_cutlass_data_type(acc_dtype) if acc_dtype is not None else cutlass.Float32
    d_dtype = _convert_to_cutlass_data_type(d_dtype) if d_dtype is not None else cutlass.BFloat16
    b_dtype = _convert_to_cutlass_data_type(b_dtype) if b_dtype is not None else None
    call = DgluCall(
        a_tensor=a_tensor,
        c_tensor=c_tensor,
        sfa_tensor=sfa_tensor,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha_tensor,
        beta_tensor=beta_tensor,
        prob_tensor=prob_tensor,
        dprob_tensor=dprob_tensor,
        b_tensor=b_tensor,
        sfb_tensor=sfb_tensor,
        generate_dbias=generate_dbias,
        b_ptrs=b_ptrs,
        sfb_ptrs=sfb_ptrs,
        n=n,
        b_dtype=b_dtype,
        b_major=b_major,
        norm_const_tensor=norm_const_tensor,
        acc_dtype=acc_dtype,
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
        epilogue_op=epilogue_op,
        use_dynamic_sched=use_dynamic_sched,
        use_single_group_runtime_offsets=use_single_group_runtime_offsets,
        current_stream=current_stream,
    )
    normalized, backend = _normalize_dglu_call(call)
    if backend is GroupedGemmBackend.BF16:
        return _grouped_gemm_dglu_bf16_call(normalized, memo_key)
    if framework == "jax":
        raise ValueError(_JAX_BLOCK_SCALED_ERROR)
    return _grouped_gemm_dglu_block_scaled_call(normalized)
