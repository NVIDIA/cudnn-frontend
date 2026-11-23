# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unified API for Grouped GEMM SReLU Kernel (SM100+)

This module provides a single API class and wrapper for grouped GEMM with a fused
(prob-weighted) squared-ReLU epilogue in MoE (Mixture of Experts) workloads, with
dense (contiguous) or discrete (per-expert pointer) weights. The backend follows the
operand dtypes:

BF16
    BF16 A/B, FP32 accumulation; C is the saved pre-activation and D the weighted
    squared-ReLU, both BF16 (``_bf16_api.py``).

Block-scaled
    FP8/FP4 A/B with block scale factors and optional SReLU output quantization
    (``_blockscaled_api.py``).
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import cutlass
from cuda.bindings import driver as cuda

from cudnn.api_base import APIBase, TupleDict, ceil_div
from cudnn._torch_stream import device_context
from cudnn.datatypes import _convert_to_cutlass_data_type, _convert_to_cutlass_data_type_or_none
from cudnn.tensor_adapter import (
    canonicalize_unit_dim_strides,
    detect_framework,
    framework_dtype,
    get_device,
    get_shape,
    get_strides,
)

from ..backend_utils import GroupedGemmBackend, _torch_stream_context, backend_cache_key, select_grouped_gemm_backend, wrapper_workspace
from ..moe_utils import MoEWeightMode

from ._bf16_options import bf16_kernel_requirement_error, check_tanh_mode, resolve_bf16_srelu_options
from ._blockscaled_api import (
    GroupedGemmSreluBlockScaledAPI,
    _JAX_SF_LAYOUT_ERROR,
)

_JAX_BF16_ERROR = (
    "The BF16 grouped GEMM SReLU backend supports torch tensors only (it has no JAX entry point); "
    "pass torch tensors. JAX arrays are served only by the block-scaled dense path of "
    "grouped_gemm_srelu_wrapper_sm100 (MXFP8 or packed FP4 b_tensor with sfa_tensor/sfb_tensor)"
)


def _block_scaled_dtype_pairs():
    # Canonical (cutlass) dtype vocabulary; select_grouped_gemm_backend canonicalizes the caller's
    # dtypes. These same-dtype pairs are what the block-scaled srelu and dsrelu classes accept.
    return {
        (dtype, dtype)
        for dtype in (
            cutlass.Float4E2M1FN,
            cutlass.Uint8,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        )
    }


def _check_block_scaled_activation_options(operation: str, tanh_clamp_scale: Optional[float], tanh_mode: str, act_input: str) -> None:
    """The block-scaled kernel evaluates tanh with MUFU tanh.approx and takes the FP32 accumulator.

    ``tanh_mode`` must be ``"approx"`` or ``"accurate"`` as on the BF16 backend. Only the clamped
    activation evaluates a tanh, so ``"accurate"`` is rejected with ``tanh_clamp_scale`` set and
    ignored without it.
    """
    check_tanh_mode(tanh_mode)
    if tanh_clamp_scale is not None and tanh_mode != "approx":
        raise ValueError(
            f"{operation}: tanh_mode={tanh_mode!r} with tanh_clamp_scale is supported only by the BF16 kernel; the block-scaled kernel "
            "evaluates tanh with MUFU tanh.approx (pass tanh_mode='approx')"
        )
    if act_input != "acc":
        raise ValueError(
            f"{operation}: act_input={act_input!r} is supported only by the BF16 kernel; the block-scaled kernel takes the activation "
            "from the FP32 accumulator"
        )


class GroupedGemmSreluSm100(APIBase):
    """Unified API for grouped GEMM SReLU operation on SM100+ GPUs.

    Computes ``C = A @ B_e^T`` per expert and ``D = prob * srelu(C)`` with
    ``srelu(x) = relu(x) ** 2`` or, with ``tanh_clamp_scale=s``, ``(s * tanh(relu(x) / s)) ** 2``.
    The backend follows the operand dtypes and is selected in ``check_support()``: BF16
    ``sample_a``/``sample_b`` (or ``b_dtype``) select the BF16 kernel, FP8/FP4 operands the
    block-scaled kernel (``D = srelu(alpha * A @ B)`` with block scale factors and optional output
    quantization).

    Weight mode is auto-detected from the constructor arguments:

    - Dense: provide ``sample_b`` (and ``sample_sfb`` for block-scaled).
    - Discrete: provide ``num_experts``, ``b_shape``, and ``b_dtype``.

    BF16 backend: ``sample_c`` is the BF16 pre-activation ``bf16(A @ B^T)`` and ``sample_d`` the
    BF16 weighted squared-ReLU; ``sample_prob`` is required, and the scale-factor, alpha, bias,
    amax and norm-const arguments must be omitted. ``tanh_mode`` (default ``"approx"``) and
    ``act_input`` select how the activation is evaluated. The BF16 ``execute()`` takes
    caller-owned scratch of ``scratch_workspace_bytes()`` bytes as ``workspace=``; see the BF16
    section of docs/operations/gemm_fusions/grouped_gemm_srelu.md.
    """

    def __init__(
        self,
        sample_a: torch.Tensor,
        # Dense mode (contiguous) -- provide these:
        sample_b: Optional[torch.Tensor] = None,
        sample_c: Optional[torch.Tensor] = None,
        sample_d: Optional[torch.Tensor] = None,
        sample_sfa: Optional[torch.Tensor] = None,
        sample_sfb: Optional[torch.Tensor] = None,
        sample_padded_offsets: Optional[torch.Tensor] = None,
        sample_alpha: Optional[torch.Tensor] = None,
        sample_d_col: Optional[torch.Tensor] = None,
        sample_bias: Optional[torch.Tensor] = None,
        # Discrete mode -- provide these instead:
        num_experts: Optional[int] = None,
        b_shape: Optional[Tuple[int, ...]] = None,
        b_dtype: Optional[torch.dtype] = None,
        # Optional SReLU output quantization output arguments
        sample_sfd_row: Optional[torch.Tensor] = None,
        sample_sfd_col: Optional[torch.Tensor] = None,
        sample_amax: Optional[torch.Tensor] = None,
        sample_norm_const: Optional[torch.Tensor] = None,
        sample_prob: Optional[torch.Tensor] = None,
        # Configuration
        acc_dtype: Optional[torch.dtype] = None,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        vector_f32: bool = False,
        m_aligned: int = 256,
        discrete_col_sfd: bool = False,
        b_major: str = "k",
        use_dynamic_sched: bool = False,
        tanh_clamp_scale: Optional[float] = None,
        tanh_mode: str = "approx",
        act_input: str = "acc",
    ):
        """Initialize the GroupedGemmSreluSm100 API.

        The arguments up to ``tanh_clamp_scale`` keep their block-scaled meaning (see
        ``GroupedGemmSreluBlockScaledAPI``); the BF16 backend uses ``sample_a``, ``sample_b`` or
        ``(num_experts, b_shape, b_dtype)``, ``sample_c``, ``sample_d``, ``sample_padded_offsets``,
        ``sample_prob`` and the configuration arguments. The arguments are stored here; the backend
        is selected and built, and the configuration validated, in ``check_support()``.

        :param tanh_mode: ``"approx"`` (default) or ``"accurate"``; any other value, ``None``
            included, raises ``ValueError`` in ``check_support()``. With ``tanh_clamp_scale`` set it
            selects the tanh: ``"approx"`` is MUFU ``tanh.approx.f32`` (~2^-11 relative error,
            capped at 1) on both backends; ``"accurate"`` is a Cephes-style FP32 tanh (~2e-7
            relative error) on the BF16 backend and raises ``ValueError`` on the block-scaled one,
            which has no accurate tanh. Without a clamp scale no tanh is evaluated and both backends
            ignore the mode. NaN pre-activations propagate to ``D`` on the BF16 backend in every
            mode; the block-scaled kernel maps them to +0 (clamped: the ``mul.rn.sat.f32`` in
            ``moe_kernel_helpers.tanh_clamp_unit``; plain: its relu select).
        :param act_input: BF16 only. ``"acc"`` (default) takes the activation from the FP32
            accumulator, ``"saved"`` from ``bf16(acc)``, the C tensor the backward reloads.
        """
        super().__init__()
        self._pending_init_kwargs = dict(locals())
        self._pending_init_kwargs.pop("self")
        self._pending_init_kwargs.pop("__class__", None)
        framework = detect_framework(sample_a)
        if sample_a is not None and framework != "torch":
            if framework != "jax":
                raise ValueError(f"Unsupported tensor framework '{framework}' for GroupedGemmSreluSm100; pass torch tensors")
            # Neither backend takes JAX arrays through the class (JAX reaches the op only through the
            # wrapper's block-scaled dense path), so they are rejected here, before any backend work.
            if _convert_to_cutlass_data_type_or_none(sample_a.dtype) is cutlass.BFloat16:
                raise ValueError(_JAX_BF16_ERROR)
            raise ValueError(f"GroupedGemmSreluSm100 does not support JAX arrays: {_JAX_SF_LAYOUT_ERROR}")
        self._implementation = None

    def check_support(self) -> bool:
        if self._implementation is None:
            kwargs = self._pending_init_kwargs
            defining_b_dtype = kwargs["sample_b"].dtype if kwargs["sample_b"] is not None else kwargs["b_dtype"]
            backend = select_grouped_gemm_backend(
                operation="grouped_gemm_srelu_sm100",
                a_dtype=getattr(kwargs["sample_a"], "dtype", None),
                b_dtype=defining_b_dtype,
                scale_controls=(
                    ("sample_sfa", kwargs["sample_sfa"]),
                    ("sample_sfb", kwargs["sample_sfb"]),
                    ("sample_alpha", kwargs["sample_alpha"]),
                    ("sample_d_col", kwargs["sample_d_col"]),
                    ("sample_bias", kwargs["sample_bias"]),
                    ("sample_sfd_row", kwargs["sample_sfd_row"]),
                    ("sample_sfd_col", kwargs["sample_sfd_col"]),
                    ("sample_amax", kwargs["sample_amax"]),
                    ("sample_norm_const", kwargs["sample_norm_const"]),
                    ("sf_vec_size", kwargs["sf_vec_size"] if kwargs["sf_vec_size"] != 16 else None),
                    ("vector_f32", kwargs["vector_f32"] if kwargs["vector_f32"] else None),
                    ("discrete_col_sfd", kwargs["discrete_col_sfd"] if kwargs["discrete_col_sfd"] else None),
                ),
                block_scaled_dtype_pairs=_block_scaled_dtype_pairs(),
            )
            self.backend = backend
            if backend is GroupedGemmBackend.BF16:
                # Rule 7: the BF16 kernel modules import cutlass.experimental (cudnn.frost.tile_dsl) at module
                # scope, so the DSL is checked before they are imported, and only on this branch.
                requirement = bf16_kernel_requirement_error("GroupedGemmSreluSm100 (BF16)")
                if requirement:
                    raise ImportError(requirement)
                from ._bf16_api import GroupedGemmSreluBf16API

                self._implementation = GroupedGemmSreluBf16API(
                    sample_a=kwargs["sample_a"],
                    sample_c=kwargs["sample_c"],
                    sample_d=kwargs["sample_d"],
                    sample_padded_offsets=kwargs["sample_padded_offsets"],
                    sample_prob=kwargs["sample_prob"],
                    sample_b=kwargs["sample_b"],
                    num_experts=kwargs["num_experts"],
                    b_shape=kwargs["b_shape"],
                    b_dtype=kwargs["b_dtype"],
                    acc_dtype=kwargs["acc_dtype"],
                    mma_tiler_mn=kwargs["mma_tiler_mn"],
                    cluster_shape_mn=kwargs["cluster_shape_mn"],
                    m_aligned=kwargs["m_aligned"],
                    b_major=kwargs["b_major"],
                    use_dynamic_sched=kwargs["use_dynamic_sched"],
                    tanh_clamp_scale=kwargs["tanh_clamp_scale"],
                    tanh_mode=kwargs["tanh_mode"],
                    act_input=kwargs["act_input"],
                )
            else:
                _check_block_scaled_activation_options("grouped_gemm_srelu_sm100", kwargs["tanh_clamp_scale"], kwargs["tanh_mode"], kwargs["act_input"])
                self._implementation = GroupedGemmSreluBlockScaledAPI(
                    sample_a=kwargs["sample_a"],
                    sample_b=kwargs["sample_b"],
                    sample_c=kwargs["sample_c"],
                    sample_d=kwargs["sample_d"],
                    sample_sfa=kwargs["sample_sfa"],
                    sample_sfb=kwargs["sample_sfb"],
                    sample_padded_offsets=kwargs["sample_padded_offsets"],
                    sample_alpha=kwargs["sample_alpha"],
                    sample_d_col=kwargs["sample_d_col"],
                    sample_bias=kwargs["sample_bias"],
                    num_experts=kwargs["num_experts"],
                    b_shape=kwargs["b_shape"],
                    b_dtype=kwargs["b_dtype"],
                    sample_sfd_row=kwargs["sample_sfd_row"],
                    sample_sfd_col=kwargs["sample_sfd_col"],
                    sample_amax=kwargs["sample_amax"],
                    sample_norm_const=kwargs["sample_norm_const"],
                    sample_prob=kwargs["sample_prob"],
                    acc_dtype=kwargs["acc_dtype"],
                    mma_tiler_mn=kwargs["mma_tiler_mn"],
                    cluster_shape_mn=kwargs["cluster_shape_mn"],
                    sf_vec_size=kwargs["sf_vec_size"],
                    vector_f32=kwargs["vector_f32"],
                    m_aligned=kwargs["m_aligned"],
                    discrete_col_sfd=kwargs["discrete_col_sfd"],
                    b_major=kwargs["b_major"],
                    use_dynamic_sched=kwargs["use_dynamic_sched"],
                    tanh_clamp_scale=kwargs["tanh_clamp_scale"],
                )
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

    def scratch_workspace_bytes(self) -> int:
        """Bytes of caller-owned, 128-byte-aligned scratch ``execute(workspace=)`` requires.

        BF16: the kernel's per-expert TMA-descriptor slots and dynamic-scheduler counter (never 0).
        Block-scaled: 0 -- that kernel keeps its own descriptor workspace, so ``workspace`` stays None.
        """
        if self._implementation is None:
            self.check_support()
        if self.backend is GroupedGemmBackend.BF16:
            return self._implementation.scratch_workspace_bytes()
        return 0

    def execute(
        self,
        a_tensor: torch.Tensor,
        sfa_tensor: Optional[torch.Tensor],
        padded_offsets: torch.Tensor,
        alpha_tensor: Optional[torch.Tensor],
        c_tensor: torch.Tensor,
        d_tensor: torch.Tensor,
        # Dense mode:
        b_tensor: Optional[torch.Tensor] = None,
        sfb_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        # Discrete mode:
        b_ptrs: Optional[torch.Tensor] = None,
        sfb_ptrs: Optional[torch.Tensor] = None,
        d_col_tensor: Optional[torch.Tensor] = None,
        sfd_row_tensor: Optional[torch.Tensor] = None,
        sfd_col_tensor: Optional[torch.Tensor] = None,
        amax_tensor: Optional[torch.Tensor] = None,
        norm_const_tensor: Optional[torch.Tensor] = None,
        prob_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        *,
        workspace: Any = None,
    ) -> None:
        """Execute the compiled kernel; see the backend implementation for each argument.

        The BF16 backend takes ``sfa_tensor=None`` and ``alpha_tensor=None``, writes ``c_tensor``
        (the BF16 pre-activation) and ``d_tensor``, requires ``prob_tensor``, and requires
        ``workspace``: a caller-owned, 128-byte-aligned device buffer of at least
        ``scratch_workspace_bytes()`` bytes, never allocated here; launches that can overlap need
        distinct workspaces. The block-scaled backend takes no ``workspace``.
        """
        if self._implementation is None:
            raise RuntimeError("Kernel not compiled; call compile() first")
        if self.backend is GroupedGemmBackend.BF16:
            controls = (
                ("sfa_tensor", sfa_tensor),
                ("alpha_tensor", alpha_tensor),
                ("sfb_tensor", sfb_tensor),
                ("bias_tensor", bias_tensor),
                ("sfb_ptrs", sfb_ptrs),
                ("d_col_tensor", d_col_tensor),
                ("sfd_row_tensor", sfd_row_tensor),
                ("sfd_col_tensor", sfd_col_tensor),
                ("amax_tensor", amax_tensor),
                ("norm_const_tensor", norm_const_tensor),
            )
            forbidden = [name for name, value in controls if value is not None]
            if forbidden:
                raise ValueError(f"grouped_gemm_srelu_sm100: BF16 forbids scale control {forbidden[0]}")
            self._implementation.execute(
                a_tensor=a_tensor,
                c_tensor=c_tensor,
                d_tensor=d_tensor,
                padded_offsets=padded_offsets,
                prob_tensor=prob_tensor,
                b_tensor=b_tensor,
                b_ptrs=b_ptrs,
                current_stream=current_stream,
                workspace=workspace,
            )
        else:
            if workspace is not None:
                raise ValueError("workspace is supported only by the BF16 kernel; the block-scaled kernel needs no caller scratch")
            self._implementation.execute(
                a_tensor=a_tensor,
                sfa_tensor=sfa_tensor,
                padded_offsets=padded_offsets,
                alpha_tensor=alpha_tensor,
                c_tensor=c_tensor,
                d_tensor=d_tensor,
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
                current_stream=current_stream,
            )
        self._is_supported = self._implementation._is_supported
        self._compiled_kernel = self._implementation._compiled_kernel


import logging

_logger = logging.getLogger(__name__)
_cache_of_GroupedGemmSreluSm100Objects = {}


def grouped_gemm_srelu_wrapper_sm100(
    a_tensor: torch.Tensor,
    b_tensor: Optional[torch.Tensor] = None,
    sfa_tensor: Optional[torch.Tensor] = None,
    sfb_tensor: Optional[torch.Tensor] = None,
    padded_offsets: Optional[torch.Tensor] = None,
    alpha_tensor: Optional[torch.Tensor] = None,
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
    use_dynamic_sched: bool = False,
    tanh_clamp_scale: Optional[float] = None,
    current_stream: Optional[cuda.CUstream] = None,
    tanh_mode: str = "approx",
    act_input: str = "acc",
) -> TupleDict:
    """Convenience wrapper for grouped GEMM SReLU operation.

    This function creates the API, compiles, and executes in one call.
    Compiled kernels are cached for reuse when called with the same configuration.

    BF16 ``a_tensor`` and ``b_tensor`` (or ``b_dtype``) select the BF16 backend: ``c_tensor`` is
    then the BF16 pre-activation ``bf16(A @ B^T)`` (saved for the backward) and ``d_tensor`` the
    BF16 weighted squared-ReLU. ``prob_tensor`` is required; ``sfa_tensor``, ``sfb_tensor``,
    ``sfb_ptrs``, ``alpha_tensor``, ``bias_tensor`` and ``norm_const_tensor`` must be omitted; the
    other outputs are ``None``. The BF16 kernel's descriptor and scheduler scratch
    (``GroupedGemmSreluSm100.scratch_workspace_bytes()``) is allocated per call on the launch
    stream. The BF16 backend takes torch tensors only; JAX arrays are served by the block-scaled
    dense path (MXFP8 or packed FP4 ``b_tensor`` with ``sfa_tensor``/``sfb_tensor``).

    Args:
        a_tensor: Input A tensor (valid_m, k, 1)
        sfa_tensor: Scale factor A
        padded_offsets: End offset per expert after padding (l,)
        alpha_tensor: Per-group scaling
        b_tensor: (Dense) Weight B tensor (n, k, l)
        sfb_tensor: (Dense) Scale factor B
        bias_tensor: Optional per-expert bias, shape ``(n, l)`` in dense mode or ``(n, num_experts)``
            in discrete mode, stride ``(1, n)``. Bias fusion requires ``mma_tiler_mn[1] == 256``.
        b_ptrs: (Discrete) 1-D int64 device tensor of per-expert B data pointers
        sfb_ptrs: (Discrete) 1-D int64 device tensor of per-expert SFB data pointers
        n: (Discrete) B weight N dimension
        b_dtype: (Discrete) B weight data type
        b_major: (Discrete) B tensor major dimension ("k" or "n")
        norm_const_tensor: Optional normalization constant. Required when using FP8
            input configurations (i.e., when a_tensor.dtype is FP8 and sfa_tensor.dtype is FP8).
            Should be None for FP4/BF16 input configurations.
        prob_tensor: Probability tensor for per-row gating (shape `(valid_m, 1, 1)`).
            This argument is required. Pass a tensor of ones when no gating is needed.
        acc_dtype: Accumulator data type
        c_dtype: Output C tensor data type
        d_dtype: Output D tensor data type
        cd_major: CD major dimension (only "n"-major layout is supported)
        mma_tiler_mn: MMA tiler shape
        cluster_shape_mn: Cluster shape
        sf_vec_size: Scale factor vector size
        vector_f32: Use vectorized f32
        m_aligned: M alignment (must be 256)
        discrete_col_sfd: Enable discrete col-major scale factor tensor
        tanh_clamp_scale: Optional soft-clamp scale ``s``. When set, computes
            ``(s * tanh(relu(x) / s)) ** 2`` instead of plain ``relu(x) ** 2``; must be
            finite and > 0. ``None`` (default) is bit-identical to today's SReLU.
        current_stream: CUDA stream
        tanh_mode: ``"approx"`` (default) or ``"accurate"``; any other value, ``None`` included,
            raises ``ValueError``. With ``tanh_clamp_scale`` set it selects the tanh: ``"approx"``
            is MUFU ``tanh.approx.f32`` (capped at 1) on both backends; ``"accurate"`` is a
            Cephes-style FP32 tanh on the BF16 backend and raises ``ValueError`` on the
            block-scaled one, which has no accurate tanh. Without ``tanh_clamp_scale`` no tanh is
            evaluated and the mode is ignored (both values share one compiled kernel). A NaN
            pre-activation reaches ``d_tensor`` as NaN on the BF16 backend in every mode; the
            block-scaled kernel maps it to +0 (clamped: the ``mul.rn.sat.f32`` in
            ``moe_kernel_helpers.tanh_clamp_unit``; plain: its relu select).
        act_input: BF16 only. ``"acc"`` (default) takes the activation from the FP32 accumulator,
            ``"saved"`` from ``bf16(acc)`` (the returned ``c_tensor``), so that a backward which
            recomputes the activation from ``c_tensor`` reproduces ``d_tensor`` bit for bit.

    Returns:
        TupleDict: A dictionary-like object containing output tensors that can also be unpacked as a tuple.
            Dictionary keys (also the unpacking order):
            - **c_tensor** (torch.Tensor): Accumulator output tensor
            - **d_tensor** (torch.Tensor): Final output tensor
            - **d_col_tensor** (torch.Tensor or None): Column-wise output tensor for low-precision D output
            - **amax_tensor** (torch.Tensor or None): Absolute maximum values (for SReLU output quantization)
            - **sfd_row_tensor** (torch.Tensor or None): Row-wise scale factors for D (FP8 only)
            - **sfd_col_tensor** (torch.Tensor or None): Column-wise scale factors for D (FP8 only)

            Example usage::

                # Dictionary-style access
                result = grouped_gemm_srelu_wrapper_sm100(...)
                c = result["c_tensor"]
                d = result["d_tensor"]

                # Tuple unpacking
                c, d, d_col, amax, sfd_row, sfd_col = grouped_gemm_srelu_wrapper_sm100(...)

                # Integer indexing
                c = result[0]  # c_tensor
                d = result[1]  # d_tensor
    """
    framework = detect_framework(a_tensor)
    if framework not in ("torch", "jax"):
        raise ValueError(f"Unsupported tensor framework '{framework}' for grouped_gemm_srelu_wrapper_sm100; pass torch tensors")
    is_dense = b_tensor is not None
    is_discrete = b_ptrs is not None
    if is_dense and is_discrete:
        raise ValueError("Provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs), not both")
    if not is_dense and not is_discrete:
        raise ValueError("Must provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs)")
    if is_discrete and (n is None or b_dtype is None):
        raise ValueError("n and b_dtype are required for discrete mode")
    backend = select_grouped_gemm_backend(
        operation="grouped_gemm_srelu_sm100",
        a_dtype=a_tensor.dtype,
        b_dtype=b_tensor.dtype if is_dense else b_dtype,
        scale_controls=(
            ("sfa_tensor", sfa_tensor),
            ("sfb_tensor", sfb_tensor),
            ("sfb_ptrs", sfb_ptrs),
            ("alpha_tensor", alpha_tensor),
            ("bias_tensor", bias_tensor),
            ("norm_const_tensor", norm_const_tensor),
            ("sf_vec_size", sf_vec_size if sf_vec_size != 16 else None),
            ("vector_f32", vector_f32 if vector_f32 else None),
            ("discrete_col_sfd", discrete_col_sfd if discrete_col_sfd else None),
        ),
        block_scaled_dtype_pairs=_block_scaled_dtype_pairs(),
    )
    if backend is GroupedGemmBackend.BF16:
        return _grouped_gemm_srelu_bf16_call(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            padded_offsets=padded_offsets,
            b_ptrs=b_ptrs,
            n=n,
            b_dtype=b_dtype,
            b_major=b_major,
            prob_tensor=prob_tensor,
            acc_dtype=acc_dtype,
            c_dtype=c_dtype,
            d_dtype=d_dtype,
            cd_major=cd_major,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            m_aligned=m_aligned,
            use_dynamic_sched=use_dynamic_sched,
            tanh_clamp_scale=tanh_clamp_scale,
            tanh_mode=tanh_mode,
            act_input=act_input,
            current_stream=current_stream,
        )
    _check_block_scaled_activation_options("grouped_gemm_srelu_sm100", tanh_clamp_scale, tanh_mode, act_input)
    return _grouped_gemm_srelu_block_scaled_call(
        a_tensor=a_tensor,
        b_tensor=b_tensor,
        sfa_tensor=sfa_tensor,
        sfb_tensor=sfb_tensor,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha_tensor,
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
        vector_f32=vector_f32,
        m_aligned=m_aligned,
        discrete_col_sfd=discrete_col_sfd,
        use_dynamic_sched=use_dynamic_sched,
        tanh_clamp_scale=tanh_clamp_scale,
        current_stream=current_stream,
    )


def _grouped_gemm_srelu_block_scaled_call(
    a_tensor: torch.Tensor,
    b_tensor: Optional[torch.Tensor] = None,
    sfa_tensor: Optional[torch.Tensor] = None,
    sfb_tensor: Optional[torch.Tensor] = None,
    padded_offsets: Optional[torch.Tensor] = None,
    alpha_tensor: Optional[torch.Tensor] = None,
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
    use_dynamic_sched: bool = False,
    tanh_clamp_scale: Optional[float] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Block-scaled path of grouped_gemm_srelu_wrapper_sm100 (arguments as documented there)."""
    from cudnn.gemm.cutedsl.grouped.unfused._bf16_api import _validate_pointer_tensor

    framework = detect_framework(a_tensor)
    if framework == "jax" and b_tensor is not None:
        from ..canonical_jax import check_jax_wrapper_options
        from .jax_api import grouped_gemm_srelu_jax_forward

        check_jax_wrapper_options(
            acc_dtype=acc_dtype,
            cd_major=cd_major,
            sf_vec_size=sf_vec_size,
            vector_f32=vector_f32,
            m_aligned=m_aligned,
            current_stream=current_stream,
            bias_tensor=bias_tensor,
            b_ptrs=b_ptrs,
            sfb_ptrs=sfb_ptrs,
            b_major=b_major,
            use_dynamic_sched=use_dynamic_sched,
            sf_vec_sizes=(16, 32),
        )
        for name, rejected in (("n", n is not None), ("b_dtype", b_dtype is not None)):
            if rejected:
                raise ValueError(f"{name} is unsupported for the JAX block-scaled path")
        return grouped_gemm_srelu_jax_forward(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            padded_offsets,
            alpha_tensor,
            prob_tensor,
            norm_const_tensor,
            c_dtype=c_dtype if c_dtype is not None else cutlass.BFloat16,
            d_dtype=d_dtype if d_dtype is not None else cutlass.BFloat16,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            discrete_col_sfd=discrete_col_sfd,
            sf_vec_size=sf_vec_size,
            tanh_clamp_scale=tanh_clamp_scale,
        )
    if framework == "jax":
        raise ValueError(f"grouped_gemm_srelu_wrapper_sm100 does not support JAX arrays: {_JAX_SF_LAYOUT_ERROR}")
    if framework != "torch":
        raise ValueError(f"Unsupported tensor framework '{framework}' for grouped_gemm_srelu_wrapper_sm100; pass torch tensors")
    import torch

    acc_dtype = _convert_to_cutlass_data_type(acc_dtype) if acc_dtype is not None else cutlass.Float32
    c_dtype = _convert_to_cutlass_data_type(c_dtype) if c_dtype is not None else cutlass.BFloat16
    d_dtype = _convert_to_cutlass_data_type(d_dtype) if d_dtype is not None else cutlass.BFloat16
    b_dtype = _convert_to_cutlass_data_type(b_dtype) if b_dtype is not None else None

    is_dense = b_tensor is not None
    is_discrete = b_ptrs is not None

    if is_dense and is_discrete:
        raise ValueError("Provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs), not both")
    if not is_dense and not is_discrete:
        raise ValueError("Must provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs)")

    valid_m, k_physical, _ = a_tensor.shape
    if is_dense:
        weight_mode = MoEWeightMode.DENSE
        n_out, _, l = b_tensor.shape
        if bias_tensor is not None and tuple(bias_tensor.shape) != (n_out, l):
            raise ValueError(f"bias_tensor must have shape {(n_out, l)}, got {tuple(bias_tensor.shape)}")
    else:
        weight_mode = MoEWeightMode.DISCRETE
        num_experts = _validate_pointer_tensor(b_ptrs, "b_ptrs")
        _validate_pointer_tensor(sfb_ptrs, "sfb_ptrs", num_experts)
        if n is None or b_dtype is None:
            raise ValueError("n and b_dtype are required for discrete mode")
        k_logical = k_physical * 2 if b_dtype in (cutlass.Float4E2M1FN, cutlass.Uint8) else k_physical
        b_shape = (n, k_logical)
        n_out = n
        l = num_experts
        if bias_tensor is not None and tuple(bias_tensor.shape) != (n_out, num_experts):
            raise ValueError(f"bias_tensor must have shape {(n_out, num_experts)}, got {tuple(bias_tensor.shape)}")

    is_fp8_input_config = _convert_to_cutlass_data_type(a_tensor.dtype) in (
        cutlass.Float8E4M3FN,
        cutlass.Float8E5M2,
    ) and _convert_to_cutlass_data_type(sfa_tensor.dtype) in (
        cutlass.Float8E8M0FNU,
        cutlass.Float8E4M3FN,
    )
    is_low_precision_output_config = d_dtype in (
        cutlass.Float8E4M3FN,
        cutlass.Float8E5M2,
        cutlass.Float4E2M1FN,
    )

    _logger.debug("grouped_gemm_srelu_wrapper_sm100: Creating output tensors")

    if cd_major == "n":
        c_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=framework_dtype(c_dtype, "torch"), device=a_tensor.device)
        d_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=framework_dtype(d_dtype, "torch"), device=a_tensor.device)
        d_col_tensor = (
            torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=framework_dtype(d_dtype, "torch"), device=a_tensor.device)
            if is_low_precision_output_config
            else None
        )
    else:
        raise ValueError(f"cd_major must be 'n', got {cd_major}")

    sfd_row_tensor = None
    sfd_col_tensor = None
    amax_tensor = None

    if is_fp8_input_config and is_low_precision_output_config and norm_const_tensor is None:
        raise ValueError(
            "norm_const_tensor is required when FP8 inputs are used with FP8 output "
            "(a_tensor is FP8 and sfa_tensor is FP8 and d_dtype is FP8). "
            "Pass a tensor with shape (1,), e.g. torch.tensor([0.01], dtype=torch.float32, device=a_tensor.device)."
        )

    if not is_low_precision_output_config:
        norm_const_tensor = None

    if is_fp8_input_config and is_low_precision_output_config:
        _logger.debug("grouped_gemm_srelu_wrapper_sm100: Detected fp8 a_dtype and sfa_dtype, constructing sfd_row_tensor and sfd_col_tensor")

        sf_dtype = sfa_tensor.dtype
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        sf_k_row = ceil_div(n_out, sf_vec_size)
        mma_shape_row = (
            1,
            ceil_div(valid_m, 128),
            ceil_div(sf_k_row, 4),
            32,
            4,
            4,
        )
        sfd_row_tensor = torch.empty(mma_shape_row, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

        sf_k_col = ceil_div(valid_m, sf_vec_size)
        mma_shape_col = (
            1,
            ceil_div(n_out, 128),
            ceil_div(sf_k_col, 4),
            32,
            4,
            4,
        )
        sfd_col_tensor = torch.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

    if d_dtype in (cutlass.BFloat16, cutlass.Float16):
        _logger.debug("grouped_gemm_srelu_wrapper_sm100: Detected bf16/float16 d_dtype, constructing amax_tensor")
        amax_tensor = torch.full((l, 1), float("-inf"), dtype=torch.float32, device=a_tensor.device)

    if prob_tensor is None:
        raise ValueError(
            "prob_tensor is required: the kernel unconditionally multiplies output by per-row gating probability. "
            "Pass a tensor of ones with shape (valid_m, 1, 1) if no gating is needed."
        )

    if valid_m == 0:
        _logger.debug("grouped_gemm_srelu_wrapper_sm100: valid_m is zero, skipping kernel execution")
        return TupleDict(
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            d_col_tensor=d_col_tensor,
            amax_tensor=amax_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
        )

    def tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype

    def stride_order(tensor: torch.Tensor) -> Tuple[int, ...]:
        return tuple(i for i, s in sorted(enumerate(tensor.stride()), key=lambda x: x[1]))

    def dynamic_tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        return None, stride_order(tensor), tensor.dtype

    def dynamic_m_tensor_signature(
        tensor: Optional[torch.Tensor], static_shape_suffix: Tuple[int, ...], dynamic_stride_dims: Tuple[int, ...] = ()
    ) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        stride_signature = tuple(None if i in dynamic_stride_dims else s for i, s in enumerate(tensor.stride()))
        return static_shape_suffix, stride_signature, tensor.dtype

    use_full_dynamic = is_dense and os.environ.get("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1") != "0"

    if is_dense:
        cache_key = (
            weight_mode,
            use_full_dynamic,
            a_tensor.shape[1:] if not use_full_dynamic else None,
            b_tensor.shape[2] if use_full_dynamic else tuple(b_tensor.shape),
            a_tensor.dtype,
            b_tensor.dtype,
            stride_order(a_tensor),
            stride_order(b_tensor),
            c_tensor.shape[1:] if not use_full_dynamic else None,
            stride_order(c_tensor),
            c_tensor.dtype,
            d_tensor.shape[1:] if not use_full_dynamic else None,
            stride_order(d_tensor),
            *(
                dynamic_tensor_signature(sfa_tensor)
                if use_full_dynamic
                else dynamic_m_tensor_signature(sfa_tensor, (sfa_tensor.shape[4], 1) if sfa_tensor is not None else None, dynamic_stride_dims=(5,))
            ),
            *(dynamic_tensor_signature(sfb_tensor) if use_full_dynamic else tensor_signature(sfb_tensor)),
            *(dynamic_tensor_signature(bias_tensor) if use_full_dynamic else tensor_signature(bias_tensor)),
            *tensor_signature(alpha_tensor),
            *tensor_signature(norm_const_tensor),
            *dynamic_m_tensor_signature(prob_tensor, (1, 1)),
            tuple(padded_offsets.shape),
            tuple(padded_offsets.stride()),
            padded_offsets.dtype,
            acc_dtype,
            d_dtype,
            cd_major,
            mma_tiler_mn,
            cluster_shape_mn,
            sf_vec_size,
            vector_f32,
            m_aligned,
            discrete_col_sfd,
            use_dynamic_sched,
            float(tanh_clamp_scale) if tanh_clamp_scale is not None else None,
        )
    else:
        cache_key = (
            weight_mode,
            a_tensor.shape[1:],
            stride_order(a_tensor),
            a_tensor.dtype,
            b_shape,
            b_dtype,
            c_tensor.shape[1:],
            stride_order(c_tensor),
            c_tensor.dtype,
            d_tensor.shape[1:],
            stride_order(d_tensor),
            *dynamic_m_tensor_signature(sfa_tensor, (sfa_tensor.shape[4], 1) if sfa_tensor is not None else None, dynamic_stride_dims=(5,)),
            *tensor_signature(bias_tensor),
            *tensor_signature(alpha_tensor),
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
            d_dtype,
            cd_major,
            mma_tiler_mn,
            cluster_shape_mn,
            sf_vec_size,
            vector_f32,
            m_aligned,
            discrete_col_sfd,
            use_dynamic_sched,
            b_major,
            num_experts,
            float(tanh_clamp_scale) if tanh_clamp_scale is not None else None,
        )

    if cache_key in _cache_of_GroupedGemmSreluSm100Objects:
        _logger.debug("grouped_gemm_srelu_wrapper_sm100: Using previously cached GroupedGemmSreluSm100 object")
        grouped_gemm_srelu = _cache_of_GroupedGemmSreluSm100Objects[cache_key]
    else:
        _logger.debug("grouped_gemm_srelu_wrapper_sm100: No previously cached object found, creating new GroupedGemmSreluSm100 object")
        if is_dense:
            grouped_gemm_srelu = GroupedGemmSreluSm100(
                sample_a=a_tensor,
                sample_sfa=sfa_tensor,
                sample_padded_offsets=padded_offsets,
                sample_alpha=alpha_tensor,
                sample_c=c_tensor,
                sample_d=d_tensor,
                sample_d_col=d_col_tensor,
                sample_b=b_tensor,
                sample_sfb=sfb_tensor,
                sample_bias=bias_tensor,
                sample_amax=amax_tensor,
                sample_sfd_row=sfd_row_tensor,
                sample_sfd_col=sfd_col_tensor,
                sample_norm_const=norm_const_tensor,
                sample_prob=prob_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                discrete_col_sfd=discrete_col_sfd,
                use_dynamic_sched=use_dynamic_sched,
                tanh_clamp_scale=tanh_clamp_scale,
            )
        else:
            grouped_gemm_srelu = GroupedGemmSreluSm100(
                sample_a=a_tensor,
                sample_sfa=sfa_tensor,
                sample_padded_offsets=padded_offsets,
                sample_alpha=alpha_tensor,
                sample_c=c_tensor,
                sample_d=d_tensor,
                sample_d_col=d_col_tensor,
                num_experts=num_experts,
                b_shape=b_shape,
                b_dtype=b_dtype,
                sample_bias=bias_tensor,
                sample_amax=amax_tensor,
                sample_sfd_row=sfd_row_tensor,
                sample_sfd_col=sfd_col_tensor,
                sample_norm_const=norm_const_tensor,
                sample_prob=prob_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                discrete_col_sfd=discrete_col_sfd,
                use_dynamic_sched=use_dynamic_sched,
                b_major=b_major,
                tanh_clamp_scale=tanh_clamp_scale,
            )

        assert grouped_gemm_srelu.check_support(), "Unsupported configuration"
        grouped_gemm_srelu.compile()
        _cache_of_GroupedGemmSreluSm100Objects[cache_key] = grouped_gemm_srelu

    if is_dense:
        grouped_gemm_srelu.execute(
            a_tensor=a_tensor,
            sfa_tensor=sfa_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            b_tensor=b_tensor,
            sfb_tensor=sfb_tensor,
            d_col_tensor=d_col_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            prob_tensor=prob_tensor,
            bias_tensor=bias_tensor,
            current_stream=current_stream,
        )
    else:
        grouped_gemm_srelu.execute(
            a_tensor=a_tensor,
            sfa_tensor=sfa_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            b_ptrs=b_ptrs,
            sfb_ptrs=sfb_ptrs,
            d_col_tensor=d_col_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            prob_tensor=prob_tensor,
            bias_tensor=bias_tensor,
            current_stream=current_stream,
        )

    return TupleDict(
        c_tensor=c_tensor,
        d_tensor=d_tensor,
        d_col_tensor=d_col_tensor,
        amax_tensor=amax_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
    )


def _srelu_tensor_signature(tensor, *, dynamic_m: bool = False) -> tuple:
    """What a compiled BF16 kernel depends on: layout, dtype and device, with M left out when dynamic."""
    if tensor is None:
        return (None,)
    device = get_device(tensor)
    shape = get_shape(tensor)
    if dynamic_m:
        # execute() checks the full layout; the compiled kernel only depends on the row shape.
        return (shape[1:], tensor.dtype, device.type, device.index)
    return (shape, canonicalize_unit_dim_strides(shape, get_strides(tensor)), tensor.dtype, device.type, device.index)


def _grouped_gemm_srelu_bf16_call(
    *,
    a_tensor,
    b_tensor,
    padded_offsets,
    b_ptrs,
    n,
    b_dtype,
    b_major,
    prob_tensor,
    acc_dtype,
    c_dtype,
    d_dtype,
    cd_major,
    mma_tiler_mn,
    cluster_shape_mn,
    m_aligned,
    use_dynamic_sched,
    tanh_clamp_scale,
    tanh_mode,
    act_input,
    current_stream,
) -> TupleDict:
    """BF16 path of grouped_gemm_srelu_wrapper_sm100."""
    from cudnn.gemm.cutedsl.grouped.unfused._bf16_api import _validate_pointer_tensor

    operation = "grouped_gemm_srelu_wrapper_sm100"
    if detect_framework(a_tensor) != "torch":
        # The public wrapper admits torch and JAX only; the BF16 backend has no JAX entry point.
        raise ValueError(_JAX_BF16_ERROR)
    # Rule 7: decline an unusable CuTe DSL by version before anything is allocated (the facade's
    # check_support() imports the BF16 kernel module behind the same gate).
    requirement = bf16_kernel_requirement_error(f"{operation} (BF16)")
    if requirement:
        raise ImportError(requirement)
    import torch

    for name, dtype, expected in (("acc_dtype", acc_dtype, cutlass.Float32), ("c_dtype", c_dtype, cutlass.BFloat16), ("d_dtype", d_dtype, cutlass.BFloat16)):
        if dtype is not None and _convert_to_cutlass_data_type(dtype) is not expected:
            raise ValueError(f"{operation}: BF16 {name} must be {expected}, got {dtype}")
    if cd_major != "n":
        raise ValueError(f"cd_major must be 'n', got {cd_major}")
    if prob_tensor is None:
        raise ValueError(f"{operation}: prob_tensor is required (shape (valid_m, 1, 1), float32); pass ones when no gating is needed")
    if padded_offsets is None:
        raise ValueError(f"{operation}: padded_offsets is required")
    # The public wrapper has checked that exactly one of b_tensor / b_ptrs is given and that
    # discrete mode carries n and b_dtype.
    is_dense = b_tensor is not None
    is_discrete = b_ptrs is not None
    a_shape = get_shape(a_tensor)
    if len(a_shape) != 3 or a_shape[2] != 1:
        raise ValueError(f"a_tensor must have shape (valid_m, k, 1), got {a_shape}")
    valid_m, k, _ = a_shape
    if is_dense:
        weight_mode = MoEWeightMode.DENSE
        b_full_shape = get_shape(b_tensor)
        if len(b_full_shape) != 3:
            raise ValueError(f"b_tensor must have shape (n, k, experts), got {b_full_shape}")
        n_out = b_full_shape[0]
        num_experts = b_full_shape[2]
        b_shape = None
        if n is not None or b_dtype is not None:
            raise ValueError("Dense mode forbids n and b_dtype")
    else:
        weight_mode = MoEWeightMode.DISCRETE
        num_experts = _validate_pointer_tensor(b_ptrs, "b_ptrs")
        n_out = n
        b_shape = (n_out, k)
    tanh_clamp_scale, tanh_mode = resolve_bf16_srelu_options(tanh_clamp_scale, tanh_mode)

    device = a_tensor.device
    if current_stream is None:
        # The launch stream on the operands' device, fixed before anything is allocated: the outputs, the
        # scratch and the launch share it even when another device is current.
        current_stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    # Allocated on the launch stream: the caching allocator hands out blocks on the strength of the
    # allocating stream's ordering, which says nothing about a kernel writing them on another one.
    with _torch_stream_context(current_stream, device):
        c_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=torch.bfloat16, device=device)
        d_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=torch.bfloat16, device=device)
    outputs = TupleDict(
        c_tensor=c_tensor,
        d_tensor=d_tensor,
        d_col_tensor=None,
        amax_tensor=None,
        sfd_row_tensor=None,
        sfd_col_tensor=None,
    )
    if valid_m == 0:
        _logger.debug("grouped_gemm_srelu_wrapper_sm100: valid_m is zero, skipping kernel execution")
        return outputs

    cache_key = backend_cache_key(
        GroupedGemmBackend.BF16,
        weight_mode,
        _srelu_tensor_signature(a_tensor, dynamic_m=True),
        _srelu_tensor_signature(prob_tensor, dynamic_m=True),
        _srelu_tensor_signature(b_tensor),
        _srelu_tensor_signature(padded_offsets),
        _srelu_tensor_signature(b_ptrs),
        n_out,
        num_experts,
        b_shape,
        None if b_dtype is None else _convert_to_cutlass_data_type(b_dtype),
        b_major,
        tuple(mma_tiler_mn),
        None if cluster_shape_mn is None else tuple(cluster_shape_mn),
        m_aligned,
        use_dynamic_sched,
        tanh_clamp_scale,
        # The plain squared-ReLU evaluates no tanh: both modes share one entry and one kernel.
        None if tanh_clamp_scale is None else tanh_mode,
        act_input,
        int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0")),
    )
    api = _cache_of_GroupedGemmSreluSm100Objects.get(cache_key)
    if api is None:
        _logger.debug("grouped_gemm_srelu_wrapper_sm100: creating a BF16 GroupedGemmSreluSm100 object")
        # check_support() and compile() pin the operands' device themselves; this context only keeps
        # the cold path on that device.
        with device_context(device):
            api = GroupedGemmSreluSm100(
                sample_a=a_tensor,
                sample_b=b_tensor,
                sample_c=c_tensor,
                sample_d=d_tensor,
                sample_padded_offsets=padded_offsets,
                num_experts=num_experts if is_discrete else None,
                b_shape=b_shape,
                b_dtype=b_dtype,
                sample_prob=prob_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                m_aligned=m_aligned,
                b_major=b_major,
                use_dynamic_sched=use_dynamic_sched,
                tanh_clamp_scale=tanh_clamp_scale,
                tanh_mode=tanh_mode,
                act_input=act_input,
            )
            if not api.check_support():
                raise RuntimeError("Unsupported BF16 configuration")
            api.compile()
        _cache_of_GroupedGemmSreluSm100Objects[cache_key] = api

    # Per-call scratch allocated on the launch stream (recipe R2): the cached API owns none.
    with wrapper_workspace("torch", api.scratch_workspace_bytes(), device, current_stream) as workspace:
        api._implementation.execute(
            a_tensor=a_tensor,
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            padded_offsets=padded_offsets,
            prob_tensor=prob_tensor,
            b_tensor=b_tensor,
            b_ptrs=b_ptrs,
            current_stream=current_stream,
            workspace=workspace,
        )
    return outputs
