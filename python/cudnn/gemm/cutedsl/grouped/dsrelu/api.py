# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unified API for Grouped GEMM dSReLU Backward Kernel (SM100+)

This module provides a single API class and wrapper for grouped GEMM with the
(prob-weighted) squared-ReLU activation gradient in MoE (Mixture of Experts)
workloads. The backend follows the operand dtypes: BF16 A/B select the BF16 kernel
(``_bf16_api.py``), FP8/FP4 operands the block-scaled kernel with optional output
quantization (``_blockscaled_api.py``).

Dense mode
    All expert weights are packed contiguously in a 3-D tensor (N, K, L).
    Callers supply ``sample_b`` (and ``sample_sfb`` for block-scaled).

Discrete mode
    Each expert has its own memory allocation.  Callers supply
    ``num_experts``, ``b_shape``, ``b_dtype``, and per-expert pointer arrays
    at execution time.
"""

from __future__ import annotations

from ..backend_utils import GroupedGemmBackend, _torch_stream_context, backend_cache_key, select_grouped_gemm_backend, wrapper_workspace
from ..moe_utils import MoEWeightMode
from cuda.bindings import driver as cuda
import logging
import os
from typing import Any, Tuple, Optional

import cutlass

from cudnn._torch_stream import device_context
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import APIBase, TupleDict, ceil_div
from cudnn.gemm.cutedsl.grouped.unfused._bf16_api import _validate_pointer_tensor
from cudnn.tensor_adapter import (
    canonicalize_unit_dim_strides,
    detect_framework,
    framework_dtype,
    get_device,
    get_shape,
    get_strides,
)

# Backend selection and the block-scaled-only argument checks are shared with the forward.
from ..srelu.api import _block_scaled_dtype_pairs, _check_block_scaled_activation_options
from ..srelu._bf16_options import bf16_kernel_requirement_error, resolve_bf16_srelu_options
from ._blockscaled_api import (
    GroupedGemmDsreluBlockScaledAPI,
    _cta_tile_m,
    _dprob_n_slots,
    _reduce_dbias_slots,
    _reduce_dprob_slots,
)

_JAX_BF16_ERROR = (
    "The BF16 grouped GEMM dSReLU backend supports torch tensors only (it has no JAX entry point); "
    "pass torch tensors. JAX arrays are served only by the block-scaled backend"
)


class GroupedGemmDsreluSm100(APIBase):
    """Unified API for grouped GEMM dSReLU backward operation on SM100+ GPUs.

    Computes ``dH = A @ B_e`` per expert fused with the gradient of ``D = prob * srelu(C)``:
    ``d_row = dH * prob * srelu'(C)``, ``dprob += sum_n srelu(C) * dH`` and optionally the
    recomputed forward output ``d_srelu = prob * srelu(C)``. The backend follows the operand
    dtypes and is selected in ``check_support()``: BF16 ``sample_a``/``sample_b`` (or
    ``b_dtype``) select the BF16 kernel, FP8/FP4 operands the block-scaled kernel (alpha scaling,
    block scale factors, dbias and optional output quantization).

    Weight mode is auto-detected from the constructor arguments:

    - **Dense**: provide ``sample_b`` (and ``sample_sfb`` for block-scaled).
    - **Discrete**: provide ``num_experts``, ``b_shape``, and ``b_dtype``.

    BF16 backend: ``sample_c`` is the BF16 pre-activation the forward saved, ``sample_d_row``
    receives the BF16 ``dZ``, ``sample_d_srelu`` (optional) the BF16 recomputed forward output,
    and ``sample_prob``/``sample_dprob`` are required; the scale-factor, alpha, dbias, amax,
    norm-const and ``sample_d_col`` arguments must be omitted. With ``deterministic=True`` the
    BF16 ``execute()`` finishes the dprob reduction itself (see ``GroupedGemmDsreluBf16API``).
    The BF16 ``execute()`` takes caller-owned scratch of ``scratch_workspace_bytes()`` bytes as
    ``workspace=``; see docs/operations/gemm_fusions/grouped_gemm_dsrelu.md.

    Example::

        # Dense mode
        api = GroupedGemmDsreluSm100(
            sample_a=a, sample_c=c,
            sample_d_row=d_row, sample_d_col=d_col,
            sample_sfa=sfa, sample_padded_offsets=offsets,
            sample_alpha=alpha,
            sample_prob=prob, sample_dprob=dprob,
            sample_b=b, sample_sfb=sfb,
        )

        # Discrete mode
        api = GroupedGemmDsreluSm100(
            sample_a=a, sample_c=c,
            sample_d_row=d_row, sample_d_col=d_col,
            sample_sfa=sfa, sample_padded_offsets=offsets,
            sample_alpha=alpha,
            sample_prob=prob, sample_dprob=dprob,
            num_experts=8, b_shape=(n, k), b_dtype=torch.uint8,
        )

        api.check_support()
        api.compile()
        api.execute(...)
    """

    def __init__(
        self,
        sample_a: torch.Tensor,
        # Dense mode (contiguous) -- provide these. sample_dbias is optional:
        sample_b: Optional[torch.Tensor] = None,
        sample_c: Optional[torch.Tensor] = None,
        sample_d_row: Optional[torch.Tensor] = None,
        sample_d_col: Optional[torch.Tensor] = None,
        sample_d_srelu: Optional[torch.Tensor] = None,
        sample_sfa: Optional[torch.Tensor] = None,
        sample_sfb: Optional[torch.Tensor] = None,
        sample_padded_offsets: Optional[torch.Tensor] = None,
        sample_alpha: Optional[torch.Tensor] = None,
        sample_prob: Optional[torch.Tensor] = None,
        sample_dprob: Optional[torch.Tensor] = None,
        sample_dbias: Optional[torch.Tensor] = None,
        sample_dprob_workspace: Optional[torch.Tensor] = None,
        sample_dbias_workspace: Optional[torch.Tensor] = None,
        # Discrete mode -- provide these instead:
        num_experts: Optional[int] = None,
        b_shape: Optional[Tuple[int, ...]] = None,
        b_dtype: Optional[torch.dtype] = None,
        # Optional quantization output arguments
        sample_sfd_row: Optional[torch.Tensor] = None,
        sample_sfd_col: Optional[torch.Tensor] = None,
        sample_sfd_col_d_srelu: Optional[torch.Tensor] = None,
        sample_amax: Optional[torch.Tensor] = None,
        sample_norm_const: Optional[torch.Tensor] = None,
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
        use_dsrelu_reuse: bool = False,
        deterministic: bool = False,
        tanh_clamp_scale: Optional[float] = None,
        tanh_mode: str = "approx",
    ):
        """Initialize the GroupedGemmDsreluSm100 API.

        The arguments up to ``tanh_clamp_scale`` keep their block-scaled meaning (see
        ``GroupedGemmDsreluBlockScaledAPI``). The arguments are stored here; the backend is
        selected and built, and the configuration validated, in ``check_support()``.

        :param use_dsrelu_reuse: The BF16 epilogue always computes ``relu(C)^2`` (clamped: ``c^2``)
            once and uses it for both ``d_srelu`` and dprob, so the BF16 backend accepts either
            value and builds the same kernel for both.
        :param tanh_mode: ``"approx"`` (default) or ``"accurate"``; any other value, ``None``
            included, raises ``ValueError`` in ``check_support()``. Like ``tanh_clamp_scale`` it
            must match the forward pass that produced ``sample_c``: the backward recomputes ``t``
            and the forward output (``d_srelu``) from ``c``. With ``tanh_clamp_scale`` set it
            selects the tanh: ``"approx"`` is MUFU ``tanh.approx.f32`` (capped at 1) on both
            backends; ``"accurate"`` is a Cephes-style FP32 tanh on the BF16 backend and raises
            ``ValueError`` on the block-scaled one, which has no accurate tanh. Without a clamp
            scale no tanh is evaluated and both backends ignore the mode. NaN pre-activations
            propagate on the BF16 backend in every mode; the block-scaled kernel maps them to +0
            (clamped: the ``mul.rn.sat.f32`` in ``moe_kernel_helpers.tanh_clamp_unit``; plain: its
            relu select).
        """
        super().__init__()
        self._pending_init_kwargs = dict(locals())
        self._pending_init_kwargs.pop("self")
        self._pending_init_kwargs.pop("__class__", None)
        framework = detect_framework(sample_a)
        if sample_a is not None and framework not in ("torch", "jax"):
            raise ValueError(f"Unsupported tensor framework '{framework}' for GroupedGemmDsreluSm100; pass torch tensors or JAX arrays")
        self._implementation = None

    def check_support(self) -> bool:
        if self._implementation is None:
            kwargs = self._pending_init_kwargs
            defining_b_dtype = kwargs["sample_b"].dtype if kwargs["sample_b"] is not None else kwargs["b_dtype"]
            backend = select_grouped_gemm_backend(
                operation="grouped_gemm_dsrelu_sm100",
                a_dtype=getattr(kwargs["sample_a"], "dtype", None),
                b_dtype=defining_b_dtype,
                scale_controls=(
                    ("sample_d_col", kwargs["sample_d_col"]),
                    ("sample_sfa", kwargs["sample_sfa"]),
                    ("sample_sfb", kwargs["sample_sfb"]),
                    ("sample_alpha", kwargs["sample_alpha"]),
                    ("sample_dbias", kwargs["sample_dbias"]),
                    ("sample_dbias_workspace", kwargs["sample_dbias_workspace"]),
                    ("sample_sfd_row", kwargs["sample_sfd_row"]),
                    ("sample_sfd_col", kwargs["sample_sfd_col"]),
                    ("sample_sfd_col_d_srelu", kwargs["sample_sfd_col_d_srelu"]),
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
                if detect_framework(kwargs["sample_a"]) == "jax":
                    raise ValueError(_JAX_BF16_ERROR)
                # Rule 7: the BF16 kernel modules import cutlass.experimental (cudnn.frost.tile_dsl) at module
                # scope, so the DSL is checked before they are imported, and only on this branch.
                requirement = bf16_kernel_requirement_error("GroupedGemmDsreluSm100 (BF16)")
                if requirement:
                    raise ImportError(requirement)
                from ._bf16_api import GroupedGemmDsreluBf16API

                self._implementation = GroupedGemmDsreluBf16API(
                    sample_a=kwargs["sample_a"],
                    sample_c=kwargs["sample_c"],
                    sample_d_row=kwargs["sample_d_row"],
                    sample_padded_offsets=kwargs["sample_padded_offsets"],
                    sample_prob=kwargs["sample_prob"],
                    sample_dprob=kwargs["sample_dprob"],
                    sample_b=kwargs["sample_b"],
                    num_experts=kwargs["num_experts"],
                    b_shape=kwargs["b_shape"],
                    b_dtype=kwargs["b_dtype"],
                    sample_d_srelu=kwargs["sample_d_srelu"],
                    sample_dprob_workspace=kwargs["sample_dprob_workspace"],
                    acc_dtype=kwargs["acc_dtype"],
                    mma_tiler_mn=kwargs["mma_tiler_mn"],
                    cluster_shape_mn=kwargs["cluster_shape_mn"],
                    m_aligned=kwargs["m_aligned"],
                    b_major=kwargs["b_major"],
                    use_dynamic_sched=kwargs["use_dynamic_sched"],
                    deterministic=kwargs["deterministic"],
                    tanh_clamp_scale=kwargs["tanh_clamp_scale"],
                    tanh_mode=kwargs["tanh_mode"],
                )
            else:
                _check_block_scaled_activation_options("grouped_gemm_dsrelu_sm100", kwargs["tanh_clamp_scale"], kwargs["tanh_mode"], "acc")
                self._implementation = GroupedGemmDsreluBlockScaledAPI(
                    sample_a=kwargs["sample_a"],
                    sample_b=kwargs["sample_b"],
                    sample_c=kwargs["sample_c"],
                    sample_d_row=kwargs["sample_d_row"],
                    sample_d_col=kwargs["sample_d_col"],
                    sample_d_srelu=kwargs["sample_d_srelu"],
                    sample_sfa=kwargs["sample_sfa"],
                    sample_sfb=kwargs["sample_sfb"],
                    sample_padded_offsets=kwargs["sample_padded_offsets"],
                    sample_alpha=kwargs["sample_alpha"],
                    sample_prob=kwargs["sample_prob"],
                    sample_dprob=kwargs["sample_dprob"],
                    sample_dbias=kwargs["sample_dbias"],
                    sample_dprob_workspace=kwargs["sample_dprob_workspace"],
                    sample_dbias_workspace=kwargs["sample_dbias_workspace"],
                    num_experts=kwargs["num_experts"],
                    b_shape=kwargs["b_shape"],
                    b_dtype=kwargs["b_dtype"],
                    sample_sfd_row=kwargs["sample_sfd_row"],
                    sample_sfd_col=kwargs["sample_sfd_col"],
                    sample_sfd_col_d_srelu=kwargs["sample_sfd_col_d_srelu"],
                    sample_amax=kwargs["sample_amax"],
                    sample_norm_const=kwargs["sample_norm_const"],
                    acc_dtype=kwargs["acc_dtype"],
                    mma_tiler_mn=kwargs["mma_tiler_mn"],
                    cluster_shape_mn=kwargs["cluster_shape_mn"],
                    sf_vec_size=kwargs["sf_vec_size"],
                    vector_f32=kwargs["vector_f32"],
                    m_aligned=kwargs["m_aligned"],
                    discrete_col_sfd=kwargs["discrete_col_sfd"],
                    b_major=kwargs["b_major"],
                    use_dynamic_sched=kwargs["use_dynamic_sched"],
                    use_dsrelu_reuse=kwargs["use_dsrelu_reuse"],
                    deterministic=kwargs["deterministic"],
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
        The deterministic dprob slots are not part of it: they stay ``dprob_workspace_tensor``.
        """
        if self._implementation is None:
            self.check_support()
        if self.backend is GroupedGemmBackend.BF16:
            return self._implementation.scratch_workspace_bytes()
        return 0

    @staticmethod
    def dprob_workspace_shape(valid_m: int, n: int, mma_tiler_mn: Tuple[int, int] = (256, 256), cluster_shape_mn: Optional[Tuple[int, int]] = None):
        """Shape of ``sample_dprob_workspace`` -- one fp32 slot per (token, N-tile)."""
        return GroupedGemmDsreluBlockScaledAPI.dprob_workspace_shape(valid_m, n, mma_tiler_mn, cluster_shape_mn)

    @staticmethod
    def dbias_workspace_shape(valid_m: int, n: int, mma_tiler_mn: Tuple[int, int] = (256, 256)):
        """Shape of ``sample_dbias_workspace`` -- one bf16 slot per (CTA M-block, n)."""
        return GroupedGemmDsreluBlockScaledAPI.dbias_workspace_shape(valid_m, n, mma_tiler_mn)

    @staticmethod
    def reduce_dprob_workspace(dprob_workspace, dprob_tensor=None, current_stream=None):
        """Collapse the dprob slots into ``(valid_m, 1, 1)``; accumulates onto dprob_tensor if given.

        The deterministic class-API sequence that works on both backends is ``execute(...)``, then
        ``reduce_dprob_workspace(slots, dprob_tensor)``, then ``slots.zero_()``. The block-scaled
        ``execute()`` adds its partials into the slots, so this call moves them into
        ``dprob_tensor`` and the slots must be zeroed before the next ``execute()``. The BF16
        ``execute()`` has already folded the slots into ``dprob_tensor`` and zeroed them, so there
        this call adds zero. The return-value form ``reduce_dprob_workspace(slots)`` is not
        portable: after a BF16 ``execute()`` it returns zeros, not dprob.
        """
        return GroupedGemmDsreluBlockScaledAPI.reduce_dprob_workspace(dprob_workspace, dprob_tensor, current_stream)

    @staticmethod
    def reduce_dbias_workspace(dbias_workspace, padded_offsets, dbias_tensor=None, mma_tiler_mn: Tuple[int, int] = (256, 256), current_stream=None):
        """Segment-sum the dbias slots per expert; accumulates onto dbias_tensor if given."""
        return GroupedGemmDsreluBlockScaledAPI.reduce_dbias_workspace(dbias_workspace, padded_offsets, dbias_tensor, mma_tiler_mn, current_stream)

    def execute(
        self,
        a_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        d_row_tensor: torch.Tensor,
        d_col_tensor: Optional[torch.Tensor],
        sfa_tensor: Optional[torch.Tensor],
        padded_offsets: torch.Tensor,
        alpha_tensor: Optional[torch.Tensor],
        prob_tensor: torch.Tensor,
        dprob_tensor: torch.Tensor,
        # Dense mode:
        b_tensor: Optional[torch.Tensor] = None,
        sfb_tensor: Optional[torch.Tensor] = None,
        dbias_tensor: Optional[torch.Tensor] = None,
        # Determinism scratch, required iff the op was built with deterministic=True:
        dprob_workspace_tensor: Optional[torch.Tensor] = None,
        dbias_workspace_tensor: Optional[torch.Tensor] = None,
        # Discrete mode:
        b_ptrs: Optional[torch.Tensor] = None,
        sfb_ptrs: Optional[torch.Tensor] = None,
        # Optional:
        sfd_row_tensor: Optional[torch.Tensor] = None,
        sfd_col_tensor: Optional[torch.Tensor] = None,
        d_srelu_tensor: Optional[torch.Tensor] = None,
        sfd_col_d_srelu_tensor: Optional[torch.Tensor] = None,
        amax_tensor: Optional[torch.Tensor] = None,
        norm_const_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        *,
        workspace: Any = None,
    ) -> None:
        """Execute the compiled kernel; see the backend implementation for each argument.

        The BF16 backend takes ``d_col_tensor=None``, ``sfa_tensor=None`` and
        ``alpha_tensor=None``, and requires ``workspace``: a caller-owned, 128-byte-aligned device
        buffer of at least ``scratch_workspace_bytes()`` bytes, never allocated here; launches that
        can overlap need distinct workspaces. The block-scaled backend takes no ``workspace``.
        """
        if self._implementation is None:
            raise RuntimeError("Kernel not compiled; call compile() first")
        if self.backend is GroupedGemmBackend.BF16:
            controls = (
                ("d_col_tensor", d_col_tensor),
                ("sfa_tensor", sfa_tensor),
                ("alpha_tensor", alpha_tensor),
                ("sfb_tensor", sfb_tensor),
                ("dbias_tensor", dbias_tensor),
                ("dbias_workspace_tensor", dbias_workspace_tensor),
                ("sfb_ptrs", sfb_ptrs),
                ("sfd_row_tensor", sfd_row_tensor),
                ("sfd_col_tensor", sfd_col_tensor),
                ("sfd_col_d_srelu_tensor", sfd_col_d_srelu_tensor),
                ("amax_tensor", amax_tensor),
                ("norm_const_tensor", norm_const_tensor),
            )
            forbidden = [name for name, value in controls if value is not None]
            if forbidden:
                raise ValueError(f"grouped_gemm_dsrelu_sm100: BF16 forbids scale control {forbidden[0]}")
            self._implementation.execute(
                a_tensor=a_tensor,
                c_tensor=c_tensor,
                d_row_tensor=d_row_tensor,
                padded_offsets=padded_offsets,
                prob_tensor=prob_tensor,
                dprob_tensor=dprob_tensor,
                b_tensor=b_tensor,
                b_ptrs=b_ptrs,
                d_srelu_tensor=d_srelu_tensor,
                dprob_workspace_tensor=dprob_workspace_tensor,
                current_stream=current_stream,
                workspace=workspace,
            )
        else:
            if workspace is not None:
                raise ValueError("workspace is supported only by the BF16 kernel; the block-scaled kernel needs no caller scratch")
            self._implementation.execute(
                a_tensor=a_tensor,
                c_tensor=c_tensor,
                d_row_tensor=d_row_tensor,
                d_col_tensor=d_col_tensor,
                sfa_tensor=sfa_tensor,
                padded_offsets=padded_offsets,
                alpha_tensor=alpha_tensor,
                prob_tensor=prob_tensor,
                dprob_tensor=dprob_tensor,
                b_tensor=b_tensor,
                sfb_tensor=sfb_tensor,
                dbias_tensor=dbias_tensor,
                dprob_workspace_tensor=dprob_workspace_tensor,
                dbias_workspace_tensor=dbias_workspace_tensor,
                b_ptrs=b_ptrs,
                sfb_ptrs=sfb_ptrs,
                sfd_row_tensor=sfd_row_tensor,
                sfd_col_tensor=sfd_col_tensor,
                d_srelu_tensor=d_srelu_tensor,
                sfd_col_d_srelu_tensor=sfd_col_d_srelu_tensor,
                amax_tensor=amax_tensor,
                norm_const_tensor=norm_const_tensor,
                current_stream=current_stream,
            )
        self._is_supported = self._implementation._is_supported
        self._compiled_kernel = self._implementation._compiled_kernel


_logger = logging.getLogger(__name__)
_cache_of_GroupedGemmDsreluSm100Objects = {}


def grouped_gemm_dsrelu_wrapper_sm100(
    a_tensor: torch.Tensor,
    b_tensor: Optional[torch.Tensor] = None,
    c_tensor: Optional[torch.Tensor] = None,
    sfa_tensor: Optional[torch.Tensor] = None,
    sfb_tensor: Optional[torch.Tensor] = None,
    padded_offsets: Optional[torch.Tensor] = None,
    alpha_tensor: Optional[torch.Tensor] = None,
    prob_tensor: Optional[torch.Tensor] = None,
    dprob_tensor: Optional[torch.Tensor] = None,
    # generate_dbias is optional in both modes:
    generate_dbias: bool = False,
    # Discrete mode:
    b_ptrs: Optional[torch.Tensor] = None,
    sfb_ptrs: Optional[torch.Tensor] = None,
    n: Optional[int] = None,
    b_dtype: Optional[torch.dtype] = None,
    b_major: str = "k",
    # Common:
    norm_const_tensor: Optional[torch.Tensor] = None,
    acc_dtype: Optional[torch.dtype] = None,
    d_dtype: Optional[torch.dtype] = None,
    cd_major: str = "n",
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
    use_dynamic_sched: bool = False,
    use_dsrelu_reuse: bool = False,
    deterministic: Optional[bool] = None,
    tanh_clamp_scale: Optional[float] = None,
    current_stream: Optional[cuda.CUstream] = None,
    tanh_mode: str = "approx",
    generate_d_srelu: bool = True,
) -> TupleDict:
    """Convenience wrapper for grouped GEMM dSReLU backward operation.

    Auto-detects dense vs. discrete mode based on which weight arguments
    are provided.

    Dense mode: provide ``b_tensor`` and ``sfb_tensor``.
    Discrete mode: provide ``b_ptrs``, ``sfb_ptrs``, ``n``, and ``b_dtype``.

    Compiled kernels are cached for reuse when called with the same configuration.

    BF16 ``a_tensor`` and ``b_tensor`` (or ``b_dtype``) select the BF16 backend: ``c_tensor`` is the
    BF16 pre-activation the forward returned as its ``c_tensor``, ``d_row_tensor`` the BF16 ``dZ``
    and ``d_srelu_tensor`` the BF16 recomputed forward output. ``prob_tensor`` is required;
    ``sfa_tensor``, ``sfb_tensor``, ``sfb_ptrs``, ``alpha_tensor``, ``norm_const_tensor`` and
    ``generate_dbias`` must be omitted; ``d_col_tensor``, ``dbias_tensor``, ``amax_tensor`` and the
    scale-factor outputs are ``None``. The BF16 kernel's descriptor and scheduler scratch
    (``GroupedGemmDsreluSm100.scratch_workspace_bytes()``) is allocated per call on the launch
    stream. The BF16 backend takes torch tensors only; JAX arrays are served by the block-scaled
    backend.

    Args:
        a_tensor: Input A tensor (valid_m, k, 1) -- gradient input
        c_tensor: Forward activations input (valid_m, n_out, 1)
        sfa_tensor: Scale factor A
        padded_offsets: End offset per expert after padding
        alpha_tensor: Per-group alpha scaling
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
        vector_f32: Use vectorized f32
        m_aligned: M alignment (must be 256)
        discrete_col_sfd: Generate discrete col-major scale factor tensor
        use_dynamic_sched: Enable dynamic tile scheduling for load balancing
        use_dsrelu_reuse: Reuse relu(C)^2 (or, when clamped, c^2) between d_srelu and dprob.
            The BF16 epilogue always does, so the BF16 backend accepts either value and builds
            the same kernel for both.
        tanh_clamp_scale: Optional soft-clamp scale ``s``, must match the forward pass that
            produced ``c_tensor``. Must be finite and > 0. ``None`` (default) is
            bit-identical to today's unclamped DSRELU.
        deterministic: Make dprob and dbias bit-exact run to run; raises NotImplementedError
            for any framework other than torch. Every returned tensor keeps its usual shape and
            dtype -- the slot workspaces and the fixed-order reductions into them are internal to
            this wrapper. ``None`` (default) follows ``torch.use_deterministic_algorithms``.
            See docs/operations/gemm_fusions/grouped_gemm_dsrelu.md#deterministic-dprob.
        current_stream: CUDA stream
        tanh_mode: ``"approx"`` (default) or ``"accurate"``; any other value, ``None`` included,
            raises ``ValueError``. Like ``tanh_clamp_scale`` it must match the forward pass that
            produced ``c_tensor``: the backward recomputes ``t`` and the forward output
            (``d_srelu_tensor``) from ``c_tensor``. With ``tanh_clamp_scale`` set it selects the
            tanh: ``"approx"`` is MUFU ``tanh.approx.f32`` (capped at 1) on both backends;
            ``"accurate"`` is a Cephes-style FP32 tanh on the BF16 backend and raises
            ``ValueError`` on the block-scaled one, which has no accurate tanh. Without
            ``tanh_clamp_scale`` no tanh is evaluated and the mode is ignored (both values share
            one compiled kernel). NaN pre-activations propagate on the BF16 backend in every mode;
            the block-scaled kernel maps them to +0 (clamped: the ``mul.rn.sat.f32`` in
            ``moe_kernel_helpers.tanh_clamp_unit``; plain: its relu select).
        generate_d_srelu: Also write ``d_srelu_tensor``, the recomputed forward output
            ``prob * srelu(C)`` (default ``True``, as before). ``False`` skips that write and
            returns ``d_srelu_tensor=None``.

    Returns:
        TupleDict with keys: d_row_tensor, d_col_tensor, d_srelu_tensor, dprob_tensor,
            dbias_tensor, amax_tensor, sfd_row_tensor, sfd_col_tensor, sfd_col_d_srelu_tensor
    """
    framework = detect_framework(a_tensor)
    if framework not in ("torch", "jax"):
        raise ValueError(f"Unsupported tensor framework '{framework}' for grouped_gemm_dsrelu_wrapper_sm100; pass torch tensors or JAX arrays")
    is_dense = b_tensor is not None
    is_discrete = b_ptrs is not None
    if is_dense and is_discrete:
        raise ValueError("Provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs), not both")
    if not is_dense and not is_discrete:
        raise ValueError("Must provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs)")
    if is_discrete and (n is None or b_dtype is None):
        raise ValueError("n and b_dtype are required for discrete mode")
    backend = select_grouped_gemm_backend(
        operation="grouped_gemm_dsrelu_sm100",
        a_dtype=a_tensor.dtype,
        b_dtype=b_tensor.dtype if is_dense else b_dtype,
        scale_controls=(
            ("sfa_tensor", sfa_tensor),
            ("sfb_tensor", sfb_tensor),
            ("sfb_ptrs", sfb_ptrs),
            ("alpha_tensor", alpha_tensor),
            ("norm_const_tensor", norm_const_tensor),
            ("generate_dbias", generate_dbias if generate_dbias else None),
            ("sf_vec_size", sf_vec_size if sf_vec_size != 16 else None),
            ("vector_f32", vector_f32 if vector_f32 else None),
            ("discrete_col_sfd", discrete_col_sfd if discrete_col_sfd else None),
        ),
        block_scaled_dtype_pairs=_block_scaled_dtype_pairs(),
    )
    if backend is GroupedGemmBackend.BF16:
        return _grouped_gemm_dsrelu_bf16_call(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            c_tensor=c_tensor,
            padded_offsets=padded_offsets,
            prob_tensor=prob_tensor,
            dprob_tensor=dprob_tensor,
            b_ptrs=b_ptrs,
            n=n,
            b_dtype=b_dtype,
            b_major=b_major,
            acc_dtype=acc_dtype,
            d_dtype=d_dtype,
            cd_major=cd_major,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            m_aligned=m_aligned,
            use_dynamic_sched=use_dynamic_sched,
            deterministic=deterministic,
            tanh_clamp_scale=tanh_clamp_scale,
            tanh_mode=tanh_mode,
            generate_d_srelu=generate_d_srelu,
            current_stream=current_stream,
        )
    _check_block_scaled_activation_options("grouped_gemm_dsrelu_sm100", tanh_clamp_scale, tanh_mode, "acc")
    return _grouped_gemm_dsrelu_block_scaled_call(
        a_tensor=a_tensor,
        b_tensor=b_tensor,
        c_tensor=c_tensor,
        sfa_tensor=sfa_tensor,
        sfb_tensor=sfb_tensor,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha_tensor,
        prob_tensor=prob_tensor,
        dprob_tensor=dprob_tensor,
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
        vector_f32=vector_f32,
        m_aligned=m_aligned,
        discrete_col_sfd=discrete_col_sfd,
        use_dynamic_sched=use_dynamic_sched,
        use_dsrelu_reuse=use_dsrelu_reuse,
        deterministic=deterministic,
        tanh_clamp_scale=tanh_clamp_scale,
        current_stream=current_stream,
        generate_d_srelu=generate_d_srelu,
    )


def _grouped_gemm_dsrelu_block_scaled_call(
    a_tensor: torch.Tensor,
    b_tensor: Optional[torch.Tensor] = None,
    c_tensor: Optional[torch.Tensor] = None,
    sfa_tensor: Optional[torch.Tensor] = None,
    sfb_tensor: Optional[torch.Tensor] = None,
    padded_offsets: Optional[torch.Tensor] = None,
    alpha_tensor: Optional[torch.Tensor] = None,
    prob_tensor: Optional[torch.Tensor] = None,
    dprob_tensor: Optional[torch.Tensor] = None,
    # generate_dbias is optional in both modes:
    generate_dbias: bool = False,
    # Discrete mode:
    b_ptrs: Optional[torch.Tensor] = None,
    sfb_ptrs: Optional[torch.Tensor] = None,
    n: Optional[int] = None,
    b_dtype: Optional[torch.dtype] = None,
    b_major: str = "k",
    # Common:
    norm_const_tensor: Optional[torch.Tensor] = None,
    acc_dtype: Optional[torch.dtype] = None,
    d_dtype: Optional[torch.dtype] = None,
    cd_major: str = "n",
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
    use_dynamic_sched: bool = False,
    use_dsrelu_reuse: bool = False,
    deterministic: Optional[bool] = None,
    tanh_clamp_scale: Optional[float] = None,
    current_stream: Optional[cuda.CUstream] = None,
    generate_d_srelu: bool = True,
) -> TupleDict:
    """Block-scaled path of grouped_gemm_dsrelu_wrapper_sm100 (arguments as documented there)."""
    framework = detect_framework(a_tensor)
    if framework not in ("torch", "jax"):
        raise ValueError(f"Unsupported tensor framework '{framework}' for grouped_gemm_dsrelu_wrapper_sm100; pass torch tensors or JAX arrays")

    acc_dtype = _convert_to_cutlass_data_type(acc_dtype) if acc_dtype is not None else cutlass.Float32
    d_dtype = _convert_to_cutlass_data_type(d_dtype) if d_dtype is not None else cutlass.BFloat16
    b_dtype = _convert_to_cutlass_data_type(b_dtype) if b_dtype is not None else None

    is_dense = b_tensor is not None
    is_discrete = b_ptrs is not None

    if is_dense and is_discrete:
        raise ValueError("Provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs), not both")
    if not is_dense and not is_discrete:
        raise ValueError("Must provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs)")

    if framework == "jax":
        if is_dense:
            from ..canonical_jax import check_jax_wrapper_options
            from .jax_api import grouped_gemm_dsrelu_jax_sm100

            check_jax_wrapper_options(
                acc_dtype=acc_dtype,
                cd_major=cd_major,
                sf_vec_size=sf_vec_size,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                current_stream=current_stream,
                dprob_tensor_buf=dprob_tensor,
                sfb_ptrs=sfb_ptrs,
                b_major=b_major,
                use_dynamic_sched=use_dynamic_sched,
                sf_vec_sizes=(16, 32),
            )
            for name, rejected in (
                ("n", n is not None),
                ("b_dtype", b_dtype is not None),
                ("generate_dbias", generate_dbias),
                ("use_dsrelu_reuse", use_dsrelu_reuse),
                ("deterministic", bool(deterministic)),
                ("tanh_clamp_scale", tanh_clamp_scale is not None),
                # grouped_gemm_dsrelu_jax_sm100 always writes d_srelu: refuse rather than return it.
                ("generate_d_srelu", not generate_d_srelu),
            ):
                if rejected:
                    raise ValueError(f"{name} is unsupported for the JAX block-scaled path")
            return grouped_gemm_dsrelu_jax_sm100(
                a_tensor,
                c_tensor,
                sfa_tensor,
                padded_offsets,
                alpha_tensor,
                prob_tensor,
                norm_const_tensor=norm_const_tensor,
                d_dtype=d_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                discrete_col_sfd=discrete_col_sfd,
                b_tensor=b_tensor,
                sfb_tensor=sfb_tensor,
            )
        if _convert_to_cutlass_data_type(a_tensor.dtype) in (cutlass.Float4E2M1FN, cutlass.Uint8) or b_dtype in (cutlass.Float4E2M1FN, cutlass.Uint8):
            raise ValueError(
                "Packed fp4 A/B tensors (float4_e2m1fn / raw uint8) are not expressible as JAX arrays "
                "(JAX has no packed fp4 dtype); use fp8 inputs from JAX, or torch tensors for fp4"
            )
    if framework == "torch":
        import torch

    valid_m, k_physical, _ = get_shape(a_tensor)

    if is_dense:
        weight_mode = MoEWeightMode.DENSE
        n_weight, _, l = b_tensor.shape
    else:
        weight_mode = MoEWeightMode.DISCRETE
        num_experts = _validate_pointer_tensor(b_ptrs, "b_ptrs")
        _validate_pointer_tensor(sfb_ptrs, "sfb_ptrs", num_experts)
        if n is None or b_dtype is None:
            raise ValueError("n and b_dtype are required for discrete mode")
        n_weight = n
        k_logical = k_physical * 2 if b_dtype in (cutlass.Float4E2M1FN, cutlass.Uint8) else k_physical
        b_shape = (n_weight, k_logical)
        l = num_experts

    n_out = n_weight

    _logger.debug("grouped_gemm_dsrelu_wrapper_sm100: Creating output tensors")

    if cd_major != "n":
        raise ValueError(f"cd_major must be 'n', got {cd_major}")

    if framework == "jax":
        import jax.numpy as jnp

    def _jax_alloc(builder):
        import jax

        # The kernel writes into these buffers on the launch stream, outside XLA's
        # tracking; materialize them before their pointers are taken.
        return jax.block_until_ready(builder())

    if framework == "torch":
        d_torch_dtype = framework_dtype(d_dtype, "torch")
        # Inside the context, like the accumulators: torch's allocator reuses a block freed on the
        # allocating stream on the strength of that stream's ordering, which says nothing about a
        # kernel writing the block on current_stream. record_stream cannot fix this direction --
        # it defers reuse after *this* tensor is freed, not the handout of an already-freed block.
        with _torch_stream_context(current_stream, a_tensor.device):
            d_row_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_torch_dtype, device=a_tensor.device)
            d_col_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_torch_dtype, device=a_tensor.device)
            d_srelu_tensor = (
                torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_torch_dtype, device=a_tensor.device) if generate_d_srelu else None
            )
    else:
        # n-major C-contiguous; the extent-1 batch dim's stride is unobservable by the kernel.
        d_jax_dtype = framework_dtype(d_dtype, "jax")
        d_row_tensor = _jax_alloc(lambda: jnp.empty((valid_m, n_out, 1), dtype=d_jax_dtype, device=a_tensor.device))
        d_col_tensor = _jax_alloc(lambda: jnp.empty((valid_m, n_out, 1), dtype=d_jax_dtype, device=a_tensor.device))
        d_srelu_tensor = _jax_alloc(lambda: jnp.empty((valid_m, n_out, 1), dtype=d_jax_dtype, device=a_tensor.device)) if generate_d_srelu else None

    sfd_row_tensor = None
    sfd_col_tensor = None
    sfd_col_d_srelu_tensor = None
    amax_tensor = None
    dbias_tensor = None

    # Deterministic mode gives dprob one slot per N-tile so each (token, tile_n) pair has a
    # single writer; the reduction back to (valid_m, 1, 1) happens after execute(). Resolved
    # here, with the other arguments, so `deterministic` is a plain bool from this point on.
    if deterministic is None:
        # Follows torch's process-wide flag. jax has no equivalent global, so there the opt-in has
        # to be explicit rather than inherited. Written as a nested branch, not
        # `framework == "torch" and torch...`: `torch` is only bound by the conditional import
        # above, so that form is one operand-reorder away from a NameError under jax.
        deterministic = False
        if framework == "torch":
            deterministic = torch.are_deterministic_algorithms_enabled()
    if deterministic and framework != "torch":
        # The slot reductions below are torch ops on torch streams. Rather than half-support it,
        # say so: a caller who asked for reproducibility should not get it silently unenforced.
        raise NotImplementedError(f"deterministic=True is only implemented for the torch framework, got {framework}")
    dprob_n_slots = _dprob_n_slots(n_out, mma_tiler_mn, cluster_shape_mn, deterministic)
    cta_tile_m = _cta_tile_m(mma_tiler_mn)
    if dprob_tensor is None:
        if framework == "torch":
            with _torch_stream_context(current_stream, a_tensor.device):
                dprob_tensor = torch.zeros((valid_m, 1, 1), dtype=torch.float32, device=a_tensor.device)
        else:
            dprob_tensor = _jax_alloc(lambda: jnp.zeros((valid_m, 1, 1), dtype=jnp.float32, device=a_tensor.device))
    # Determinism scratch, allocated here so the wrapper's own signature does not change with
    # the flag. torch-only, rejected for jax above.
    dprob_workspace = None
    if deterministic:
        with _torch_stream_context(current_stream, a_tensor.device):
            dprob_workspace = torch.zeros((valid_m, dprob_n_slots, 1), dtype=torch.float32, device=a_tensor.device)

    if _convert_to_cutlass_data_type(a_tensor.dtype) in (
        cutlass.Float8E4M3FN,
        cutlass.Float8E5M2,
    ) and _convert_to_cutlass_data_type(
        sfa_tensor.dtype
    ) in (cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN):
        _logger.debug("grouped_gemm_dsrelu_wrapper_sm100: Detected fp8 config, constructing sfd tensors")

        sf_dtype = sfa_tensor.dtype
        sf_k_row = ceil_div(n_out, sf_vec_size)
        mma_shape_row = (1, ceil_div(valid_m, 128), ceil_div(sf_k_row, 4), 32, 4, 4)
        sf_k_col = ceil_div(valid_m, sf_vec_size)
        mma_shape_col = (1, ceil_div(n_out, 128), ceil_div(sf_k_col, 4), 32, 4, 4)
        if framework == "torch":
            mma_permute_order = (3, 4, 1, 5, 2, 0)
            with _torch_stream_context(current_stream, a_tensor.device):
                sfd_row_tensor = torch.empty(mma_shape_row, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)
                sfd_col_tensor = torch.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)
                if generate_d_srelu:
                    sfd_col_d_srelu_tensor = torch.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)
        else:
            # Physical C-contiguous atom allocations: JAX cannot express the permuted view;
            # the kernel rebuilds the SF layout from the GEMM shapes and consumes only the
            # SF base pointer, so the physical form is byte-identical.
            sfd_row_tensor = _jax_alloc(lambda: jnp.empty(mma_shape_row, dtype=sf_dtype, device=a_tensor.device))
            sfd_col_tensor = _jax_alloc(lambda: jnp.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device))
            if generate_d_srelu:
                sfd_col_d_srelu_tensor = _jax_alloc(lambda: jnp.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device))

    if d_dtype in (cutlass.BFloat16, cutlass.Float16):
        _logger.debug("grouped_gemm_dsrelu_wrapper_sm100: Constructing amax_tensor")
        if framework == "torch":
            with _torch_stream_context(current_stream, a_tensor.device):
                amax_tensor = torch.full((l, 1), float("-inf"), dtype=torch.float32, device=a_tensor.device)
        else:
            amax_tensor = _jax_alloc(lambda: jnp.full((l, 1), float("-inf"), dtype=jnp.float32, device=a_tensor.device))
    # Deterministic dbias is a separate buffer, not a reshaped one: the kernel writes fp32
    # dbias keeps its (expert, n, 1) bf16 shape in both modes; under determinism the kernel fills
    # the M-block scratch instead and _reduce_dbias_slots segment-sums it into this tensor.
    dbias_workspace = None
    if generate_dbias:
        if framework == "torch":
            with _torch_stream_context(current_stream, a_tensor.device):
                dbias_tensor = torch.zeros((l, n_out, 1), dtype=torch.bfloat16, device=a_tensor.device)
                if deterministic:
                    dbias_workspace = torch.zeros((ceil_div(valid_m, cta_tile_m), n_out, 1), dtype=torch.bfloat16, device=a_tensor.device)
        else:
            dbias_tensor = _jax_alloc(lambda: jnp.zeros((l, n_out, 1), dtype=framework_dtype(cutlass.BFloat16, "jax"), device=a_tensor.device))
    dbias_kernel_tensor = dbias_workspace if dbias_workspace is not None else dbias_tensor

    if valid_m == 0:
        _logger.debug("grouped_gemm_dsrelu_wrapper_sm100: valid_m is zero, skipping kernel execution")
        return TupleDict(
            d_row_tensor=d_row_tensor,
            d_col_tensor=d_col_tensor,
            d_srelu_tensor=d_srelu_tensor,
            # Reduced here too, so a zero-token call returns the documented (valid_m, 1, 1)
            # shape rather than leaking the slot dimension.
            dprob_tensor=_reduce_dprob_slots(dprob_workspace, dprob_tensor, current_stream),
            dbias_tensor=_reduce_dbias_slots(dbias_workspace, dbias_tensor, current_stream, padded_offsets, cta_tile_m),
            amax_tensor=amax_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            sfd_col_d_srelu_tensor=sfd_col_d_srelu_tensor,
        )

    # ---- Build cache key ----
    def stride_order(tensor: torch.Tensor) -> Tuple[int, ...]:
        return tuple(i for i, s in sorted(enumerate(get_strides(tensor)), key=lambda x: x[1]))

    def tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        return get_shape(tensor), get_strides(tensor), _convert_to_cutlass_data_type(tensor.dtype)

    def dynamic_tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        return None, stride_order(tensor), _convert_to_cutlass_data_type(tensor.dtype)

    def dynamic_m_tensor_signature(
        tensor: Optional[torch.Tensor], static_shape_suffix: Optional[Tuple[int, ...]], dynamic_stride_dims: Tuple[int, ...] = ()
    ) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        stride_signature = tuple(None if i in dynamic_stride_dims else s for i, s in enumerate(get_strides(tensor)))
        return static_shape_suffix, stride_signature, _convert_to_cutlass_data_type(tensor.dtype)

    def _sf_is_physical(tensor) -> bool:
        shape = get_shape(tensor)
        return not (len(shape) == 6 and shape[0] == 32 and shape[1] == 4 and shape[3] == 4)

    def dynamic_m_sf_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        """M-independent signature of an SF tensor in either atom form (see _sf_desc_is_physical)."""
        if tensor is None:
            return None, None, None
        shape = get_shape(tensor)
        if not _sf_is_physical(tensor):
            # torch-style permuted view: M' at dim 2, M-dependent stride at dim 5
            return dynamic_m_tensor_signature(tensor, (shape[4], 1), dynamic_stride_dims=(5,))
        # physical C-contiguous form (e.g. JAX): M' at dim 1, M-dependent stride at dim 0
        static_shape = (shape[0], None, *shape[2:])
        return dynamic_m_tensor_signature(tensor, static_shape, dynamic_stride_dims=(0,))

    def dynamic_sfd_col_tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        shape = get_shape(tensor)
        if not _sf_is_physical(tensor):
            static_shape = (shape[0], shape[1], shape[2], shape[3], shape[5])
            return dynamic_m_tensor_signature(tensor, static_shape, dynamic_stride_dims=(2, 5))
        # physical C-contiguous form (e.g. JAX): M_rest at dim 2, M-dependent strides at dims 0-1
        static_shape = (shape[0], shape[1], None, *shape[3:])
        return dynamic_m_tensor_signature(tensor, static_shape, dynamic_stride_dims=(0, 1))

    # Deterministic dbias indexes by M-block, so its dim 0 follows valid_m and keying on the full
    # shape would recompile for every distinct token count -- the trap _dprob_n_slots documents
    # for dprob's slot extent, and _make_dbias_fake already keeps that extent symbolic. Key it
    # the way dprob is keyed: M dynamic, the rest static. Strides are (n_out, 1, 1) either way,
    # so they carry no M. Non-deterministic dbias is (expert_cnt, n_out, 1) and static already.
    dbias_signature = (
        dynamic_m_tensor_signature(dbias_kernel_tensor, tuple(get_shape(dbias_kernel_tensor)[1:]))
        if deterministic and dbias_kernel_tensor is not None
        else tensor_signature(dbias_kernel_tensor)
    )
    # Used on both arms, not just the non-full-dynamic one. Every other tensor may drop its shape
    # under full dynamic because the kernel compiles with symbolic extents, but _make_dbias_fake
    # frees only dim 0 -- dbias's n stays baked -- and nothing else in the dense key carries n
    # (b_tensor.shape[2] is l, and dprob_n_slots is too coarse: n=384 and n=512 both give 2).
    # Verified: with the shape dropped, n=384 then n=512 reuses the first kernel and CuTeDSL
    # rejects the call with "Mismatched dbias_tensor.shape[1] ... expected to be 384" (job 469460).
    # The kernel writes the scratch under determinism, so that is what the key has to describe.
    dprob_kernel_tensor = dprob_workspace if dprob_workspace is not None else dprob_tensor

    use_full_dynamic = is_dense and os.environ.get("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1") != "0"

    if is_dense:
        cache_key = (
            weight_mode,
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
            *(dynamic_m_tensor_signature(prob_tensor, (1, 1)) if not use_full_dynamic else dynamic_tensor_signature(prob_tensor)),
            *(
                dynamic_m_tensor_signature(dprob_kernel_tensor, tuple(dprob_kernel_tensor.shape[1:]))
                if not use_full_dynamic
                else dynamic_tensor_signature(dprob_kernel_tensor)
            ),
            # Explicit, because the signature above does not carry it under use_full_dynamic:
            # that path drops the shape entirely, yet the slot count is baked into the
            # compiled descriptor. n_out=512 and n_out=513 give 2 and 3 slots with identical
            # stride order, so without this the second call would reuse a kernel built for
            # the wrong dprob extent.
            dprob_n_slots,
            *dbias_signature,
            *(dynamic_m_tensor_signature(d_srelu_tensor, (n_out, 1)) if not use_full_dynamic else dynamic_tensor_signature(d_srelu_tensor)),
            *(dynamic_tensor_signature(sfb_tensor) if use_full_dynamic else tensor_signature(sfb_tensor)),
            *(dynamic_tensor_signature(sfd_col_d_srelu_tensor) if use_full_dynamic else tensor_signature(sfd_col_d_srelu_tensor)),
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
            vector_f32,
            m_aligned,
            discrete_col_sfd,
            use_dynamic_sched,
            use_dsrelu_reuse,
            deterministic,
            float(tanh_clamp_scale) if tanh_clamp_scale is not None else None,
        )
    else:
        cache_key = (
            weight_mode,
            *dynamic_m_tensor_signature(a_tensor, get_shape(a_tensor)[1:], dynamic_stride_dims=(2,)),
            b_shape,
            b_dtype,
            *dynamic_m_tensor_signature(c_tensor, get_shape(c_tensor)[1:], dynamic_stride_dims=(2,)),
            *dynamic_m_sf_signature(sfa_tensor),
            *tensor_signature(alpha_tensor),
            *dynamic_m_tensor_signature(prob_tensor, (1, 1)),
            *dynamic_m_tensor_signature(dprob_kernel_tensor, tuple(dprob_kernel_tensor.shape[1:])),
            *dbias_signature,
            *dynamic_m_tensor_signature(d_srelu_tensor, (n_out, 1), dynamic_stride_dims=(2,)),
            *dynamic_sfd_col_tensor_signature(sfd_col_d_srelu_tensor),
            *tensor_signature(norm_const_tensor),
            *tensor_signature(b_ptrs),
            *tensor_signature(sfb_ptrs),
            *tensor_signature(padded_offsets),
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
            use_dsrelu_reuse,
            deterministic,
            b_major,
            num_experts,
            float(tanh_clamp_scale) if tanh_clamp_scale is not None else None,
        )

    # ---- Cache lookup or create + compile ----
    if cache_key in _cache_of_GroupedGemmDsreluSm100Objects:
        _logger.debug("grouped_gemm_dsrelu_wrapper_sm100: Using cached object")
        api = _cache_of_GroupedGemmDsreluSm100Objects[cache_key]
    else:
        _logger.debug("grouped_gemm_dsrelu_wrapper_sm100: Creating new object")
        if is_dense:
            api = GroupedGemmDsreluSm100(
                sample_a=a_tensor,
                sample_c=c_tensor,
                sample_d_row=d_row_tensor,
                sample_d_col=d_col_tensor,
                sample_d_srelu=d_srelu_tensor,
                sample_sfa=sfa_tensor,
                sample_padded_offsets=padded_offsets,
                sample_alpha=alpha_tensor,
                sample_prob=prob_tensor,
                sample_dprob=dprob_tensor,
                sample_dbias=dbias_tensor,
                sample_dprob_workspace=dprob_workspace,
                sample_dbias_workspace=dbias_workspace,
                sample_b=b_tensor,
                sample_sfb=sfb_tensor,
                sample_sfd_row=sfd_row_tensor,
                sample_sfd_col=sfd_col_tensor,
                sample_sfd_col_d_srelu=sfd_col_d_srelu_tensor,
                sample_amax=amax_tensor,
                sample_norm_const=norm_const_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                discrete_col_sfd=discrete_col_sfd,
                use_dynamic_sched=use_dynamic_sched,
                use_dsrelu_reuse=use_dsrelu_reuse,
                deterministic=deterministic,
                tanh_clamp_scale=tanh_clamp_scale,
            )
        else:
            api = GroupedGemmDsreluSm100(
                sample_a=a_tensor,
                sample_c=c_tensor,
                sample_d_row=d_row_tensor,
                sample_d_col=d_col_tensor,
                sample_d_srelu=d_srelu_tensor,
                sample_sfa=sfa_tensor,
                sample_padded_offsets=padded_offsets,
                sample_alpha=alpha_tensor,
                sample_prob=prob_tensor,
                sample_dprob=dprob_tensor,
                sample_dbias=dbias_tensor,
                sample_dprob_workspace=dprob_workspace,
                sample_dbias_workspace=dbias_workspace,
                num_experts=num_experts,
                b_shape=b_shape,
                b_dtype=b_dtype,
                sample_sfd_row=sfd_row_tensor,
                sample_sfd_col=sfd_col_tensor,
                sample_sfd_col_d_srelu=sfd_col_d_srelu_tensor,
                sample_amax=amax_tensor,
                sample_norm_const=norm_const_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                discrete_col_sfd=discrete_col_sfd,
                b_major=b_major,
                use_dynamic_sched=use_dynamic_sched,
                use_dsrelu_reuse=use_dsrelu_reuse,
                deterministic=deterministic,
                tanh_clamp_scale=tanh_clamp_scale,
            )

        if not api.check_support():
            raise RuntimeError("Unsupported configuration")
        api.compile()
        _cache_of_GroupedGemmDsreluSm100Objects[cache_key] = api

    # ---- Execute ----
    if is_dense:
        api.execute(
            a_tensor=a_tensor,
            c_tensor=c_tensor,
            d_row_tensor=d_row_tensor,
            d_col_tensor=d_col_tensor,
            d_srelu_tensor=d_srelu_tensor,
            sfa_tensor=sfa_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            prob_tensor=prob_tensor,
            dprob_tensor=dprob_tensor,
            dbias_tensor=dbias_tensor,
            dprob_workspace_tensor=dprob_workspace,
            dbias_workspace_tensor=dbias_workspace,
            b_tensor=b_tensor,
            sfb_tensor=sfb_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            sfd_col_d_srelu_tensor=sfd_col_d_srelu_tensor,
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
            d_srelu_tensor=d_srelu_tensor,
            sfa_tensor=sfa_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            prob_tensor=prob_tensor,
            dprob_tensor=dprob_tensor,
            dbias_tensor=dbias_tensor,
            dprob_workspace_tensor=dprob_workspace,
            dbias_workspace_tensor=dbias_workspace,
            b_ptrs=b_ptrs,
            sfb_ptrs=sfb_ptrs,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            sfd_col_d_srelu_tensor=sfd_col_d_srelu_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            current_stream=current_stream,
        )

    return TupleDict(
        d_row_tensor=d_row_tensor,
        d_col_tensor=d_col_tensor,
        d_srelu_tensor=d_srelu_tensor,
        dprob_tensor=_reduce_dprob_slots(dprob_workspace, dprob_tensor, current_stream),
        dbias_tensor=_reduce_dbias_slots(dbias_workspace, dbias_tensor, current_stream, padded_offsets, cta_tile_m),
        amax_tensor=amax_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
        sfd_col_d_srelu_tensor=sfd_col_d_srelu_tensor,
    )


def _dsrelu_tensor_signature(tensor, *, dynamic_m: bool = False) -> tuple:
    """What a compiled BF16 kernel depends on: layout, dtype and device, with M left out when dynamic."""
    if tensor is None:
        return (None,)
    device = get_device(tensor)
    shape = get_shape(tensor)
    if dynamic_m:
        # execute() checks the full layout; the compiled kernel only depends on the row shape.
        return (shape[1:], tensor.dtype, device.type, device.index)
    return (shape, canonicalize_unit_dim_strides(shape, get_strides(tensor)), tensor.dtype, device.type, device.index)


def _grouped_gemm_dsrelu_bf16_call(
    *,
    a_tensor,
    b_tensor,
    c_tensor,
    padded_offsets,
    prob_tensor,
    dprob_tensor,
    b_ptrs,
    n,
    b_dtype,
    b_major,
    acc_dtype,
    d_dtype,
    cd_major,
    mma_tiler_mn,
    cluster_shape_mn,
    m_aligned,
    use_dynamic_sched,
    deterministic,
    tanh_clamp_scale,
    tanh_mode,
    generate_d_srelu,
    current_stream,
) -> TupleDict:
    """BF16 path of grouped_gemm_dsrelu_wrapper_sm100."""
    operation = "grouped_gemm_dsrelu_wrapper_sm100"
    if detect_framework(a_tensor) != "torch":
        # The public wrapper admits torch and JAX only; the BF16 backend has no JAX entry point.
        raise ValueError(_JAX_BF16_ERROR)
    # Rule 7: decline an unusable CuTe DSL by version before anything is allocated (the facade's
    # check_support() imports the BF16 kernel module behind the same gate).
    requirement = bf16_kernel_requirement_error(f"{operation} (BF16)")
    if requirement:
        raise ImportError(requirement)
    import torch

    for name, dtype, expected in (("acc_dtype", acc_dtype, cutlass.Float32), ("d_dtype", d_dtype, cutlass.BFloat16)):
        if dtype is not None and _convert_to_cutlass_data_type(dtype) is not expected:
            raise ValueError(f"{operation}: BF16 {name} must be {expected}, got {dtype}")
    if cd_major != "n":
        raise ValueError(f"cd_major must be 'n', got {cd_major}")
    if c_tensor is None:
        raise ValueError(f"{operation}: c_tensor (the BF16 pre-activation the forward saved) is required")
    if prob_tensor is None:
        raise ValueError(f"{operation}: prob_tensor is required (shape (valid_m, 1, 1), float32)")
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
    if deterministic is None:
        deterministic = torch.are_deterministic_algorithms_enabled()
    deterministic = bool(deterministic)

    device = a_tensor.device
    if current_stream is None:
        # The launch stream on the operands' device, fixed before anything is allocated: the outputs, the
        # dprob buffers, the scratch and the launch share it even when another device is current.
        current_stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    resolved_cluster = cluster_shape_mn if cluster_shape_mn is not None else ((2, 1) if mma_tiler_mn[0] == 256 else (1, 1))
    dprob_n_slots = _dprob_n_slots(n_out, mma_tiler_mn, resolved_cluster, True)
    # Allocated on the launch stream, as on the block-scaled path: the caching allocator hands out
    # blocks on the strength of the allocating stream's ordering.
    with _torch_stream_context(current_stream, device):
        d_row_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=torch.bfloat16, device=device)
        d_srelu_tensor = (
            torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=torch.bfloat16, device=device) if generate_d_srelu else None
        )
        if dprob_tensor is None:
            dprob_tensor = torch.zeros((valid_m, 1, 1), dtype=torch.float32, device=device)
        dprob_workspace = torch.zeros((valid_m, dprob_n_slots, 1), dtype=torch.float32, device=device) if deterministic else None
    outputs = TupleDict(
        d_row_tensor=d_row_tensor,
        d_col_tensor=None,
        d_srelu_tensor=d_srelu_tensor,
        dprob_tensor=dprob_tensor,
        dbias_tensor=None,
        amax_tensor=None,
        sfd_row_tensor=None,
        sfd_col_tensor=None,
        sfd_col_d_srelu_tensor=None,
    )
    if valid_m == 0:
        _logger.debug("grouped_gemm_dsrelu_wrapper_sm100: valid_m is zero, skipping kernel execution")
        return outputs

    cache_key = backend_cache_key(
        GroupedGemmBackend.BF16,
        weight_mode,
        _dsrelu_tensor_signature(a_tensor, dynamic_m=True),
        _dsrelu_tensor_signature(c_tensor, dynamic_m=True),
        _dsrelu_tensor_signature(prob_tensor, dynamic_m=True),
        _dsrelu_tensor_signature(dprob_tensor, dynamic_m=True),
        _dsrelu_tensor_signature(b_tensor),
        _dsrelu_tensor_signature(padded_offsets),
        _dsrelu_tensor_signature(b_ptrs),
        n_out,
        num_experts,
        b_shape,
        None if b_dtype is None else _convert_to_cutlass_data_type(b_dtype),
        b_major,
        tuple(mma_tiler_mn),
        None if cluster_shape_mn is None else tuple(cluster_shape_mn),
        m_aligned,
        use_dynamic_sched,
        deterministic,
        generate_d_srelu,
        tanh_clamp_scale,
        # The plain squared-ReLU evaluates no tanh: both modes share one entry and one kernel.
        None if tanh_clamp_scale is None else tanh_mode,
        int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0")),
    )
    api = _cache_of_GroupedGemmDsreluSm100Objects.get(cache_key)
    if api is None:
        _logger.debug("grouped_gemm_dsrelu_wrapper_sm100: creating a BF16 GroupedGemmDsreluSm100 object")
        # check_support() and compile() pin the operands' device themselves; this context only keeps
        # the cold path on that device.
        with device_context(device):
            api = GroupedGemmDsreluSm100(
                sample_a=a_tensor,
                sample_b=b_tensor,
                sample_c=c_tensor,
                sample_d_row=d_row_tensor,
                sample_d_srelu=d_srelu_tensor,
                sample_padded_offsets=padded_offsets,
                sample_prob=prob_tensor,
                sample_dprob=dprob_tensor,
                sample_dprob_workspace=dprob_workspace,
                num_experts=num_experts if is_discrete else None,
                b_shape=b_shape,
                b_dtype=b_dtype,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                m_aligned=m_aligned,
                b_major=b_major,
                use_dynamic_sched=use_dynamic_sched,
                deterministic=deterministic,
                tanh_clamp_scale=tanh_clamp_scale,
                tanh_mode=tanh_mode,
            )
            if not api.check_support():
                raise RuntimeError("Unsupported BF16 configuration")
            api.compile()
        _cache_of_GroupedGemmDsreluSm100Objects[cache_key] = api

    # Per-call scratch allocated on the launch stream (recipe R2): the cached API owns none.
    with wrapper_workspace("torch", api.scratch_workspace_bytes(), device, current_stream) as workspace:
        api._implementation.execute(
            a_tensor=a_tensor,
            c_tensor=c_tensor,
            d_row_tensor=d_row_tensor,
            padded_offsets=padded_offsets,
            prob_tensor=prob_tensor,
            dprob_tensor=dprob_tensor,
            b_tensor=b_tensor,
            b_ptrs=b_ptrs,
            d_srelu_tensor=d_srelu_tensor,
            dprob_workspace_tensor=dprob_workspace,
            current_stream=current_stream,
            workspace=workspace,
        )
    return outputs
