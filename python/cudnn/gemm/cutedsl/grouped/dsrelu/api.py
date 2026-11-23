# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unified API for Grouped GEMM dSReLU Backward Kernel (SM100+)

This module provides a single API class that supports both contiguous (dense)
and discrete weight modes for block-scaled grouped GEMM with dSReLU activation
gradient in MoE (Mixture of Experts) workloads.

Dense mode
    All expert weights are packed contiguously in a 3-D tensor (N, K, L).
    Callers supply ``sample_b`` and ``sample_sfb``.

Discrete mode
    Each expert has its own memory allocation.  Callers supply
    ``num_experts``, ``b_shape``, ``b_dtype``, and per-expert pointer arrays
    at execution time.
"""

from __future__ import annotations

from ..backend_utils import _torch_stream_context
from ..moe_utils import MoEWeightMode
from cuda.bindings import driver as cuda
import logging
import os
from typing import Tuple, Optional

import cutlass

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import TupleDict, ceil_div
from cudnn.gemm.cutedsl.grouped.unfused._bf16_api import _validate_pointer_tensor
from cudnn.tensor_adapter import (
    detect_framework,
    framework_dtype,
    get_shape,
    get_strides,
)


from ._blockscaled_api import (
    GroupedGemmDsreluSm100,
    _cta_tile_m,
    _dprob_n_slots,
    _reduce_dbias_slots,
    _reduce_dprob_slots,
)

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
) -> TupleDict:
    """Convenience wrapper for grouped GEMM dSReLU backward operation.

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
        use_dsrelu_reuse: Reuse relu(C)^2 (or, when clamped, c^2) between d_srelu and dprob
        tanh_clamp_scale: Optional soft-clamp scale ``s``, must match the forward pass that
            produced ``c_tensor``. Must be finite and > 0. ``None`` (default) is
            bit-identical to today's unclamped DSRELU.
        deterministic: Make dprob and dbias bit-exact run to run; raises NotImplementedError
            for any framework other than torch. Every returned tensor keeps its usual shape and
            dtype -- the slot workspaces and the fixed-order reductions into them are internal to
            this wrapper. ``None`` (default) follows ``torch.use_deterministic_algorithms``.
            See docs/operations/gemm_fusions/grouped_gemm_dsrelu.md#deterministic-dprob.
        current_stream: CUDA stream

    Returns:
        TupleDict with keys: d_row_tensor, d_col_tensor, dprob_tensor,
            dbias_tensor, amax_tensor, sfd_row_tensor, sfd_col_tensor
    """
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
            d_srelu_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_torch_dtype, device=a_tensor.device)
    else:
        # n-major C-contiguous; the extent-1 batch dim's stride is unobservable by the kernel.
        d_jax_dtype = framework_dtype(d_dtype, "jax")
        d_row_tensor = _jax_alloc(lambda: jnp.empty((valid_m, n_out, 1), dtype=d_jax_dtype, device=a_tensor.device))
        d_col_tensor = _jax_alloc(lambda: jnp.empty((valid_m, n_out, 1), dtype=d_jax_dtype, device=a_tensor.device))
        d_srelu_tensor = _jax_alloc(lambda: jnp.empty((valid_m, n_out, 1), dtype=d_jax_dtype, device=a_tensor.device))

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
                sfd_col_d_srelu_tensor = torch.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)
        else:
            # Physical C-contiguous atom allocations: JAX cannot express the permuted view;
            # the kernel rebuilds the SF layout from the GEMM shapes and consumes only the
            # SF base pointer, so the physical form is byte-identical.
            sfd_row_tensor = _jax_alloc(lambda: jnp.empty(mma_shape_row, dtype=sf_dtype, device=a_tensor.device))
            sfd_col_tensor = _jax_alloc(lambda: jnp.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device))
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
