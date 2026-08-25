# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX-native (XLA custom call) entry point for the discrete-weight block-scaled
grouped GEMM GLU forward (SwiGLU/GeGLU), built on :func:`cudnn.jax.call`.

FP8 inputs only (the packed-fp4 uint8 container dtype is not expressible as JAX
arrays) and no bias (its (n, experts) column-major layout is likewise
inexpressible; ``bias`` is a compile-time ``None`` inside the adapter). This
discrete kernel is JAX-expressible where the contiguous block-scaled GLU kernel is
not: SFB arrives as per-expert base pointers and SFA/SFD travel in the physical
C-contiguous atom shape ``(1, MN', K', 32, 4, 4)`` — the kernel rebuilds every
scale-factor layout from the A/D shapes and consumes only the SF base pointers.
The per-expert B/SFB pointers travel as regular device arrays whose *values* are
raw addresses — the referenced weight/scale buffers are not visible to XLA, so the
caller must keep them alive (and unmoved) across every execution of the traced
computation. ``padded_offsets`` values cannot be host-validated under tracing;
malformed offsets are the caller's responsibility here (the eager wrapper
validates them).
"""

import os
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.nvgpu import OperandMajorMode

from cudnn.api_base import ceil_div
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import framework_dtype
from cudnn.jax import TensorSpec, call, gemm_operand_spec, neg_inf_init, zeros_init
from ...grouped.unfused.jax_api import _pointer_count, _prob_spec
from .discrete_B_blockscaled_grouped_gemm_glu_bias import BlockScaledDiscreteWeightGroupedGemmBiasKernel

# cache_key -> (kernel instance, max_active_clusters, workspace_bytes); reusing the
# instance keeps cutlass_call's compile cache warm (its FunctionSpec keys on the
# constexpr kwargs).
_kernel_cache: dict = {}

_fp8_dtypes = (cutlass.Float8E4M3FN, cutlass.Float8E5M2)
_c_dtypes = (cutlass.Float32, cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.Float8E5M2)
_d_dtypes = (cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.Float8E5M2)
_amax_dtypes = (cutlass.BFloat16, cutlass.Float16)

_JAX_FP4_ERROR = (
    "packed-fp4 inputs are not expressible as JAX arrays for this API "
    "(JAX has no packed fp4 dtype and the compiled kernel entry point requires float4_e2m1fn_x2 tensors); "
    "use torch tensors for FP4, or FP8 inputs for JAX"
)


def _sf_atom_spec() -> TensorSpec:
    # Physical C-contiguous (1, MN', K', 32, 4, 4) scale-factor atom allocation:
    # explicit row-major ranks because the unit dim makes leading-dim inference
    # ambiguous. The kernel rebuilds the SF layout from the GEMM shapes and consumes
    # only the base pointer, so no permuted view is needed.
    return TensorSpec(layout=(5, 4, 3, 2, 1, 0))


def _sf_byte_zeros_init(shape_dtype: jax.ShapeDtypeStruct) -> jax.Array:
    # float8_e8m0fnu has no representable zero; zero the raw bytes instead
    # (byte 0x00 decodes to 2^-127) via a free uint8 bitcast.
    return jnp.zeros(shape_dtype.shape, jnp.uint8).view(shape_dtype.dtype)


def _as_e8m0_array(scale: Any) -> Any:
    """Present a uint8 E8M0-bit-pattern array as float8_e8m0fnu (free bitcast, jit-safe)."""
    if _convert_to_cutlass_data_type(scale.dtype) is cutlass.Uint8:
        import ml_dtypes

        return scale.view(ml_dtypes.float8_e8m0fnu)
    return scale


@cute.jit
def _discrete_swiglu_adapter(
    stream,
    a,
    b_ptrs,
    sfb_ptrs,
    sfa,
    padded_offsets,
    alpha,
    prob,
    norm_const,
    c,
    d,
    d_col,
    sfd_row,
    sfd_col,
    amax,
    workspace,
    *,
    kernel,
    n,
    k,
    mac,
    has_amax,
    linear_offset,
    geglu_alpha,
    glu_clamp_max,
    glu_clamp_min,
):
    # Discrete-mode b/sfb are raw pointers to the device int64[] of per-expert base
    # addresses; the packed uint8 (or int64) input buffers recast for free.
    b_arg = cute.recast_ptr(b_ptrs.iterator, dtype=cutlass.Int64)
    sfb_arg = cute.recast_ptr(sfb_ptrs.iterator, dtype=cutlass.Int64)
    # amax exists only for 16-bit d (eager parity: fp8 d compiles with amax=None);
    # the dummy fp8-mode buffer is never passed to the kernel.
    amax_arg = None
    if cutlass.const_expr(has_amax):
        amax_arg = amax
    kernel(
        a=a,
        b_ptrs=b_arg,
        sfb_ptrs=sfb_arg,
        n=cutlass.Int32(n),
        k=cutlass.Int32(k),
        b_stride_size=cutlass.Int64(k),  # uniform k-major per-expert (n, k) weights
        b_major_mode=OperandMajorMode.K,
        workspace_ptr=workspace.iterator,
        c=c,
        d=d,
        d_col=d_col,
        sfa=sfa,
        sfd_row_tensor=sfd_row,
        sfd_col_tensor=sfd_col,
        amax_tensor=amax_arg,
        norm_const_tensor=norm_const,
        padded_offsets=padded_offsets,
        alpha=alpha,
        prob=prob,
        bias=None,
        max_active_clusters=mac,
        stream=stream,
        linear_offset=cutlass.Float32(linear_offset),
        geglu_alpha=cutlass.Float32(geglu_alpha),
        glu_clamp_max=cutlass.Float32(glu_clamp_max),
        glu_clamp_min=cutlass.Float32(glu_clamp_min),
    )


def discrete_grouped_gemm_swiglu_jax_sm100(
    a_tensor: Any,
    b_ptrs: Any,
    sfa_tensor: Any,
    sfb_ptrs: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    prob_tensor: Any,
    norm_const_tensor: Any,
    n: int,
    bias_tensor: Optional[Any] = None,
    c_dtype: Any = cutlass.BFloat16,
    d_dtype: Any = cutlass.BFloat16,
    acc_dtype: Any = cutlass.Float32,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 32,
    vector_f32: bool = False,
    discrete_col_sfd: bool = False,
    act_func: str = "swiglu",
    linear_offset: Optional[float] = None,
    geglu_alpha: float = 1.702,
    glu_clamp_max: float = 7.0,
    glu_clamp_min: float = -7.0,
    use_dynamic_sched: bool = False,
) -> dict:
    """Discrete-weight block-scaled grouped GEMM GLU forward as an XLA custom call.

    Same contract as the eager wrapper's FP8 JAX mode: A ``(m, k, 1)`` k-major
    C-contiguous fp8 (e4m3/e5m2), SFA in the physical C-contiguous E8M0 atom shape
    ``(1, m/128, ceil(ceil(k/sf_vec_size)/4), 32, 4, 4)`` (``float8_e8m0fnu``, or
    ``uint8`` bit patterns), ``padded_offsets (experts,)`` int32 cumulative
    256-aligned row offsets, ``alpha (experts,)`` float32, ``prob (m, 1, 1)``
    float32, ``norm_const (1,)`` float32, and ``b_ptrs``/``sfb_ptrs`` holding
    per-expert ``(n, k)`` k-major weight / SFB atom base addresses (packed
    little-endian uint8, 8 bytes per pointer — or int64 with x64 mode). ``n`` is
    the full weight N before the GLU split; ``d``/``d_col`` come back
    ``(m, n // 2, 1)``. The scalar GLU knobs (``linear_offset`` — defaulting per
    ``act_func`` — ``geglu_alpha``, ``glu_clamp_max``, ``glu_clamp_min``) are
    compile-time constants of the traced call. Rows at/past ``padded_offsets[-1]``
    come back zero-filled (the outputs are donated zero-initialized buffers).

    Returns a dict with the eager wrapper's keys: ``c_tensor (m, n, 1)``,
    ``d_tensor``/``d_col_tensor (m, n//2, 1)``, ``sfd_row_tensor``/``sfd_col_tensor``
    (physical E8M0 atom shape), and ``amax_tensor ((experts, 1) float32,
    -inf-initialized)`` — ``None`` unless ``d_dtype`` is bf16/fp16.

    Rejected for JAX (as in the eager wrapper): packed-fp4 inputs/outputs, bias,
    and the contiguous kernel's non-k-major weight layouts.
    """
    c_dtype = _convert_to_cutlass_data_type(c_dtype)
    d_dtype = _convert_to_cutlass_data_type(d_dtype)
    acc_dtype = _convert_to_cutlass_data_type(acc_dtype)

    if bias_tensor is not None:
        raise ValueError(
            "bias_tensor is not expressible as JAX arrays (its (n, experts) column-major layout has no row-major equivalent); omit bias for JAX inputs"
        )
    ab_dtype = _convert_to_cutlass_data_type(a_tensor.dtype)
    if ab_dtype in (cutlass.Uint8, cutlass.Float4E2M1FN):
        raise ValueError(_JAX_FP4_ERROR)
    if ab_dtype not in _fp8_dtypes:
        raise ValueError(f"a_tensor must be float8_e4m3fn or float8_e5m2, got {a_tensor.dtype}")

    if len(a_tensor.shape) != 3 or a_tensor.shape[2] != 1:
        raise ValueError(f"a_tensor must have shape (m, k, 1), got {tuple(a_tensor.shape)}")
    m, k, _ = a_tensor.shape
    if m % 256 != 0:
        raise ValueError(f"a_tensor M dimension must be 256-aligned, got {m}")
    if n is None or n <= 0 or n % 2 != 0:
        raise ValueError(f"n must be positive and even (gate+up combined width), got {n}")
    n_out = n // 2

    if c_dtype not in _c_dtypes:
        raise ValueError(f"c_dtype must be FP32, FP16, BF16, or FP8 for JAX (packed fp4 has no JAX dtype), got {c_dtype}")
    if d_dtype not in _d_dtypes:
        raise ValueError(f"d_dtype must be FP16, BF16, or FP8 for JAX (packed fp4 has no JAX dtype), got {d_dtype}")
    if acc_dtype is not cutlass.Float32:
        raise ValueError(f"acc_dtype must be float32, got {acc_dtype}")
    if sf_vec_size != 32:
        raise ValueError(f"fp8 inputs require sf_vec_size 32, got {sf_vec_size}")
    if act_func not in ("swiglu", "geglu"):
        raise ValueError(f"act_func must be 'swiglu' or 'geglu', got {act_func}")
    if linear_offset is None:
        linear_offset = 1.0 if act_func == "geglu" else 0.0

    sfa_tensor = _as_e8m0_array(sfa_tensor)
    if _convert_to_cutlass_data_type(sfa_tensor.dtype) is not cutlass.Float8E8M0FNU:
        raise ValueError(f"sfa_tensor must be float8_e8m0fnu (or uint8 bit patterns) for fp8 inputs, got {sfa_tensor.dtype}")
    rest_k = ceil_div(ceil_div(k, sf_vec_size), 4)
    expected_sfa = (1, ceil_div(m, 128), rest_k, 32, 4, 4)
    if tuple(sfa_tensor.shape) != expected_sfa:
        raise ValueError(f"sfa_tensor must use the physical C-contiguous atom shape {expected_sfa}, got {tuple(sfa_tensor.shape)}")

    expert_cnt = _pointer_count(b_ptrs)
    sfb_cnt = _pointer_count(sfb_ptrs, "sfb_ptrs")
    if sfb_cnt != expert_cnt:
        raise ValueError(f"sfb_ptrs length mismatch: expected {expert_cnt} pointers, got {sfb_cnt}")
    if expert_cnt <= 0 or expert_cnt > 1024:
        raise ValueError(f"expert count must be in [1, 1024], got {expert_cnt}")
    if tuple(padded_offsets.shape) != (expert_cnt,):
        raise ValueError(f"padded_offsets must have shape ({expert_cnt},), got {tuple(padded_offsets.shape)}")
    if _convert_to_cutlass_data_type(padded_offsets.dtype) is not cutlass.Int32:
        raise ValueError(f"padded_offsets must have dtype int32, got {padded_offsets.dtype}")
    if tuple(alpha_tensor.shape) != (expert_cnt,) or _convert_to_cutlass_data_type(alpha_tensor.dtype) is not cutlass.Float32:
        raise ValueError(f"alpha_tensor must be ({expert_cnt},) float32, got {tuple(alpha_tensor.shape)} {alpha_tensor.dtype}")
    if tuple(prob_tensor.shape) != (m, 1, 1) or _convert_to_cutlass_data_type(prob_tensor.dtype) is not cutlass.Float32:
        raise ValueError(f"prob_tensor must be ({m}, 1, 1) float32, got {tuple(prob_tensor.shape)} {prob_tensor.dtype}")
    if tuple(norm_const_tensor.shape) != (1,) or _convert_to_cutlass_data_type(norm_const_tensor.dtype) is not cutlass.Float32:
        raise ValueError(f"norm_const_tensor must be (1,) float32, got {tuple(norm_const_tensor.shape)} {norm_const_tensor.dtype}")

    use_2cta_instrs = mma_tiler_mn[0] == 256
    cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if use_2cta_instrs else (1, 1)))
    if mma_tiler_mn[1] != 256:
        raise ValueError(f"MMA tiler N must be 256, got {mma_tiler_mn[1]}")

    if not BlockScaledDiscreteWeightGroupedGemmBiasKernel.can_implement(
        ab_dtype,
        cutlass.Float8E8M0FNU,
        sf_vec_size,
        acc_dtype,
        d_dtype,
        use_2cta_instrs,
        tuple(mma_tiler_mn),
        cluster_shape_mn,
        m,
        n,
        k,
        expert_cnt,
        "k",
        "k",
        "n",
        BlockScaledDiscreteWeightGroupedGemmBiasKernel.FIX_PAD_SIZE,
    ):
        raise ValueError("Unsupported discrete grouped GEMM SwiGLU tile, cluster, alignment, or layout configuration")

    has_amax = d_dtype in _amax_dtypes

    cache_key = (
        expert_cnt,
        acc_dtype,
        tuple(mma_tiler_mn),
        cluster_shape_mn,
        sf_vec_size,
        vector_f32,
        discrete_col_sfd,
        act_func,
        use_dynamic_sched,
    )
    entry = _kernel_cache.get(cache_key)
    if entry is None:
        kernel = BlockScaledDiscreteWeightGroupedGemmBiasKernel(
            sf_vec_size=sf_vec_size,
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=vector_f32,
            generate_sfd=True,  # the fp8 JAX path always generates SFD (E8M0 scale factors)
            discrete_col_sfd=discrete_col_sfd,
            expert_cnt=expert_cnt,
            act_func=act_func,
            enable_bias=False,
            use_dynamic_sched=use_dynamic_sched,
        )
        overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]) - overlap_margin
        if mac <= 0:
            raise ValueError("max_active_clusters must be > 0 after applying CUDNNFE_CLUSTER_OVERLAP_MARGIN")
        entry = (kernel, mac, max(kernel.get_workspace_bytes(), 1))
        _kernel_cache[cache_key] = entry
    kernel, mac, workspace_bytes = entry

    sf_jax_dtype = framework_dtype(cutlass.Float8E8M0FNU, "jax")
    operand = gemm_operand_spec()
    sf_spec = _sf_atom_spec()
    c_out, d_out, d_col_out, sfd_row_out, sfd_col_out, amax_out, _workspace = call(
        _discrete_swiglu_adapter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((m, n, 1), framework_dtype(c_dtype, "jax")),
            jax.ShapeDtypeStruct((m, n_out, 1), framework_dtype(d_dtype, "jax")),
            jax.ShapeDtypeStruct((m, n_out, 1), framework_dtype(d_dtype, "jax")),
            jax.ShapeDtypeStruct((1, ceil_div(m, 128), ceil_div(ceil_div(n_out, sf_vec_size), 4), 32, 4, 4), sf_jax_dtype),
            jax.ShapeDtypeStruct((1, ceil_div(n_out, 128), ceil_div(ceil_div(m, sf_vec_size), 4), 32, 4, 4), sf_jax_dtype),
            jax.ShapeDtypeStruct((expert_cnt, 1), jnp.float32),
            jax.ShapeDtypeStruct((workspace_bytes,), jnp.uint8),
        ),
        input_spec=(operand, None, None, sf_spec, None, None, _prob_spec(), None),
        output_spec=(operand, operand, operand, sf_spec, sf_spec, TensorSpec(layout=(1, 0)), None),
        # All outputs donated: c/d/d_col/sfd for the unit-dim layout specs (and
        # defined bytes past the last offset); amax because the kernel accumulates
        # into it (atomic max over a -inf-initialized buffer, eager parity); the
        # workspace because the helper kernel writes the per-expert TMA descriptors
        # into it (XLA inputs are immutable).
        initialized_outputs={
            0: zeros_init,
            1: zeros_init,
            2: zeros_init,
            3: _sf_byte_zeros_init,
            4: _sf_byte_zeros_init,
            5: neg_inf_init if has_amax else zeros_init,
            6: zeros_init,
        },
        kernel=kernel,
        n=int(n),
        k=int(k),
        mac=mac,
        has_amax=has_amax,
        linear_offset=float(linear_offset),
        geglu_alpha=float(geglu_alpha),
        glu_clamp_max=float(glu_clamp_max),
        glu_clamp_min=float(glu_clamp_min),
    )(a_tensor, b_ptrs, sfb_ptrs, sfa_tensor, padded_offsets, alpha_tensor, prob_tensor, norm_const_tensor)

    return {
        "c_tensor": c_out,
        "d_tensor": d_out,
        "d_col_tensor": d_col_out,
        "amax_tensor": amax_out if has_amax else None,
        "sfd_row_tensor": sfd_row_out,
        "sfd_col_tensor": sfd_col_out,
    }
