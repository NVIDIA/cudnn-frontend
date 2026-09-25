# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX-native (XLA custom call) entry point for the discrete-weight block-scaled
grouped GEMM dGLU backward (dSwiGLU/dGeGLU), built on :func:`cudnn.jax.call`.

FP8 inputs only (the packed-fp4 uint8 container dtype is not expressible as JAX
arrays). This discrete kernel is JAX-expressible where the contiguous block-scaled
dGLU kernel is not: SFB arrives as per-expert base pointers and SFA/SFD travel in
the physical C-contiguous atom shape ``(1, MN', K', 32, 4, 4)`` — the kernel
rebuilds every scale-factor layout from the A/D shapes and consumes only the SF
base pointers. The per-expert B/SFB pointers travel as regular device arrays whose
*values* are raw addresses — the referenced weight/scale buffers are not visible
to XLA, so the caller must keep them alive (and unmoved) across every execution of
the traced computation. ``dprob`` and ``dbias`` are kernel-accumulated outputs, so
unlike the eager wrapper they are not caller-provided buffers here: both are
donated zero-initialized outputs of the custom call. ``padded_offsets`` values
cannot be host-validated under tracing; malformed offsets are the caller's
responsibility here (the eager wrapper validates them).
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
from ..swiglu.jax_api import _as_e8m0_array, _sf_atom_spec, _sf_byte_zeros_init, _JAX_FP4_ERROR
from .discrete_B_blockscaled_grouped_gemm_dglu_dbias import BlockScaledDiscreteWeightDgluDbiasGroupedGemmKernel

# cache_key -> (kernel instance, max_active_clusters, workspace_bytes); reusing the
# instance keeps cutlass_call's compile cache warm (its FunctionSpec keys on the
# constexpr kwargs).
_kernel_cache: dict = {}

_fp8_dtypes = (cutlass.Float8E4M3FN, cutlass.Float8E5M2)
_c_dtypes = (cutlass.Float32, cutlass.Float16, cutlass.BFloat16)
_d_dtypes = (cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.Float8E5M2)
_amax_dtypes = (cutlass.BFloat16, cutlass.Float16)


def _epilogue_identity(x):
    return x


def _epilogue_relu(x):
    return cute.where(x > 0, x, cute.full_like(x, 0))


def _epilogue_srelu(x):
    return cute.where(x > 0, x, cute.full_like(x, 0)) ** 2


_EPILOGUE_OPS = {
    None: _epilogue_identity,
    "none": _epilogue_identity,
    "identity": _epilogue_identity,
    "relu": _epilogue_relu,
    "srelu": _epilogue_srelu,
}


@cute.jit
def _discrete_dswiglu_adapter(
    stream,
    a,
    b_ptrs,
    sfb_ptrs,
    c,
    sfa,
    padded_offsets,
    alpha,
    beta,
    prob,
    norm_const,
    d_row,
    d_col,
    dprob,
    sfd_row,
    sfd_col,
    amax,
    dbias,
    workspace,
    *,
    kernel,
    n,
    k,
    mac,
    has_amax,
    has_dbias,
    epilogue,
    linear_offset,
    geglu_alpha,
    glu_clamp_max,
    glu_clamp_min,
):
    # Discrete-mode b/sfb are raw pointers to the device int64[] of per-expert base
    # addresses; the packed uint8 (or int64) input buffers recast for free.
    b_arg = cute.recast_ptr(b_ptrs.iterator, dtype=cutlass.Int64)
    sfb_arg = cute.recast_ptr(sfb_ptrs.iterator, dtype=cutlass.Int64)
    # amax exists only for 16-bit d and dbias only with generate_dbias (eager
    # parity: the other configurations compile with compile-time Nones); the
    # dummy placeholder buffers are never passed to the kernel.
    amax_arg = None
    if cutlass.const_expr(has_amax):
        amax_arg = amax
    dbias_arg = None
    if cutlass.const_expr(has_dbias):
        dbias_arg = dbias
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
        d=d_row,
        d_col=d_col,
        sfa=sfa,
        sfd_row_tensor=sfd_row,
        sfd_col_tensor=sfd_col,
        amax_tensor=amax_arg,
        norm_const_tensor=norm_const,
        padded_offsets=padded_offsets,
        alpha=alpha,
        beta=beta,
        prob=prob,
        dprob=dprob,
        linear_offset=cutlass.Float32(linear_offset),
        dbias_tensor=dbias_arg,
        max_active_clusters=mac,
        stream=stream,
        epilogue_op=epilogue,
        geglu_alpha=cutlass.Float32(geglu_alpha),
        glu_clamp_max=cutlass.Float32(glu_clamp_max),
        glu_clamp_min=cutlass.Float32(glu_clamp_min),
    )


def discrete_grouped_gemm_dswiglu_jax_sm100(
    a_tensor: Any,
    b_ptrs: Any,
    c_tensor: Any,
    sfa_tensor: Any,
    sfb_ptrs: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    beta_tensor: Any,
    prob_tensor: Any,
    norm_const_tensor: Any,
    n: int,
    generate_dbias: bool = False,
    d_dtype: Any = cutlass.BFloat16,
    acc_dtype: Any = cutlass.Float32,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 32,
    vector_f32: bool = False,
    discrete_col_sfd: bool = False,
    act_func: str = "dswiglu",
    epilogue_op: Optional[str] = None,
    linear_offset: Optional[float] = None,
    geglu_alpha: float = 1.702,
    glu_clamp_max: float = 7.0,
    glu_clamp_min: float = -7.0,
    use_dynamic_sched: bool = False,
) -> dict:
    """Discrete-weight block-scaled grouped GEMM dGLU backward as an XLA custom call.

    Same contract as the eager wrapper's FP8 JAX mode: A ``(m, k, 1)`` k-major
    C-contiguous fp8 (e4m3/e5m2) gradient input, C ``(m, 2n, 1)`` n-major forward
    activations (fp32/fp16/bf16), SFA in the physical C-contiguous E8M0 atom shape
    ``(1, m/128, ceil(ceil(k/sf_vec_size)/4), 32, 4, 4)`` (``float8_e8m0fnu``, or
    ``uint8`` bit patterns), ``padded_offsets (experts,)`` int32 cumulative
    256-aligned row offsets, ``alpha``/``beta (experts,)`` float32, ``prob
    (m, 1, 1)`` float32, ``norm_const (1,)`` float32, and ``b_ptrs``/``sfb_ptrs``
    holding per-expert ``(n, k)`` k-major weight / SFB atom base addresses (packed
    little-endian uint8, 8 bytes per pointer — or int64 with x64 mode). ``n`` is
    the per-expert weight N (half the activation width); ``d_row``/``d_col`` come
    back ``(m, 2n, 1)``. The scalar GLU knobs (``linear_offset`` — defaulting per
    ``act_func`` — ``geglu_alpha``, ``glu_clamp_max``, ``glu_clamp_min``) and
    ``epilogue_op`` are compile-time constants of the traced call. Rows at/past
    ``padded_offsets[-1]`` come back zero-filled (the outputs are donated
    zero-initialized buffers, matching the eager contract of a caller-zeroed
    ``dprob``); ``dprob`` accumulates through floating-point atomics, so its
    values are not bitwise-deterministic across runs.

    Returns a dict with the eager wrapper's keys: ``d_row_tensor``/``d_col_tensor
    (m, 2n, 1)``, ``dprob_tensor ((m, 1, 1) float32)``, ``dbias_tensor
    ((experts, 2n, 1) bfloat16, None unless generate_dbias)``, ``amax_tensor
    ((experts, 2, 1) float32, -inf-initialized, None unless d_dtype is
    bf16/fp16)``, and ``sfd_row_tensor``/``sfd_col_tensor`` (physical E8M0 atom
    shape; written only for fp8 ``d_dtype``, zero bytes otherwise).

    Rejected for JAX (as in the eager wrapper): packed-fp4 inputs/outputs and the
    contiguous kernel's non-k-major weight layouts.
    """
    d_dtype = _convert_to_cutlass_data_type(d_dtype)
    acc_dtype = _convert_to_cutlass_data_type(acc_dtype)

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
    if n is None or n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    n_out = 2 * n
    if tuple(c_tensor.shape) != (m, n_out, 1):
        raise ValueError(f"c_tensor must have shape ({m}, {n_out}, 1), got {tuple(c_tensor.shape)}")
    c_dtype = _convert_to_cutlass_data_type(c_tensor.dtype)
    if c_dtype not in _c_dtypes:
        raise ValueError(f"c_tensor must be FP32, FP16, or BF16, got {c_dtype}")
    if d_dtype not in _d_dtypes:
        raise ValueError(f"d_dtype must be FP16, BF16, or FP8 for JAX (packed fp4 has no JAX dtype), got {d_dtype}")
    if acc_dtype is not cutlass.Float32:
        raise ValueError(f"acc_dtype must be float32, got {acc_dtype}")
    if sf_vec_size != 32:
        raise ValueError(f"fp8 inputs require sf_vec_size 32, got {sf_vec_size}")
    if act_func not in ("dswiglu", "dgeglu"):
        raise ValueError(f"act_func must be 'dswiglu' or 'dgeglu', got {act_func}")
    if epilogue_op not in _EPILOGUE_OPS:
        raise ValueError(f"Invalid epilogue operation: {epilogue_op}. Valid: None, 'relu', 'srelu'")
    if linear_offset is None:
        linear_offset = 1.0 if act_func == "dgeglu" else 0.0

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
    if tuple(beta_tensor.shape) != (expert_cnt,) or _convert_to_cutlass_data_type(beta_tensor.dtype) is not cutlass.Float32:
        raise ValueError(f"beta_tensor must be ({expert_cnt},) float32, got {tuple(beta_tensor.shape)} {beta_tensor.dtype}")
    if tuple(prob_tensor.shape) != (m, 1, 1) or _convert_to_cutlass_data_type(prob_tensor.dtype) is not cutlass.Float32:
        raise ValueError(f"prob_tensor must be ({m}, 1, 1) float32, got {tuple(prob_tensor.shape)} {prob_tensor.dtype}")
    if tuple(norm_const_tensor.shape) != (1,) or _convert_to_cutlass_data_type(norm_const_tensor.dtype) is not cutlass.Float32:
        raise ValueError(f"norm_const_tensor must be (1,) float32, got {tuple(norm_const_tensor.shape)} {norm_const_tensor.dtype}")

    use_2cta_instrs = mma_tiler_mn[0] == 256
    cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if use_2cta_instrs else (1, 1)))
    if mma_tiler_mn[1] != 256:
        raise ValueError(f"MMA tiler N must be 256, got {mma_tiler_mn[1]}")

    if not BlockScaledDiscreteWeightDgluDbiasGroupedGemmKernel.can_implement(
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
        BlockScaledDiscreteWeightDgluDbiasGroupedGemmKernel.FIX_PAD_SIZE,
        act_func,
    ):
        raise ValueError("Unsupported discrete grouped GEMM dSwiGLU tile, cluster, alignment, or layout configuration")

    # SFD is generated only for the fp8-in/fp8-out E8M0 configuration; 16-bit d
    # accumulates per-expert amax instead (both mirror the eager wrapper).
    has_amax = d_dtype in _amax_dtypes
    if not (d_dtype in _fp8_dtypes) and discrete_col_sfd:
        discrete_col_sfd = False  # eager parity: ignored when SFD is not generated

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
        kernel = BlockScaledDiscreteWeightDgluDbiasGroupedGemmKernel(
            sf_vec_size=sf_vec_size,
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=vector_f32,
            discrete_col_sfd=discrete_col_sfd,
            expert_cnt=expert_cnt,
            act_func=act_func,
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
    prob_spec = _prob_spec()
    sf_spec = _sf_atom_spec()
    # amax/dbias keep the adapter arity fixed across configurations: when disabled,
    # a minimal placeholder buffer is donated (amax keeps its tiny real shape) and
    # the kernel receives a compile-time None instead.
    dbias_shape = (expert_cnt, n_out, 1) if generate_dbias else (1, 1, 1)
    outputs = call(
        _discrete_dswiglu_adapter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((m, n_out, 1), framework_dtype(d_dtype, "jax")),
            jax.ShapeDtypeStruct((m, n_out, 1), framework_dtype(d_dtype, "jax")),
            jax.ShapeDtypeStruct((m, 1, 1), jnp.float32),  # dprob (kernel-accumulated)
            jax.ShapeDtypeStruct((1, ceil_div(m, 128), ceil_div(ceil_div(n_out, sf_vec_size), 4), 32, 4, 4), sf_jax_dtype),
            jax.ShapeDtypeStruct((1, ceil_div(n_out, 128), ceil_div(ceil_div(m, sf_vec_size), 4), 32, 4, 4), sf_jax_dtype),
            jax.ShapeDtypeStruct((expert_cnt, 2, 1), jnp.float32),
            jax.ShapeDtypeStruct(dbias_shape, framework_dtype(cutlass.BFloat16, "jax")),
            jax.ShapeDtypeStruct((workspace_bytes,), jnp.uint8),
        ),
        input_spec=(operand, None, None, operand, sf_spec, None, None, None, prob_spec, None),
        output_spec=(operand, operand, prob_spec, sf_spec, sf_spec, operand, operand, None),
        # All outputs donated: d_row/d_col/sfd for the unit-dim layout specs (and
        # defined bytes past the last offset); dprob/dbias because the kernel
        # accumulates into them (atomic add) and expects zeroed buffers; amax
        # because the kernel accumulates into it (atomic max over a
        # -inf-initialized buffer, eager parity); the workspace because the helper
        # kernel writes the per-expert TMA descriptors into it (XLA inputs are
        # immutable).
        initialized_outputs={
            0: zeros_init,
            1: zeros_init,
            2: zeros_init,
            3: _sf_byte_zeros_init,
            4: _sf_byte_zeros_init,
            5: neg_inf_init if has_amax else zeros_init,
            6: zeros_init,
            7: zeros_init,
        },
        kernel=kernel,
        n=int(n),
        k=int(k),
        mac=mac,
        has_amax=has_amax,
        has_dbias=bool(generate_dbias),
        epilogue=_EPILOGUE_OPS[epilogue_op],
        linear_offset=float(linear_offset),
        geglu_alpha=float(geglu_alpha),
        glu_clamp_max=float(glu_clamp_max),
        glu_clamp_min=float(glu_clamp_min),
    )(a_tensor, b_ptrs, sfb_ptrs, c_tensor, sfa_tensor, padded_offsets, alpha_tensor, beta_tensor, prob_tensor, norm_const_tensor)

    d_row_out, d_col_out, dprob_out, sfd_row_out, sfd_col_out, amax_out, dbias_out, _workspace = outputs
    return {
        "d_row_tensor": d_row_out,
        "d_col_tensor": d_col_out,
        "dprob_tensor": dprob_out,
        "dbias_tensor": dbias_out if generate_dbias else None,
        "amax_tensor": amax_out if has_amax else None,
        "sfd_row_tensor": sfd_row_out,
        "sfd_col_tensor": sfd_col_out,
    }
