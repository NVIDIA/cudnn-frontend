# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX-native (XLA custom call) entry point for the BF16 SM100 grouped GEMM dGLU
backward (discrete weight mode), built on :func:`cudnn.jax.call`.

BF16 backend and discrete mode only: dense mode's expert-outermost strided B has
no row-major JAX equivalent, and the block-scaled backend's MMA-interleaved
scale-factor layouts cannot be presented as row-major JAX arrays. The per-expert
weight pointers travel as a regular device array whose *values* are raw addresses
— the referenced weight buffers are not visible to XLA, so the caller must keep
them alive (and unmoved) across every execution of the traced computation.
``dprob`` and ``dbias`` are kernel-accumulated outputs, so unlike the eager
wrapper they are not caller-provided buffers here: both are donated
zero-initialized outputs of the custom call. ``padded_offsets`` values cannot be
host-validated under tracing; malformed offsets are the caller's responsibility
here (the eager wrapper validates them).
"""

import os
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.nvgpu import OperandMajorMode

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import framework_dtype
from cudnn.jax import call, gemm_operand_spec, zeros_init
from ..moe_utils import MoEWeightMode
from ..unfused.jax_api import _pointer_count, _prob_spec
from .moe_grouped_gemm_dglu_dbias import MoEGroupedGemmDgluDbiasBf16Kernel

# cache_key -> (kernel instance, max_active_clusters, workspace_bytes); reusing the
# instance keeps cutlass_call's compile cache warm (its FunctionSpec keys on the
# constexpr kwargs).
_kernel_cache: dict = {}

_output_dtypes = (cutlass.BFloat16, cutlass.Float16, cutlass.Float32)

_JAX_BLOCK_SCALED_ERROR = (
    "the block-scaled grouped GEMM dGLU backend is not expressible as JAX arrays "
    "(its scale-factor tensors use an MMA-interleaved layout with no row-major equivalent); "
    "only the BF16 backend supports JAX inputs"
)


@cute.jit
def _grouped_dglu_bf16_adapter(stream, a, c, b_ptrs, padded_offsets, alpha, beta, prob, d, dprob, workspace, *, kernel, n, k, mac, linear_offset):
    # Discrete-mode b is a raw pointer to the device int64[] of per-expert base
    # addresses; the packed uint8 (or int64) input buffer recasts for free.
    b_arg = cute.recast_ptr(b_ptrs.iterator, dtype=cutlass.Int64)
    kernel(
        a=a,
        b=b_arg,
        n=cutlass.Int32(n),
        k=cutlass.Int32(k),
        b_stride_size=cutlass.Int64(k),  # uniform k-major per-expert (n, k) weights
        b_major_mode=OperandMajorMode.K,
        workspace_ptr=workspace.iterator,
        c=c,
        d=d,
        padded_offsets=padded_offsets,
        alpha=alpha,
        beta=beta,
        prob=prob,
        dprob=dprob,
        linear_offset=cutlass.Float32(linear_offset),
        dbias_tensor=None,
        max_active_clusters=mac,
        stream=stream,
    )


@cute.jit
def _grouped_dglu_bf16_dbias_adapter(stream, a, c, b_ptrs, padded_offsets, alpha, beta, prob, d, dprob, dbias, workspace, *, kernel, n, k, mac, linear_offset):
    b_arg = cute.recast_ptr(b_ptrs.iterator, dtype=cutlass.Int64)
    kernel(
        a=a,
        b=b_arg,
        n=cutlass.Int32(n),
        k=cutlass.Int32(k),
        b_stride_size=cutlass.Int64(k),  # uniform k-major per-expert (n, k) weights
        b_major_mode=OperandMajorMode.K,
        workspace_ptr=workspace.iterator,
        c=c,
        d=d,
        padded_offsets=padded_offsets,
        alpha=alpha,
        beta=beta,
        prob=prob,
        dprob=dprob,
        linear_offset=cutlass.Float32(linear_offset),
        dbias_tensor=dbias,
        max_active_clusters=mac,
        stream=stream,
    )


def grouped_gemm_dglu_jax_sm100(
    a_tensor: Any,
    c_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    beta_tensor: Any,
    b_ptrs: Any,
    n: int,
    prob_tensor: Any,
    d_dtype: Any = cutlass.BFloat16,
    acc_dtype: Any = cutlass.Float32,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    vector_f32: bool = False,
    act_func: str = "dswiglu",
    linear_offset: Optional[float] = None,
    generate_dbias: bool = False,
    use_dynamic_sched: bool = False,
) -> Tuple[Any, Any, Optional[Any]]:
    """BF16 grouped GEMM dGLU backward (discrete weights) as an XLA custom call.

    Same contract as the eager wrapper's BF16 discrete mode: A ``(m, k, 1)`` k-major
    C-contiguous bfloat16, C ``(m, 2n, 1)`` n-major forward pre-activations,
    ``padded_offsets (experts,)`` int32 cumulative 256-aligned row offsets, ``alpha``
    and ``beta`` ``(experts,)`` float32, ``prob (m, 1, 1)`` float32, and ``b_ptrs``
    holding per-expert ``(n, k)`` k-major bfloat16 weight base addresses (packed
    little-endian uint8, 8 bytes per pointer — or int64 with x64 mode). ``n`` is the
    per-expert weight N (half the pre-activation width). ``linear_offset`` defaults
    per ``act_func`` (1.0 for ``"dgeglu"``, 0.0 for ``"dswiglu"``) and is a
    compile-time constant of the traced call. Returns ``(d_row_tensor, dprob_tensor,
    dbias_tensor)`` with ``dbias_tensor`` None unless ``generate_dbias``; rows
    at/past ``padded_offsets[-1]`` come back zero-filled (the outputs are donated
    zero-initialized buffers, matching the eager contract of a caller-zeroed
    ``dprob``).
    """
    d_dtype = _convert_to_cutlass_data_type(d_dtype)
    acc_dtype = _convert_to_cutlass_data_type(acc_dtype)

    if len(a_tensor.shape) != 3 or a_tensor.shape[2] != 1:
        raise ValueError(f"a_tensor must have shape (m, k, 1), got {tuple(a_tensor.shape)}")
    m, k, _ = a_tensor.shape
    if m % 256 != 0:
        raise ValueError(f"a_tensor M dimension must be 256-aligned, got {m}")
    if _convert_to_cutlass_data_type(a_tensor.dtype) is not cutlass.BFloat16:
        raise ValueError(f"a_tensor must have dtype bfloat16, got {a_tensor.dtype}; " + _JAX_BLOCK_SCALED_ERROR)
    if n is None or n <= 0 or n % 32 != 0:
        raise ValueError(f"n must be positive and divisible by 32, got {n}")
    two_n = 2 * n
    if tuple(c_tensor.shape) != (m, two_n, 1):
        raise ValueError(f"c_tensor must have shape ({m}, {two_n}, 1), got {tuple(c_tensor.shape)}")
    c_dtype = _convert_to_cutlass_data_type(c_tensor.dtype)
    if c_dtype not in _output_dtypes or d_dtype not in _output_dtypes:
        raise ValueError(f"c_tensor/d_dtype must be BF16, FP16, or FP32, got {c_dtype}/{d_dtype}; " + _JAX_BLOCK_SCALED_ERROR)
    if acc_dtype is not cutlass.Float32:
        raise ValueError(f"acc_dtype must be float32, got {acc_dtype}")
    if act_func not in ("dswiglu", "dgeglu"):
        raise ValueError(f"act_func must be 'dswiglu' or 'dgeglu', got {act_func}")
    if linear_offset is None:
        linear_offset = 1.0 if act_func == "dgeglu" else 0.0

    expert_cnt = _pointer_count(b_ptrs)
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

    use_2cta_instrs = mma_tiler_mn[0] == 256
    cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if use_2cta_instrs else (1, 1)))

    if not MoEGroupedGemmDgluDbiasBf16Kernel.can_implement(
        cutlass.BFloat16,
        c_dtype,
        d_dtype,
        acc_dtype,
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
        MoEGroupedGemmDgluDbiasBf16Kernel.FIX_PAD_SIZE,
        act_func,
    ):
        raise ValueError("Unsupported BF16 grouped GEMM dGLU tile, cluster, alignment, or layout configuration")

    cache_key = (
        expert_cnt,
        c_dtype,
        d_dtype,
        acc_dtype,
        tuple(mma_tiler_mn),
        cluster_shape_mn,
        vector_f32,
        act_func,
        use_dynamic_sched,
    )
    entry = _kernel_cache.get(cache_key)
    if entry is None:
        kernel = MoEGroupedGemmDgluDbiasBf16Kernel(
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=vector_f32,
            expert_cnt=expert_cnt,
            weight_mode=MoEWeightMode.DISCRETE,
            use_dynamic_sched=use_dynamic_sched,
            act_func=act_func,
        )
        overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]) - overlap_margin
        if mac <= 0:
            raise ValueError("max_active_clusters must be > 0 after applying CUDNNFE_CLUSTER_OVERLAP_MARGIN")
        entry = (kernel, mac, max(kernel.get_workspace_bytes(), 1))
        _kernel_cache[cache_key] = entry
    kernel, mac, workspace_bytes = entry

    operand = gemm_operand_spec()
    prob_spec = _prob_spec()
    output_shape_dtype = [
        jax.ShapeDtypeStruct((m, two_n, 1), framework_dtype(d_dtype, "jax")),
        jax.ShapeDtypeStruct((m, 1, 1), jnp.float32),  # dprob (kernel-accumulated)
    ]
    output_spec = [operand, prob_spec]
    if generate_dbias:
        output_shape_dtype.append(jax.ShapeDtypeStruct((expert_cnt, two_n, 1), framework_dtype(cutlass.BFloat16, "jax")))
        output_spec.append(operand)
    output_shape_dtype.append(jax.ShapeDtypeStruct((workspace_bytes,), jnp.uint8))
    output_spec.append(None)

    results = call(
        _grouped_dglu_bf16_dbias_adapter if generate_dbias else _grouped_dglu_bf16_adapter,
        output_shape_dtype=tuple(output_shape_dtype),
        input_spec=(operand, operand, None, None, None, None, prob_spec),
        output_spec=tuple(output_spec),
        # All outputs donated: d for the trailing-unit-dim layout spec (and defined
        # bytes past the last offset); dprob/dbias because the kernel accumulates
        # into them (atomic add) and expects zeroed buffers; the workspace because
        # the helper kernel writes the per-expert TMA descriptors into it (XLA
        # inputs are immutable).
        initialized_outputs={index: zeros_init for index in range(len(output_shape_dtype))},
        kernel=kernel,
        n=int(n),
        k=int(k),
        mac=mac,
        linear_offset=float(linear_offset),
    )(a_tensor, c_tensor, b_ptrs, padded_offsets, alpha_tensor, beta_tensor, prob_tensor)

    if generate_dbias:
        d_row_tensor, dprob_tensor, dbias_tensor, _workspace = results
        return d_row_tensor, dprob_tensor, dbias_tensor
    d_row_tensor, dprob_tensor, _workspace = results
    return d_row_tensor, dprob_tensor, None
