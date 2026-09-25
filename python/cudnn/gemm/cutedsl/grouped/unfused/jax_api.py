# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX-native (XLA custom call) entry point for the unfused BF16 grouped GEMM
(discrete weight mode), built on :func:`cudnn.jax.call`.

Discrete mode only (dense mode's expert-outermost strided B has no row-major JAX
equivalent) and no bias (its (n, experts) column-major layout is likewise
inexpressible; ``bias`` is a compile-time ``None`` inside the adapter). The
per-expert weight pointers travel as a regular device array whose *values* are
raw addresses — the referenced weight buffers are not visible to XLA, so the
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

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import framework_dtype
from cudnn.jax import TensorSpec, call, gemm_operand_spec
from ..moe_utils import MoEWeightMode
from .moe_grouped_gemm import MoEGroupedGemmBf16Kernel

# cache_key -> (kernel instance, max_active_clusters, workspace_bytes); reusing the
# instance keeps cutlass_call's compile cache warm (its FunctionSpec keys on the
# constexpr kwargs).
_kernel_cache: dict = {}

_output_dtypes = (cutlass.BFloat16, cutlass.Float16, cutlass.Float32)


def _prob_spec() -> TensorSpec:
    # (m, 1, 1) with m innermost: explicit ranks because trailing unit dims make
    # leading-dim inference ambiguous
    return TensorSpec(layout=(0, 1, 2))


@cute.jit
def _grouped_bf16_adapter(stream, a, b_ptrs, padded_offsets, alpha, prob, d, c, workspace, *, kernel, n, k, mac):
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
        bias=None,
        prob=prob,
        max_active_clusters=mac,
        stream=stream,
    )


def _pointer_count(b_ptrs: Any, name: str = "b_ptrs") -> int:
    """Tracing-safe pointer-array shape/dtype check; returns the pointer count."""
    shape = tuple(b_ptrs.shape)
    if len(shape) != 1:
        raise ValueError(f"{name} must be 1-D, got shape={shape}")
    dtype = _convert_to_cutlass_data_type(b_ptrs.dtype)
    if dtype is cutlass.Int64:
        return shape[0]
    if dtype is cutlass.Uint8:
        if shape[0] % 8 != 0:
            raise ValueError(f"{name} packed uint8 length must be a multiple of 8, got {shape[0]}")
        return shape[0] // 8
    raise ValueError(f"{name} must be int64 (or, without x64 mode, packed uint8), got {b_ptrs.dtype}")


def grouped_gemm_jax_sm100(
    a_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    b_ptrs: Any,
    n: int,
    prob_tensor: Any,
    c_dtype: Any = cutlass.BFloat16,
    d_dtype: Any = cutlass.BFloat16,
    acc_dtype: Any = cutlass.Float32,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    vector_f32: bool = False,
    generate_c: bool = False,
    use_dynamic_sched: bool = False,
) -> Tuple[Any, Optional[Any]]:
    """Unfused BF16 grouped GEMM (discrete weights) as an XLA custom call.

    Same contract as the eager wrapper's discrete mode: A ``(m, k, 1)`` k-major
    C-contiguous bfloat16, ``padded_offsets (experts,)`` int32 cumulative 256-aligned
    row offsets, ``alpha (experts,)`` float32, ``prob (m, 1, 1)`` float32, and
    ``b_ptrs`` holding per-expert ``(n, k)`` k-major bfloat16 weight base addresses
    (packed little-endian uint8, 8 bytes per pointer — or int64 with x64 mode).
    Returns ``(d_tensor, c_tensor)`` with ``c_tensor`` None unless ``generate_c``.
    Rows at/past ``padded_offsets[-1]`` are unspecified.
    """
    c_dtype = _convert_to_cutlass_data_type(c_dtype)
    d_dtype = _convert_to_cutlass_data_type(d_dtype)
    acc_dtype = _convert_to_cutlass_data_type(acc_dtype)

    if len(a_tensor.shape) != 3 or a_tensor.shape[2] != 1:
        raise ValueError(f"a_tensor must have shape (m, k, 1), got {tuple(a_tensor.shape)}")
    m, k, _ = a_tensor.shape
    if m % 256 != 0:
        raise ValueError(f"a_tensor M dimension must be 256-aligned, got {m}")
    if _convert_to_cutlass_data_type(a_tensor.dtype) is not cutlass.BFloat16:
        raise ValueError(f"a_tensor must have dtype bfloat16, got {a_tensor.dtype}")
    if n is None or n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if c_dtype not in _output_dtypes or d_dtype not in _output_dtypes:
        raise ValueError(f"c_dtype/d_dtype must be BF16, FP16, or FP32, got {c_dtype}/{d_dtype}")
    if acc_dtype is not cutlass.Float32:
        raise ValueError(f"acc_dtype must be float32, got {acc_dtype}")

    expert_cnt = _pointer_count(b_ptrs)
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

    use_2cta_instrs = mma_tiler_mn[0] == 256
    cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if use_2cta_instrs else (1, 1)))

    if not MoEGroupedGemmBf16Kernel.can_implement(
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
        MoEGroupedGemmBf16Kernel.FIX_PAD_SIZE,
    ):
        raise ValueError("Unsupported BF16 grouped GEMM tile, cluster, alignment, or layout configuration")

    cache_key = (
        expert_cnt,
        c_dtype,
        d_dtype,
        acc_dtype,
        tuple(mma_tiler_mn),
        cluster_shape_mn,
        vector_f32,
        generate_c,
        use_dynamic_sched,
    )
    entry = _kernel_cache.get(cache_key)
    if entry is None:
        kernel = MoEGroupedGemmBf16Kernel(
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=vector_f32,
            generate_c=generate_c,
            enable_bias=False,
            expert_cnt=expert_cnt,
            weight_mode=MoEWeightMode.DISCRETE,
            use_dynamic_sched=use_dynamic_sched,
        )
        overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]) - overlap_margin
        if mac <= 0:
            raise ValueError("max_active_clusters must be > 0 after applying CUDNNFE_CLUSTER_OVERLAP_MARGIN")
        entry = (kernel, mac, max(kernel.get_workspace_bytes(), 1))
        _kernel_cache[cache_key] = entry
    kernel, mac, workspace_bytes = entry

    operand = gemm_operand_spec()
    d_tensor, c_tensor, _workspace = call(
        _grouped_bf16_adapter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((m, n, 1), framework_dtype(d_dtype, "jax")),
            jax.ShapeDtypeStruct((m, n, 1), framework_dtype(c_dtype, "jax")),
            jax.ShapeDtypeStruct((workspace_bytes,), jnp.uint8),
        ),
        input_spec=(operand, None, None, None, _prob_spec()),
        output_spec=(operand, operand, None),
        # Only zero what the kernel does not write. Zero-filling an output the
        # kernel overwrites is a full-size device write on every dispatch, and it
        # scales with the output -- the dominant host-visible cost of the JAX path.
        # Nothing here qualifies: the kernel writes d/c over the addressed rows and the
        # descriptor helper writes the workspace before the kernel reads it. Rows at or
        # past padded_offsets[-1] are consequently unspecified, matching what the torch
        # wrapper has always returned from empty_strided.
        kernel=kernel,
        n=int(n),
        k=int(k),
        mac=mac,
    )(a_tensor, b_ptrs, padded_offsets, alpha_tensor, prob_tensor)

    return d_tensor, (c_tensor if generate_c else None)
