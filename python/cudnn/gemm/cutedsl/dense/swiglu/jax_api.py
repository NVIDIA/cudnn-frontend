# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX-native (XLA custom call) entry point for the GEMM + SwiGLU kernels.

See dense/amax/jax_api.py for the integration pattern: the kernel variant is compiled
with the TVM-FFI environment stream (runs on XLA's compute stream), all outputs are
donated pre-initialized operands, and the call composes with jax.jit.
"""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

import cutlass

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import framework_dtype
from cudnn.gemm.cutedsl._jax_ffi import get_or_register_env_stream_target, make_row_major_desc as _make_desc
from .api import GemmSwigluSm100

_registered_targets = {}


def gemm_swiglu_jax_sm100(
    a_tensor: Any,
    b_tensor: Any,
    alpha: float = 1.0,
    ab12_dtype: Any = cutlass.Float32,
    c_dtype: Any = cutlass.Float16,
    acc_dtype: Any = cutlass.Float32,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    ### Quantize only arguments
    sfa_tensor: Optional[Any] = None,
    sfb_tensor: Optional[Any] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    ab12_stages: int = 4,
) -> Tuple[Any, Any]:
    """GEMM + SwiGLU as an XLA custom call; usable eagerly or under jax.jit.

    A (M, K, 1) and B (N, K, 1) are k-major C-contiguous JAX arrays (or tracers).
    Returns a plain ``(ab12_tensor, c_tensor)`` tuple of fresh n-major JAX arrays.
    ``alpha`` is a static (trace-time) parameter.

    Supports the non-quantized kernel only: the quantized kernel's compiled signature
    carries None-typed parameters the XLA FFI bridge cannot supply -- use
    ``gemm_swiglu_wrapper_sm100`` (eager) for blockscaled MXFP8 inputs from JAX.
    """
    ab12_dtype = _convert_to_cutlass_data_type(ab12_dtype)
    c_dtype = _convert_to_cutlass_data_type(c_dtype)
    acc_dtype = _convert_to_cutlass_data_type(acc_dtype)

    m, _, l = a_tensor.shape
    n, _, _ = b_tensor.shape
    if l != 1:
        raise ValueError("JAX inputs must have batch dim L == 1; batch-outermost (L-major) layouts are not expressible as JAX arrays")

    if sfa_tensor is not None or sfb_tensor is not None:
        # The quantized kernel's compiled signature carries explicit None-typed
        # parameters (amax/sfc/norm_const) that the XLA FFI bridge cannot supply.
        raise NotImplementedError(
            "gemm_swiglu_jax_sm100 currently supports the non-quantized kernel only; "
            "use gemm_swiglu_wrapper_sm100 (eager) for blockscaled MXFP8 inputs from JAX"
        )

    cache_key = (
        tuple(a_tensor.shape),
        tuple(b_tensor.shape),
        _convert_to_cutlass_data_type(a_tensor.dtype),
        _convert_to_cutlass_data_type(b_tensor.dtype),
        alpha,
        ab12_dtype,
        c_dtype,
        acc_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
    )

    def make_gemm():
        return GemmSwigluSm100(
            sample_a=_make_desc(tuple(a_tensor.shape), a_tensor.dtype, "sample_a"),
            sample_b=_make_desc(tuple(b_tensor.shape), b_tensor.dtype, "sample_b"),
            sample_ab12=_make_desc((m, n, l), ab12_dtype, "sample_ab12"),
            sample_c=_make_desc((m, n // 2, l), c_dtype, "sample_c"),
            alpha=alpha,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
        )

    # arg_spec: operands first (the donated output buffers land in the kernel's
    # destination-passing slots), then alpha as an XLA call attribute.
    target = get_or_register_env_stream_target(
        _registered_targets,
        cache_key,
        make_gemm,
        "cudnn.gemm_swiglu_sm100",
        arg_spec=("args", "attrs.alpha"),
    )

    ab12_jax_dtype = framework_dtype(ab12_dtype, "jax")
    c_jax_dtype = framework_dtype(c_dtype, "jax")
    ab12_buf = jnp.zeros((m, n, l), dtype=ab12_jax_dtype)
    c_buf = jnp.zeros((m, n // 2, l), dtype=c_jax_dtype)
    out_types = (
        jax.ShapeDtypeStruct((m, n, l), ab12_jax_dtype),
        jax.ShapeDtypeStruct((m, n // 2, l), c_jax_dtype),
    )

    # Kernel signature: (a, b, ab12, c, alpha[, stream from the TVM-FFI environment]).
    ab12_tensor, c_tensor = jax.ffi.ffi_call(
        target,
        out_types,
        input_output_aliases={2: 0, 3: 1},
    )(a_tensor, b_tensor, ab12_buf, c_buf, alpha=float(alpha))

    return ab12_tensor, c_tensor
