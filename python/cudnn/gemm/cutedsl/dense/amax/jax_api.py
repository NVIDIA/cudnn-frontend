# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX-native (XLA custom call) entry point for the blockscaled GEMM + amax kernel.

Unlike the eager path (``gemm_amax_wrapper_sm100`` with JAX arrays), this entry point
integrates with XLA via jax-tvm-ffi: the kernel is compiled with the TVM-FFI environment
stream (so it runs on XLA's compute stream, correctly ordered with surrounding ops), the
output buffers are managed by XLA through donation, and the call is ``jax.jit``-compatible.
No manual synchronization or block_until_ready is needed around it.

This module imports jax/jax_tvm_ffi at import time; it is only loaded when the
``gemm_amax_jax_sm100`` symbol is requested.
"""

from typing import Any, Tuple

import jax
import jax.numpy as jnp
import jax_tvm_ffi

import cutlass

from cudnn.api_base import TensorDesc
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import Device, framework_dtype
from .api import GemmAmaxSm100

# cache_key -> (registered XLA target name, GemmAmaxSm100, compiled tvm-ffi callable).
# The object references keep the compiled kernel alive alongside the global registration.
_registered_targets = {}


def _c_contiguous_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    strides, acc = [1] * len(shape), 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    return tuple(strides)


def _make_desc(shape: Tuple[int, ...], dtype: Any, name: str) -> TensorDesc:
    """Descriptor for a C-contiguous (row-major) JAX buffer, from shape/dtype only.

    Built directly from aval metadata so this works for jax.jit tracers as well as
    concrete arrays (tracers expose .shape/.dtype but no device or DLPack).
    """
    stride = _c_contiguous_strides(shape)
    return TensorDesc(
        dtype=_convert_to_cutlass_data_type(dtype),
        shape=shape,
        stride=stride,
        stride_order=TensorDesc._compute_stride_order(shape, stride),
        device=Device("cuda", 0),
        name=name,
    )


def gemm_amax_jax_sm100(
    a_tensor: Any,
    b_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    c_dtype: Any = cutlass.Float32,
    acc_dtype: Any = cutlass.Float32,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    sf_vec_size: int = 32,
) -> Tuple[Any, Any]:
    """Blockscaled GEMM + amax as an XLA custom call; usable eagerly or under jax.jit.

    Arguments are JAX arrays (or tracers): A (M, K, 1) and B (N, K, 1) k-major
    C-contiguous, SFA/SFB in the physical atom shape (1, MN', K', 32, 4, 4).
    Returns a plain ``(c_tensor, amax_tensor)`` tuple of fresh JAX arrays (a plain
    tuple, not a TupleDict, so the result is a valid JAX pytree under jit); C is n-major.

    Note: calling this eagerly re-traces the ffi_call each time -- prefer calling it
    from inside a jitted function in hot loops.
    """
    c_dtype = _convert_to_cutlass_data_type(c_dtype)
    acc_dtype = _convert_to_cutlass_data_type(acc_dtype)

    m, _, l = a_tensor.shape
    n, _, _ = b_tensor.shape
    if l != 1:
        raise ValueError("JAX inputs must have batch dim L == 1; batch-outermost (L-major) layouts are not expressible as JAX arrays")

    cache_key = (
        tuple(a_tensor.shape),
        tuple(b_tensor.shape),
        tuple(sfa_tensor.shape),
        tuple(sfb_tensor.shape),
        _convert_to_cutlass_data_type(a_tensor.dtype),
        _convert_to_cutlass_data_type(b_tensor.dtype),
        _convert_to_cutlass_data_type(sfa_tensor.dtype),
        _convert_to_cutlass_data_type(sfb_tensor.dtype),
        c_dtype,
        acc_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
    )

    entry = _registered_targets.get(cache_key)
    if entry is None:
        gemm = GemmAmaxSm100(
            sample_a=_make_desc(tuple(a_tensor.shape), a_tensor.dtype, "sample_a"),
            sample_b=_make_desc(tuple(b_tensor.shape), b_tensor.dtype, "sample_b"),
            sample_sfa=_make_desc(tuple(sfa_tensor.shape), sfa_tensor.dtype, "sample_sfa"),
            sample_sfb=_make_desc(tuple(sfb_tensor.shape), sfb_tensor.dtype, "sample_sfb"),
            sample_c=_make_desc((m, n, l), c_dtype, "sample_c"),
            sample_amax=_make_desc((1, 1, 1), cutlass.Float32, "sample_amax"),
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
        )
        assert gemm.check_support()
        compiled = gemm._compile_kernel(use_tvm_ffi_env_stream=True)
        target = f"cudnn.gemm_amax_sm100.{len(_registered_targets)}"
        # arg_spec=["args"]: only the operands are passed to the kernel; the result
        # buffers arrive as the two donated trailing operands (input_output_aliases below),
        # matching the kernel's destination-passing signature (a, b, sfa, sfb, c, amax).
        jax_tvm_ffi.register_ffi_target(target, compiled, arg_spec=["args"], platform="gpu", allow_cuda_graph=True)
        entry = (target, gemm, compiled)
        _registered_targets[cache_key] = entry
    target = entry[0]

    c_jax_dtype = framework_dtype(c_dtype, "jax")
    c_buf = jnp.zeros((m, n, l), dtype=c_jax_dtype)
    # Zero-init is a valid amax identity: the kernel accumulates max(|c|) >= 0 via a
    # signed-integer atomic max of non-negative float bit patterns.
    amax_buf = jnp.zeros((1, 1, 1), dtype=jnp.float32)

    c_tensor, amax_tensor = jax.ffi.ffi_call(
        target,
        (
            jax.ShapeDtypeStruct((m, n, l), c_jax_dtype),
            jax.ShapeDtypeStruct((1, 1, 1), jnp.float32),
        ),
        input_output_aliases={4: 0, 5: 1},
    )(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_buf, amax_buf)

    return c_tensor, amax_tensor
