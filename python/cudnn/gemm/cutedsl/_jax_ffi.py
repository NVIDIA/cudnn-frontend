# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the jax.jit-compatible entry points of the CuTeDSL GEMM APIs.

These entry points integrate with XLA via jax-tvm-ffi: the kernel variant is compiled
with the TVM-FFI environment stream (so it runs on XLA's compute stream), all outputs
are donated pre-initialized operands (input_output_aliases + arg_spec=["args"]) matching
the kernels' destination-passing signatures, and calls compose with jax.jit.

This module imports jax/jax_tvm_ffi at import time; it is only loaded from the
per-API jax_api modules, which are lazily exported.
"""

from typing import Any, Callable, Tuple

import jax_tvm_ffi

from cudnn.api_base import TensorDesc
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import Device


def c_contiguous_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    strides, acc = [1] * len(shape), 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    return tuple(strides)


def make_row_major_desc(shape: Tuple[int, ...], dtype: Any, name: str) -> TensorDesc:
    """Descriptor for a C-contiguous (row-major) JAX buffer, from shape/dtype only.

    Built from aval metadata so this works for jax.jit tracers as well as concrete
    arrays (tracers expose .shape/.dtype but no device or DLPack).
    """
    shape = tuple(shape)
    stride = c_contiguous_strides(shape)
    return TensorDesc(
        dtype=_convert_to_cutlass_data_type(dtype),
        shape=shape,
        stride=stride,
        stride_order=TensorDesc._compute_stride_order(shape, stride),
        device=Device("cuda", 0),
        name=name,
    )


def get_or_register_env_stream_target(
    registry: dict,
    cache_key: Any,
    make_gemm: Callable[[], Any],
    target_prefix: str,
    arg_spec: Tuple[str, ...] = ("args",),
) -> str:
    """Compile the env-stream kernel variant for cache_key (once) and register it as an
    XLA FFI target; return the registered target name.

    The registry keeps (target, gemm, compiled) so the compiled kernel stays alive
    alongside the global registration.
    """
    entry = registry.get(cache_key)
    if entry is None:
        gemm = make_gemm()
        assert gemm.check_support()
        compiled = gemm._compile_kernel(use_tvm_ffi_env_stream=True)
        target = f"{target_prefix}.{len(registry)}"
        jax_tvm_ffi.register_ffi_target(target, compiled, arg_spec=list(arg_spec), platform="gpu", allow_cuda_graph=True)
        registry[cache_key] = (target, gemm, compiled)
        entry = registry[cache_key]
    return entry[0]
