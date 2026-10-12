# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared casts from the existing SM80 backward auxiliary accumulators."""

from functools import lru_cache
import hashlib
import math
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached, template_key

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
_THREADS = 256


@cute.kernel
def _kernel(src, dst, strides, shape: cutlass.Constexpr):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    index = cutlass.Int64(block) * _THREADS + cutlass.Int64(thread)
    if index < math.prod(shape):
        a, b, c, d = shape
        dim = index % d
        col = index // d % c
        row = index // (d * c) % b
        batch = index // (d * c * b)
        source = cutlass.make_array_view(cute.make_tensor(src, cute.make_layout((math.prod(shape),), stride=(1,))))
        target = cutlass.make_array_view(cute.make_tensor(dst, cute.make_layout(shape, stride=strides)))
        target[batch, row, col, dim] = source[index].to(dst.dtype)


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(src, dst, strides, shape: cutlass.Constexpr, stream: driver.CUstream):
    _kernel(src, dst, strides, shape).launch(grid=((math.prod(shape) + _THREADS - 1) // _THREADS, 1, 1), block=(_THREADS, 1, 1), stream=stream)


@lru_cache(maxsize=128)
def compile_cast(shape, dtype):
    key = template_key(globals(), locals(), "compile_cast")
    elem = {"float16": cutlass.Float16, "bfloat16": cutlass.BFloat16, "float32": cutlass.Float32}[dtype]
    return compile_cached(
        _host,
        cute.runtime.make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        cute.runtime.make_ptr(elem, 16, cute.AddressSpace.gmem, assumed_align=elem.width // 8),
        (cutlass.Int64(0),) * 4,
        shape,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=key,
        symbol="frost_sdpa_sm80_bwd_aux_cast",
    )
