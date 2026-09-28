# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interleave the existing SM80 standalone RoPE table with full-precision trig."""

from functools import lru_cache
import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached, template_key

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
_THREADS = 256


@cute.kernel
def _kernel(source, target, strides, rows: cutlass.Constexpr, width: cutlass.Constexpr):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    index = cutlass.Int64(block) * _THREADS + cutlass.Int64(thread)
    if index < rows * width:
        src = cutlass.make_array_view(cute.make_tensor(source, cute.make_layout((rows, width), stride=strides)))
        dst = cutlass.make_array_view(cute.make_tensor(target, cute.make_layout((rows * width * 2,), stride=(1,))))
        angle = src[index // width, index % width]
        # Match Torch's float32 conversion followed by cos/sin, including
        # large-angle range reduction. Approximate PTX trig is not equivalent.
        dst[index * 2] = cute.math.cos(angle)
        dst[index * 2 + 1] = cute.math.sin(angle)


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(source, target, strides, rows: cutlass.Constexpr, width: cutlass.Constexpr, stream: driver.CUstream):
    _kernel(source, target, strides, rows, width).launch(grid=((rows * width + _THREADS - 1) // _THREADS, 1, 1), block=(_THREADS, 1, 1), stream=stream)


@lru_cache(maxsize=128)
def compile_table(rows, width):
    key = template_key(globals(), locals(), "compile_table")
    ptr = cute.runtime.make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4)
    return compile_cached(
        _host,
        ptr,
        ptr,
        (cutlass.Int64(0), cutlass.Int64(0)),
        rows,
        width,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=key,
        symbol="frost_sdpa_sm80_rope_table",
    )
