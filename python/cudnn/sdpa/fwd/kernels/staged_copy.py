# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Bitwise gather/scatter for the existing standalone conversion layouts."""

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
def _copy_kernel(srcs, dsts, src_strides, dst_strides, shapes: cutlass.Constexpr, items_per_thread: cutlass.Constexpr):
    block, role, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    for i in cutlass.range_constexpr(len(shapes)):
        if role == i:
            b, s, h, d = shapes[i]
            for item in cutlass.range_constexpr(items_per_thread):
                index = cutlass.Int64(block) * (_THREADS * items_per_thread) + cutlass.Int64(thread) + item * _THREADS
                if index < b * s * h * d:
                    dim = index % d
                    head = index // d % h
                    seq = index // (d * h) % s
                    batch = index // (d * h * s)
                    src = cutlass.make_array_view(cute.make_tensor(srcs[i], cute.make_layout(shapes[i], stride=src_strides[i])))
                    dst = cutlass.make_array_view(cute.make_tensor(dsts[i], cute.make_layout(shapes[i], stride=dst_strides[i])))
                    dst[batch, seq, head, dim] = src[batch, seq, head, dim]


_copy_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(srcs, dsts, src_strides, dst_strides, shapes: cutlass.Constexpr, items_per_thread: cutlass.Constexpr, stream: driver.CUstream):
    tile = _THREADS * items_per_thread
    blocks = max((b * s * h * d + tile - 1) // tile for b, s, h, d in shapes)
    _copy_kernel(srcs, dsts, src_strides, dst_strides, shapes, items_per_thread).launch(grid=(blocks, len(shapes), 1), block=(_THREADS, 1, 1), stream=stream)


@lru_cache(maxsize=128)
def compile_copy(shapes, widths, items_per_thread=1):
    """Shapes are a fixed plan contract; pointers and Int64 strides bind per call."""
    key = template_key(globals(), locals(), "compile_copy")
    elem = {1: cutlass.Uint8, 2: cutlass.Uint16, 4: cutlass.Uint32}
    ptrs = tuple(cute.runtime.make_ptr(elem[w], 16, cute.AddressSpace.gmem, assumed_align=w) for w in widths)
    strides = tuple((cutlass.Int64(0),) * 4 for _ in shapes)
    return compile_cached(
        _host,
        ptrs,
        ptrs,
        strides,
        strides,
        shapes,
        items_per_thread,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=key,
        symbol="frost_sdpa_staged_copy",
    )
