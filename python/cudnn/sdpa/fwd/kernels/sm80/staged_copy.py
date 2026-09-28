# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise SM80 input gathers with zero padding to the vector/flavor width."""

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
def _kernel(srcs, dsts, strides, shapes: cutlass.Constexpr, t_q: cutlass.Int64, t_kv: cutlass.Int64):
    block, role, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    index = cutlass.Int64(block) * _THREADS + cutlass.Int64(thread)
    for i in cutlass.range_constexpr(len(shapes)):
        if role == i:
            b, s, h, d, padded_d = shapes[i]
            if cutlass.const_expr(s == -1):
                s = t_q
            elif cutlass.const_expr(s == -2):
                s = t_kv
            if index < b * s * h * padded_d:
                dim = index % padded_d
                head = index // padded_d % h
                seq = index // (padded_d * h) % s
                batch = index // (padded_d * h * s)
                src = cutlass.make_array_view(cute.make_tensor(srcs[i], cute.make_layout((b, s, h, d), stride=strides[i])))
                dst = cutlass.make_array_view(cute.make_tensor(dsts[i], cute.make_layout((b * s * h * padded_d,), stride=(1,))))
                if dim < d:
                    dst[index] = src[batch, seq, head, dim]
                else:
                    dst[index] = cutlass.Uint16(0)


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(srcs, dsts, strides, shapes: cutlass.Constexpr, stream: driver.CUstream):
    blocks = max((b * s * h * padded_d + _THREADS - 1) // _THREADS for b, s, h, d, padded_d in shapes)
    _kernel(srcs, dsts, strides, shapes, cutlass.Int64(0), cutlass.Int64(0)).launch(grid=(blocks, len(shapes), 1), block=(_THREADS, 1, 1), stream=stream)


@cute.jit
def _packed_host(srcs, dsts, strides, t_q: cutlass.Int64, t_kv: cutlass.Int64, shapes: cutlass.Constexpr, stream: driver.CUstream):
    words = cutlass.Int64(0)
    for i in cutlass.range_constexpr(len(shapes)):
        b, seq, h, d, padded_d = shapes[i]
        tokens = t_q if cutlass.const_expr(seq == -1) else t_kv
        words = cute.math.max(words, tokens * h * padded_d)
    # Only the packed token counts vary at execute. Keep head/flavor widths
    # static in the device kernel so dense copies do not pay dynamic Int64
    # division for every element just to share this host with packed copies.
    _kernel(srcs, dsts, strides, shapes, t_q, t_kv).launch(grid=((words + _THREADS - 1) // _THREADS, len(shapes), 1), block=(_THREADS, 1, 1), stream=stream)


@lru_cache(maxsize=128)
def compile_gather(shapes, *, packed=False):
    key = template_key(globals(), locals(), "compile_gather")
    ptrs = tuple(cute.runtime.make_ptr(cutlass.Uint16, 16, cute.AddressSpace.gmem, assumed_align=2) for _ in shapes)
    strides = tuple((cutlass.Int64(0),) * 4 for _ in shapes)
    args = (ptrs, ptrs, strides, cutlass.Int64(0), cutlass.Int64(0), shapes) if packed else (ptrs, ptrs, strides, shapes)
    return compile_cached(
        _packed_host if packed else _host,
        *args,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=key,
        symbol="frost_sdpa_sm80_staged_gather",
    )
