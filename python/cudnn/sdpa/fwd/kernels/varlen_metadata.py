# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the shared SDPA torch wrapper's lengths and element offsets together."""

from functools import lru_cache
import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached, template_key

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


@cute.kernel
def _kernel(q, kv, offsets, lengths, n, prefix_strides, token_strides, offset_pitch, length_pitch, n_q: cutlass.Constexpr):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    i = cutlass.Int64(block) * 256 + cutlass.Int64(thread)
    if i <= n:
        q_view = cutlass.make_array_view(cute.make_tensor(q, cute.make_layout((n * prefix_strides[0] + 1,), stride=(1,))))
        kv_view = cutlass.make_array_view(cute.make_tensor(kv, cute.make_layout((n * prefix_strides[1] + 1,), stride=(1,))))
        out = cutlass.make_array_view(cute.make_tensor(offsets, cute.make_layout((len(token_strides), n + 1), stride=(offset_pitch, 1))))
        # Widen the prefix before multiplication, including int32 input columns.
        # Array rank-one indexing is flat, including one-element tuples.
        q_offset, kv_offset = i * prefix_strides[0], i * prefix_strides[1]
        q_value, kv_value = cutlass.Int64(q_view[q_offset]), cutlass.Int64(kv_view[kv_offset])
        for role in cutlass.range_constexpr(len(token_strides)):
            if cutlass.const_expr(role < n_q):
                out[role, i] = q_value * token_strides[role]
            else:
                out[role, i] = kv_value * token_strides[role]
        if i < n:
            lens = cutlass.make_array_view(cute.make_tensor(lengths, cute.make_layout((2, n), stride=(length_pitch, 1))))
            lens[0, i] = cutlass.Int32(cutlass.Int64(q_view[q_offset + prefix_strides[0]]) - q_value)
            lens[1, i] = cutlass.Int32(cutlass.Int64(kv_view[kv_offset + prefix_strides[1]]) - kv_value)


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(q, kv, offsets, lengths, n, prefix_strides, token_strides, offset_pitch, length_pitch, n_q: cutlass.Constexpr, stream: driver.CUstream):
    _kernel(q, kv, offsets, lengths, n, prefix_strides, token_strides, offset_pitch, length_pitch, n_q).launch(
        grid=((n + 256) // 256, 1, 1), block=(256, 1, 1), stream=stream
    )


@lru_cache(maxsize=128)
def compile_metadata(q_dtype, kv_dtype, n_q, n_kv, device_index, arch):
    key = template_key(globals(), locals(), "compile_metadata")
    types = {"torch.int32": cutlass.Int32, "torch.int64": cutlass.Int64}

    def pointer(dtype):
        return cute.runtime.make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=dtype.width // 8)

    return compile_cached(
        _host,
        pointer(types[q_dtype]),
        pointer(types[kv_dtype]),
        pointer(cutlass.Int64),
        pointer(cutlass.Int32),
        cutlass.Int64(0),
        (cutlass.Int64(0), cutlass.Int64(0)),
        (cutlass.Int64(0),) * (n_q + n_kv),
        cutlass.Int64(0),
        cutlass.Int64(0),
        n_q,
        driver.CUstream(0),
        options=f"--enable-tvm-ffi --gpu-arch {arch}",
        cache_key=key,
        symbol="cudnn_sdpa_varlen_metadata",
    )
