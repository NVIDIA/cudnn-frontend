# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare compact Int32 lengths and Float32 sinks in one pointer launch."""

from functools import lru_cache
import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached, template_key

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


@cute.kernel
def _kernel(q, kv, sink, out_q, out_kv, out_sink, counts, strides):
    block, _, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    index = cutlass.Int64(block) * 256 + cutlass.Int64(thread)
    sources, outputs = (q, kv, sink), (out_q, out_kv, out_sink)
    for role in cutlass.range_constexpr(3):
        if cutlass.const_expr(sources[role] is not None):
            if index < counts[role]:
                span = (counts[role] - 1) * strides[role] + 1
                source = cutlass.make_array_view(cute.make_tensor(sources[role], cute.make_layout((span,), stride=(1,))))
                output = cutlass.make_array_view(cute.make_tensor(outputs[role], cute.make_layout((counts[role],), stride=(1,))))
                value = source[index * strides[role]]
                if cutlass.const_expr(role == 2):
                    output[index] = cutlass.Float32(value)
                else:
                    output[index] = cutlass.Int32(value)


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(q, kv, sink, out_q, out_kv, out_sink, counts, strides, stream: driver.CUstream):
    extent = cutlass.max(counts[0], cutlass.max(counts[1], counts[2]))
    _kernel(q, kv, sink, out_q, out_kv, out_sink, counts, strides).launch(grid=((extent + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)


@lru_cache(maxsize=128)
def compile_metadata(dtypes, device_index, arch):
    key = template_key(globals(), locals(), "compile_metadata")
    types = {
        "torch.int32": cutlass.Int32,
        "torch.int64": cutlass.Int64,
        "torch.float32": cutlass.Float32,
        "torch.float16": cutlass.Float16,
        "torch.bfloat16": cutlass.BFloat16,
    }

    def pointer(dtype):
        return cute.runtime.make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=dtype.width // 8)

    inputs = tuple(pointer(types[dtype]) if dtype is not None else None for dtype in dtypes)
    outputs = tuple(pointer(cutlass.Float32 if role == 2 else cutlass.Int32) if dtype is not None else None for role, dtype in enumerate(dtypes))
    return compile_cached(
        _host,
        *inputs,
        *outputs,
        (cutlass.Int64(0),) * 3,
        (cutlass.Int64(0),) * 3,
        driver.CUstream(0),
        options=f"--enable-tvm-ffi --gpu-arch {arch}",
        cache_key=key,
        symbol="cudnn_sdpa_forward_metadata",
    )
