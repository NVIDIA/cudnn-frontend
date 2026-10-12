# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy packed Stats and zero padded rows in one prepared launch."""

from functools import lru_cache
import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

from cudnn.frost.compiled_cache import compile_cached, template_key

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


@cute.kernel
def _kernel(lse, prefix, output, batch, tokens, heads, max_seqlen, strides, prefix_stride):
    block, head_id, sequence_id = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    position = cutlass.Int64(block) * 256 + cutlass.Int64(thread)
    head, sequence = cutlass.Int64(head_id), cutlass.Int64(sequence_id)
    count = batch * heads * max_seqlen
    if position < max_seqlen:
        cu = cutlass.make_array_view(cute.make_tensor(prefix, cute.make_layout((batch * prefix_stride + 1,), stride=(1,))))
        start = cutlass.Int64(cu[sequence * prefix_stride])
        end = cutlass.Int64(cu[(sequence + 1) * prefix_stride])
        value = cutlass.Float32(0)
        if position < end - start and start + position < tokens:
            source = cutlass.make_array_view(cute.make_tensor(lse, cute.make_layout((tokens, heads), stride=strides)))
            value = source[start + position, head]
        target = cutlass.make_array_view(cute.make_tensor(output, cute.make_layout((count,), stride=(1,))))
        index = (sequence * heads + head) * max_seqlen + position
        target[index] = value


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(lse, prefix, output, batch, tokens, heads, max_seqlen, strides, prefix_stride, stream: driver.CUstream):
    _kernel(lse, prefix, output, batch, tokens, heads, max_seqlen, strides, prefix_stride).launch(
        grid=((max_seqlen + 255) // 256, heads, batch), block=(256, 1, 1), stream=stream
    )


@lru_cache(maxsize=128)
def compile_packed_lse(prefix_dtype, device_index, arch):
    key = template_key(globals(), locals(), "compile_packed_lse")
    prefix_type = {"torch.int32": cutlass.Int32, "torch.int64": cutlass.Int64}[prefix_dtype]

    def pointer(dtype):
        return cute.runtime.make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=dtype.width // 8)

    return compile_cached(
        _host,
        pointer(cutlass.Float32),
        pointer(prefix_type),
        pointer(cutlass.Float32),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        (cutlass.Int64(0), cutlass.Int64(0)),
        cutlass.Int64(0),
        driver.CUstream(0),
        options=f"--enable-tvm-ffi --gpu-arch {arch}",
        cache_key=key,
        symbol="cudnn_sdpa_packed_lse",
    )
