# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Initialize the packed convenience wrappers' owned output buffers together."""

import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
_THREADS = 256
_ITEMS = 4


@cute.kernel
def _kernel(outputs, counts):
    block, role, _ = cute.arch.block_idx()
    thread, _, _ = cute.arch.thread_idx()
    for i in cutlass.range_constexpr(len(outputs)):
        if role == i:
            dst = cutlass.make_array_view(cute.make_tensor(outputs[i], cute.make_layout((counts[i],), stride=(1,))))
            for item in cutlass.range_constexpr(_ITEMS):
                index = cutlass.Int64(block) * (_THREADS * _ITEMS) + cutlass.Int64(thread) + item * _THREADS
                if index < counts[i]:
                    dst[index] = cutlass.Uint32(0)


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(outputs, counts, stream: driver.CUstream):
    words = cutlass.Int64(0)
    for i in cutlass.range_constexpr(len(outputs)):
        words = cute.math.max(words, counts[i])
    _kernel(outputs, counts).launch(grid=((words + _THREADS * _ITEMS - 1) // (_THREADS * _ITEMS), len(outputs), 1), block=(_THREADS, 1, 1), stream=stream)


@cute.jit
def zero_outputs(tensors, stream: driver.CUstream):
    """Clear compact, whole-word regions already owned by the wrapper or plan."""
    pointers, counts = (), ()
    for i in cutlass.range_constexpr(len(tensors)):
        tensor = tensors[i]
        if cutlass.const_expr(tensor is not None):
            pointers += (cute.make_ptr(cutlass.Uint32, tensor.iterator.toint(), cute.AddressSpace.gmem, assumed_align=4),)
            counts += (cutlass.Int64(cute.size(tensor)) * (tensor.element_type.width // 8) // 4,)
    _host(pointers, counts, stream)
