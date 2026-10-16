# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block permutations for explicit grouping; imported only after the DSL gate."""

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cudnn.frost.compiled_cache import compile_cached


@cute.kernel
def _reorder_kernel(source, target, permutation, scatter: cutlass.Constexpr):
    block, head, batch = cute.arch.block_idx()
    original = permutation[batch, head, block]
    src_block = block if scatter else original
    dst_block = original if scatter else block
    tid = cute.arch.thread_idx()[0]
    for iteration in cutlass.range_constexpr(4):
        element = (tid + iteration * 256) * 8
        row, column = element // 128, element % 128
        src_offset = ((cutlass.Int64(batch) * source.shape[1] + src_block * 64 + row) * source.shape[2] + head) * 128 + column
        dst_offset = ((cutlass.Int64(batch) * target.shape[1] + dst_block * 64 + row) * target.shape[2] + head) * 128 + column
        values = (source.iterator.raw_ptr() + src_offset).load(count=8, alignment=16)
        (target.iterator.raw_ptr() + dst_offset).store(values, alignment=16)


_reorder_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _reorder_stats_kernel(source, target, permutation, scatter: cutlass.Constexpr):
    block, head, batch = cute.arch.block_idx()
    row = cute.arch.thread_idx()[0]
    original = permutation[batch, head, block]
    src_block = block if scatter else original
    dst_block = original if scatter else block
    src_offset = (cutlass.Int64(batch) * source.shape[1] + head) * source.shape[2] + src_block * 64 + row
    dst_offset = (cutlass.Int64(batch) * target.shape[1] + head) * target.shape[2] + dst_block * 64 + row
    target.iterator[dst_offset] = source.iterator[src_offset]


_reorder_stats_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _reorder_host(source, target, permutation, scatter: cutlass.Constexpr, stream):
    grid = (permutation.shape[2], permutation.shape[1], permutation.shape[0])
    if cutlass.const_expr(len(source.shape) == 4):
        _reorder_kernel(source, target, permutation, scatter).launch(grid=grid, block=256, stream=stream)
    else:
        _reorder_stats_kernel(source, target, permutation, scatter).launch(grid=grid, block=64, stream=stream)


def reorder(source, target, permutation, *, scatter):
    # Cache by geometry, never sparse data or pointer. The shared cache also
    # keys the active CUDA device/context; do not wrap this in an lru_cache.
    key = f"bsa_grouped_reorder:{tuple(source.shape)}:{tuple(source.stride())}:{tuple(permutation.shape)}:{scatter}"
    args = [from_dlpack(t.detach(), assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic() for t in (source, target, permutation)]
    fn = compile_cached(
        _reorder_host,
        *args,
        scatter,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        cache_key=key,
        options="--enable-tvm-ffi",
    )
    fn(source, target, permutation, cuda.CUstream(torch.cuda.current_stream(source.device).cuda_stream))
