# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact each selection row on the launch stream."""

import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cuda.bindings import driver as cuda

from cudnn.frost.compiled_cache import compile_cached, template_key

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


@cute.kernel
def frost_qsa_compact(sel: cute.Tensor, lo: cute.Tensor, hi: cute.Tensor, first: cute.Tensor, last: cute.Tensor, ids: cute.Tensor, counts: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    row = cute.arch.block_idx()[0] * 4 + tid // 32
    lane = tid % 32
    if row < sel.shape[0]:
        low, high = lo[row], hi[row]
        count = Int32(0)
        lower_lanes = (Uint32(1) << Uint32(lane)) - Uint32(1)
        for start in cutlass.range(0, Int32(sel.shape[1]), 32, unroll=1):
            block = start + lane
            selected = Boolean(False)
            if block < sel.shape[1]:
                selected = (sel[Int64(row), Int64(block)] != 0) & (last[block] >= low) & (first[block] <= high) & (low <= high)
            ballot = Uint32(cute.arch.vote_ballot_sync(selected))
            if selected:
                offset = count + Int32(cute.arch.popc(ballot & lower_lanes))
                ids[Int64(row), Int64(offset)] = block
            count = count + Int32(cute.arch.popc(ballot))
        if lane == 0:
            counts[row] = count


frost_qsa_compact.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _compact_host(sel, lo, hi, first, last, ids, counts, stream: cuda.CUstream):
    frost_qsa_compact(sel, lo, hi, first, last, ids, counts).launch(grid=(cute.ceil_div(sel.shape[0], 4), 1, 1), block=(128, 1, 1), stream=stream)


def compile_kernel(arch):
    key = template_key(globals(), {"arch": arch}, "compile_kernel")
    total, blocks = cute.sym_int64(), cute.sym_int64()

    def tensor(dtype, shape, alignment=4):
        return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=alignment)

    sel = tensor(cutlass.Uint8, (total, blocks), 1)
    rows, spans = tensor(Int32, (total,)), tensor(Int32, (blocks,))
    ids = tensor(Int32, (total, blocks))
    return compile_cached(
        _compact_host,
        sel,
        rows,
        rows,
        spans,
        spans,
        ids,
        rows,
        make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch}a",
        cache_key=key,
        symbol="qsa_compact",
    )
