# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One compiled launcher for the indexed forward's dense and gathered query tiles, on caller storage."""

import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cuda.bindings import driver as cuda

from cudnn._cutlass_compat import SmemAllocator
from cudnn.frost.compiled_cache import compile_cached, template_key
from cudnn.QSA import _forward
from cudnn.QSA.kernels import fprop_prepare, fprop_sm100
from . import _metadata

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@cute.kernel
def frost_qsa_indexed_mask(ids, counts, hi, bits, modes, ranges):
    # one warp per row of a dense tile: the key mask words of its groups, counted from the tile's first key chunk
    tid = Int32(cute.arch.thread_idx()[0])
    lane, warp = tid % 32, tid // 32
    row = Int64(cute.arch.block_idx()[0]) * 4 + warp
    words = SmemAllocator().allocate_tensor(Int32, cute.make_layout(4 * fprop_prepare.MASK_WORDS), byte_alignment=16)
    begin, stop = Int32(0), Int32(0)
    if row < bits.shape[0]:
        tile = row // fprop_prepare.TILE_M
        if modes[tile] == 0:
            begin, stop = ranges[tile, 0], ranges[tile, 1] * 4
    for word in cutlass.range(lane, stop, 32, unroll=1):
        words[warp * fprop_prepare.MASK_WORDS + word] = Int32(0)
    cute.arch.sync_threads()
    if stop > 0:
        high = hi[row] - begin
        for column in cutlass.range(lane, counts[row], 32, unroll=1):
            key = ids[row, Int64(column)] - begin
            count = cute.math.min(key + 3, high) - key + 1
            span = (Int32(1) << count) - 1
            shift = key & 31
            word = warp * fprop_prepare.MASK_WORDS + (key >> 5)
            cute.arch.atomic_or((words.iterator + word).llvm_ptr, span << shift, sem="relaxed", scope="cta")
            if shift + count > 32:
                cute.arch.atomic_or((words.iterator + word + 1).llvm_ptr, span >> (32 - shift), sem="relaxed", scope="cta")
    cute.arch.sync_threads()
    for word in cutlass.range(lane, stop, 32, unroll=1):
        bits[row, Int64(word)] = words[warp * fprop_prepare.MASK_WORDS + word]


frost_qsa_indexed_mask.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _launch(
    q,
    k,
    v,
    indices,
    out,
    out32,
    lse,
    ids,
    counts,
    lo,
    hi,
    first,
    last,
    partials,
    modes,
    ranges,
    bits,
    order,
    scale: Float32,
    stream: cuda.CUstream,
    arch: cutlass.Constexpr,
    table: cutlass.Constexpr,
):
    _metadata.frost_qsa_indexed_metadata(indices, ids, counts, lo, hi, first, last, table, True).launch(
        grid=(ids.shape[0], 1, 1), block=(256, 1, 1), stream=stream
    )
    flat_k = cute.make_tensor(k.iterator, cute.make_layout((cute.size(k),)))
    flat_v = cute.make_tensor(v.iterator, cute.make_layout((cute.size(v),)))
    fprop_prepare.tile_routes(flat_k, flat_v, lo, hi, partials, modes, ranges, order, stream)
    frost_qsa_indexed_mask(ids, counts, hi, bits, modes, ranges).launch(grid=(cute.ceil_div(ids.shape[0], 4), 1, 1), block=(128, 1, 1), stream=stream)
    _forward._launch_tiles(q, k, v, bits, ranges, modes, order, out, out32, lse, ids, counts, lo, hi, first, last, scale, stream, arch)


def compile_kernel(hq, arch, capacity):
    key = template_key(
        globals(),
        {
            **locals(),
            "sources": (_metadata.FROST_SOURCE_DIGEST, _forward.FROST_SOURCE_DIGEST, *_forward.SOURCE_DIGESTS),
            "sm107": fprop_sm100._FPROP_SM107_SOURCE_DIGEST if arch == 107 else None,
        },
        "compile_kernel",
    )
    total, width, columns, pieces, tiles = (cute.sym_int64() for _ in range(5))

    def tensor(dtype, shape, alignment=4):
        return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=alignment)

    q = tensor(cutlass.BFloat16, (total, hq, 256), 16)
    kv = tensor(cutlass.BFloat16, (total, 1, 256), 16)
    rows = tensor(Int32, (total,))
    return compile_cached(
        _launch,
        q,
        kv,
        kv,
        tensor(Int32, (total, width)),
        q,
        tensor(Float32, (total, hq, 256), 16),
        tensor(Float32, (hq, total)),
        tensor(Int32, (total, columns)),
        *(rows,) * 5,
        tensor(Int32, (pieces,)),
        tensor(Int32, (tiles,)),
        tensor(Int32, (tiles, 2)),
        tensor(Int32, (total, fprop_prepare.MASK_WORDS), 16),
        tensor(Int32, (tiles,)),
        Float32(1),
        make_fake_stream(use_tvm_ffi_env_stream=False),
        arch,
        _metadata.table_size(capacity),
        cache_key=key,
        symbol="qsa_indexed_forward",
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch}a",
    )


def route_layouts(total):
    """Names and shapes of the scratch behind the tile routes and the dense tiles' key mask rows."""
    tiles = -(-total // fprop_prepare.TILE_M)
    return [("partials", (-(-total // 4) * 4,)), ("modes", (tiles,)), ("ranges", (tiles, 2)), ("bits", (total, fprop_prepare.MASK_WORDS)), ("order", (tiles,))]


def launch(compiled, *tensors, scale, stream):
    """Launch on caller-owned outputs and scratch, without allocating."""
    compiled(*tensors, Float32(scale), stream)
