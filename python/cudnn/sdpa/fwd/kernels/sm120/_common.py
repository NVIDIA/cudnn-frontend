# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""PTX helpers shared by the SM120 SDPA-forward kernels.

``sm120/prefill_f16.py``, ``sm120/prefill_fp8.py``, ``sm120/prefill_mxfp8.py``,
and the D=256 specialization ``sm120/prefill_d256_f16.py`` are self-contained
kernels that differ in head-tile geometry, MMA shape, and warp roles; the
lane-level primitives below are shared between them (``sfo_atom_offset`` by the
two block-scaled-O kernels, FP8 and MXFP8; ``zero_dead_q_rows`` by the three D=512
templates) and live here so a fix lands once.
"""

from typing import Type

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import Pointer
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cudnn.sdpa.fwd.kernels._mixed_kv import _packed_asm
from cudnn.frost.tile_dsl.swizzle import swizzle_xor

from cutlass.experimental import primitives as prims

SCHED_L2_BUDGET_BYTES = 50 * 1024 * 1024


@cute.jit
def nvvm_threadquad_reduction_max(val: cutlass.Float32) -> cutlass.Float32:
    """Butterfly thread-quad (4 lanes) reduction max via shfl.sync.bfly."""
    val = cute.arch.fmax(
        val,
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=val,
            offset=2,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        ),
    )
    val = cute.arch.fmax(
        val,
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=val,
            offset=1,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        ),
    )
    return val


@cute.jit
def nvvm_threadquad_reduction_sum(val: cutlass.Float32) -> cutlass.Float32:
    """Butterfly thread-quad (4 lanes) reduction sum via shfl.sync.bfly."""
    val = val + prims.shfl_sync(
        thread_mask=0xFFFFFFFF,
        val=val,
        offset=2,
        mask_and_clamp=0x1F,
        kind=prims.Shfl.BFLY,
    )
    val = val + prims.shfl_sync(
        thread_mask=0xFFFFFFFF,
        val=val,
        offset=1,
        mask_and_clamp=0x1F,
        kind=prims.Shfl.BFLY,
    )
    return val


@cute.jit
def pack_to_i32(
    src: tuple,
    dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
) -> cutlass.Int32:
    """Pack four 8-bit or two 16-bit values into one 32-bit register."""
    vals = cutlass.Vector.from_elements(src, dtype)
    return vals.bitcast(cutlass.Int32)[0]


def ceil_div(a: int, b: int) -> int:
    """Return the ceiling division of a by b."""
    return (a + b - 1) // b


@cute.jit
def zero_dead_q_rows(frag, row_lo, live_rows: cutlass.Constexpr[int]):
    """Zero a packed tile's dead rows in one ``ldmatrix.x4`` A fragment of a TMA-staged
    Q tile: the packed TMA box holds only the live rows, so theirs are stale slab bytes.
    Registers 0 and 2 hold row ``row_lo`` (the warp's first row + ``lane // 4``),
    registers 1 and 3 that row + 8."""
    keep_lo = cutlass.Int32(-1) if row_lo < live_rows else cutlass.Int32(0)
    keep_hi = cutlass.Int32(-1) if row_lo + 8 < live_rows else cutlass.Int32(0)
    return cutlass.Vector.from_elements((frag[0] & keep_lo, frag[1] & keep_hi, frag[2] & keep_lo, frag[3] & keep_hi), cutlass.Int32)


def sfo_atom_offset(plane, r, c, cols):
    """Byte offset of scale (r, c) in a [rows, cols] SF_O matrix stored in the F8_128x4 atom
    order at ``plane``: ``plane + (r//128)*128*cols + (c//4)*512 + (r%32)*16 + ((r//32)%4)*4 + c%4``."""
    off = plane + (r >> cutlass.Int32(7)) * (cols << cutlass.Int32(7)) + (c >> cutlass.Int32(2)) * cutlass.Int32(512)
    return off + (r & cutlass.Int32(31)) * cutlass.Int32(16) + ((r >> cutlass.Int32(5)) & cutlass.Int32(3)) * cutlass.Int32(4) + (c & cutlass.Int32(3))


@cute.jit
def copy_paged_half_tile(
    s_dst,
    pool: cute.Tensor,
    page_table,
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    seq_coord: cutlass.Int32,
    seqlen_k: cutlass.Int32,
    page_size: cutlass.Constexpr[int],
    kv_tile: cutlass.Constexpr[int],
    head_tile: cutlass.Constexpr[int],
    chunk_elems: cutlass.Constexpr[int],
    producer_warps: cutlass.Constexpr[int] = 1,
) -> None:
    """Producer warps fill a half MMA tile from half or FP8 page storage.

    The caller owns publication: cp.async arrivals for half copies, ordinary
    release arrivals for the FP8 widening stores. Page lookups stay within
    live rows; dead vectors read no payload. Eight-element chunks fit the
    half SMEM swizzle.
    """
    # Read the caller's table in the attention CTA. Each half vector lands
    # directly in the existing MMA tile; dead rows never read a table entry
    # or pool payload. Copies and widening both zero-fill those SMEM rows.
    src = Pointer(cutlass.make_array_view(pool).data_ptr(), dtype=pool.element_type)
    lane = cute.arch.lane_idx()
    vectors_per_row = head_tile // 8
    producer_threads = 32 * producer_warps
    producer_lane = (cute.arch.warp_idx() % producer_warps) * 32 + lane
    assert kv_tile * vectors_per_row % producer_threads == 0
    # Each producer warp caches live page IDs before walking its vectors.
    # Include a possible unaligned first page; retain the scalar walk when
    # the largest possible span would exceed the warp's shuffle sources.
    cache_pages = (kv_tile + page_size - 2) // page_size + 1 <= 32
    if cutlass.const_expr(cache_pages):
        first_page = seq_coord // page_size
        lane_page = cutlass.Int32(0)
        if seq_coord < seqlen_k:
            live_rows = cute.math.min(cutlass.Int32(kv_tile), seqlen_k - seq_coord)
            last_page = (seq_coord + live_rows - 1) // page_size
            table_col = first_page + cute.math.min(lane, last_page - first_page)
            lane_page = cutlass.Int32(page_table[batch_idx, table_col])
    # Keep the generated loader bounded as the tile grows. Fully unrolling
    # dozens of vectors per lane duplicates the page-table/address machinery
    # and makes large paged prefill tiles much slower than the rolled walk.
    for wave in cutlass.range(kv_tile * vectors_per_row // producer_threads, unroll=1):
        chunk = wave * producer_threads + producer_lane
        row = chunk // vectors_per_row
        col = (chunk % vectors_per_row) * 8
        logical_row = seq_coord + row
        if cutlass.const_expr(cache_pages):
            # All lanes shuffle before the per-vector live predicate diverges.
            page_id = cute.arch.shuffle_sync(lane_page, logical_row // page_size - first_page)
        # The logical width is an eight-element multiple. A complete
        # 16-byte vector is either live or zero-filled; head padding
        # must never read the next head or poisoned storage slack.
        live = cutlass.Int32((logical_row < seqlen_k) & (col < pool.shape[3]))
        offset = cutlass.Int64(0)
        if live != 0:
            if cutlass.const_expr(cache_pages):
                page = cutlass.Int64(page_id)
            else:
                page = cutlass.Int64(page_table[batch_idx, logical_row // page_size])
            offset = page * cutlass.Int64(pool.stride[0])
            offset += cutlass.Int64(logical_row % page_size) * cutlass.Int64(pool.stride[1])
            offset += cutlass.Int64(head_idx) * cutlass.Int64(pool.stride[2]) + cutlass.Int64(col)
        physical_row = (col // chunk_elems) * kv_tile + row
        physical_col = swizzle_xor(physical_row, col % chunk_elems, chunk_elems, 2)
        dst = s_dst + physical_row * chunk_elems + physical_col
        if cutlass.const_expr(pool.element_type in (cutlass.Float8E4M3FN, cutlass.Float8E5M2)):
            inline_ptx(_packed_asm(pool.element_type, s_dst.dtype, global_src=True), read_only_args=[src + offset, dst, live])
        else:
            prims.cp_async_shared_global(dst, src + offset, 16, prims.LoadCacheModifier.CG, cp_size=live * 16)


@cute.jit
def load_paged_half_tile(
    s_dst: cutlass.Array,
    mbar: cutlass.Array,
    pool: cute.Tensor,
    page_table,
    batch_idx: cutlass.Int32,
    head_idx: cutlass.Int32,
    seq_coord: cutlass.Int32,
    seqlen_k: cutlass.Int32,
    page_size: cutlass.Constexpr[int],
    kv_tile: cutlass.Constexpr[int],
    head_tile: cutlass.Constexpr[int],
    chunk_elems: cutlass.Constexpr[int],
    producer_warps: cutlass.Constexpr[int] = 1,
    async_publication: cutlass.Constexpr[bool] = False,
) -> None:
    """Fill one paged half tile, then publish every producer arrival."""
    copy_paged_half_tile(
        s_dst.data_ptr(), pool, page_table, batch_idx, head_idx, seq_coord, seqlen_k, page_size, kv_tile, head_tile, chunk_elems, producer_warps
    )
    if cutlass.const_expr(pool.element_type in (cutlass.Float8E4M3FN, cutlass.Float8E5M2)):
        # Ordinary release arrivals publish the generic widening stores.
        prims.mbarrier_arrive(mbar)
    else:
        prims.cp_async_commit_group()
        if cutlass.const_expr(not async_publication):
            prims.cp_async_wait_group(0)
        # Async-tracked arrivals publish cp.async writes; .noinc accounts
        # for all producer lanes counted by barrier initialization/rearm.
        prims.cp_async_mbarrier_arrive(mbar, noinc=True)


@cute.jit
def thd_row_origin(offsets, specs: cutlass.Constexpr, port: cutlass.Constexpr[int], batch, token_stride, prefix):
    """Resolve one physical token origin without changing logical lengths."""
    origin = cutlass.Int64(prefix)
    if cutlass.const_expr(specs is not None and specs[port] is not None):
        origin = cutlass.Int64(cutlass.make_array_view(offsets[port])[batch]) * cutlass.Int64(specs[port][1]) // cutlass.Int64(token_stride)
    return origin
