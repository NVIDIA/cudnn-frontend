# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native FP8 cache conversion into the half-input attention MMA layouts."""

import cutlass
import cutlass.cute as cute
from cudnn.frost.tile_dsl.swizzle import swizzle_xor
from cutlass.cute.arch.nvvm_wrappers import inline_ptx


def _packed_asm(src_dtype, out_dtype):
    # Packed widening preserves every finite FP8 value, including subnormals.
    # BF16 widening needs PTX 9.2, supplied by the supported DSL 4.7 compiler.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
    src = "e4m3x2" if src_dtype == cutlass.Float8E4M3FN else "e5m2x2"
    dst = "bf16x2" if out_dtype == cutlass.BFloat16 else "f16x2"
    return (
        "{ .reg .b32 x0, x1, y0, y1, y2, y3; .reg .b16 p0, p1, p2, p3; "
        "ld.shared.v2.b32 {x0, x1}, [$0]; "
        "mov.b32 {p0, p1}, x0; mov.b32 {p2, p3}, x1; "
        + " ".join(f"cvt.rn.{dst}.{src} y{i}, p{i};" for i in range(4))
        + " st.shared.v4.b32 [$1], {y0, y1, y2, y3}; }"
    )


@cute.jit
def convert_fp8_smem_tile(src, dst, dst_swizzle: cutlass.Constexpr[int], out_dtype: cutlass.Constexpr):
    """Convert one byte-TMA tile to its corresponding half MMA SMEM layout.

    The source and destination have identical logical subtiles, with one and
    two bytes per element respectively. TMA swizzles therefore differ by a
    factor of two; reusing the raw offset would permute rows silently. Every
    finite E4M3/E5M2 value is exactly representable in either half type.
    Each lane widens eight contiguous values with four packed conversions,
    one 64-bit load and one 128-bit store. Eight elements stay within both
    swizzles' 16-byte chunks, so the vector preserves the logical permutation.
    Called by all 32 load-warp lanes after TMA completion. The caller owns
    publication to the async consumer and the raw-slab reuse barrier.
    """
    width = cutlass.const_expr(dst.tma_granu_elems)
    if cutlass.const_expr(width * 2 != dst_swizzle):
        raise ValueError("mixed FP8 KV: destination swizzle must match the half-width TMA subtile")
    rows = cutlass.const_expr(dst.tma_subtile_stride_elems // width)
    lane = cute.arch.thread_idx()[0] % cutlass.Int32(32)
    # Eight chunks amortize loop/index work while preserving the warp-owned stores.
    for base in cutlass.range(0, dst.elems_per_stage, 256, unroll=8):
        logical = base + lane * cutlass.Int32(8)
        col = logical % cutlass.Int32(width)
        row = (logical // cutlass.Int32(width)) % cutlass.Int32(rows)
        outer = (logical // cutlass.Int32(width * rows)) * cutlass.Int32(width * rows)
        src_idx = outer + row * cutlass.Int32(width) + swizzle_xor(row, col, width, 1)
        dst_idx = outer + row * cutlass.Int32(width) + swizzle_xor(row, col, width, 2)
        inline_ptx(
            _packed_asm(src.base.dtype, out_dtype),
            read_only_args=[src.base.subview(src_idx).data_ptr(), dst.base.subview(dst_idx).data_ptr()],
        )
