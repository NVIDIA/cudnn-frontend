# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL implementation of HSTU LMSD forward.

The output is laid out as three contiguous D-wide segments:

* segment 0: dropout(SiLU(u))
* segment 1: dropout(x)
* segment 2: dropout(LayerNorm(x) * SiLU(u))

One int8 mask stores the three keep decisions in bits 2, 1, and 0,
respectively. See the benchmark README for the tuning history behind the
shipping configuration.
"""

from cuda.bindings import driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as _cm

from ._common import LOG2E, MASK_LMSD, MASK_SILU, MASK_X, domain_offset_i64

VECTOR_SIZE = 8
ROWS_PER_CTA = 2
THREADS_PER_ROW = 32
MIN_BLOCKS_PER_MP = 0

M0, M1 = 0xD2511F53, 0xCD9E8D57
W0, W1 = 0x9E3779B9, 0xBB67AE85
MASK32 = 0xFFFFFFFF
ROUNDS = 10


class LnMulDropoutForward:
    """Compile-time configuration and device code for LMSD forward.

    The class follows QuACK's split between a JIT-callable launch layer and a
    ``kernel`` method. Performance-sensitive choices remain compile-time
    attributes. Tensor names use the same scope convention: ``m`` for whole
    tensors, ``g`` for tiled global-memory views, ``t`` for per-thread
    partitions, and ``r`` for register fragments.
    """

    def __init__(self):
        self.threads_per_row = THREADS_PER_ROW
        self.rows_per_cta = ROWS_PER_CTA
        self.vector_size = VECTOR_SIZE
        self.min_blocks_per_mp = MIN_BLOCKS_PER_MP

    @cute.kernel
    def kernel(
        self,
        gX: cute.Tensor,
        gU: cute.Tensor,
        gW: cute.Tensor,
        gB: cute.Tensor,
        gSiluOut: cute.Tensor,
        gXOut: cute.Tensor,
        gLmsdOut: cute.Tensor,
        gMask: cute.Tensor,
        gMean: cute.Tensor,
        gRstd: cute.Tensor,
        eps: cutlass.Float32,
        drop: cutlass.Float32,
        thresh: cutlass.Uint32,
        thr_layout: cute.Layout,
        val_layout: cute.Layout,
        num_column_tiles: cutlass.Constexpr,
        seed: cutlass.Int64,
        nrows: cutlass.Int32,
        row_off: cutlass.Int32,
        ncols: cutlass.Int32,
    ):
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        # A CTA owns rows_per_cta warps, and each warp owns one row.
        thread_in_row = thread_idx % self.threads_per_row
        rows_per_block = self.rows_per_cta * 32 // self.threads_per_row
        row = block_idx * rows_per_block + thread_idx // self.threads_per_row
        valid = row < nrows
        if valid:
            # Phase 1: map this thread to its row and column fragments.
            tensor_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type)
            mask_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gMask.element_type)
            thread_copy = cute.make_tiled_copy_tv(tensor_copy_atom, thr_layout, val_layout).get_slice(thread_in_row)
            mask_thread_copy = cute.make_tiled_copy_tv(mask_copy_atom, thr_layout, val_layout).get_slice(thread_in_row)

            row_coord = ((0, 0), (row, 0))
            gX_row = domain_offset_i64(row_coord, gX)
            gU_row = domain_offset_i64(row_coord, gU)
            gSiluOut_row = domain_offset_i64(row_coord, gSiluOut)
            gXOut_row = domain_offset_i64(row_coord, gXOut)
            gLmsdOut_row = domain_offset_i64(row_coord, gLmsdOut)
            gMask_row = domain_offset_i64(row_coord, gMask)

            # Python lists here are JIT-time containers unrolled by
            # range_constexpr; they are not dynamically allocated on the GPU.
            rX_reduction_tiles, rU_tiles = [], []
            tXgOutput_tiles, tXgMask_tiles = [], []
            tXgX_compute_tiles, tXgW_tiles, tXgB_tiles = [], [], []
            for j in cutlass.range_constexpr(num_column_tiles):
                row_tile_coord = ((None, None), (0, j))
                for src, src_coord, dst in (
                    (gX_row, row_tile_coord, rX_reduction_tiles),
                    (gU_row, row_tile_coord, rU_tiles),
                ):
                    t = thread_copy.partition_S(src[src_coord])
                    f = cute.make_fragment_like(t)
                    cute.copy(tensor_copy_atom, t, f)
                    dst.append(f)
                # Keep the partition, not the fragment, so the compute phase can
                # reread x after the reduction with a shorter register live range.
                tXgX_compute_tiles.append(thread_copy.partition_S(gX_row[row_tile_coord]))
                param_coord = ((None, None), (0, j))
                tXgW_tiles.append(thread_copy.partition_S(gW[param_coord]))
                tXgB_tiles.append(thread_copy.partition_S(gB[param_coord]))
                tXgOutput_tiles.append(
                    (
                        thread_copy.partition_S(gSiluOut_row[row_tile_coord]),
                        thread_copy.partition_S(gXOut_row[row_tile_coord]),
                        thread_copy.partition_S(gLmsdOut_row[row_tile_coord]),
                    )
                )
                tXgMask_tiles.append(mask_thread_copy.partition_S(gMask_row[row_tile_coord]))

            # Phase 2: compute row-wise LayerNorm statistics.
            sum_x = cutlass.Float32(0.0)
            sum_sq_x = cutlass.Float32(0.0)
            for j in cutlass.range_constexpr(num_column_tiles):
                for e in cutlass.range_constexpr(self.vector_size):
                    v = rX_reduction_tiles[j][e].to(cutlass.Float32)
                    sum_x = sum_x + v
                    sum_sq_x = sum_sq_x + v * v
            # The butterfly stays inside the aligned threads-per-row subgroup.
            for off in cutlass.range_constexpr(self.threads_per_row.bit_length() - 1):
                sum_x = sum_x + cute.arch.shuffle_sync_bfly(sum_x, 1 << off)
                sum_sq_x = sum_sq_x + cute.arch.shuffle_sync_bfly(sum_sq_x, 1 << off)

            inv_d = cutlass.Float32(1.0) / ncols.to(cutlass.Float32)
            mean = sum_x * inv_d
            var = _cm.max(sum_sq_x * inv_d - mean * mean, cutlass.Float32(0.0))
            rstd = cutlass.Float32(1.0) / _cm.sqrt(var + eps)
            scale = cutlass.Float32(1.0) / (cutlass.Float32(1.0) - drop)

            k0 = cutlass.Uint32(seed & MASK32)
            k1 = cutlass.Uint32((seed >> 32) & MASK32)

            # Phase 3: generate three Philox streams, evaluate LMSD, and write
            # segment 0 (SiLU), segment 1 (x), segment 2 (LMSD), and the mask.
            for j in cutlass.range_constexpr(num_column_tiles):
                tXgW = tXgW_tiles[j]
                tXgB = tXgB_tiles[j]
                tXgSiluOut, tXgXOut, tXgLmsdOut = tXgOutput_tiles[j]
                tXgMask = tXgMask_tiles[j]
                rU = rU_tiles[j]
                rX = cute.make_fragment_like(tXgX_compute_tiles[j])
                cute.copy(tensor_copy_atom, tXgX_compute_tiles[j], rX)
                rW = cute.make_fragment_like(tXgW)
                rB = cute.make_fragment_like(tXgB)
                cute.copy(tensor_copy_atom, tXgW, rW)
                cute.copy(tensor_copy_atom, tXgB, rB)
                rSiluOut = cute.make_fragment_like(tXgSiluOut)
                rXOut = cute.make_fragment_like(tXgXOut)
                rLmsdOut = cute.make_fragment_like(tXgLmsdOut)
                rMask = cute.make_fragment_like(tXgMask)
                # A thread owns vector_size contiguous columns in tile j.
                # Each mask plane uses full 32-bit samples: plane 0 is LMSD,
                # plane 1 is x, and plane 2 is SiLU(u).
                philox_block = cutlass.Uint32(j * self.threads_per_row) + cutlass.Uint32(thread_in_row)
                philox_words = []
                for mask_plane in cutlass.range_constexpr(3):
                    for h in cutlass.range_constexpr(2):
                        c0 = cutlass.Uint32(row) + cutlass.Uint32(row_off)
                        c1 = philox_block * cutlass.Uint32(2) + cutlass.Uint32(h)
                        c2 = cutlass.Uint32(mask_plane)
                        c3 = cutlass.Uint32(0)
                        kk0, kk1 = k0, k1
                        m0c = cutlass.Uint32(M0)
                        m1c = cutlass.Uint32(M1)
                        for _r in cutlass.range_constexpr(ROUNDS):
                            # One mul.wide.u32 yields both halves of each
                            # product, so no second multiply is required.
                            p0 = cute.arch.mul_wide(m0c, c0)
                            p1 = cute.arch.mul_wide(m1c, c2)
                            hi0 = (p0 >> cutlass.Uint64(32)).to(cutlass.Uint32)
                            lo0 = (p0 & cutlass.Uint64(MASK32)).to(cutlass.Uint32)
                            hi1 = (p1 >> cutlass.Uint64(32)).to(cutlass.Uint32)
                            lo1 = (p1 & cutlass.Uint64(MASK32)).to(cutlass.Uint32)
                            c0 = hi1 ^ c1 ^ kk0
                            c1 = lo1
                            c2 = hi0 ^ c3 ^ kk1
                            c3 = lo0
                            kk0 = kk0 + cutlass.Uint32(W0)
                            kk1 = kk1 + cutlass.Uint32(W1)
                        philox_words.append((c0, c1, c2, c3))
                for e in cutlass.range_constexpr(self.vector_size):
                    bits = cutlass.Int8(0)
                    for mask_plane in cutlass.range_constexpr(3):
                        word = philox_words[mask_plane * 2 + e // 4][e % 4]
                        keep = word >= thresh
                        bits = bits | (cutlass.Int8(1 << mask_plane) if keep else cutlass.Int8(0))
                    rMask[e] = bits
                # Evaluate pairs with packed FP32 arithmetic. Accumulation and
                # output conversion retain the shipping numerical order.
                one2 = (cutlass.Float32(1.0), cutlass.Float32(1.0))
                mean2 = (mean, mean)
                rstd2 = (rstd, rstd)
                scale2 = (scale, scale)
                negative_log2e2 = cutlass.Float32(-LOG2E)
                for pair in cutlass.range_constexpr(self.vector_size // 2):
                    e0 = 2 * pair
                    e1 = e0 + 1
                    xf2 = (rX[e0].to(cutlass.Float32), rX[e1].to(cutlass.Float32))
                    uf2 = (rU[e0].to(cutlass.Float32), rU[e1].to(cutlass.Float32))
                    weight2 = (rW[e0].to(cutlass.Float32), rW[e1].to(cutlass.Float32))
                    bias2 = (rB[e0].to(cutlass.Float32), rB[e1].to(cutlass.Float32))
                    xhat2 = cute.arch.mul_packed_f32x2(cute.arch.sub_packed_f32x2(xf2, mean2), rstd2)
                    layer_norm2 = cute.arch.fma_packed_f32x2(xhat2, weight2, bias2)
                    exp2 = (cute.arch.exp2(uf2[0] * negative_log2e2), cute.arch.exp2(uf2[1] * negative_log2e2))
                    denominator2 = cute.arch.add_packed_f32x2(exp2, one2)
                    silu2 = (_cm.div(uf2[0], denominator2[0], approx=True), _cm.div(uf2[1], denominator2[1], approx=True))
                    scaled_silu2 = cute.arch.mul_packed_f32x2(silu2, scale2)
                    scaled_x2 = cute.arch.mul_packed_f32x2(xf2, scale2)
                    scaled_lmsd2 = cute.arch.mul_packed_f32x2(layer_norm2, scaled_silu2)
                    zero = cutlass.Float32(0.0)
                    for q in cutlass.range_constexpr(2):
                        mask_bits = rMask[e0 + q].to(cutlass.Int32)
                        rSiluOut[e0 + q] = (scaled_silu2[q] if (mask_bits & MASK_SILU) != 0 else zero).to(gX.element_type)
                        rXOut[e0 + q] = (scaled_x2[q] if (mask_bits & MASK_X) != 0 else zero).to(gX.element_type)
                        rLmsdOut[e0 + q] = (scaled_lmsd2[q] if (mask_bits & MASK_LMSD) != 0 else zero).to(gX.element_type)
                cute.copy(tensor_copy_atom, rSiluOut, tXgSiluOut)
                cute.copy(tensor_copy_atom, rXOut, tXgXOut)
                cute.copy(tensor_copy_atom, rLmsdOut, tXgLmsdOut)
                cute.copy(mask_copy_atom, rMask, tXgMask)

            if thread_in_row == 0:
                gMean[row] = mean
                gRstd[row] = rstd

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mU: cute.Tensor,
        mW: cute.Tensor,
        mB: cute.Tensor,
        mSiluOut: cute.Tensor,
        mXOut: cute.Tensor,
        mLmsdOut: cute.Tensor,
        mMask: cute.Tensor,
        mMean: cute.Tensor,
        mRstd: cute.Tensor,
        seed: cutlass.Int64,
        nrows: cutlass.Int32,
        row_off: cutlass.Int32,
        eps: cutlass.Float32,
        drop: cutlass.Float32,
        thresh: cutlass.Uint32,
        ncols: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        # The copy tile represents one row. The separate row layout describes
        # the two physical warps (and therefore two rows) owned by each CTA.
        thr_layout = cute.make_ordered_layout((1, self.threads_per_row), order=(1, 0))
        row_thr_layout = cute.make_ordered_layout(
            (self.rows_per_cta * 32 // self.threads_per_row, 32),
            order=(1, 0),
        )
        val_layout = cute.make_ordered_layout((1, self.vector_size), order=(1, 0))
        _row_tiler, _ = cute.make_layout_tv(row_thr_layout, val_layout)
        grid_n = cute.size(cute.zipped_divide(mX, _row_tiler), mode=[1, 0])
        tiler, _ = cute.make_layout_tv(thr_layout, val_layout)
        tile = lambda tensor: cute.zipped_divide(tensor, tiler)
        gX = tile(mX)
        # Weight and bias are public rank-1 tensors. The device kernel only
        # indexes parameter row zero, so construct that singleton mode as a
        # zero-stride CuTe view instead of materializing a repeated tensor.
        param_layout = cute.make_layout((1, cute.size(mW)), stride=(0, 1))
        mW2 = cute.make_tensor(mW.iterator, param_layout)
        mB2 = cute.make_tensor(mB.iterator, param_layout)
        self.kernel(
            gX,
            tile(mU),
            tile(mW2),
            tile(mB2),
            tile(mSiluOut),
            tile(mXOut),
            tile(mLmsdOut),
            tile(mMask),
            mMean,
            mRstd,
            eps,
            drop,
            thresh,
            thr_layout,
            val_layout,
            cute.size(gX, mode=[1, 1]),
            seed,
            nrows,
            row_off,
            ncols,
        ).launch(
            grid=(grid_n, 1, 1),
            block=(self.rows_per_cta * 32, 1, 1),
            min_blocks_per_mp=self.min_blocks_per_mp,
            stream=stream,
        )
