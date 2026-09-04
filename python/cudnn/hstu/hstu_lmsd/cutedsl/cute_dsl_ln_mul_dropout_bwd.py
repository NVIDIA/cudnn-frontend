# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL implementation of HSTU LMSD backward.

The three D-wide segments of ``dy`` are gradients of dropout(SiLU(u)),
dropout(x), and dropout(LayerNorm(x) * SiLU(u)), in that order.

The persistent grid-stride main kernel emits per-CTA dW/dB partials. The
companion ``LnMulDropoutGradReduce`` kernel reduces those partials to the
public gradients without depending on Triton.
"""

from cuda.bindings import driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as _cm

from ._common import LOG2E, MASK_LMSD, MASK_SILU, MASK_X, domain_offset_i64

VECTOR_SIZE = 8
ROWS_PER_CTA = 1
TARGET_TILES = 13568
# X, mask, DX, and DU retain the faster tiled Int32 addressing path. With
# D=512, this limit keeps their largest flattened element offset <= 2^31 - 1.
MAX_NUM_ROWS = (1 << 31) // 512

MIN_BLOCKS_PER_MP = 12


class LnMulDropoutBackward:
    """Compile-time configuration and device code for LMSD backward.

    The JIT-callable object owns launch policy and its ``kernel`` method owns
    device-side work. Tensor names use the same scope convention: ``m`` for
    whole tensors, ``g`` for tiled global-memory views, ``t`` for per-thread
    partitions, ``r`` for register fragments, and ``s`` for shared memory.
    """

    def __init__(self):
        self.min_blocks_per_mp = MIN_BLOCKS_PER_MP
        self.rows_per_cta = ROWS_PER_CTA
        assert self.rows_per_cta == 1, "this kernel requires one row per CTA"
        self.vector_size = VECTOR_SIZE

    @cute.kernel
    def kernel(
        self,
        gDYSilu: cute.Tensor,
        gDYX: cute.Tensor,
        gDYLmsd: cute.Tensor,
        gX: cute.Tensor,
        gU: cute.Tensor,
        gW: cute.Tensor,
        gB: cute.Tensor,
        gMask: cute.Tensor,
        gDX: cute.Tensor,
        gDU: cute.Tensor,
        gMean: cute.Tensor,
        gRstd: cute.Tensor,
        gDW: cute.Tensor,
        gDB: cute.Tensor,
        drop: cutlass.Float32,
        thr_layout: cute.Layout,
        val_layout: cute.Layout,
        num_column_tiles: cutlass.Constexpr,
        ncols: cutlass.Int32,
        nblk: cutlass.Int32,
        grid: cutlass.Int32,
    ):
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        thread_in_row = thread_idx % 64
        warp_in_row = thread_in_row // 32
        reduction_smem = cute.make_tensor(
            cute.arch.alloc_smem(cutlass.Float32, self.rows_per_cta * 4),
            cute.make_layout(self.rows_per_cta * 4),
        )

        # Phase 1: establish copy and row-reduction resources. The wide X/U/dY
        # row streams have no useful L1 reuse. Bypass L1 while
        # retaining L2; keep the compact byte mask on the default cache policy.
        wide_load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            gX.element_type,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL,
            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_NORMAL,
            l2_prefetch_size=cute.nvgpu.L2PrefetchSize.NONE,
        )
        mask_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gMask.element_type)
        tensor_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type)
        fp32_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gDW.element_type)
        thread_copy = cute.make_tiled_copy_tv(tensor_copy_atom, thr_layout, val_layout).get_slice(thread_idx)
        mask_thread_copy = cute.make_tiled_copy_tv(mask_copy_atom, thr_layout, val_layout).get_slice(thread_idx)
        fp32_thread_copy = cute.make_tiled_copy_tv(fp32_copy_atom, thr_layout, val_layout).get_slice(thread_idx)

        scale = cutlass.Float32(1.0) / (cutlass.Float32(1.0) - drop)
        inv_d = cutlass.Float32(1.0) / ncols.to(cutlass.Float32)

        # Phase 2: retain W/B and parameter-gradient accumulators across the
        # persistent row loop.
        rW_tiles, rB_tiles = [], []
        for j in cutlass.range_constexpr(num_column_tiles):
            for src, dst in ((gW, rW_tiles), (gB, rB_tiles)):
                t = thread_copy.partition_S(src[((None, None), (0, j))])
                f = cute.make_fragment_like(t)
                cute.copy(wide_load_atom, t, f)
                dst.append(f)
        rDW_accum = cute.make_rmem_tensor(num_column_tiles * self.vector_size, cutlass.Float32)
        rDB_accum = cute.make_rmem_tensor(num_column_tiles * self.vector_size, cutlass.Float32)
        for e in cutlass.range_constexpr(num_column_tiles * self.vector_size):
            rDW_accum[e] = cutlass.Float32(0.0)
            rDB_accum[e] = cutlass.Float32(0.0)

        # These tensors stay below the public API's flattened Int32 offset
        # limit, so partition them once and retain the faster tiled path.
        tXgX_rows, tXgMask_rows = [], []
        for j in cutlass.range_constexpr(num_column_tiles):
            all_rows = ((None, None), (None, j))
            tXgX_rows.append(thread_copy.partition_S(gX[all_rows]))
            tXgMask_rows.append(mask_thread_copy.partition_S(gMask[all_rows]))

        # Phase 3: process rows assigned to this persistent CTA.
        for row_block in cutlass.range(block_idx, nblk, grid):
            row = row_block
            row_coord = ((0, 0), (row_block, 0))
            gDYSilu_row = domain_offset_i64(row_coord, gDYSilu)
            gDYX_row = domain_offset_i64(row_coord, gDYX)
            gDYLmsd_row = domain_offset_i64(row_coord, gDYLmsd)
            gU_row = domain_offset_i64(row_coord, gU)
            mean = gMean[row]
            rstd = gRstd[row]
            norm_bias = -mean * rstd
            sum_xhat_wdy = cutlass.Float32(0.0)
            sum_wdy = cutlass.Float32(0.0)
            rXhat_tiles, rWdy_tiles, rDirectDX_tiles = [], [], []
            for j in cutlass.range_constexpr(num_column_tiles):
                tile_coord = ((None, None), (row_block, j))
                row_tile_coord = ((None, None), (0, j))
                tXgX = tXgX_rows[j][None, None, None, row_block]
                tXgU = thread_copy.partition_S(gU_row[row_tile_coord])
                tXgDYSilu = thread_copy.partition_S(gDYSilu_row[row_tile_coord])
                tXgDYX = thread_copy.partition_S(gDYX_row[row_tile_coord])
                tXgDYLmsd = thread_copy.partition_S(gDYLmsd_row[row_tile_coord])
                tXgMask = tXgMask_rows[j][None, None, None, row_block]

                rX = cute.make_fragment_like(tXgX)
                rU = cute.make_fragment_like(tXgU)
                rDYSilu = cute.make_fragment_like(tXgDYSilu)
                rDYX = cute.make_fragment_like(tXgDYX)
                rDYLmsd = cute.make_fragment_like(tXgDYLmsd)
                rMask = cute.make_fragment_like(tXgMask)
                cute.copy(wide_load_atom, tXgX, rX)
                cute.copy(wide_load_atom, tXgU, rU)
                cute.copy(wide_load_atom, tXgDYSilu, rDYSilu)
                cute.copy(wide_load_atom, tXgDYX, rDYX)
                cute.copy(wide_load_atom, tXgDYLmsd, rDYLmsd)
                cute.copy(mask_copy_atom, tXgMask, rMask)

                tXgDU = thread_copy.partition_D(gDU[tile_coord])
                rDU = cute.make_fragment_like(tXgDU)
                rXhat = cute.make_rmem_tensor(self.vector_size, cutlass.Float32)
                rWdy = cute.make_rmem_tensor(self.vector_size, cutlass.Float32)
                rDirectDX = cute.make_rmem_tensor(self.vector_size, cutlass.Float32)

                for e in cutlass.range_constexpr(self.vector_size):
                    xf = rX[e].to(cutlass.Float32)
                    uf = rU[e].to(cutlass.Float32)
                    mb = rMask[e].to(cutlass.Int32)
                    zero = cutlass.Float32(0.0)
                    direct_du = rDYSilu[e].to(cutlass.Float32) * scale if (mb & MASK_SILU) != 0 else zero
                    direct_dx = rDYX[e].to(cutlass.Float32) * scale if (mb & MASK_X) != 0 else zero
                    grad_lmsd = rDYLmsd[e].to(cutlass.Float32) * scale if (mb & MASK_LMSD) != 0 else zero

                    xh = _cm.fma(xf, rstd, norm_bias)
                    ln = _cm.fma(xh, rW_tiles[j][e].to(cutlass.Float32), rB_tiles[j][e].to(cutlass.Float32))
                    den = cutlass.Float32(1.0) + cute.arch.exp2(-uf * cutlass.Float32(LOG2E))
                    sig = cute.arch.rcp_approx(den)
                    silu = uf * sig
                    dsilu = _cm.fma(silu, cutlass.Float32(1.0) - sig, sig)

                    du_from_lmsd = grad_lmsd * ln * dsilu
                    grad_layer_norm = grad_lmsd * silu
                    rDU[e] = (du_from_lmsd + direct_du * dsilu).to(gX.element_type)

                    wd = rW_tiles[j][e].to(cutlass.Float32) * grad_layer_norm
                    xh_e = xh
                    sum_xhat_wdy = sum_xhat_wdy + xh_e * wd
                    sum_wdy = sum_wdy + wd
                    rDW_accum[j * self.vector_size + e] = rDW_accum[j * self.vector_size + e] + grad_layer_norm * xh_e
                    rDB_accum[j * self.vector_size + e] = rDB_accum[j * self.vector_size + e] + grad_layer_norm
                    rXhat[e] = xh_e
                    rWdy[e] = wd
                    rDirectDX[e] = direct_dx

                cute.copy(tensor_copy_atom, rDU, tXgDU)
                rXhat_tiles.append(rXhat)
                rWdy_tiles.append(rWdy)
                rDirectDX_tiles.append(rDirectDX)

            # Reduce the two LayerNorm gradient statistics across both warps.
            for off in cutlass.range_constexpr(5):
                sum_xhat_wdy = sum_xhat_wdy + cute.arch.shuffle_sync_bfly(sum_xhat_wdy, 1 << off)
                sum_wdy = sum_wdy + cute.arch.shuffle_sync_bfly(sum_wdy, 1 << off)
            if thread_in_row % 32 == 0:
                reduction_smem[warp_in_row * 2 + 0] = sum_xhat_wdy
                reduction_smem[warp_in_row * 2 + 1] = sum_wdy
            cute.arch.sync_threads()
            sum_xhat_wdy = reduction_smem[0] + reduction_smem[2]
            sum_wdy = reduction_smem[1] + reduction_smem[3]
            cute.arch.sync_threads()
            sum_xhat_wdy = sum_xhat_wdy * inv_d
            sum_wdy = sum_wdy * inv_d

            # Complete dX once the row statistics are available.
            for j in cutlass.range_constexpr(num_column_tiles):
                tile_coord = ((None, None), (row_block, j))
                tXgDX = thread_copy.partition_D(gDX[tile_coord])
                rDX = cute.make_fragment_like(tXgDX)
                for e in cutlass.range_constexpr(self.vector_size):
                    rDX[e] = (rDirectDX_tiles[j][e] + (rWdy_tiles[j][e] - (rXhat_tiles[j][e] * sum_xhat_wdy + sum_wdy)) * rstd).to(gX.element_type)
                cute.copy(tensor_copy_atom, rDX, tXgDX)

        # Phase 4: write this CTA's dW/dB partials for the companion reduction.
        for j in cutlass.range_constexpr(num_column_tiles):
            tile_coord = ((None, None), (block_idx, j))
            tXgDW = fp32_thread_copy.partition_S(gDW[tile_coord])
            tXgDB = fp32_thread_copy.partition_S(gDB[tile_coord])
            rDW = cute.make_fragment_like(tXgDW)
            rDB = cute.make_fragment_like(tXgDB)
            for e in cutlass.range_constexpr(self.vector_size):
                rDW[e] = rDW_accum[j * self.vector_size + e]
                rDB[e] = rDB_accum[j * self.vector_size + e]
            cute.copy(fp32_copy_atom, rDW, tXgDW)
            cute.copy(fp32_copy_atom, rDB, tXgDB)

    @cute.jit
    def __call__(
        self,
        mDYSilu: cute.Tensor,
        mDYX: cute.Tensor,
        mDYLmsd: cute.Tensor,
        mX: cute.Tensor,
        mU: cute.Tensor,
        mW: cute.Tensor,
        mB: cute.Tensor,
        mMask: cute.Tensor,
        mDX: cute.Tensor,
        mDU: cute.Tensor,
        mMean: cute.Tensor,
        mRstd: cute.Tensor,
        mDW: cute.Tensor,
        mDB: cute.Tensor,
        drop: cutlass.Float32,
        ncols: cutlass.Int32,
        nblk: cutlass.Int32,
        grid: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        thr_layout = cute.make_ordered_layout((self.rows_per_cta, 64), order=(1, 0))
        val_layout = cute.make_ordered_layout((1, self.vector_size), order=(1, 0))
        tiler, _ = cute.make_layout_tv(thr_layout, val_layout)
        tile = lambda tensor: cute.zipped_divide(tensor, tiler)
        gX = tile(mX)
        param_layout = cute.make_layout((1, cute.size(mW)), stride=(0, 1))
        mW2 = cute.make_tensor(mW.iterator, param_layout)
        mB2 = cute.make_tensor(mB.iterator, param_layout)
        self.kernel(
            tile(mDYSilu),
            tile(mDYX),
            tile(mDYLmsd),
            gX,
            tile(mU),
            tile(mW2),
            tile(mB2),
            tile(mMask),
            tile(mDX),
            tile(mDU),
            mMean,
            mRstd,
            tile(mDW),
            tile(mDB),
            drop,
            thr_layout,
            val_layout,
            cute.size(gX, mode=[1, 1]),
            ncols,
            nblk,
            grid,
        ).launch(
            grid=(grid, 1, 1),
            block=(cute.size(thr_layout), 1, 1),
            smem=self.rows_per_cta * 4 * 4,
            min_blocks_per_mp=self.min_blocks_per_mp,
            stream=stream,
        )


class LnMulDropoutGradReduce:
    """Reduce persistent-CTA dW/dB partials to the public gradients."""

    threads = 256
    columns_per_cta = 4
    warps = threads // 32
    rows_per_cta = threads // columns_per_cta
    prefetch_batch = 16

    @cute.kernel
    def kernel(
        self,
        mDW: cute.Tensor,
        mDB: cute.Tensor,
        mFinalDW: cute.Tensor,
        mFinalDB: cute.Tensor,
        nrows: cutlass.Int32,
        ncols: cutlass.Int32,
    ):
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        lane = thread_idx % 32
        warp = thread_idx // 32
        column_in_cta = lane % self.columns_per_cta
        row_in_warp = lane // self.columns_per_cta
        column = block_idx * self.columns_per_cta + column_in_cta
        row = warp * 8 + row_in_warp

        dw = cutlass.Float32(0.0)
        db = cutlass.Float32(0.0)
        rDW = cute.make_rmem_tensor(self.prefetch_batch, cutlass.Float32)
        rDB = cute.make_rmem_tensor(self.prefetch_batch, cutlass.Float32)
        for row_base in cutlass.range(
            row,
            nrows,
            self.rows_per_cta * self.prefetch_batch,
            unroll=1,
        ):
            # Expose independent global loads before consuming them, then add
            # in the original row, row+64, ... order. This retains bitwise
            # results while hiding the dependent workspace-read latency.
            for item in cutlass.range_constexpr(self.prefetch_batch):
                rDW[item] = cutlass.Float32(0.0)
                rDB[item] = cutlass.Float32(0.0)
                partial_row = row_base + item * self.rows_per_cta
                if partial_row < nrows and column < ncols:
                    rDW[item] = mDW[(partial_row, column)]
                    rDB[item] = mDB[(partial_row, column)]
            for item in cutlass.range_constexpr(self.prefetch_batch):
                dw = dw + rDW[item]
                db = db + rDB[item]

        # Lanes 0/4/.../28 own one column. These XOR distances reduce the
        # eight row positions while keeping the four columns independent.
        for offset in cutlass.range_constexpr(3):
            delta = 1 << (offset + 2)
            dw = dw + cute.arch.shuffle_sync_bfly(dw, delta)
            db = db + cute.arch.shuffle_sync_bfly(db, delta)

        partials = cute.make_tensor(
            cute.arch.alloc_smem(
                cutlass.Float32,
                self.warps * self.columns_per_cta * 2,
            ),
            cute.make_layout((self.warps, self.columns_per_cta, 2)),
        )
        if row_in_warp == 0:
            partials[(warp, column_in_cta, 0)] = dw
            partials[(warp, column_in_cta, 1)] = db
        cute.arch.sync_threads()

        if thread_idx < self.columns_per_cta:
            final_dw = cutlass.Float32(0.0)
            final_db = cutlass.Float32(0.0)
            for source_warp in cutlass.range_constexpr(self.warps):
                final_dw = final_dw + partials[(source_warp, thread_idx, 0)]
                final_db = final_db + partials[(source_warp, thread_idx, 1)]
            output_column = block_idx * self.columns_per_cta + thread_idx
            if output_column < ncols:
                mFinalDW[output_column] = final_dw.to(mFinalDW.element_type)
                mFinalDB[output_column] = final_db.to(mFinalDB.element_type)

    @cute.jit
    def __call__(
        self,
        mDW: cute.Tensor,
        mDB: cute.Tensor,
        mFinalDW: cute.Tensor,
        mFinalDB: cute.Tensor,
        nrows: cutlass.Int32,
        ncols: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        grid = (cute.size(mFinalDW) + self.columns_per_cta - 1) // self.columns_per_cta
        self.kernel(mDW, mDB, mFinalDW, mFinalDB, nrows, ncols).launch(
            grid=(grid, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.warps * self.columns_per_cta * 2 * 4,
            stream=stream,
        )
