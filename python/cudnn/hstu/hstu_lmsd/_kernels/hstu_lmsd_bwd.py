# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL implementation of HSTU LMSD backward.

The optional activated-U and X segments of ``dy`` precede the mandatory LMSD
segment. Dropout, SiLU, and dWeight work are compile-time specializations.

The persistent grid-stride main kernel emits per-CTA dW/dB partials. The
companion ``HSTULMSDGradReduce`` kernel reduces those partials to the
public gradients without depending on Triton.
"""

from cuda.bindings import driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as _cm
from cutlass import const_expr

from ._common import (
    _DROPOUT_KEEP_LMSD_BIT,
    _DROPOUT_KEEP_U_BIT,
    _DROPOUT_KEEP_X_BIT,
    LOG2E,
    domain_offset_i64,
    is_supported_hidden_size,
)
from ._config import HSTULMSDBwdConfig, HSTULMSDGradReduceConfig, WARP_SIZE


class HSTULMSDBackward:
    """Compile-time configuration and device code for LMSD backward.

    The JIT-callable object owns launch policy and its ``kernel`` method owns
    device-side work. Tensor names use the same scope convention: ``m`` for
    whole tensors, ``g`` for tiled global-memory views, ``t`` for per-thread
    partitions, ``r`` for register fragments, and ``s`` for shared memory.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        apply_u_silu: bool = True,
        concat_u: bool = True,
        concat_x: bool = True,
        has_dropout: bool = True,
        compute_dweight: bool = True,
        config: HSTULMSDBwdConfig | None = None,
    ):
        if not is_supported_hidden_size(hidden_size):
            raise ValueError(f"unsupported HSTU LMSD hidden size: {hidden_size}")
        self.config = HSTULMSDBwdConfig.from_hidden_size(hidden_size) if config is None else config
        self.hidden_size = hidden_size
        self.min_blocks_per_mp = self.config.min_blocks_per_mp
        self.rows_per_cta = self.config.rows_per_cta
        assert self.rows_per_cta == 1, "this kernel requires one row per CTA"
        self.threads_per_row = self.config.threads_per_row
        self.vector_size = self.config.vector_size
        self.warps_per_row = self.threads_per_row // WARP_SIZE
        self.apply_u_silu = apply_u_silu
        self.concat_u = concat_u
        self.concat_x = concat_x
        self.has_dropout = has_dropout
        self.compute_dweight = compute_dweight

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
        thread_in_row = thread_idx % self.threads_per_row
        warp_in_row = thread_in_row // WARP_SIZE
        reduction_smem = cute.make_tensor(
            cute.arch.alloc_smem(cutlass.Float32, self.rows_per_cta * self.warps_per_row * 2),
            cute.make_layout(self.rows_per_cta * self.warps_per_row * 2),
        )

        # Phase 1: establish copy and row-reduction resources. The wide X/U/dY
        # row streams have no useful L1 reuse. Bypass L1 while
        # retaining L2; keep the compact byte mask on the default cache policy.
        wide_load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            gX.element_type,
            num_bits_per_copy=self.vector_size * 16,
            load_cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL,
            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_NORMAL,
            l2_prefetch_size=cute.nvgpu.L2PrefetchSize.NONE,
        )
        mask_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gMask.element_type) if const_expr(self.has_dropout) else None
        tensor_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type)
        fp32_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gDB.element_type)
        thread_copy = cute.make_tiled_copy_tv(tensor_copy_atom, thr_layout, val_layout).get_slice(thread_idx)
        mask_thread_copy = cute.make_tiled_copy_tv(mask_copy_atom, thr_layout, val_layout).get_slice(thread_idx) if const_expr(self.has_dropout) else None
        fp32_thread_copy = cute.make_tiled_copy_tv(fp32_copy_atom, thr_layout, val_layout).get_slice(thread_idx)

        scale = cutlass.Float32(1.0) / (cutlass.Float32(1.0) - drop) if const_expr(self.has_dropout) else cutlass.Float32(1.0)
        inv_d = cutlass.Float32(1.0) / ncols.to(cutlass.Float32)

        # Phase 2: retain W/B and parameter-gradient accumulators across the
        # persistent row loop.
        rW_tiles, rB_tiles = [], []
        for j in cutlass.range_constexpr(num_column_tiles):
            for src, dst in ((gW, rW_tiles), (gB, rB_tiles)):
                t = thread_copy.partition_S(src[((None, None), (0, j))])
                f = cute.make_fragment_like(t)
                vector_index = j * self.threads_per_row + thread_in_row
                if const_expr(self.hidden_size % (self.threads_per_row * self.vector_size) == 0) or vector_index < self.hidden_size // self.vector_size:
                    cute.copy(wide_load_atom, t, f)
                dst.append(f)
        rDW_accum = cute.make_rmem_tensor(num_column_tiles * self.vector_size, cutlass.Float32) if const_expr(self.compute_dweight) else None
        rDB_accum = cute.make_rmem_tensor(num_column_tiles * self.vector_size, cutlass.Float32)
        for e in cutlass.range_constexpr(num_column_tiles * self.vector_size):
            if const_expr(self.compute_dweight):
                rDW_accum[e] = cutlass.Float32(0.0)
            rDB_accum[e] = cutlass.Float32(0.0)

        tXgMask_rows = []
        for j in cutlass.range_constexpr(num_column_tiles):
            all_rows = ((None, None), (None, j))
            if const_expr(self.has_dropout):
                tXgMask_rows.append(mask_thread_copy.partition_S(gMask[all_rows]))

        # Phase 3: process rows assigned to this persistent CTA.
        for row_block in cutlass.range(block_idx, nblk, grid):
            row = row_block
            row_coord = ((0, 0), (row_block, 0))
            gDYSilu_row = domain_offset_i64(row_coord, gDYSilu) if const_expr(self.concat_u) else None
            gDYX_row = domain_offset_i64(row_coord, gDYX) if const_expr(self.concat_x) else None
            gDYLmsd_row = domain_offset_i64(row_coord, gDYLmsd)
            gX_row = domain_offset_i64(row_coord, gX)
            gU_row = domain_offset_i64(row_coord, gU)
            mean = gMean[row]
            rstd = gRstd[row]
            norm_bias = -mean * rstd
            sum_xhat_wdy = cutlass.Float32(0.0)
            sum_wdy = cutlass.Float32(0.0)
            rXhat_tiles, rWdy_tiles, rDirectDX_tiles = [], [], []
            for j in cutlass.range_constexpr(num_column_tiles):
                rXhat = cute.make_rmem_tensor(self.vector_size, cutlass.Float32)
                rWdy = cute.make_rmem_tensor(self.vector_size, cutlass.Float32)
                rDirectDX = cute.make_rmem_tensor(self.vector_size, cutlass.Float32)
                vector_index = j * self.threads_per_row + thread_in_row
                if const_expr(self.hidden_size % (self.threads_per_row * self.vector_size) == 0) or vector_index < self.hidden_size // self.vector_size:
                    tile_coord = ((None, None), (row_block, j))
                    row_tile_coord = ((None, None), (0, j))
                    tXgX = thread_copy.partition_S(gX_row[row_tile_coord])
                    tXgU = thread_copy.partition_S(gU_row[row_tile_coord])
                    tXgDYSilu = thread_copy.partition_S(gDYSilu_row[row_tile_coord]) if const_expr(self.concat_u) else None
                    tXgDYX = thread_copy.partition_S(gDYX_row[row_tile_coord]) if const_expr(self.concat_x) else None
                    tXgDYLmsd = thread_copy.partition_S(gDYLmsd_row[row_tile_coord])
                    tXgMask = tXgMask_rows[j][None, None, None, row_block] if const_expr(self.has_dropout) else None

                    rX = cute.make_fragment_like(tXgX)
                    rU = cute.make_fragment_like(tXgU)
                    rDYSilu = cute.make_fragment_like(tXgDYSilu) if const_expr(self.concat_u) else None
                    rDYX = cute.make_fragment_like(tXgDYX) if const_expr(self.concat_x) else None
                    rDYLmsd = cute.make_fragment_like(tXgDYLmsd)
                    rMask = cute.make_fragment_like(tXgMask) if const_expr(self.has_dropout) else None
                    cute.copy(wide_load_atom, tXgX, rX)
                    cute.copy(wide_load_atom, tXgU, rU)
                    if const_expr(self.concat_u):
                        cute.copy(wide_load_atom, tXgDYSilu, rDYSilu)
                    if const_expr(self.concat_x):
                        cute.copy(wide_load_atom, tXgDYX, rDYX)
                    cute.copy(wide_load_atom, tXgDYLmsd, rDYLmsd)
                    if const_expr(self.has_dropout):
                        cute.copy(mask_copy_atom, tXgMask, rMask)

                    tXgDU = thread_copy.partition_D(gDU[tile_coord])
                    rDU = cute.make_fragment_like(tXgDU)
                    for e in cutlass.range_constexpr(self.vector_size):
                        xf = rX[e].to(cutlass.Float32)
                        uf = rU[e].to(cutlass.Float32)
                        zero = cutlass.Float32(0.0)
                        direct_du = zero
                        direct_dx = zero
                        if const_expr(self.has_dropout):
                            mb = rMask[e].to(cutlass.Int32)
                            if const_expr(self.concat_u):
                                direct_du = rDYSilu[e].to(cutlass.Float32) * scale if (mb & _DROPOUT_KEEP_U_BIT) != 0 else zero
                            if const_expr(self.concat_x):
                                direct_dx = rDYX[e].to(cutlass.Float32) * scale if (mb & _DROPOUT_KEEP_X_BIT) != 0 else zero
                            grad_lmsd = rDYLmsd[e].to(cutlass.Float32) * scale if (mb & _DROPOUT_KEEP_LMSD_BIT) != 0 else zero
                        else:
                            if const_expr(self.concat_u):
                                direct_du = rDYSilu[e].to(cutlass.Float32)
                            if const_expr(self.concat_x):
                                direct_dx = rDYX[e].to(cutlass.Float32)
                            grad_lmsd = rDYLmsd[e].to(cutlass.Float32)

                        xh = _cm.fma(xf, rstd, norm_bias)
                        ln = _cm.fma(xh, rW_tiles[j][e].to(cutlass.Float32), rB_tiles[j][e].to(cutlass.Float32))
                        if const_expr(self.apply_u_silu):
                            den = cutlass.Float32(1.0) + cute.arch.exp2(-uf * cutlass.Float32(LOG2E))
                            sig = cute.arch.rcp_approx(den)
                            activated_u = uf * sig
                            dactivated_u = _cm.fma(activated_u, cutlass.Float32(1.0) - sig, sig)
                        else:
                            activated_u = uf
                            dactivated_u = cutlass.Float32(1.0)

                        du_from_lmsd = grad_lmsd * ln * dactivated_u
                        grad_layer_norm = grad_lmsd * activated_u
                        rDU[e] = (du_from_lmsd + direct_du * dactivated_u).to(gX.element_type)

                        wd = rW_tiles[j][e].to(cutlass.Float32) * grad_layer_norm
                        xh_e = xh
                        sum_xhat_wdy = sum_xhat_wdy + xh_e * wd
                        sum_wdy = sum_wdy + wd
                        if const_expr(self.compute_dweight):
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
            if thread_in_row % WARP_SIZE == 0:
                reduction_smem[warp_in_row * 2 + 0] = sum_xhat_wdy
                reduction_smem[warp_in_row * 2 + 1] = sum_wdy
            cute.arch.sync_threads()
            sum_xhat_wdy = cutlass.Float32(0.0)
            sum_wdy = cutlass.Float32(0.0)
            for source_warp in cutlass.range_constexpr(self.warps_per_row):
                sum_xhat_wdy = sum_xhat_wdy + reduction_smem[source_warp * 2]
                sum_wdy = sum_wdy + reduction_smem[source_warp * 2 + 1]
            cute.arch.sync_threads()
            sum_xhat_wdy = sum_xhat_wdy * inv_d
            sum_wdy = sum_wdy * inv_d

            # Complete dX once the row statistics are available.
            for j in cutlass.range_constexpr(num_column_tiles):
                vector_index = j * self.threads_per_row + thread_in_row
                if const_expr(self.hidden_size % (self.threads_per_row * self.vector_size) == 0) or vector_index < self.hidden_size // self.vector_size:
                    tile_coord = ((None, None), (row_block, j))
                    tXgDX = thread_copy.partition_D(gDX[tile_coord])
                    rDX = cute.make_fragment_like(tXgDX)
                    for e in cutlass.range_constexpr(self.vector_size):
                        rDX[e] = (rDirectDX_tiles[j][e] + (rWdy_tiles[j][e] - (rXhat_tiles[j][e] * sum_xhat_wdy + sum_wdy)) * rstd).to(gX.element_type)
                    cute.copy(tensor_copy_atom, rDX, tXgDX)

        # Phase 4: write this CTA's dW/dB partials for the companion reduction.
        for j in cutlass.range_constexpr(num_column_tiles):
            vector_index = j * self.threads_per_row + thread_in_row
            if const_expr(self.hidden_size % (self.threads_per_row * self.vector_size) == 0) or vector_index < self.hidden_size // self.vector_size:
                tile_coord = ((None, None), (block_idx, j))
                tXgDW = fp32_thread_copy.partition_S(gDW[tile_coord]) if const_expr(self.compute_dweight) else None
                tXgDB = fp32_thread_copy.partition_S(gDB[tile_coord])
                rDW = cute.make_fragment_like(tXgDW) if const_expr(self.compute_dweight) else None
                rDB = cute.make_fragment_like(tXgDB)
                for e in cutlass.range_constexpr(self.vector_size):
                    if const_expr(self.compute_dweight):
                        rDW[e] = rDW_accum[j * self.vector_size + e]
                    rDB[e] = rDB_accum[j * self.vector_size + e]
                if const_expr(self.compute_dweight):
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
        thr_layout = cute.make_ordered_layout((self.rows_per_cta, self.threads_per_row), order=(1, 0))
        val_layout = cute.make_ordered_layout((1, self.vector_size), order=(1, 0))
        tiler, _ = cute.make_layout_tv(thr_layout, val_layout)
        tile = lambda tensor: cute.zipped_divide(tensor, tiler)
        gX = tile(mX)
        param_layout = cute.make_layout((1, cute.size(mW)), stride=(0, 1))
        mW2 = cute.make_tensor(mW.iterator, param_layout)
        mB2 = cute.make_tensor(mB.iterator, param_layout)
        self.kernel(
            tile(mDYSilu) if const_expr(self.concat_u) else None,
            tile(mDYX) if const_expr(self.concat_x) else None,
            tile(mDYLmsd),
            gX,
            tile(mU),
            tile(mW2),
            tile(mB2),
            tile(mMask) if const_expr(self.has_dropout) else None,
            tile(mDX),
            tile(mDU),
            mMean,
            mRstd,
            tile(mDW) if const_expr(self.compute_dweight) else None,
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
            smem=self.rows_per_cta * self.warps_per_row * 2 * 4,
            min_blocks_per_mp=self.min_blocks_per_mp,
            stream=stream,
        )


class HSTULMSDGradReduce:
    """Reduce persistent-CTA dW/dB partials to the public gradients."""

    def __init__(self, compute_dweight: bool = True, config: HSTULMSDGradReduceConfig | None = None):
        self.config = HSTULMSDGradReduceConfig() if config is None else config
        self.threads = self.config.threads
        self.columns_per_cta = self.config.columns_per_cta
        self.warps = self.config.warps
        self.rows_per_warp = self.config.rows_per_warp
        self.rows_per_cta = self.config.rows_per_cta
        self.prefetch_batch = self.config.prefetch_batch
        self.compute_dweight = compute_dweight

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
        lane = thread_idx % WARP_SIZE
        warp = thread_idx // WARP_SIZE
        column_in_cta = lane % self.columns_per_cta
        row_in_warp = lane // self.columns_per_cta
        column = block_idx * self.columns_per_cta + column_in_cta
        row = warp * self.rows_per_warp + row_in_warp

        dw = cutlass.Float32(0.0)
        db = cutlass.Float32(0.0)
        rDW = cute.make_rmem_tensor(self.prefetch_batch, cutlass.Float32) if const_expr(self.compute_dweight) else None
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
                if const_expr(self.compute_dweight):
                    rDW[item] = cutlass.Float32(0.0)
                rDB[item] = cutlass.Float32(0.0)
                partial_row = row_base + item * self.rows_per_cta
                if partial_row < nrows and column < ncols:
                    if const_expr(self.compute_dweight):
                        rDW[item] = mDW[(partial_row, column)]
                    rDB[item] = mDB[(partial_row, column)]
            for item in cutlass.range_constexpr(self.prefetch_batch):
                if const_expr(self.compute_dweight):
                    dw = dw + rDW[item]
                db = db + rDB[item]

        # Lanes 0/4/.../28 own one column. These XOR distances reduce the
        # eight row positions while keeping the four columns independent.
        for offset in cutlass.range_constexpr(self.rows_per_warp.bit_length() - 1):
            delta = self.columns_per_cta << offset
            if const_expr(self.compute_dweight):
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
            if const_expr(self.compute_dweight):
                partials[(warp, column_in_cta, 0)] = dw
            partials[(warp, column_in_cta, 1)] = db
        cute.arch.sync_threads()

        if thread_idx < self.columns_per_cta:
            final_dw = cutlass.Float32(0.0)
            final_db = cutlass.Float32(0.0)
            for source_warp in cutlass.range_constexpr(self.warps):
                if const_expr(self.compute_dweight):
                    final_dw = final_dw + partials[(source_warp, thread_idx, 0)]
                final_db = final_db + partials[(source_warp, thread_idx, 1)]
            output_column = block_idx * self.columns_per_cta + thread_idx
            if output_column < ncols:
                if const_expr(self.compute_dweight):
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
        grid = (cute.size(mFinalDB) + self.columns_per_cta - 1) // self.columns_per_cta
        self.kernel(mDW, mDB, mFinalDW, mFinalDB, nrows, ncols).launch(
            grid=(grid, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.warps * self.columns_per_cta * 2 * 4,
            stream=stream,
        )
