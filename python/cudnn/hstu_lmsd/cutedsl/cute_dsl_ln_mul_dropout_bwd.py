# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL implementation of HSTU LMSD backward.

The persistent grid-stride main kernel emits per-CTA dW/dB partials. The
companion ``LnMulDropoutGradReduce`` kernel reduces those partials to the
public gradients without depending on Triton.
"""

from cuda.bindings import driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as _cm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.runtime import from_dlpack

VEC = 8
ROWS = 1
ALIGN = 16
TARGET_TILES = 13568

FAST_DIV = True
MIN_BLOCKS_PER_MP = 12


@dsl_user_op
def _domain_offset_i64(
    coord: cute.Coord,
    tensor: cute.Tensor,
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    """Rebase a tensor with a 64-bit byte offset, preserving its layout."""
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(tensor.stride)
    assert len(flat_coord_i64) == len(flat_stride)
    element_offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride))
    assert isinstance(tensor.iterator, cute.Pointer)
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint()
        + element_offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


class LnMulDropoutBackward:
    """Compile-time configuration and device code for LMSD backward.

    Following QuACK's CuTe DSL structure, the JIT-callable object owns launch
    policy and its ``kernel`` method owns device-side work.  Optional paths are
    immutable instance attributes instead of mutable module globals. Tensor
    names use the same scope convention: ``m`` for whole tensors, ``g`` for
    tiled global-memory views, ``t`` for per-thread partitions, ``r`` for
    registers, and ``s`` for shared memory.
    """

    def __init__(self, compute_y: bool):
        self.compute_y = compute_y
        self.fast_div = FAST_DIV
        self.min_blocks_per_mp = MIN_BLOCKS_PER_MP
        self.rows_per_cta = ROWS
        self.vector_size = VEC

    @cute.kernel
    def kernel(
        self,
        gDY0: cute.Tensor,
        gDY1: cute.Tensor,
        gDY2: cute.Tensor,
        gX: cute.Tensor,
        gU: cute.Tensor,
        gW: cute.Tensor,
        gB: cute.Tensor,
        gM: cute.Tensor,
        gDX: cute.Tensor,
        gDU: cute.Tensor,
        gY0: cute.Tensor,
        gY1: cute.Tensor,
        gY2: cute.Tensor,
        gMean: cute.Tensor,
        gRstd: cute.Tensor,
        gDW: cute.Tensor,
        gDB: cute.Tensor,
        drop: cutlass.Float32,
        thr_layout: cute.Layout,
        val_layout: cute.Layout,
        NTILE: cutlass.Constexpr,
        nrows: cutlass.Int32,
        ncols: cutlass.Int32,
        nblk: cutlass.Int32,
        iters: cutlass.Int32,
        grid: cutlass.Int32,
    ):
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        thread_in_row = thread_idx % 64
        row_in_cta = thread_idx // 64
        warp_in_row = thread_in_row // 32
        reduction_smem = cute.make_tensor(
            cute.arch.alloc_smem(cutlass.Float32, self.rows_per_cta * 4),
            cute.make_layout(self.rows_per_cta * 4),
        )

        # The wide X/U/dY row streams have no useful L1 reuse. Bypass L1 while
        # retaining L2; keep the compact byte mask on the default cache policy.
        wide_load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(), gX.element_type,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL,
            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_NORMAL,
            l2_prefetch_size=cute.nvgpu.L2PrefetchSize.NONE)
        mask_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gM.element_type
        )
        tensor_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gX.element_type
        )
        fp32_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gDW.element_type
        )
        thread_copy = cute.make_tiled_copy_tv(
            tensor_copy_atom, thr_layout, val_layout
        ).get_slice(thread_idx)
        mask_thread_copy = cute.make_tiled_copy_tv(
            mask_copy_atom, thr_layout, val_layout
        ).get_slice(thread_idx)
        fp32_thread_copy = cute.make_tiled_copy_tv(
            fp32_copy_atom, thr_layout, val_layout
        ).get_slice(thread_idx)

        scale = cutlass.Float32(1.0) / (cutlass.Float32(1.0) - drop)
        inv_d = cutlass.Float32(1.0) / ncols.to(cutlass.Float32)

        rW_tiles, rB_tiles = [], []
        for j in cutlass.range_constexpr(NTILE):
            for src, dst in ((gW, rW_tiles), (gB, rB_tiles)):
                t = thread_copy.partition_S(src[((None, None), (0, j))])
                f = cute.make_fragment_like(t)
                cute.copy(wide_load_atom, t, f)
                dst.append(f)
        rDW_accum = cute.make_rmem_tensor(
            NTILE * self.vector_size, cutlass.Float32
        )
        rDB_accum = cute.make_rmem_tensor(
            NTILE * self.vector_size, cutlass.Float32
        )
        for e in cutlass.range_constexpr(NTILE * self.vector_size):
            rDW_accum[e] = cutlass.Float32(0.0)
            rDB_accum[e] = cutlass.Float32(0.0)

        for it in cutlass.range(iters):
            row_block = block_idx + it * grid
            if row_block < nblk:
                row = row_block * self.rows_per_cta + row_in_cta
                valid = row < nrows
                if valid:
                    row_coord = ((0, 0), (row_block, 0))
                    gDY0_row = _domain_offset_i64(row_coord, gDY0)
                    gDY1_row = _domain_offset_i64(row_coord, gDY1)
                    gDY2_row = _domain_offset_i64(row_coord, gDY2)
                    gU_row = _domain_offset_i64(row_coord, gU)
                    gY0_row = _domain_offset_i64(row_coord, gY0)
                    gY1_row = _domain_offset_i64(row_coord, gY1)
                    gY2_row = _domain_offset_i64(row_coord, gY2)
                    mean = gMean[row]
                    rstd = gRstd[row]
                    norm_bias = -mean * rstd
                    sum_xhat_wdy = cutlass.Float32(0.0)
                    sum_wdy = cutlass.Float32(0.0)
                    rXhat_tiles, rWdy_tiles, rDirectDX_tiles = [], [], []
                    for j in cutlass.range_constexpr(NTILE):
                        tile_coord = ((None, None), (row_block, j))
                        row_tile_coord = ((None, None), (0, j))
                        tXgX = thread_copy.partition_S(gX[tile_coord])
                        tXgU = thread_copy.partition_S(gU_row[row_tile_coord])
                        tXgDY0 = thread_copy.partition_S(gDY0_row[row_tile_coord])
                        tXgDY1 = thread_copy.partition_S(gDY1_row[row_tile_coord])
                        tXgDY2 = thread_copy.partition_S(gDY2_row[row_tile_coord])
                        tXgMask = mask_thread_copy.partition_S(gM[tile_coord])

                        rX = cute.make_fragment_like(tXgX)
                        rU = cute.make_fragment_like(tXgU)
                        rDY0 = cute.make_fragment_like(tXgDY0)
                        rDY1 = cute.make_fragment_like(tXgDY1)
                        rDY2 = cute.make_fragment_like(tXgDY2)
                        rMask = cute.make_fragment_like(tXgMask)
                        cute.copy(wide_load_atom, tXgX, rX)
                        cute.copy(wide_load_atom, tXgU, rU)
                        cute.copy(wide_load_atom, tXgDY0, rDY0)
                        cute.copy(wide_load_atom, tXgDY1, rDY1)
                        cute.copy(wide_load_atom, tXgDY2, rDY2)
                        cute.copy(mask_copy_atom, tXgMask, rMask)

                        tXgY0 = thread_copy.partition_S(gY0_row[row_tile_coord])
                        tXgY1 = thread_copy.partition_S(gY1_row[row_tile_coord])
                        tXgY2 = thread_copy.partition_S(gY2_row[row_tile_coord])
                        tXgDU = thread_copy.partition_S(gDU[tile_coord])
                        rY0 = cute.make_fragment_like(tXgY0)
                        rY1 = cute.make_fragment_like(tXgY1)
                        rY2 = cute.make_fragment_like(tXgY2)
                        rDU = cute.make_fragment_like(tXgDU)
                        rXhat = cute.make_rmem_tensor(self.vector_size, cutlass.Float32)
                        rWdy = cute.make_rmem_tensor(self.vector_size, cutlass.Float32)
                        rDirectDX = cute.make_rmem_tensor(self.vector_size, cutlass.Float32)

                        for e in cutlass.range_constexpr(self.vector_size):
                            xf = rX[e].to(cutlass.Float32)
                            uf = rU[e].to(cutlass.Float32)
                            mb = rMask[e].to(cutlass.Int32)
                            zero = cutlass.Float32(0.0)
                            du_v = (rDY0[e].to(cutlass.Float32) * scale
                                    if (mb & 4) != 0 else zero)
                            dx_v = (rDY1[e].to(cutlass.Float32) * scale
                                    if (mb & 2) != 0 else zero)
                            dy_v = (rDY2[e].to(cutlass.Float32) * scale
                                    if (mb & 1) != 0 else zero)

                            xh = _cm.fma(xf, rstd, norm_bias)
                            ln = _cm.fma(xh, rW_tiles[j][e].to(cutlass.Float32),
                                         rB_tiles[j][e].to(cutlass.Float32))
                            den = (cutlass.Float32(1.0)
                                   + cute.arch.exp2(-uf * cutlass.Float32(
                                       1.4426950408889634)))
                            if cutlass.const_expr(self.fast_div):
                                sig = cute.arch.rcp_approx(den)
                            else:
                                sig = cutlass.Float32(1.0) / den
                            silu = uf * sig
                            dsilu = _cm.fma(silu, cutlass.Float32(1.0) - sig, sig)

                            du_y = dy_v * ln * dsilu
                            dy_v = dy_v * silu
                            rDU[e] = (du_y + du_v * dsilu).to(gX.element_type)

                            silu_s = silu * scale
                            rY0[e] = (silu_s if (mb & 4) != 0 else zero
                                      ).to(gX.element_type)
                            rY1[e] = (xf * scale if (mb & 2) != 0 else zero
                                      ).to(gX.element_type)
                            rY2[e] = (ln * silu_s if (mb & 1) != 0 else zero
                                      ).to(gX.element_type)

                            wd = rW_tiles[j][e].to(cutlass.Float32) * dy_v
                            xh_e = xh
                            sum_xhat_wdy = sum_xhat_wdy + xh_e * wd
                            sum_wdy = sum_wdy + wd
                            rDW_accum[j * self.vector_size + e] = rDW_accum[j * self.vector_size + e] + dy_v * xh_e
                            rDB_accum[j * self.vector_size + e] = rDB_accum[j * self.vector_size + e] + dy_v
                            rXhat[e] = xh_e
                            rWdy[e] = wd
                            rDirectDX[e] = dx_v

                        cute.copy(tensor_copy_atom, rDU, tXgDU)
                        if cutlass.const_expr(self.compute_y):
                            cute.copy(tensor_copy_atom, rY0, tXgY0)
                            cute.copy(tensor_copy_atom, rY1, tXgY1)
                            cute.copy(tensor_copy_atom, rY2, tXgY2)
                        rXhat_tiles.append(rXhat)
                        rWdy_tiles.append(rWdy)
                        rDirectDX_tiles.append(rDirectDX)

                    for off in cutlass.range_constexpr(5):
                        sum_xhat_wdy = sum_xhat_wdy + cute.arch.shuffle_sync_bfly(sum_xhat_wdy, 1 << off)
                        sum_wdy = sum_wdy + cute.arch.shuffle_sync_bfly(sum_wdy, 1 << off)
                    if thread_in_row % 32 == 0:
                        reduction_smem[row_in_cta * 4 + warp_in_row * 2 + 0] = sum_xhat_wdy
                        reduction_smem[row_in_cta * 4 + warp_in_row * 2 + 1] = sum_wdy
                    cute.arch.sync_threads()
                    sum_xhat_wdy = reduction_smem[row_in_cta * 4 + 0] + reduction_smem[row_in_cta * 4 + 2]
                    sum_wdy = reduction_smem[row_in_cta * 4 + 1] + reduction_smem[row_in_cta * 4 + 3]
                    cute.arch.sync_threads()
                    sum_xhat_wdy = sum_xhat_wdy * inv_d
                    sum_wdy = sum_wdy * inv_d

                    for j in cutlass.range_constexpr(NTILE):
                        tile_coord = ((None, None), (row_block, j))
                        tXgDX = thread_copy.partition_S(gDX[tile_coord])
                        rDX = cute.make_fragment_like(tXgDX)
                        for e in cutlass.range_constexpr(self.vector_size):
                            rDX[e] = (rDirectDX_tiles[j][e]
                                      + (rWdy_tiles[j][e] - (rXhat_tiles[j][e] * sum_xhat_wdy + sum_wdy)) * rstd
                                      ).to(gX.element_type)
                        cute.copy(tensor_copy_atom, rDX, tXgDX)

        for j in cutlass.range_constexpr(NTILE):
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
        mDY0: cute.Tensor,
        mDY1: cute.Tensor,
        mDY2: cute.Tensor,
        mX: cute.Tensor,
        mU: cute.Tensor,
        mW: cute.Tensor,
        mB: cute.Tensor,
        mMask: cute.Tensor,
        mDX: cute.Tensor,
        mDU: cute.Tensor,
        mY0: cute.Tensor,
        mY1: cute.Tensor,
        mY2: cute.Tensor,
        mMean: cute.Tensor,
        mRstd: cute.Tensor,
        mDW: cute.Tensor,
        mDB: cute.Tensor,
        drop: cutlass.Float32,
        nrows: cutlass.Int32,
        ncols: cutlass.Int32,
        nblk: cutlass.Int32,
        iters: cutlass.Int32,
        grid: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        thr_layout = cute.make_ordered_layout(
            (self.rows_per_cta, 64), order=(1, 0)
        )
        val_layout = cute.make_ordered_layout(
            (1, self.vector_size), order=(1, 0)
        )
        tiler, _ = cute.make_layout_tv(thr_layout, val_layout)
        tile = lambda tensor: cute.zipped_divide(tensor, tiler)
        gX = tile(mX)
        param_layout = cute.make_layout((1, cute.size(mW)), stride=(0, 1))
        mW2 = cute.make_tensor(mW.iterator, param_layout)
        mB2 = cute.make_tensor(mB.iterator, param_layout)
        self.kernel(
            tile(mDY0),
            tile(mDY1),
            tile(mDY2),
            gX,
            tile(mU),
            tile(mW2),
            tile(mB2),
            tile(mMask),
            tile(mDX),
            tile(mDU),
            tile(mY0),
            tile(mY1),
            tile(mY2),
            mMean,
            mRstd,
            tile(mDW),
            tile(mDB),
            drop,
            thr_layout,
            val_layout,
            cute.size(gX, mode=[1, 1]),
            nrows,
            ncols,
            nblk,
            iters,
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


_COMPILED: dict[tuple, object] = {}


def _compiled(
    key: tuple,
    args: tuple,
    *,
    compute_y: bool,
):
    fn = _COMPILED.get(key)
    if fn is None:
        op = LnMulDropoutBackward(compute_y=compute_y)
        fn = cute.compile(op, *args)
        _COMPILED[key] = fn
    return fn


def ln_mul_dropout_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    random_mask: torch.Tensor,
    dropout_ratio: float,
    compute_y: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (dx, du, y, dw_partial, db_partial), matching the CUDA reference.

    y is an empty tensor when compute_y is False, matching the Triton path.

    dw/db are **per-tile partial sums**. The cross-tile reduction is left to the
    existing Triton kernel (_ln_mul_dropout_bwd_dwdb, measured at 0.017 ms), so
    it is not reimplemented here.
    """
    N, D = x.shape
    assert D % (VEC * 32) == 0, f"D must be a multiple of {VEC * 32}, got {D}"
    dy, x, u = dy.detach(), x.detach(), u.detach()
    weight, bias = weight.detach(), bias.detach()
    mean, rstd, random_mask = mean.detach(), rstd.detach(), random_mask.detach()

    dev = x.device
    dx = torch.empty((N, D), dtype=x.dtype, device=dev)
    du = torch.empty((N, D), dtype=x.dtype, device=dev)
    if compute_y:
        y = torch.empty((N, 3 * D), dtype=x.dtype, device=dev)
        y0, y1, y2 = y[:, :D], y[:, D:2 * D], y[:, 2 * D:]
    else:
        # never written; dx stands in so the kernel still gets three valid
        # tensor arguments without an 8.42 GB allocation
        y = torch.empty(0, dtype=x.dtype, device=dev)
        y0 = y1 = y2 = dx

    grid = max(1, TARGET_TILES // ROWS)
    n_tiles = grid * ROWS
    dw_p = torch.empty((n_tiles, D), dtype=torch.float32, device=dev)
    db_p = torch.empty((n_tiles, D), dtype=torch.float32, device=dev)

    w2 = weight.unsqueeze(0).expand(ROWS, D).contiguous()
    b2 = bias.unsqueeze(0).expand(ROWS, D).contiguous()
    kw = {"assumed_align": ALIGN}

    nblk = (N + ROWS - 1) // ROWS
    iters = (nblk + grid - 1) // grid
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    args = (
        from_dlpack(dy[:, :D], **kw), from_dlpack(dy[:, D:2 * D], **kw),
        from_dlpack(dy[:, 2 * D:], **kw),
        from_dlpack(x, **kw), from_dlpack(u, **kw),
        from_dlpack(w2, **kw), from_dlpack(b2, **kw),
        from_dlpack(random_mask, **kw),
        from_dlpack(dx, **kw), from_dlpack(du, **kw),
        from_dlpack(y0, **kw), from_dlpack(y1, **kw),
        from_dlpack(y2, **kw),
        from_dlpack(mean), from_dlpack(rstd),
        from_dlpack(dw_p, **kw), from_dlpack(db_p, **kw),
        cutlass.Float32(dropout_ratio), cutlass.Int32(N),
        cutlass.Int32(D), cutlass.Int32(nblk), cutlass.Int32(iters),
        cutlass.Int32(grid), stream,
    )
    key = (
        N, D, x.dtype, FAST_DIV, MIN_BLOCKS_PER_MP, compute_y,
        "i64_single_launch",
    )
    _compiled(key, args, compute_y=compute_y)(*args)
    return dx, du, y, dw_p, db_p
