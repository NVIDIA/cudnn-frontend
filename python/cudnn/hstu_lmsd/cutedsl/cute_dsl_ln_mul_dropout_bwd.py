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
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from cudnn.datatypes import _convert_to_cutlass_data_type

VEC = 8
ROWS = 1
ALIGN = 16
TARGET_TILES = 13568
# X, mask, DX, and DU retain the faster tiled Int32 addressing path. With
# D=512, this limit keeps their largest flattened element offset <= 2^31 - 1.
MAX_NUM_ROWS = (1 << 31) // 512

# The current low-level wrapper and public API require D == 512.

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
        tensor.iterator.toint() + element_offset * tensor.element_type.width // 8,
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
        assert self.rows_per_cta == 1, "this kernel requires one row per CTA"
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

        # The wide X/U/dY row streams have no useful L1 reuse. Bypass L1 while
        # retaining L2; keep the compact byte mask on the default cache policy.
        wide_load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            gX.element_type,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL,
            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_NORMAL,
            l2_prefetch_size=cute.nvgpu.L2PrefetchSize.NONE,
        )
        mask_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gM.element_type)
        tensor_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type)
        fp32_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gDW.element_type)
        thread_copy = cute.make_tiled_copy_tv(tensor_copy_atom, thr_layout, val_layout).get_slice(thread_idx)
        mask_thread_copy = cute.make_tiled_copy_tv(mask_copy_atom, thr_layout, val_layout).get_slice(thread_idx)
        fp32_thread_copy = cute.make_tiled_copy_tv(fp32_copy_atom, thr_layout, val_layout).get_slice(thread_idx)

        scale = cutlass.Float32(1.0) / (cutlass.Float32(1.0) - drop)
        inv_d = cutlass.Float32(1.0) / ncols.to(cutlass.Float32)

        rW_tiles, rB_tiles = [], []
        for j in cutlass.range_constexpr(NTILE):
            for src, dst in ((gW, rW_tiles), (gB, rB_tiles)):
                t = thread_copy.partition_S(src[((None, None), (0, j))])
                f = cute.make_fragment_like(t)
                cute.copy(wide_load_atom, t, f)
                dst.append(f)
        rDW_accum = cute.make_rmem_tensor(NTILE * self.vector_size, cutlass.Float32)
        rDB_accum = cute.make_rmem_tensor(NTILE * self.vector_size, cutlass.Float32)
        for e in cutlass.range_constexpr(NTILE * self.vector_size):
            rDW_accum[e] = cutlass.Float32(0.0)
            rDB_accum[e] = cutlass.Float32(0.0)

        # These tensors stay below the public API's flattened Int32 offset
        # limit, so partition them once and retain the faster tiled path.
        tXgX_rows, tXgMask_rows = [], []
        for j in cutlass.range_constexpr(NTILE):
            all_rows = ((None, None), (None, j))
            tXgX_rows.append(thread_copy.partition_S(gX[all_rows]))
            tXgMask_rows.append(mask_thread_copy.partition_S(gM[all_rows]))

        for row_block in cutlass.range(block_idx, nblk, grid):
            row = row_block
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
                tXgX = tXgX_rows[j][None, None, None, row_block]
                tXgU = thread_copy.partition_S(gU_row[row_tile_coord])
                tXgDY0 = thread_copy.partition_S(gDY0_row[row_tile_coord])
                tXgDY1 = thread_copy.partition_S(gDY1_row[row_tile_coord])
                tXgDY2 = thread_copy.partition_S(gDY2_row[row_tile_coord])
                tXgMask = tXgMask_rows[j][None, None, None, row_block]

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

                tXgY0 = thread_copy.partition_D(gY0_row[row_tile_coord])
                tXgY1 = thread_copy.partition_D(gY1_row[row_tile_coord])
                tXgY2 = thread_copy.partition_D(gY2_row[row_tile_coord])
                tXgDU = thread_copy.partition_D(gDU[tile_coord])
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
                    du_v = rDY0[e].to(cutlass.Float32) * scale if (mb & 4) != 0 else zero
                    dx_v = rDY1[e].to(cutlass.Float32) * scale if (mb & 2) != 0 else zero
                    dy_v = rDY2[e].to(cutlass.Float32) * scale if (mb & 1) != 0 else zero

                    xh = _cm.fma(xf, rstd, norm_bias)
                    ln = _cm.fma(xh, rW_tiles[j][e].to(cutlass.Float32), rB_tiles[j][e].to(cutlass.Float32))
                    den = cutlass.Float32(1.0) + cute.arch.exp2(-uf * cutlass.Float32(1.4426950408889634))
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
                    rY0[e] = (silu_s if (mb & 4) != 0 else zero).to(gX.element_type)
                    rY1[e] = (xf * scale if (mb & 2) != 0 else zero).to(gX.element_type)
                    rY2[e] = (ln * silu_s if (mb & 1) != 0 else zero).to(gX.element_type)

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
                reduction_smem[warp_in_row * 2 + 0] = sum_xhat_wdy
                reduction_smem[warp_in_row * 2 + 1] = sum_wdy
            cute.arch.sync_threads()
            sum_xhat_wdy = reduction_smem[0] + reduction_smem[2]
            sum_wdy = reduction_smem[1] + reduction_smem[3]
            cute.arch.sync_threads()
            sum_xhat_wdy = sum_xhat_wdy * inv_d
            sum_wdy = sum_wdy * inv_d

            for j in cutlass.range_constexpr(NTILE):
                tile_coord = ((None, None), (row_block, j))
                tXgDX = thread_copy.partition_D(gDX[tile_coord])
                rDX = cute.make_fragment_like(tXgDX)
                for e in cutlass.range_constexpr(self.vector_size):
                    rDX[e] = (rDirectDX_tiles[j][e] + (rWdy_tiles[j][e] - (rXhat_tiles[j][e] * sum_xhat_wdy + sum_wdy)) * rstd).to(gX.element_type)
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


_COMPILED: dict[tuple, object] = {}


def _plan_signature(tensor: torch.Tensor, *, dynamic_rows: bool) -> tuple:
    shape = tuple(tensor.shape)
    if dynamic_rows:
        shape = (None, *shape[1:])
    return shape, tuple(tensor.stride()), tensor.dtype, tensor.device


def _compiled(
    key: tuple,
    compile_args: tuple,
    *,
    compute_y: bool,
):
    fn = _COMPILED.get(key)
    if fn is None:
        op = LnMulDropoutBackward(compute_y=compute_y)
        fn = cute.compile(op, *compile_args, options="--enable-tvm-ffi")
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
    if x.ndim != 2:
        raise ValueError(f"x must be rank 2, got shape {tuple(x.shape)}")
    N, D = x.shape
    if not 1 <= N <= MAX_NUM_ROWS:
        raise ValueError(f"N must be in [1, {MAX_NUM_ROWS}], got {N}")
    if D != 512:
        raise ValueError(f"D must be 512, got {D}")
    if dy.shape != (N, 3 * D):
        raise ValueError(f"dy must have shape ({N}, {3 * D}), got {tuple(dy.shape)}")
    if u.shape != x.shape or random_mask.shape != x.shape:
        raise ValueError(f"u and random_mask must have shape {tuple(x.shape)}")
    if weight.shape != (D,) or bias.shape != (D,):
        raise ValueError(f"weight and bias must have shape ({D},)")
    if mean.shape != (N,) or rstd.shape != (N,):
        raise ValueError(f"mean and rstd must have shape ({N},)")
    if x.dtype != torch.bfloat16 or any(tensor.dtype != x.dtype for tensor in (dy, u, weight, bias)):
        raise ValueError("dy, x, u, weight, and bias must all have dtype torch.bfloat16")
    if mean.dtype != torch.float32 or rstd.dtype != torch.float32:
        raise ValueError("mean and rstd must have dtype torch.float32")
    if random_mask.dtype != torch.int8:
        raise ValueError("random_mask must have dtype torch.int8")
    dy, x, u = dy.detach(), x.detach(), u.detach()
    weight, bias = weight.detach(), bias.detach()
    mean, rstd, random_mask = mean.detach(), rstd.detach(), random_mask.detach()

    dev = x.device
    if dev.type != "cuda" or any(tensor.device != dev for tensor in (dy, u, weight, bias, mean, rstd, random_mask)):
        raise ValueError("all inputs must be CUDA tensors on the same device")
    if tuple(x.stride()) != (D, 1) or tuple(random_mask.stride()) != (D, 1):
        raise ValueError(f"x and random_mask must have stride ({D}, 1)")
    if dy.stride(1) != 1 or dy.stride(0) < 3 * D or dy.stride(0) % 8 != 0:
        raise ValueError("dy must have unit innermost stride and 16-byte-aligned, non-overlapping rows")
    if u.stride(1) != 1 or u.stride(0) < D or u.stride(0) % 8 != 0:
        raise ValueError("u must have unit innermost stride and 16-byte-aligned, non-overlapping rows")
    if tuple(weight.stride()) != (1,) or tuple(bias.stride()) != (1,) or tuple(mean.stride()) != (1,) or tuple(rstd.stride()) != (1,):
        raise ValueError("weight, bias, mean, and rstd must be contiguous")
    if any(tensor.data_ptr() % ALIGN != 0 for tensor in (dy, x, u, weight, bias, mean, rstd, random_mask)):
        raise ValueError(f"all inputs must be {ALIGN}-byte aligned")
    dx = torch.empty((N, D), dtype=x.dtype, device=dev)
    du = torch.empty((N, D), dtype=x.dtype, device=dev)
    if compute_y:
        y = torch.empty((N, 3 * D), dtype=x.dtype, device=dev)
        y0, y1, y2 = y[:, :D], y[:, D : 2 * D], y[:, 2 * D :]
    else:
        # never written; dx stands in so the kernel still gets three valid
        # tensor arguments without an 8.42 GB allocation
        y = torch.empty(0, dtype=x.dtype, device=dev)
        y0 = y1 = y2 = dx

    grid = max(1, TARGET_TILES // ROWS)
    n_tiles = grid * ROWS
    dw_p = torch.empty((n_tiles, D), dtype=torch.float32, device=dev)
    db_p = torch.empty((n_tiles, D), dtype=torch.float32, device=dev)

    nblk = (N + ROWS - 1) // ROWS
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    launch_args = (
        dy[:, :D],
        dy[:, D : 2 * D],
        dy[:, 2 * D :],
        x,
        u,
        weight,
        bias,
        random_mask,
        dx,
        du,
        y0,
        y1,
        y2,
        mean,
        rstd,
        dw_p,
        db_p,
        cutlass.Float32(dropout_ratio),
        cutlass.Int32(D),
        cutlass.Int32(nblk),
        cutlass.Int32(grid),
        stream,
    )
    rows = cute.sym_int()
    dtype = _convert_to_cutlass_data_type(x.dtype)
    fake_dx = make_fake_tensor(dtype, (rows, D), stride=(D, 1), assumed_align=ALIGN)
    if compute_y:
        fake_y = tuple(make_fake_tensor(dtype, (rows, D), stride=(3 * D, 1), assumed_align=ALIGN) for _ in range(3))
    else:
        fake_y = (fake_dx, fake_dx, fake_dx)
    compile_args = (
        make_fake_tensor(dtype, (rows, D), stride=tuple(dy.stride()), assumed_align=ALIGN),
        make_fake_tensor(dtype, (rows, D), stride=tuple(dy.stride()), assumed_align=ALIGN),
        make_fake_tensor(dtype, (rows, D), stride=tuple(dy.stride()), assumed_align=ALIGN),
        make_fake_tensor(dtype, (rows, D), stride=tuple(x.stride()), assumed_align=ALIGN),
        make_fake_tensor(dtype, (rows, D), stride=tuple(u.stride()), assumed_align=ALIGN),
        make_fake_tensor(dtype, (D,), stride=tuple(weight.stride()), assumed_align=ALIGN),
        make_fake_tensor(dtype, (D,), stride=tuple(bias.stride()), assumed_align=ALIGN),
        make_fake_tensor(cutlass.Int8, (rows, D), stride=tuple(random_mask.stride()), assumed_align=ALIGN),
        fake_dx,
        make_fake_tensor(dtype, (rows, D), stride=(D, 1), assumed_align=ALIGN),
        *fake_y,
        make_fake_tensor(cutlass.Float32, (rows,), stride=tuple(mean.stride()), assumed_align=ALIGN),
        make_fake_tensor(cutlass.Float32, (rows,), stride=tuple(rstd.stride()), assumed_align=ALIGN),
        make_fake_tensor(cutlass.Float32, (n_tiles, D), stride=(D, 1), assumed_align=ALIGN),
        make_fake_tensor(cutlass.Float32, (n_tiles, D), stride=(D, 1), assumed_align=ALIGN),
        cutlass.Float32(0.0),
        cutlass.Int32(D),
        cutlass.Int32(1),
        cutlass.Int32(grid),
        make_fake_stream(use_tvm_ffi_env_stream=False),
    )
    key = (
        "hstu_lmsd_bwd_dynamic_v1",
        _plan_signature(dy, dynamic_rows=True),
        _plan_signature(x, dynamic_rows=True),
        _plan_signature(u, dynamic_rows=True),
        _plan_signature(weight, dynamic_rows=False),
        _plan_signature(bias, dynamic_rows=False),
        _plan_signature(mean, dynamic_rows=True),
        _plan_signature(rstd, dynamic_rows=True),
        _plan_signature(random_mask, dynamic_rows=True),
        VEC,
        ROWS,
        TARGET_TILES,
        FAST_DIV,
        MIN_BLOCKS_PER_MP,
        compute_y,
        "i64_single_launch",
    )
    with torch.cuda.device(dev):
        compiled = _compiled(key, compile_args, compute_y=compute_y)
    compiled(*launch_args)
    return dx, du, y, dw_p, db_p
