# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL implementation of HSTU LMSD forward.

The output contains optional activated-U and X segments followed by the
mandatory LMSD segment. With all options enabled, it is laid out as:

* segment 0: dropout(SiLU(u))
* segment 1: dropout(x)
* segment 2: dropout(LayerNorm(x) * SiLU(u))

When dropout is enabled, one int8 mask stores the active keep decisions in
bits 2, 1, and 0, respectively.
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
from ._config import HSTULMSDFwdConfig

_PHILOX_M0, _PHILOX_M1 = 0xD2511F53, 0xCD9E8D57
_PHILOX_W0, _PHILOX_W1 = 0x9E3779B9, 0xBB67AE85
_PHILOX_MASK32 = 0xFFFFFFFF
_PHILOX_ROUNDS = 10


class HSTULMSDForward:
    """Compile-time configuration and device code for LMSD forward.

    Four warps process four independent rows per CTA.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        apply_u_silu: bool = True,
        concat_u: bool = True,
        concat_x: bool = True,
        has_dropout: bool = True,
        config: HSTULMSDFwdConfig | None = None,
    ):
        if not is_supported_hidden_size(hidden_size):
            raise ValueError(f"unsupported HSTU LMSD hidden size: {hidden_size}")
        self.config = HSTULMSDFwdConfig.from_hidden_size(hidden_size) if config is None else config
        self.hidden_size = hidden_size
        self.threads_per_row = self.config.threads_per_row
        self.rows_per_cta = self.config.rows_per_cta
        self.vector_size = self.config.vector_size
        self.min_blocks_per_mp = self.config.min_blocks_per_mp
        self.apply_u_silu = apply_u_silu
        self.concat_u = concat_u
        self.concat_x = concat_x
        self.has_dropout = has_dropout

    @cute.jit
    def _generate_dropout_mask(
        self,
        rMask: cute.Tensor,
        row: cutlass.Int32,
        philox_block: cutlass.Uint32,
        key0: cutlass.Uint32,
        key1: cutlass.Uint32,
        thresh: cutlass.Uint32,
    ):
        for element in cutlass.range_constexpr(self.vector_size):
            rMask[element] = cutlass.Int8(0)
        philox_groups = self.vector_size // 4
        for mask_plane in cutlass.range_constexpr(3):
            plane_enabled = mask_plane == 0 or (mask_plane == 1 and self.concat_x) or (mask_plane == 2 and self.concat_u)
            if const_expr(plane_enabled):
                for group in cutlass.range_constexpr(philox_groups):
                    counter0 = cutlass.Uint32(row)
                    counter1 = philox_block * cutlass.Uint32(philox_groups) + cutlass.Uint32(group)
                    counter2 = cutlass.Uint32(mask_plane)
                    counter3 = cutlass.Uint32(0)
                    round_key0, round_key1 = key0, key1
                    multiplier0 = cutlass.Uint32(_PHILOX_M0)
                    multiplier1 = cutlass.Uint32(_PHILOX_M1)
                    for _round in cutlass.range_constexpr(_PHILOX_ROUNDS):
                        product0 = cute.arch.mul_wide(multiplier0, counter0)
                        product1 = cute.arch.mul_wide(multiplier1, counter2)
                        high0 = (product0 >> cutlass.Uint64(32)).to(cutlass.Uint32)
                        low0 = (product0 & cutlass.Uint64(_PHILOX_MASK32)).to(cutlass.Uint32)
                        high1 = (product1 >> cutlass.Uint64(32)).to(cutlass.Uint32)
                        low1 = (product1 & cutlass.Uint64(_PHILOX_MASK32)).to(cutlass.Uint32)
                        counter0 = high1 ^ counter1 ^ round_key0
                        counter1 = low1
                        counter2 = high0 ^ counter3 ^ round_key1
                        counter3 = low0
                        round_key0 = round_key0 + cutlass.Uint32(_PHILOX_W0)
                        round_key1 = round_key1 + cutlass.Uint32(_PHILOX_W1)
                    words = (counter0, counter1, counter2, counter3)
                    for word_index in cutlass.range_constexpr(4):
                        element = group * 4 + word_index
                        bit = cutlass.Int8(1 << mask_plane) if words[word_index] >= thresh else cutlass.Int8(0)
                        rMask[element] = rMask[element] | bit

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
        ncols: cutlass.Int32,
        num_row_blocks: cutlass.Int32,
        num_iterations: cutlass.Int32,
        grid_size: cutlass.Int32,
    ):
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        lane = thread_idx % self.threads_per_row
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        tensor_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type)
        mask_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gMask.element_type) if const_expr(self.has_dropout) else None
        thread_copy = cute.make_tiled_copy_tv(tensor_copy_atom, thr_layout, val_layout).get_slice(lane)
        mask_thread_copy = cute.make_tiled_copy_tv(mask_copy_atom, thr_layout, val_layout).get_slice(lane) if const_expr(self.has_dropout) else None

        inv_d = cutlass.Float32(1.0) / ncols.to(cutlass.Float32)
        scale = cutlass.Float32(1.0) / (cutlass.Float32(1.0) - drop) if const_expr(self.has_dropout) else cutlass.Float32(1.0)
        if const_expr(self.has_dropout):
            key0 = cutlass.Uint32(seed & _PHILOX_MASK32)
            key1 = cutlass.Uint32((seed >> 32) & _PHILOX_MASK32)

        # Runtime loop bounds preserve one compiled binary across all supported
        # row counts while retaining the persistent grid-stride schedule.
        for iteration in cutlass.range(num_iterations):
            row_block = block_idx + iteration * grid_size
            if row_block < num_row_blocks:
                row = row_block * self.rows_per_cta + warp_idx
                if row < nrows:
                    row_coord = ((0, 0), (row, 0))
                    gX_row = domain_offset_i64(row_coord, gX)
                    sum_x = cutlass.Float32(0.0)
                    sum_sq_x = cutlass.Float32(0.0)
                    for column_tile in cutlass.range_constexpr(num_column_tiles):
                        row_tile_coord = ((None, None), (0, column_tile))
                        tXgReduction = thread_copy.partition_S(gX_row[row_tile_coord])
                        rXReduction = cute.make_fragment_like(tXgReduction)
                        if const_expr(self.hidden_size % (self.threads_per_row * self.vector_size) == 0):
                            cute.copy(tensor_copy_atom, tXgReduction, rXReduction)
                        else:
                            for element in cutlass.range_constexpr(self.vector_size):
                                rXReduction[element] = cutlass.BFloat16(0.0)
                            vector_index = column_tile * self.threads_per_row + lane
                            if vector_index < self.hidden_size // self.vector_size:
                                cute.copy(tensor_copy_atom, tXgReduction, rXReduction)
                        for element in cutlass.range_constexpr(self.vector_size):
                            value = rXReduction[element].to(cutlass.Float32)
                            sum_x = sum_x + value
                            sum_sq_x = sum_sq_x + value * value

                    for offset in cutlass.range_constexpr(self.threads_per_row.bit_length() - 1):
                        sum_x = sum_x + cute.arch.shuffle_sync_bfly(sum_x, 1 << offset)
                        sum_sq_x = sum_sq_x + cute.arch.shuffle_sync_bfly(sum_sq_x, 1 << offset)

                    mean = sum_x * inv_d
                    variance = cute.arch.fmax(sum_sq_x * inv_d - mean * mean, cutlass.Float32(0.0))
                    rstd = cutlass.Float32(1.0) / _cm.sqrt(variance + eps)

                    # Optional auxiliary segments and the mandatory LMSD segment
                    # are views into one compact output allocation.
                    gSiluOut_row = domain_offset_i64(row_coord, gSiluOut) if const_expr(self.concat_u) else None
                    gXOut_row = domain_offset_i64(row_coord, gXOut) if const_expr(self.concat_x) else None
                    gLmsdOut_row = domain_offset_i64(row_coord, gLmsdOut)
                    gU_row = domain_offset_i64(row_coord, gU)

                    for column_tile in cutlass.range_constexpr(num_column_tiles):
                        vector_index = column_tile * self.threads_per_row + lane
                        if const_expr(self.hidden_size % (self.threads_per_row * self.vector_size) == 0) or vector_index < self.hidden_size // self.vector_size:
                            global_coord = ((None, None), (row, column_tile))
                            local_coord = ((None, None), (0, column_tile))
                            tXgSiluOut = thread_copy.partition_S(gSiluOut_row[local_coord]) if const_expr(self.concat_u) else None
                            tXgXOut = thread_copy.partition_S(gXOut_row[local_coord]) if const_expr(self.concat_x) else None
                            tXgLmsdOut = thread_copy.partition_S(gLmsdOut_row[local_coord])
                            tXgMask = mask_thread_copy.partition_S(gMask[global_coord]) if const_expr(self.has_dropout) else None
                            rSiluOut = cute.make_fragment_like(tXgSiluOut) if const_expr(self.concat_u) else None
                            rXOut = cute.make_fragment_like(tXgXOut) if const_expr(self.concat_x) else None
                            rLmsdOut = cute.make_fragment_like(tXgLmsdOut)
                            rMask = cute.make_fragment_like(tXgMask) if const_expr(self.has_dropout) else None

                            # Reload X only after the reduction fragment is dead,
                            # limiting the register live range in the output pass.
                            tXgX = thread_copy.partition_S(gX_row[local_coord])
                            tXgU = thread_copy.partition_S(gU_row[local_coord])
                            rX = cute.make_fragment_like(tXgX)
                            rU = cute.make_fragment_like(tXgU)
                            cute.copy(tensor_copy_atom, tXgX, rX)
                            cute.copy(tensor_copy_atom, tXgU, rU)

                            if const_expr(self.has_dropout):
                                philox_block = cutlass.Uint32(column_tile * self.threads_per_row) + cutlass.Uint32(lane)
                                self._generate_dropout_mask(rMask, row, philox_block, key0, key1, thresh)

                            # W/B do not participate in Philox. Loading them here
                            # shortens their live range through the integer loop.
                            parameter_coord = ((None, None), (0, column_tile))
                            tXgW = thread_copy.partition_S(gW[parameter_coord])
                            tXgB = thread_copy.partition_S(gB[parameter_coord])
                            rW = cute.make_fragment_like(tXgW)
                            rB = cute.make_fragment_like(tXgB)
                            cute.copy(tensor_copy_atom, tXgW, rW)
                            cute.copy(tensor_copy_atom, tXgB, rB)

                            zero = cutlass.Float32(0.0)
                            for element in cutlass.range_constexpr(self.vector_size):
                                x_value = rX[element].to(cutlass.Float32)
                                u_value = rU[element].to(cutlass.Float32)
                                weight = rW[element].to(cutlass.Float32)
                                bias = rB[element].to(cutlass.Float32)
                                layer_norm = (x_value - mean) * rstd * weight + bias
                                if const_expr(self.apply_u_silu):
                                    denominator = cutlass.Float32(1.0) + cute.arch.exp2(-u_value * cutlass.Float32(LOG2E))
                                    activated_u = _cm.div(u_value, denominator, approx=True)
                                else:
                                    activated_u = u_value
                                if const_expr(self.has_dropout):
                                    mask_bits = rMask[element].to(cutlass.Int32)
                                    dropped_u = activated_u * scale if (mask_bits & _DROPOUT_KEEP_U_BIT) != 0 else zero
                                    dropped_x = x_value * scale if (mask_bits & _DROPOUT_KEEP_X_BIT) != 0 else zero
                                    dropped_lmsd = layer_norm * activated_u * scale if (mask_bits & _DROPOUT_KEEP_LMSD_BIT) != 0 else zero
                                else:
                                    dropped_u = activated_u
                                    dropped_x = x_value
                                    dropped_lmsd = layer_norm * activated_u
                                if const_expr(self.concat_u):
                                    rSiluOut[element] = dropped_u.to(gX.element_type)
                                if const_expr(self.concat_x):
                                    rXOut[element] = dropped_x.to(gX.element_type)
                                rLmsdOut[element] = dropped_lmsd.to(gX.element_type)

                            if const_expr(self.concat_x):
                                cute.copy(tensor_copy_atom, rXOut, tXgXOut)
                            if const_expr(self.concat_u):
                                cute.copy(tensor_copy_atom, rSiluOut, tXgSiluOut)
                            cute.copy(tensor_copy_atom, rLmsdOut, tXgLmsdOut)
                            if const_expr(self.has_dropout):
                                cute.copy(mask_copy_atom, rMask, tXgMask)

                    if lane == 0:
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
        ncols: cutlass.Int32,
        eps: cutlass.Float32,
        drop: cutlass.Float32,
        thresh: cutlass.Uint32,
        num_row_blocks: cutlass.Int32,
        num_iterations: cutlass.Int32,
        grid_size: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        thr_layout = cute.make_ordered_layout((1, self.threads_per_row), order=(1, 0))
        val_layout = cute.make_ordered_layout((1, self.vector_size), order=(1, 0))
        tiler, _ = cute.make_layout_tv(thr_layout, val_layout)
        tile = lambda tensor: cute.zipped_divide(tensor, tiler)

        gX = tile(mX)
        gSiluOut = tile(mSiluOut) if const_expr(self.concat_u) else None
        param_layout = cute.make_layout((1, cute.size(mW)), stride=(0, 1))
        mW2 = cute.make_tensor(mW.iterator, param_layout)
        mB2 = cute.make_tensor(mB.iterator, param_layout)

        self.kernel(
            gX,
            tile(mU),
            tile(mW2),
            tile(mB2),
            gSiluOut,
            tile(mXOut) if const_expr(self.concat_x) else None,
            tile(mLmsdOut),
            tile(mMask) if const_expr(self.has_dropout) else None,
            mMean,
            mRstd,
            eps,
            drop,
            thresh,
            thr_layout,
            val_layout,
            (self.hidden_size + self.threads_per_row * self.vector_size - 1) // (self.threads_per_row * self.vector_size),
            seed,
            nrows,
            ncols,
            num_row_blocks,
            num_iterations,
            grid_size,
        ).launch(
            grid=(grid_size, 1, 1),
            block=(self.rows_per_cta * self.threads_per_row, 1, 1),
            smem=0,
            min_blocks_per_mp=self.min_blocks_per_mp,
            stream=stream,
        )
