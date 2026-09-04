# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL implementation of HSTU LMSD forward.

The output is laid out as three contiguous D-wide segments:

* segment 0: dropout(SiLU(u))
* segment 1: dropout(x)
* segment 2: dropout(LayerNorm(x) * SiLU(u))

One int8 mask stores the three keep decisions in bits 2, 1, and 0,
respectively. U is copied asynchronously to per-warp shared memory while the
warp computes the LayerNorm statistics from X.
"""

from cuda.bindings import driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as _cm
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.utils as _cutils

from ._common import LOG2E, MASK_LMSD, MASK_SILU, MASK_X, domain_offset_i64, offset_tensor_i64

VECTOR_SIZE = 8
THREADS_PER_ROW = 32
ROWS_PER_CTA = 4
BLOCKS_PER_SM = 192
MIN_BLOCKS_PER_MP = 7

M0, M1 = 0xD2511F53, 0xCD9E8D57
W0, W1 = 0x9E3779B9, 0xBB67AE85
MASK32 = 0xFFFFFFFF
ROUNDS = 10


@cute.struct
class ForwardSharedStorage:
    """One barrier and one 512-element U row for each warp in the CTA."""

    barriers: cute.struct.MemRange[cutlass.Int64, ROWS_PER_CTA]
    u_rows: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, ROWS_PER_CTA * 512], 128]


class LnMulDropoutForward:
    """Compile-time configuration and device code for LMSD forward.

    Four warps process four independent rows per CTA. Each warp starts an
    asynchronous 1024-byte U copy, reduces X while that copy is in flight,
    then consumes U from its private shared-memory row.
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
        mU: cute.Tensor,
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
        u_transaction_bytes: cutlass.Constexpr,
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
        lane = thread_idx % 32
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        storage = _cutils.SmemAllocator().allocate(ForwardSharedStorage)
        shared_u_layout = cute.make_layout(
            (self.rows_per_cta, 1, 512, 1),
            stride=(512, 512, 1, 512),
        )
        shared_u_all = cute.make_tensor(storage.u_rows.data_ptr(), shared_u_layout)
        shared_u = shared_u_all[(warp_idx, None, None, None)]

        barrier = storage.barriers.data_ptr() + warp_idx
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(barrier, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_warp()

        tensor_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type)
        mask_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gMask.element_type)
        bulk_u_copy_atom = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), mU.element_type)
        thread_copy = cute.make_tiled_copy_tv(tensor_copy_atom, thr_layout, val_layout).get_slice(lane)
        mask_thread_copy = cute.make_tiled_copy_tv(mask_copy_atom, thr_layout, val_layout).get_slice(lane)

        inv_d = cutlass.Float32(1.0) / ncols.to(cutlass.Float32)
        scale = cutlass.Float32(1.0) / (cutlass.Float32(1.0) - drop)
        key0 = cutlass.Uint32(seed & MASK32)
        key1 = cutlass.Uint32((seed >> 32) & MASK32)

        # Runtime loop bounds preserve one compiled binary across all supported
        # row counts while retaining v63's persistent grid-stride schedule.
        for iteration in cutlass.range(num_iterations):
            row_block = block_idx + iteration * grid_size
            if row_block < num_row_blocks:
                row = row_block * self.rows_per_cta + warp_idx
                if row < nrows:
                    # Each warp owns its barrier and shared row. Issue U first;
                    # the X reduction below is independent work that hides it.
                    mU_row = domain_offset_i64((row, 0), mU)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(barrier, u_transaction_bytes)
                        cute.copy(
                            bulk_u_copy_atom,
                            mU_row[(0, None)],
                            shared_u[(0, None, 0)],
                            mbar_ptr=barrier,
                        )

                    sum_x = cutlass.Float32(0.0)
                    sum_sq_x = cutlass.Float32(0.0)
                    for column_tile in cutlass.range_constexpr(num_column_tiles):
                        tile_coord = ((None, None), (row, column_tile))
                        tXgReduction = thread_copy.partition_S(gX[tile_coord])
                        rXReduction = cute.make_fragment_like(tXgReduction)
                        cute.copy(tensor_copy_atom, tXgReduction, rXReduction)
                        for element in cutlass.range_constexpr(self.vector_size):
                            value = rXReduction[element].to(cutlass.Float32)
                            sum_x = sum_x + value
                            sum_sq_x = sum_sq_x + value * value

                    for offset in cutlass.range_constexpr(self.threads_per_row.bit_length() - 1):
                        sum_x = sum_x + cute.arch.shuffle_sync_bfly(sum_x, 1 << offset)
                        sum_sq_x = sum_sq_x + cute.arch.shuffle_sync_bfly(sum_sq_x, 1 << offset)

                    mean = sum_x * inv_d
                    variance = _cm.max(sum_sq_x * inv_d - mean * mean, cutlass.Float32(0.0))
                    rstd = cutlass.Float32(1.0) / _cm.sqrt(variance + eps)

                    # The three segments share one [N, 3D] allocation. Rebase
                    # its first segment once, then use constant segment offsets.
                    row_coord = ((0, 0), (row, 0))
                    gSiluOut_row = domain_offset_i64(row_coord, gSiluOut)
                    gXOut_row = offset_tensor_i64(gSiluOut_row, cutlass.Int64(512))
                    gLmsdOut_row = offset_tensor_i64(gSiluOut_row, cutlass.Int64(1024))

                    # This is the first U consumer. Barrier parity flips each
                    # time the single shared stage is reused.
                    cute.arch.mbarrier_wait(barrier, iteration % 2)
                    shared_u_vectors = cute.zipped_divide(shared_u[(0, None, None)], (self.vector_size,))

                    for column_tile in cutlass.range_constexpr(num_column_tiles):
                        global_coord = ((None, None), (row, column_tile))
                        local_coord = ((None, None), (0, column_tile))
                        tXgSiluOut = thread_copy.partition_S(gSiluOut_row[local_coord])
                        tXgXOut = thread_copy.partition_S(gXOut_row[local_coord])
                        tXgLmsdOut = thread_copy.partition_S(gLmsdOut_row[local_coord])
                        tXgMask = mask_thread_copy.partition_S(gMask[global_coord])
                        rSiluOut = cute.make_fragment_like(tXgSiluOut)
                        rXOut = cute.make_fragment_like(tXgXOut)
                        rLmsdOut = cute.make_fragment_like(tXgLmsdOut)
                        rMask = cute.make_fragment_like(tXgMask)

                        # Match the established reread-X schedule: the reduction
                        # fragment is dead before X is fetched for the output pass.
                        tXgX = thread_copy.partition_S(gX[global_coord])
                        rX = cute.make_fragment_like(tXgX)
                        cute.copy(tensor_copy_atom, tXgX, rX)

                        for element in cutlass.range_constexpr(self.vector_size):
                            rMask[element] = cutlass.Int8(0)
                        philox_block = cutlass.Uint32(column_tile * self.threads_per_row) + cutlass.Uint32(lane)
                        # Consume each Philox4 result immediately so six tuples
                        # are not simultaneously live in registers.
                        for mask_plane in cutlass.range_constexpr(3):
                            for half in cutlass.range_constexpr(2):
                                counter0 = cutlass.Uint32(row)
                                counter1 = philox_block * cutlass.Uint32(2) + cutlass.Uint32(half)
                                counter2 = cutlass.Uint32(mask_plane)
                                counter3 = cutlass.Uint32(0)
                                round_key0, round_key1 = key0, key1
                                multiplier0 = cutlass.Uint32(M0)
                                multiplier1 = cutlass.Uint32(M1)
                                for _round in cutlass.range_constexpr(ROUNDS):
                                    product0 = cute.arch.mul_wide(multiplier0, counter0)
                                    product1 = cute.arch.mul_wide(multiplier1, counter2)
                                    high0 = (product0 >> cutlass.Uint64(32)).to(cutlass.Uint32)
                                    low0 = (product0 & cutlass.Uint64(MASK32)).to(cutlass.Uint32)
                                    high1 = (product1 >> cutlass.Uint64(32)).to(cutlass.Uint32)
                                    low1 = (product1 & cutlass.Uint64(MASK32)).to(cutlass.Uint32)
                                    counter0 = high1 ^ counter1 ^ round_key0
                                    counter1 = low1
                                    counter2 = high0 ^ counter3 ^ round_key1
                                    counter3 = low0
                                    round_key0 = round_key0 + cutlass.Uint32(W0)
                                    round_key1 = round_key1 + cutlass.Uint32(W1)
                                words = (counter0, counter1, counter2, counter3)
                                for word_index in cutlass.range_constexpr(4):
                                    element = half * 4 + word_index
                                    bit = cutlass.Int8(1 << mask_plane) if words[word_index] >= thresh else cutlass.Int8(0)
                                    rMask[element] = rMask[element] | bit

                        # W/B do not participate in Philox. Loading them here
                        # shortens their live range through the integer loop.
                        parameter_coord = ((None, None), (0, column_tile))
                        tXgW = thread_copy.partition_S(gW[parameter_coord])
                        tXgB = thread_copy.partition_S(gB[parameter_coord])
                        rW = cute.make_fragment_like(tXgW)
                        rB = cute.make_fragment_like(tXgB)
                        cute.copy(tensor_copy_atom, tXgW, rW)
                        cute.copy(tensor_copy_atom, tXgB, rB)

                        shared_u_vector = shared_u_vectors[(None, (column_tile * self.threads_per_row + lane, 0))]
                        rU = cute.make_fragment_like(shared_u_vector)
                        cute.autovec_copy(shared_u_vector, rU)

                        zero = cutlass.Float32(0.0)
                        for element in cutlass.range_constexpr(self.vector_size):
                            x_value = rX[element].to(cutlass.Float32)
                            u_value = rU[element].to(cutlass.Float32)
                            weight = rW[element].to(cutlass.Float32)
                            bias = rB[element].to(cutlass.Float32)
                            layer_norm = (x_value - mean) * rstd * weight + bias
                            denominator = cutlass.Float32(1.0) + cute.arch.exp2(-u_value * cutlass.Float32(LOG2E))
                            silu = _cm.div(u_value, denominator, approx=True)
                            mask_bits = rMask[element].to(cutlass.Int32)
                            scaled_silu = silu * scale
                            rSiluOut[element] = (scaled_silu if (mask_bits & MASK_SILU) != 0 else zero).to(gX.element_type)
                            rXOut[element] = (x_value * scale if (mask_bits & MASK_X) != 0 else zero).to(gX.element_type)
                            rLmsdOut[element] = (layer_norm * scaled_silu if (mask_bits & MASK_LMSD) != 0 else zero).to(gX.element_type)

                        # This order is part of the measured v63 schedule.
                        cute.copy(tensor_copy_atom, rXOut, tXgXOut)
                        cute.copy(tensor_copy_atom, rSiluOut, tXgSiluOut)
                        cute.copy(tensor_copy_atom, rLmsdOut, tXgLmsdOut)
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
        gSiluOut = tile(mSiluOut)
        param_layout = cute.make_layout((1, cute.size(mW)), stride=(0, 1))
        mW2 = cute.make_tensor(mW.iterator, param_layout)
        mB2 = cute.make_tensor(mB.iterator, param_layout)
        u_stage_layout = cute.make_ordered_layout((1, 512), order=(1, 0))
        u_transaction_bytes = cute.size_in_bytes(cutlass.BFloat16, u_stage_layout)

        self.kernel(
            gX,
            mU,
            tile(mW2),
            tile(mB2),
            gSiluOut,
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
            u_transaction_bytes,
            cute.size(gSiluOut, mode=[1, 1]),
            seed,
            nrows,
            ncols,
            num_row_blocks,
            num_iterations,
            grid_size,
        ).launch(
            grid=(grid_size, 1, 1),
            block=(self.rows_per_cta * 32, 1, 1),
            smem=ForwardSharedStorage.size_in_bytes(),
            min_blocks_per_mp=self.min_blocks_per_mp,
            stream=stream,
        )
