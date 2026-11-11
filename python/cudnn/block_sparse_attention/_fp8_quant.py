# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private CuTe DSL Sage FP8 quantization for contiguous BHSD tensors."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import BFloat16, Float8E4M3FN, Float32, Int32
from cudnn._cutlass_compat import SmemAllocator

from cudnn.block_sparse_attention.csrc.utils.cute_dsl_utils import to_cute_tensor


class _SageFp8Quantizer:
    """Quantize Q/K/V with the scale and rounding contract used by Sage."""

    HEAD_DIM = 128
    Q_THREADS = 32
    Q_ELEMS_PER_LANE = HEAD_DIM // Q_THREADS
    K_BLOCK_ROWS = 16
    K_BLOCK_ELEMS = K_BLOCK_ROWS * HEAD_DIM
    K_THREADS = 256
    K_ELEMS_PER_THREAD = K_BLOCK_ELEMS // K_THREADS
    K_WARPS = K_THREADS // 32
    REDUCTION_THREADS = 128
    ROWS_PER_CHUNK = 256
    SM90_ROWS_PER_CHUNK = 64

    def __init__(self, is_sm120: bool = False, v_block_size: int = 0, v_layout: str = "bhsd", is_sm90: bool = False, k_quant_block_size: int = 16):
        self.is_sm120 = is_sm120
        self.k_quant_block_size = k_quant_block_size
        self.v_block_size = v_block_size
        if v_block_size:
            from cudnn.block_sparse_attention.csrc.fwd.sm120_blk128.sage_v_quant import SageFp8VQuantizerSm120Blk128

            self.v_quantizer = SageFp8VQuantizerSm120Blk128()
        self.v_layout = v_layout
        self.is_sm90 = is_sm90

    @cute.jit
    def _warp_amax(self, value: Float32):
        # Floating-point redux is a Blackwell instruction. Hopper uses shuffles.
        if cutlass.const_expr(self.is_sm90 or self.is_sm120):
            return cute.arch.warp_reduction_max(value, threads_in_group=32)
        else:
            return cute.arch.warp_redux_sync(value, "fmax")

    @cute.kernel
    def _quantize_v_bhds_kernel(
        self,
        mV: cute.Tensor,
        mVFp8: cute.Tensor,
        mVScale: cute.Tensor,
        num_heads: Int32,
        seqlen_k: Int32,
    ):
        """Fuse Sage V quantization with BHSD -> BHDS storage conversion."""
        tid = cute.arch.thread_idx()[0]
        tile, head, batch = cute.arch.block_idx()
        # Four adjacent channels per word: no sub-word shared store races.
        # Padding the row stride breaks the bank conflict of a plain transpose.
        shared = utils.SmemAllocator().allocate_tensor(cutlass.Uint32, cute.make_layout((64, 33), stride=(33, 1)), byte_alignment=16)
        channel = (tid % 32) * 4
        values = cute.make_rmem_tensor(cute.make_layout(4), Float8E4M3FN)
        reciprocal = cute.make_rmem_tensor(cute.make_layout(4), Float32)
        for c in cutlass.range_constexpr(4):
            reciprocal[c] = cute.math.div(Float32(1.0), mVScale[head * 128 + channel + c], full=True)
        raw_values = cute.make_rmem_tensor(cute.make_layout(4), BFloat16)
        for r in cutlass.range_constexpr(16):
            row = tid // 32 + r * 4
            base = ((batch * num_heads + head) * seqlen_k + tile * 64 + row) * 128 + channel
            cute.autovec_copy(cute.make_tensor((mV.iterator + base).align(8), cute.make_layout(4)), raw_values)
            values.store((raw_values.load().to(Float32) * reciprocal.load()).to(Float8E4M3FN))
            shared[row, tid % 32] = cute.recast_tensor(values, cutlass.Uint32)[0]
        cute.arch.sync_threads()

        packed = cute.make_rmem_tensor(cute.make_layout(16), cutlass.Uint8)
        for chunk in cutlass.range_constexpr(4):
            task = tid + chunk * 128
            dim = task // 4
            row_start = (task % 4) * 16
            for r in cutlass.range_constexpr(16):
                packed[r] = (shared[row_start + r, dim // 4] >> ((dim % 4) * 8)).to(cutlass.Uint8)
            offset = ((batch * num_heads + head) * 128 + dim) * seqlen_k + tile * 64 + row_start
            output = cute.make_tensor(cute.recast_ptr(mVFp8.iterator + offset, dtype=cutlass.Uint8).align(16), cute.make_layout(16))
            cute.autovec_copy(packed, output)

    @cute.kernel
    def _kv_partial_sm90_kernel(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        k_partial: cute.Tensor,
        v_partial: cute.Tensor,
        num_heads: Int32,
        seqlen_k: Int32,
        num_chunks: Int32,
        num_v_chunks: Int32,
    ):
        # Each lane loads eight adjacent BF16 channels (one 128-bit load).
        # Eight independent row stripes replace the serial per-channel scan.
        tid = cute.arch.thread_idx()[0]
        chunk, head, batch = cute.arch.block_idx()
        dim = (tid % 16) * 8
        row = tid // 16
        sums = cute.make_rmem_tensor(cute.make_layout(8), Float32)
        maxima = cute.make_rmem_tensor_like(sums)
        values = cute.make_rmem_tensor(cute.make_layout(8), BFloat16)
        sums.fill(0.0)
        maxima.fill(0.0)
        for r in cutlass.range_constexpr(8):
            offset = ((batch * num_heads + head) * seqlen_k + chunk * self.SM90_ROWS_PER_CHUNK + row + r * 8) * 128 + dim
            source_k = cute.make_tensor((mK.iterator + offset).align(16), cute.make_layout(8))
            cute.autovec_copy(source_k, values)
            sums.store(sums.load() + values.load().to(Float32))
            source_v = cute.make_tensor((mV.iterator + offset).align(16), cute.make_layout(8))
            cute.autovec_copy(source_v, values)
            for d in cutlass.range_constexpr(8):
                x = values[d].to(Float32)
                maxima[d] = cute.arch.fmax(maxima[d], cute.arch.fmax(x, -x))
        smem = utils.SmemAllocator()
        shared_sum = smem.allocate_tensor(Float32, cute.make_layout((8, 128), stride=(128, 1)))
        shared_max = smem.allocate_tensor(Float32, cute.make_layout((8, 128), stride=(128, 1)))
        cute.autovec_copy(sums, cute.make_tensor((shared_sum.iterator + row * 128 + dim).align(16), cute.make_layout(8)))
        cute.autovec_copy(maxima, cute.make_tensor((shared_max.iterator + row * 128 + dim).align(16), cute.make_layout(8)))
        cute.arch.sync_threads()
        total = Float32(0.0)
        maximum = Float32(0.0)
        for r in cutlass.range_constexpr(8):
            total += shared_sum[r, tid]
            maximum = cute.arch.fmax(maximum, shared_max[r, tid])
        k_partial[((batch * num_heads + head) * num_chunks + chunk) * 128 + tid] = total
        v_partial[(head * num_v_chunks + batch * num_chunks + chunk) * 128 + tid] = maximum

    @cute.kernel
    def _finalize_sm90_kernel(
        self,
        partial: cute.Tensor,
        output: cute.Tensor,
        num_chunks: Int32,
        divisor: Float32,
        is_max: cutlass.Constexpr,
    ):
        # Thirty-two stripes reduce eight adjacent channels per CTA.
        tid = cute.arch.thread_idx()[0]
        task = cute.arch.block_idx()[0]
        lane = tid % 8
        shard = tid // 8
        group = task // 16
        dim = (task % 16) * 8 + lane
        value = Float32(0.0)
        for chunk in cutlass.range(shard, num_chunks, 32, unroll=8):
            x = partial[(group * num_chunks + chunk) * 128 + dim]
            if cutlass.const_expr(is_max):
                value = cute.arch.fmax(value, x)
            else:
                value += x
        shared = utils.SmemAllocator().allocate_tensor(Float32, cute.make_layout((32, 8), stride=(8, 1)))
        shared[shard, lane] = value
        cute.arch.sync_threads()
        if shard == 0:
            total = Float32(0.0)
            for i in cutlass.range_constexpr(32):
                if cutlass.const_expr(is_max):
                    total = cute.arch.fmax(total, shared[i, lane])
                else:
                    total += shared[i, lane]
            if cutlass.const_expr(is_max):
                total = cute.arch.fmax(total, Float32(1.0e-3))
            output[group * 128 + dim] = cute.math.div(total, divisor, full=True)

    @cute.kernel
    def _quantize_q_kernel(
        self,
        mQ: cute.Tensor,
        mQFp8: cute.Tensor,
        mQScale: cute.Tensor,
    ):
        lane_idx = cute.arch.lane_idx()
        row_idx = cute.arch.block_idx()[0]
        if cutlass.const_expr(self.is_sm90):
            row_idx = row_idx * 4 + cute.arch.warp_idx()
        row_base = row_idx * self.HEAD_DIM + lane_idx * self.Q_ELEMS_PER_LANE

        values = cute.make_rmem_tensor(
            cute.make_layout((self.Q_ELEMS_PER_LANE,)),
            BFloat16,
        )
        if cutlass.const_expr(self.is_sm90):
            cute.autovec_copy(cute.make_tensor((mQ.iterator + row_base).align(8), cute.make_layout(4)), values)
        local_amax = Float32(0.0)
        for elem_idx in cutlass.range_constexpr(self.Q_ELEMS_PER_LANE):
            if cutlass.const_expr(not self.is_sm90):
                values[elem_idx] = mQ[row_base + elem_idx]
            value = values[elem_idx]
            value_f32 = value.to(Float32)
            local_amax = cute.arch.fmax(
                local_amax,
                cute.arch.fmax(value_f32, -value_f32),
            )

        row_amax = self._warp_amax(local_amax)
        row_scale = cute.math.div(
            cute.arch.fmax(row_amax, Float32(1.0e-3)),
            Float32(448.0),
            full=True,
        )
        if lane_idx == 0:
            mQScale[row_idx] = row_scale

        # Match the customer recipe exactly: scale, reciprocal, and multiply
        # are each rounded through BF16 before the E4M3 conversion.
        scale_bf16 = row_scale.to(BFloat16)
        reciprocal_bf16 = (BFloat16(1.0) / scale_bf16).to(BFloat16)
        result = cute.make_rmem_tensor(cute.make_layout(4), Float8E4M3FN)
        for elem_idx in cutlass.range_constexpr(self.Q_ELEMS_PER_LANE):
            scaled_bf16 = (values[elem_idx] * reciprocal_bf16).to(BFloat16)
            result[elem_idx] = scaled_bf16.to(Float8E4M3FN)
            if cutlass.const_expr(not self.is_sm90):
                mQFp8[row_base + elem_idx] = result[elem_idx]
        if cutlass.const_expr(self.is_sm90):
            cute.autovec_copy(result, cute.make_tensor((mQFp8.iterator + row_base).align(4), cute.make_layout(4)))

    @cute.kernel
    def _k_mean_partial_kernel(
        self,
        mK: cute.Tensor,
        mKMeanPartial: cute.Tensor,
        seqlen_k: Int32,
        num_chunks: Int32,
    ):
        dim_idx = cute.arch.thread_idx()[0]
        task_idx = cute.arch.block_idx()[0]
        group_idx = task_idx // num_chunks
        chunk_idx = task_idx - group_idx * num_chunks
        seq_start = chunk_idx * self.ROWS_PER_CHUNK
        group_base = group_idx * seqlen_k * self.HEAD_DIM

        partial_sum = Float32(0.0)
        for row_offset in cutlass.range(self.ROWS_PER_CHUNK, unroll=1):
            seq_idx = seq_start + row_offset
            if seq_idx < seqlen_k:
                partial_sum += mK[group_base + seq_idx * self.HEAD_DIM + dim_idx].to(Float32)
        mKMeanPartial[task_idx * self.HEAD_DIM + dim_idx] = partial_sum

    @cute.kernel
    def _k_mean_finalize_kernel(
        self,
        mKMeanPartial: cute.Tensor,
        mKMean: cute.Tensor,
        num_chunks: Int32,
        seqlen_k_float: Float32,
    ):
        dim_idx = cute.arch.thread_idx()[0]
        group_idx = cute.arch.block_idx()[0]
        partial_base = group_idx * num_chunks * self.HEAD_DIM

        total = Float32(0.0)
        for chunk_idx in cutlass.range(num_chunks, unroll=1):
            total += mKMeanPartial[partial_base + chunk_idx * self.HEAD_DIM + dim_idx]
        mKMean[group_idx * self.HEAD_DIM + dim_idx] = cute.math.div(total, seqlen_k_float, full=True)

    @cute.kernel
    def _quantize_q_four_rows_kernel(self, mQ: cute.Tensor, mQFp8: cute.Tensor, mQScale: cute.Tensor, num_rows: Int32):
        """Vectorized four-warp CTA with the unchanged Sage Q rounding."""
        row = cute.arch.block_idx()[0] * 4 + cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        if row < num_rows:
            offset = row * 128 + lane * 4
            values = cute.make_rmem_tensor(cute.make_layout(4), BFloat16)
            cute.autovec_copy(cute.make_tensor((mQ.iterator + offset).align(8), cute.make_layout(4)), values)
            maximum = Float32(0.0)
            for c in cutlass.range_constexpr(4):
                x = values[c].to(Float32)
                maximum = cute.arch.fmax(maximum, cute.arch.fmax(x, -x))
            scale = cute.math.div(cute.arch.fmax(self._warp_amax(maximum), Float32(1.0e-3)), Float32(448.0), full=True)
            if lane == 0:
                mQScale[row] = scale
            reciprocal = (BFloat16(1.0) / scale.to(BFloat16)).to(BFloat16)
            result = cute.make_rmem_tensor(cute.make_layout(4), Float8E4M3FN)
            for c in cutlass.range_constexpr(4):
                result[c] = (values[c] * reciprocal).to(BFloat16).to(Float8E4M3FN)
            cute.autovec_copy(result, cute.make_tensor((mQFp8.iterator + offset).align(4), cute.make_layout(4)))

    @cute.kernel
    def _quantize_k_per_token_kernel(self, mK: cute.Tensor, mKFp8: cute.Tensor, mKScale: cute.Tensor, mKMean: cute.Tensor, seqlen_k: Int32, num_rows: Int32):
        """Four independent rows per CTA; no cross-warp reduction or scratch."""
        row = cute.arch.block_idx()[0] * 4 + cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        if row < num_rows:
            channel = lane * 4
            offset = row * 128 + channel
            mean_offset = (row // seqlen_k) * 128 + channel
            values = cute.make_rmem_tensor(cute.make_layout(4), BFloat16)
            means = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            cute.autovec_copy(cute.make_tensor((mK.iterator + offset).align(8), cute.make_layout(4)), values)
            cute.autovec_copy(cute.make_tensor((mKMean.iterator + mean_offset).align(16), cute.make_layout(4)), means)
            centered = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            centered.store(values.load().to(Float32) - means.load())
            maximum = Float32(0.0)
            for c in cutlass.range_constexpr(4):
                maximum = cute.arch.fmax(maximum, cute.arch.fmax(centered[c], -centered[c]))
            scale = cute.math.div(cute.arch.fmax(self._warp_amax(maximum), Float32(1.0e-3)), Float32(448.0), full=True)
            if lane == 0:
                mKScale[row] = scale
            result = cute.make_rmem_tensor(cute.make_layout(4), Float8E4M3FN)
            for c in cutlass.range_constexpr(4):
                result[c] = cute.math.div(centered[c], scale, full=True).to(Float8E4M3FN)
            cute.autovec_copy(result, cute.make_tensor((mKFp8.iterator + offset).align(4), cute.make_layout(4)))

    @cute.kernel
    def _quantize_k_kernel(
        self,
        mK: cute.Tensor,
        mKFp8: cute.Tensor,
        mKScale: cute.Tensor,
        mKMean: cute.Tensor,
        seqlen_k: Int32,
        num_k_blocks: Int32,
    ):
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        task_idx = cute.arch.block_idx()[0]
        group_idx = task_idx // num_k_blocks
        block_idx = task_idx - group_idx * num_k_blocks
        seq_start = block_idx * self.K_BLOCK_ROWS
        group_base = group_idx * seqlen_k * self.HEAD_DIM
        local_offset = thread_idx * self.K_ELEMS_PER_THREAD

        smem = SmemAllocator()
        warp_amax = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.K_WARPS,)),
            byte_alignment=4,
        )
        block_scale = smem.allocate_tensor(
            Float32,
            cute.make_layout((1,)),
            byte_alignment=4,
        )

        centered = cute.make_rmem_tensor(
            cute.make_layout((self.K_ELEMS_PER_THREAD,)),
            Float32,
        )
        raw_values = cute.make_rmem_tensor(cute.make_layout(8), BFloat16)
        mean_values = cute.make_rmem_tensor(cute.make_layout(8), Float32)
        if cutlass.const_expr(self.is_sm90):
            offset = group_base + seq_start * 128 + local_offset
            cute.autovec_copy(cute.make_tensor((mK.iterator + offset).align(16), cute.make_layout(8)), raw_values)
            mean_offset = group_idx * 128 + local_offset % 128
            cute.autovec_copy(cute.make_tensor((mKMean.iterator + mean_offset).align(16), cute.make_layout(8)), mean_values)
        local_amax = Float32(0.0)
        for elem_idx in cutlass.range_constexpr(self.K_ELEMS_PER_THREAD):
            block_offset = local_offset + elem_idx
            row_offset = block_offset // self.HEAD_DIM
            dim_idx = block_offset - row_offset * self.HEAD_DIM
            seq_idx = seq_start + row_offset
            is_valid = seq_idx < seqlen_k
            raw_value = Float32(0.0)
            if cutlass.const_expr(self.is_sm90):
                centered_value = raw_values[elem_idx].to(Float32) - mean_values[elem_idx]
            else:
                if is_valid:
                    raw_value = mK[group_base + seq_idx * self.HEAD_DIM + dim_idx].to(Float32)
                centered_value = raw_value - mKMean[group_idx * self.HEAD_DIM + dim_idx]
            centered[elem_idx] = centered_value
            local_amax = cute.arch.fmax(
                local_amax,
                cute.arch.fmax(centered_value, -centered_value),
            )

        per_warp_amax = self._warp_amax(local_amax)
        if lane_idx == 0:
            warp_amax[warp_idx] = per_warp_amax
        cute.arch.barrier()

        cta_amax = Float32(0.0)
        if lane_idx < self.K_WARPS:
            cta_amax = warp_amax[lane_idx]
        cta_amax = self._warp_amax(cta_amax)
        if thread_idx == 0:
            block_scale[0] = cute.math.div(
                cute.arch.fmax(cta_amax, Float32(1.0e-3)),
                Float32(448.0),
                full=True,
            )
            mKScale[task_idx] = block_scale[0]
        cute.arch.barrier()

        scale = block_scale[0]
        result = cute.make_rmem_tensor(cute.make_layout(8), Float8E4M3FN)
        for elem_idx in cutlass.range_constexpr(self.K_ELEMS_PER_THREAD):
            block_offset = local_offset + elem_idx
            row_offset = block_offset // self.HEAD_DIM
            dim_idx = block_offset - row_offset * self.HEAD_DIM
            seq_idx = seq_start + row_offset
            if seq_idx < seqlen_k:
                output_idx = group_base + seq_idx * self.HEAD_DIM + dim_idx
                result[elem_idx] = cute.math.div(centered[elem_idx], scale, full=True).to(Float8E4M3FN)
                if cutlass.const_expr(not self.is_sm90):
                    mKFp8[output_idx] = result[elem_idx]
        if cutlass.const_expr(self.is_sm90):
            offset = group_base + seq_start * 128 + local_offset
            cute.autovec_copy(result, cute.make_tensor((mKFp8.iterator + offset).align(8), cute.make_layout(8)))

    @cute.kernel
    def _v_amax_partial_kernel(
        self,
        mV: cute.Tensor,
        mVAmaxPartial: cute.Tensor,
        batch_size: Int32,
        num_heads: Int32,
        seqlen_k: Int32,
        num_chunks: Int32,
    ):
        dim_idx = cute.arch.thread_idx()[0]
        task_idx = cute.arch.block_idx()[0]
        head_idx = task_idx // num_chunks
        chunk_idx = task_idx - head_idx * num_chunks
        row_start = chunk_idx * self.ROWS_PER_CHUNK
        total_rows = batch_size * seqlen_k

        local_amax = Float32(0.0)
        for row_offset in cutlass.range(self.ROWS_PER_CHUNK, unroll=1):
            flat_row = row_start + row_offset
            if flat_row < total_rows:
                batch_idx = flat_row // seqlen_k
                seq_idx = flat_row - batch_idx * seqlen_k
                input_idx = ((batch_idx * num_heads + head_idx) * seqlen_k + seq_idx) * self.HEAD_DIM + dim_idx
                value = mV[input_idx].to(Float32)
                local_amax = cute.arch.fmax(
                    local_amax,
                    cute.arch.fmax(value, -value),
                )
        mVAmaxPartial[task_idx * self.HEAD_DIM + dim_idx] = local_amax

    @cute.kernel
    def _v_scale_finalize_kernel(
        self,
        mVAmaxPartial: cute.Tensor,
        mVScale: cute.Tensor,
        num_chunks: Int32,
    ):
        dim_idx = cute.arch.thread_idx()[0]
        head_idx = cute.arch.block_idx()[0]
        partial_base = head_idx * num_chunks * self.HEAD_DIM

        amax = Float32(0.0)
        for chunk_idx in cutlass.range(num_chunks, unroll=1):
            amax = cute.arch.fmax(
                amax,
                mVAmaxPartial[partial_base + chunk_idx * self.HEAD_DIM + dim_idx],
            )
        mVScale[head_idx * self.HEAD_DIM + dim_idx] = cute.math.div(
            cute.arch.fmax(amax, Float32(1.0e-3)),
            Float32(448.0),
            full=True,
        )

    @cute.kernel
    def _quantize_v_kernel(
        self,
        mV: cute.Tensor,
        mVFp8: cute.Tensor,
        mVScale: cute.Tensor,
        num_heads: Int32,
        seqlen_k: Int32,
    ):
        dim_idx = cute.arch.thread_idx()[0]
        row_idx = cute.arch.block_idx()[0]
        group_idx = row_idx // seqlen_k
        head_idx = group_idx - (group_idx // num_heads) * num_heads
        input_idx = row_idx * self.HEAD_DIM + dim_idx

        # Keep the V reciprocal and product in FP32 until the E4M3 conversion.
        reciprocal = cute.math.div(Float32(1.0), mVScale[head_idx * self.HEAD_DIM + dim_idx], full=True)
        mVFp8[input_idx] = (mV[input_idx].to(Float32) * reciprocal).to(Float8E4M3FN)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mQFp8: cute.Tensor,
        mKFp8: cute.Tensor,
        mVFp8: cute.Tensor,
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
        mVScale: cute.Tensor,
        mKMeanPartial: cute.Tensor,
        mKMean: cute.Tensor,
        mVAmaxPartial: cute.Tensor,
        batch_size: Int32,
        num_heads: Int32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        num_k_chunks: Int32,
        num_v_chunks: Int32,
        seqlen_k_float: Float32,
        stream: cuda.CUstream,
    ):
        num_q_rows = batch_size * num_heads * seqlen_q
        num_groups = batch_size * num_heads
        num_k_blocks = (seqlen_k + self.K_BLOCK_ROWS - 1) // self.K_BLOCK_ROWS

        if cutlass.const_expr(self.k_quant_block_size == 1):
            self._quantize_q_four_rows_kernel(mQ, mQFp8, mQScale, num_q_rows).launch(grid=((num_q_rows + 3) // 4, 1, 1), block=(128, 1, 1), stream=stream)
        else:
            self._quantize_q_kernel(mQ, mQFp8, mQScale).launch(
                grid=(num_q_rows // 4 if self.is_sm90 else num_q_rows, 1, 1),
                block=(128 if self.is_sm90 else self.Q_THREADS, 1, 1),
                stream=stream,
            )
        if cutlass.const_expr(self.is_sm90):
            self._kv_partial_sm90_kernel(mK, mV, mKMeanPartial, mVAmaxPartial, num_heads, seqlen_k, num_k_chunks, num_v_chunks).launch(
                grid=(num_k_chunks, num_heads, batch_size), block=(128, 1, 1), stream=stream
            )
        else:
            self._k_mean_partial_kernel(
                mK,
                mKMeanPartial,
                seqlen_k,
                num_k_chunks,
            ).launch(
                grid=(num_groups * num_k_chunks, 1, 1),
                block=(self.REDUCTION_THREADS, 1, 1),
                stream=stream,
            )
        if cutlass.const_expr(self.is_sm90):
            self._finalize_sm90_kernel(mKMeanPartial, mKMean, num_k_chunks, seqlen_k_float, False).launch(
                grid=(num_groups * 16, 1, 1), block=(256, 1, 1), stream=stream
            )
        else:
            self._k_mean_finalize_kernel(
                mKMeanPartial,
                mKMean,
                num_k_chunks,
                seqlen_k_float,
            ).launch(
                grid=(num_groups, 1, 1),
                block=(self.REDUCTION_THREADS, 1, 1),
                stream=stream,
            )
        if cutlass.const_expr(self.k_quant_block_size == 1):
            self._quantize_k_per_token_kernel(mK, mKFp8, mKScale, mKMean, seqlen_k, num_groups * seqlen_k).launch(
                grid=((num_groups * seqlen_k + 3) // 4, 1, 1), block=(128, 1, 1), stream=stream
            )
        else:
            self._quantize_k_kernel(
                mK,
                mKFp8,
                mKScale,
                mKMean,
                seqlen_k,
                num_k_blocks,
            ).launch(
                grid=(num_groups * num_k_blocks, 1, 1),
                block=(self.K_THREADS, 1, 1),
                smem=self.K_WARPS * 4 + 4,
                stream=stream,
            )
        if cutlass.const_expr(not self.is_sm90):
            self._v_amax_partial_kernel(
                mV,
                mVAmaxPartial,
                batch_size,
                num_heads,
                seqlen_k,
                num_v_chunks,
            ).launch(
                grid=(num_heads * num_v_chunks, 1, 1),
                block=(self.REDUCTION_THREADS, 1, 1),
                stream=stream,
            )
        if cutlass.const_expr(self.is_sm90):
            self._finalize_sm90_kernel(mVAmaxPartial, mVScale, num_v_chunks, Float32(448.0), True).launch(
                grid=(num_heads * 16, 1, 1), block=(256, 1, 1), stream=stream
            )
        else:
            self._v_scale_finalize_kernel(
                mVAmaxPartial,
                mVScale,
                num_v_chunks,
            ).launch(
                grid=(num_heads, 1, 1),
                block=(self.REDUCTION_THREADS, 1, 1),
                stream=stream,
            )
        if cutlass.const_expr(self.v_layout == "bhds"):
            self._quantize_v_bhds_kernel(mV, mVFp8, mVScale, num_heads, seqlen_k).launch(
                grid=(seqlen_k // 64, num_heads, batch_size), block=(128, 1, 1), stream=stream
            )
        elif cutlass.const_expr(self.v_block_size == 128):
            self.v_quantizer(mV, mVFp8, mVScale, batch_size, num_heads, seqlen_k, stream)
        else:
            self._quantize_v_kernel(
                mV,
                mVFp8,
                mVScale,
                num_heads,
                seqlen_k,
            ).launch(
                grid=(num_groups * seqlen_k, 1, 1),
                block=(self.REDUCTION_THREADS, 1, 1),
                stream=stream,
            )


def _to_dynamic_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return to_cute_tensor(
        tensor.view(-1),
        assumed_align=16,
        fully_dynamic=True,
        enable_tvm_ffi=False,
    )


def _quantize_sage_bhsd(
    q_bhsd: torch.Tensor,
    k_bhsd: torch.Tensor,
    v_bhsd: torch.Tensor,
    *,
    v_block_size: int = 0,
    v_layout: str = "bhsd",
    k_quant_block_size: int = 16,
):
    """Quantize BF16 Q/K/V; optionally store V as [B, H, KV-block, D, token]."""
    if v_block_size not in (0, 128):
        raise ValueError("v_block_size must be 0 or 128")
    if type(k_quant_block_size) is not int or k_quant_block_size not in (1, 16):
        raise ValueError("k_quant_block_size must be 1 or 16")
    if v_layout not in {"bhsd", "bhds"}:
        raise ValueError("v_layout must be 'bhsd' or 'bhds'")
    if v_block_size and v_layout != "bhsd":
        raise ValueError("blocked V quantization cannot be combined with BHDS layout")
    tensors = (q_bhsd, k_bhsd, v_bhsd)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("Q, K, and V must be rank-4 BHSD tensors")
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise TypeError("Sage FP8 quantization currently requires BF16 inputs")
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("Q, K, and V must be CUDA tensors")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Q, K, and V must use contiguous BHSD storage")
    if any(tensor.device != q_bhsd.device for tensor in tensors[1:]):
        raise ValueError("Q, K, and V must be on the same CUDA device")

    batch_size, num_heads, seqlen_q, head_dim = q_bhsd.shape
    if batch_size < 1 or num_heads < 1 or seqlen_q < 1:
        raise ValueError("Sage FP8 quantization requires positive batch, head, and sequence counts")
    if head_dim != _SageFp8Quantizer.HEAD_DIM:
        raise ValueError("Sage FP8 quantization requires D=128")
    if k_bhsd.shape[:2] != (batch_size, num_heads) or k_bhsd.shape[-1] != head_dim:
        raise ValueError("K must match Q batch, heads, and head dimension")
    if v_bhsd.shape != k_bhsd.shape:
        raise ValueError("V must have the same shape as K")

    seqlen_k = k_bhsd.shape[2]
    if seqlen_k < 1:
        raise ValueError("Sage FP8 quantization requires positive batch, head, and sequence counts")
    if v_layout == "bhds" and seqlen_k % 64:
        raise ValueError("BHDS V quantization requires Sk to be a multiple of 64")
    capability = torch.cuda.get_device_capability(q_bhsd.device)
    is_sm120 = capability[0] == 12
    if k_quant_block_size == 1 and not is_sm120:
        raise NotImplementedError("per-token K quantization requires SM120")
    if v_block_size and not is_sm120:
        raise NotImplementedError("blocked V quantization requires SM120")
    if not is_sm120 and capability not in {(10, 0), (10, 3)} and (seqlen_q % 64 or seqlen_k % 64):
        raise ValueError("Q and K/V sequence lengths must be multiples of 64")

    num_groups = batch_size * num_heads
    rows_per_chunk = _SageFp8Quantizer.SM90_ROWS_PER_CHUNK if torch.cuda.get_device_capability(q_bhsd.device)[0] == 9 else _SageFp8Quantizer.ROWS_PER_CHUNK
    num_k_chunks = (seqlen_k + rows_per_chunk - 1) // rows_per_chunk
    num_v_chunks = (batch_size * seqlen_k + rows_per_chunk - 1) // rows_per_chunk
    num_k_blocks = (seqlen_k + k_quant_block_size - 1) // k_quant_block_size
    device = q_bhsd.device

    with torch.cuda.device(device):
        q_fp8 = torch.empty_like(q_bhsd, dtype=torch.float8_e4m3fn)
        k_fp8 = torch.empty_like(k_bhsd, dtype=torch.float8_e4m3fn)
        if v_block_size:
            v_fp8 = torch.empty(
                (batch_size, num_heads, (seqlen_k + 127) // 128, head_dim, 128),
                dtype=torch.float8_e4m3fn,
                device=device,
            )
        elif v_layout == "bhds":
            v_fp8 = torch.empty((batch_size, num_heads, head_dim, seqlen_k), device=device, dtype=torch.float8_e4m3fn)
        else:
            v_fp8 = torch.empty_like(v_bhsd, dtype=torch.float8_e4m3fn)
        q_scale = torch.empty(
            (batch_size, num_heads, seqlen_q),
            dtype=torch.float32,
            device=device,
        )
        k_scale = torch.empty(
            (batch_size, num_heads, num_k_blocks),
            dtype=torch.float32,
            device=device,
        )
        v_scale = torch.empty(
            (num_heads, head_dim),
            dtype=torch.float32,
            device=device,
        )
        k_mean_partial = torch.empty(
            (num_groups, num_k_chunks, head_dim),
            dtype=torch.float32,
            device=device,
        )
        k_mean = torch.empty(
            (num_groups, head_dim),
            dtype=torch.float32,
            device=device,
        )
        v_amax_partial = torch.empty(
            (num_heads, num_v_chunks, head_dim),
            dtype=torch.float32,
            device=device,
        )

        runtime_tensors = (
            q_bhsd,
            k_bhsd,
            v_bhsd,
            q_fp8,
            k_fp8,
            v_fp8,
            q_scale,
            k_scale,
            v_scale,
            k_mean_partial,
            k_mean,
            v_amax_partial,
        )
        cute_tensors = tuple(_to_dynamic_cute_tensor(tensor) for tensor in runtime_tensors)
        current_stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        compile_key = (device.index, capability, v_block_size, v_layout, k_quant_block_size)
        if compile_key not in _quantize_sage_bhsd.compile_cache:
            quantizer = _SageFp8Quantizer(
                is_sm120=is_sm120,
                v_block_size=v_block_size,
                v_layout=v_layout,
                is_sm90=torch.cuda.get_device_capability(device)[0] == 9,
                k_quant_block_size=k_quant_block_size,
            )
            _quantize_sage_bhsd.compile_cache[compile_key] = cute.compile(
                quantizer,
                *cute_tensors,
                Int32(batch_size),
                Int32(num_heads),
                Int32(seqlen_q),
                Int32(seqlen_k),
                Int32(num_k_chunks),
                Int32(num_v_chunks),
                Float32(seqlen_k),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            )

        _quantize_sage_bhsd.compile_cache[compile_key](
            *cute_tensors,
            Int32(batch_size),
            Int32(num_heads),
            Int32(seqlen_q),
            Int32(seqlen_k),
            Int32(num_k_chunks),
            Int32(num_v_chunks),
            Float32(seqlen_k),
            current_stream,
        )

    if not (q_fp8.is_contiguous() and k_fp8.is_contiguous() and v_fp8.is_contiguous()):
        raise RuntimeError("quantized Q, K, and V storage must remain contiguous")
    return q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale


_quantize_sage_bhsd.compile_cache = {}

__all__ = []
