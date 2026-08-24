# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Private CuTe DSL Sage FP8 quantization for contiguous BHSD tensors."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import BFloat16, Float8E4M3FN, Float32, Int32

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

    @cute.kernel
    def _quantize_q_kernel(
        self,
        mQ: cute.Tensor,
        mQFp8: cute.Tensor,
        mQScale: cute.Tensor,
    ):
        lane_idx = cute.arch.thread_idx()[0]
        row_idx = cute.arch.block_idx()[0]
        row_base = row_idx * self.HEAD_DIM + lane_idx * self.Q_ELEMS_PER_LANE

        values = cute.make_rmem_tensor(
            cute.make_layout((self.Q_ELEMS_PER_LANE,)),
            BFloat16,
        )
        local_amax = Float32(0.0)
        for elem_idx in cutlass.range_constexpr(self.Q_ELEMS_PER_LANE):
            value = mQ[row_base + elem_idx]
            values[elem_idx] = value
            value_f32 = value.to(Float32)
            local_amax = cute.arch.fmax(
                local_amax,
                cute.arch.fmax(value_f32, -value_f32),
            )

        row_amax = cute.arch.warp_redux_sync(local_amax, "fmax")
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
        for elem_idx in cutlass.range_constexpr(self.Q_ELEMS_PER_LANE):
            scaled_bf16 = (values[elem_idx] * reciprocal_bf16).to(BFloat16)
            mQFp8[row_base + elem_idx] = scaled_bf16.to(Float8E4M3FN)

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

        smem = utils.SmemAllocator()
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
        local_amax = Float32(0.0)
        for elem_idx in cutlass.range_constexpr(self.K_ELEMS_PER_THREAD):
            block_offset = local_offset + elem_idx
            row_offset = block_offset // self.HEAD_DIM
            dim_idx = block_offset - row_offset * self.HEAD_DIM
            seq_idx = seq_start + row_offset
            is_valid = seq_idx < seqlen_k
            raw_value = Float32(0.0)
            if is_valid:
                raw_value = mK[group_base + seq_idx * self.HEAD_DIM + dim_idx].to(Float32)
            centered_value = raw_value - mKMean[group_idx * self.HEAD_DIM + dim_idx]
            centered[elem_idx] = centered_value
            local_amax = cute.arch.fmax(
                local_amax,
                cute.arch.fmax(centered_value, -centered_value),
            )

        per_warp_amax = cute.arch.warp_redux_sync(local_amax, "fmax")
        if lane_idx == 0:
            warp_amax[warp_idx] = per_warp_amax
        cute.arch.barrier()

        cta_amax = Float32(0.0)
        if lane_idx < self.K_WARPS:
            cta_amax = warp_amax[lane_idx]
        cta_amax = cute.arch.warp_redux_sync(cta_amax, "fmax")
        if thread_idx == 0:
            block_scale[0] = cute.math.div(
                cute.arch.fmax(cta_amax, Float32(1.0e-3)),
                Float32(448.0),
                full=True,
            )
            mKScale[task_idx] = block_scale[0]
        cute.arch.barrier()

        scale = block_scale[0]
        for elem_idx in cutlass.range_constexpr(self.K_ELEMS_PER_THREAD):
            block_offset = local_offset + elem_idx
            row_offset = block_offset // self.HEAD_DIM
            dim_idx = block_offset - row_offset * self.HEAD_DIM
            seq_idx = seq_start + row_offset
            if seq_idx < seqlen_k:
                output_idx = group_base + seq_idx * self.HEAD_DIM + dim_idx
                mKFp8[output_idx] = cute.math.div(centered[elem_idx], scale, full=True).to(Float8E4M3FN)

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

        reciprocal_bf16 = cute.math.div(
            Float32(1.0),
            mVScale[head_idx * self.HEAD_DIM + dim_idx],
            full=True,
        ).to(BFloat16)
        scaled_bf16 = (mV[input_idx] * reciprocal_bf16).to(BFloat16)
        mVFp8[input_idx] = scaled_bf16.to(Float8E4M3FN)

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

        self._quantize_q_kernel(mQ, mQFp8, mQScale).launch(
            grid=(num_q_rows, 1, 1),
            block=(self.Q_THREADS, 1, 1),
            stream=stream,
        )
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
        self._v_scale_finalize_kernel(
            mVAmaxPartial,
            mVScale,
            num_v_chunks,
        ).launch(
            grid=(num_heads, 1, 1),
            block=(self.REDUCTION_THREADS, 1, 1),
            stream=stream,
        )
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
):
    """Quantize BF16 BHSD Q/K/V using private CuTe DSL kernels."""
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
    is_sm120 = torch.cuda.get_device_capability(q_bhsd.device)[0] == 12
    if not is_sm120 and (batch_size != 1 or num_heads not in (4, 8)):
        raise ValueError("Sage FP8 v1 requires B=1, H in {4, 8}, and D=128")
    if not is_sm120 and (seqlen_q % 64 or seqlen_k % 64):
        raise ValueError("Q and K/V sequence lengths must be multiples of 64")

    num_groups = batch_size * num_heads
    num_k_chunks = (seqlen_k + _SageFp8Quantizer.ROWS_PER_CHUNK - 1) // _SageFp8Quantizer.ROWS_PER_CHUNK
    num_v_chunks = (batch_size * seqlen_k + _SageFp8Quantizer.ROWS_PER_CHUNK - 1) // _SageFp8Quantizer.ROWS_PER_CHUNK
    num_k_blocks = (seqlen_k + _SageFp8Quantizer.K_BLOCK_ROWS - 1) // _SageFp8Quantizer.K_BLOCK_ROWS
    device = q_bhsd.device

    with torch.cuda.device(device):
        q_fp8 = torch.empty_like(q_bhsd, dtype=torch.float8_e4m3fn)
        k_fp8 = torch.empty_like(k_bhsd, dtype=torch.float8_e4m3fn)
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
        compile_key = torch.cuda.get_device_capability(device)
        if compile_key not in _quantize_sage_bhsd.compile_cache:
            quantizer = _SageFp8Quantizer()
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
        raise RuntimeError("quantized Q, K, and V must remain BHSD-contiguous")
    return q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale


_quantize_sage_bhsd.compile_cache = {}

__all__ = []
