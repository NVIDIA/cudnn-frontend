# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HSTU backward specialized for one query per sequence."""

import operator
from typing import Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm

from .utils import tanhf, warp_reduce


def _atomic_add_bf16x2(ptr, val_lo: Float32, val_hi: Float32, *, loc=None, ip=None):
    """Packed BF16x2 atomic add used to combine split-KV dQ partials."""
    llvm.inline_asm(
        None,
        [ptr, val_hi.ir_value(loc=loc, ip=ip), val_lo.ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 packed; cvt.rn.bf16x2.f32 packed, $1, $2; red.global.add.noftz.bf16x2 [$0], packed; }",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class HSTUAttentionBackwardQlen1Sm100:
    """Q-major CUDA-core kernel for HSTU qlen=1.

    One or more CTAs own a batch/head pair. Their warps traverse disjoint KV
    rows, use contiguous 64-bit per-lane K/V and dK/dV transfers, and write
    dK/dV without atomics. The unsplit path writes dQ once after a CTA-local
    reduction; split-KV paths atomically combine one dQ partial per CTA.
    """

    def __init__(
        self,
        element_dtype: Type[cutlass.Numeric],
        head_dim: int,
        num_threads: int = 256,
        split_kv: int = 1,
        rows_per_warp: int = 1,
        is_local: bool = False,
    ):
        assert 0 < num_threads <= 1024 and num_threads % cute.arch.WARP_SIZE == 0
        assert head_dim in (64, 128, 256)
        assert split_kv in (1, 2, 4, 8, 13, 16, 22, 26, 32, 64)
        assert rows_per_warp in (1, 2, 4)
        self.element_dtype = element_dtype
        self.head_dim = head_dim
        self.num_threads = num_threads
        self.num_warps = num_threads // cute.arch.WARP_SIZE
        self.split_kv = split_kv
        self.rows_per_warp = rows_per_warp
        self.is_local = is_local
        self.lanes_per_row = cute.arch.WARP_SIZE // rows_per_warp
        self.values_per_lane = self.head_dim // self.lanes_per_row
        self.smem_bytes = self.num_warps * self.head_dim * Float32.width // 8

    @cute.kernel
    def kernel(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        dO: cute.Tensor,
        dQ: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        window_size_left: Int32,
        window_size_right: Int32,
        alpha: Float32,
        scaling_seqlen: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        batch_idx, head_idx, split_idx = cute.arch.block_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        lane_in_row = lane_idx % self.lanes_per_row
        row_in_warp = lane_idx // self.lanes_per_row

        q_begin = cu_seqlens_q[batch_idx]
        q_end = cu_seqlens_q[batch_idx + 1]
        k_begin = cu_seqlens_k[batch_idx]
        k_end = cu_seqlens_k[batch_idx + 1]
        has_query = q_begin < q_end
        seqlen_q = q_end - q_begin
        seqlen_k = k_end - k_begin

        # HSTU aligns Q to the right edge of K. With one query, a local window
        # is therefore one contiguous interval near the end of the sequence.
        # Keep partitioning the complete K range so every dK/dV row is written
        # exactly once; rows outside the interval take the zero-store path.
        attended_begin = k_begin
        attended_end = k_end
        if cutlass.const_expr(self.is_local):
            aligned_q_row = seqlen_k - seqlen_q
            attended_begin = k_begin + max(Int32(0), aligned_q_row - window_size_left)
            attended_end = k_begin + min(seqlen_k, aligned_q_row + Int32(1) + window_size_right)

        rQ = cute.make_rmem_tensor(cute.make_layout((self.values_per_lane,)), Float32)
        rdO = cute.make_rmem_tensor(cute.make_layout((self.values_per_lane,)), Float32)
        rdQ = cute.make_rmem_tensor(cute.make_layout((self.values_per_lane,)), Float32)
        for value_idx in cutlass.range_constexpr(self.values_per_lane):
            dim_idx = lane_in_row * self.values_per_lane + value_idx
            if has_query:
                rQ[value_idx] = Q[q_begin, head_idx, dim_idx].to(Float32)
                rdO[value_idx] = dO[q_begin, head_idx, dim_idx].to(Float32)
            else:
                rQ[value_idx] = Float32(0.0)
                rdO[value_idx] = Float32(0.0)
            rdQ[value_idx] = Float32(0.0)

        inv_scaling = cute.arch.rcp_approx(scaling_seqlen)
        grad_scale = alpha * inv_scaling
        vector_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.element_dtype,
            num_bits_per_copy=self.values_per_lane * self.element_dtype.width,
        )
        vector_copy = cute.make_tiled_copy_tv(
            vector_copy_atom,
            cute.make_layout((1, self.lanes_per_row)),
            cute.make_layout((1, self.values_per_lane)),
        )
        thread_copy = vector_copy.get_slice(lane_in_row)
        row_layout = cute.make_layout((1, self.head_dim), stride=(0, 1))
        rows_per_split = (k_end - k_begin + self.split_kv - 1) // self.split_kv
        split_begin = min(k_end, k_begin + split_idx * rows_per_split)
        split_end = min(k_end, split_begin + rows_per_split)
        # For qlen=1 the right-aligned local interval is a suffix.  First zero
        # the rows preceding that suffix with vector stores, then run the
        # expensive QK / dO.V math only over the attended rows.  This preserves
        # explicit zero gradients outside the window without burning CUDA-core
        # arithmetic on the full KV sequence.
        if cutlass.const_expr(self.is_local):
            # Keep the bandwidth-dominated zero stores in one contiguous range
            # per CTA.  Only the attended suffix below is interleaved.
            zero_end = min(split_end, attended_begin)
            zero_k_base = split_begin + warp_idx * self.rows_per_warp
            while zero_k_base < zero_end:
                zero_k_idx = zero_k_base + row_in_warp
                zero_row_valid = zero_k_idx < zero_end
                safe_zero_k_idx = min(zero_k_idx, zero_end - 1)
                zero_row_offset_dk = cute.assume(
                    safe_zero_k_idx * dK.stride[0] + head_idx * dK.stride[1],
                    divby=128 // self.element_dtype.width,
                )
                zero_row_offset_dv = cute.assume(
                    safe_zero_k_idx * dV.stride[0] + head_idx * dV.stride[1],
                    divby=128 // self.element_dtype.width,
                )
                gdK_zero = cute.make_tensor(dK.iterator + zero_row_offset_dk, row_layout)
                gdV_zero = cute.make_tensor(dV.iterator + zero_row_offset_dv, row_layout)
                tDgdK_zero = thread_copy.partition_D(gdK_zero)
                tDgdV_zero = thread_copy.partition_D(gdV_zero)
                tDrdK_zero = cute.make_fragment_like(tDgdK_zero)
                tDrdV_zero = cute.make_fragment_like(tDgdV_zero)
                for value_idx in cutlass.range_constexpr(self.values_per_lane):
                    tDrdK_zero[value_idx] = dK.element_type(0.0)
                    tDrdV_zero[value_idx] = dV.element_type(0.0)
                if zero_row_valid:
                    cute.copy(vector_copy_atom, tDrdK_zero, tDgdK_zero)
                    cute.copy(vector_copy_atom, tDrdV_zero, tDgdV_zero)
                zero_k_base += self.num_warps * self.rows_per_warp

        compute_begin = split_begin
        compute_end = split_end
        warp_k_base = compute_begin + warp_idx * self.rows_per_warp
        row_stride = self.num_warps * self.rows_per_warp
        has_attended_rows = compute_begin < compute_end
        if cutlass.const_expr(self.is_local):
            # Interleave small row groups across split CTAs.  A contiguous
            # partition puts a short suffix almost entirely in the final CTA,
            # leaving the other split CTAs to perform only zero stores.
            compute_begin = attended_begin
            compute_end = attended_end
            warp_k_base = compute_begin + (split_idx * self.num_warps + warp_idx) * self.rows_per_warp
            row_stride = self.num_warps * self.split_kv * self.rows_per_warp
            has_attended_rows = compute_begin + split_idx * self.num_warps * self.rows_per_warp < compute_end
        while warp_k_base < compute_end:
            k_idx = warp_k_base + row_in_warp
            row_valid = k_idx < compute_end
            safe_k_idx = min(k_idx, compute_end - 1)
            row_offset_k = cute.assume(
                safe_k_idx * K.stride[0] + head_idx * K.stride[1],
                divby=128 // self.element_dtype.width,
            )
            row_offset_v = cute.assume(
                safe_k_idx * V.stride[0] + head_idx * V.stride[1],
                divby=128 // self.element_dtype.width,
            )
            gK = cute.make_tensor(K.iterator + row_offset_k, row_layout)
            gV = cute.make_tensor(V.iterator + row_offset_v, row_layout)
            tKgK = thread_copy.partition_S(gK)
            tVgV = thread_copy.partition_S(gV)
            tKrK = cute.make_fragment_like(tKgK)
            tVrV = cute.make_fragment_like(tVgV)
            for value_idx in cutlass.range_constexpr(self.values_per_lane):
                tKrK[value_idx] = self.element_dtype(0.0)
                tVrV[value_idx] = self.element_dtype(0.0)
            if row_valid and has_query:
                cute.copy(vector_copy_atom, tKgK, tKrK)
                cute.copy(vector_copy_atom, tVgV, tVrV)
            rK = tKrK.load().to(Float32)
            rV = tVrV.load().to(Float32)
            qk = Float32(0.0)
            dov = Float32(0.0)
            for value_idx in cutlass.range_constexpr(self.values_per_lane):
                qk += rQ[value_idx] * rK[value_idx]
                dov += rdO[value_idx] * rV[value_idx]
            qk = warp_reduce(qk, operator.add, self.lanes_per_row)
            dov = warp_reduce(dov, operator.add, self.lanes_per_row)
            score = alpha * qk
            score_tanh = tanhf(score * Float32(0.5))
            sigmoid = Float32(0.5) * score_tanh + Float32(0.5)
            weight = score * sigmoid
            dsilu = sigmoid + weight * (Float32(1.0) - sigmoid)
            ds = dov * dsilu

            rdK = cute.make_rmem_tensor(cute.make_layout((self.values_per_lane,)), dK.element_type)
            for value_idx in cutlass.range_constexpr(self.values_per_lane):
                rdK[value_idx] = dK.element_type(ds * grad_scale * rQ[value_idx])
                rdQ[value_idx] += ds * rK[value_idx]
            row_offset_dk = cute.assume(safe_k_idx * dK.stride[0] + head_idx * dK.stride[1], divby=128 // self.element_dtype.width)
            gdK = cute.make_tensor(dK.iterator + row_offset_dk, row_layout)
            tDgdK = thread_copy.partition_D(gdK)
            tDrdK = cute.make_fragment_like(tDgdK)
            tDrdK.store(rdK.load())
            if row_valid:
                cute.copy(vector_copy_atom, tDrdK, tDgdK)

            rdV = cute.make_rmem_tensor(cute.make_layout((self.values_per_lane,)), dV.element_type)
            for value_idx in cutlass.range_constexpr(self.values_per_lane):
                rdV[value_idx] = dV.element_type(weight * inv_scaling * rdO[value_idx])
            row_offset_dv = cute.assume(safe_k_idx * dV.stride[0] + head_idx * dV.stride[1], divby=128 // self.element_dtype.width)
            gdV = cute.make_tensor(dV.iterator + row_offset_dv, row_layout)
            tDgdV = thread_copy.partition_D(gdV)
            tDrdV = cute.make_fragment_like(tDgdV)
            tDrdV.store(rdV.load())
            if row_valid:
                cute.copy(vector_copy_atom, tDrdV, tDgdV)
            warp_k_base += row_stride

        smem = utils.SmemAllocator()
        sdQ = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.head_dim, self.num_warps), stride=(1, self.head_dim)),
            byte_alignment=16,
        )
        for value_idx in cutlass.range_constexpr(self.values_per_lane):
            if cutlass.const_expr(self.rows_per_warp >= 2):
                rdQ[value_idx] += cute.arch.shuffle_sync_bfly(rdQ[value_idx], offset=self.lanes_per_row)
            if cutlass.const_expr(self.rows_per_warp == 4):
                rdQ[value_idx] += cute.arch.shuffle_sync_bfly(rdQ[value_idx], offset=2 * self.lanes_per_row)
            if lane_idx < self.lanes_per_row:
                dim_idx = lane_in_row * self.values_per_lane + value_idx
                sdQ[dim_idx, warp_idx] = rdQ[value_idx]
        cute.arch.barrier()

        if cutlass.const_expr(self.split_kv == 1):
            if tidx < self.head_dim and has_query:
                dq_sum = Float32(0.0)
                for reduce_warp in cutlass.range_constexpr(self.num_warps):
                    dq_sum += sdQ[tidx, reduce_warp]
                dQ[q_begin, head_idx, tidx] = dQ.element_type(dq_sum * grad_scale)
        elif tidx < self.head_dim // 2 and has_query and has_attended_rows:
            dim_idx = tidx * 2
            dq_lo = Float32(0.0)
            dq_hi = Float32(0.0)
            for reduce_warp in cutlass.range_constexpr(self.num_warps):
                dq_lo += sdQ[dim_idx, reduce_warp]
                dq_hi += sdQ[dim_idx + 1, reduce_warp]
            row_offset_dq = cute.assume(
                q_begin * dQ.stride[0] + head_idx * dQ.stride[1],
                divby=128 // self.element_dtype.width,
            )
            _atomic_add_bf16x2(
                (dQ.iterator + row_offset_dq + dim_idx).llvm_ptr,
                dq_lo * grad_scale,
                dq_hi * grad_scale,
            )

    @cute.jit
    def __call__(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        dO: cute.Tensor,
        dQ: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        cu_seqlens_k: cute.Tensor,
        window_size_left: Int32,
        window_size_right: Int32,
        batch_size: Int32,
        num_heads: Int32,
        alpha: Float32,
        scaling_seqlen: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            Q,
            K,
            V,
            dO,
            dQ,
            dK,
            dV,
            cu_seqlens_q,
            cu_seqlens_k,
            window_size_left,
            window_size_right,
            alpha,
            scaling_seqlen,
        ).launch(
            grid=(batch_size, num_heads, self.split_kv),
            block=(self.num_threads, 1, 1),
            smem=self.smem_bytes,
            stream=stream,
        )
