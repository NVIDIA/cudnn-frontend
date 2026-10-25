# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mode-specific radix selection with token expansion in the same CTA."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.memory import SmemAllocator

from cudnn.deepseek_sparse_attention.indexer_top_k.indexer_top_k_decode_varlen import IndexerTopKKernelVarlenDecode


class PooledSelectKernel(IndexerTopKKernelVarlenDecode):
    def __init__(self, max_pools, selected_pools, pool_size, mode, score_dtype=cutlass.Float32):
        super().__init__(score_dtype, max_pools, selected_pools, return_val=False, large_occupancy=mode == "prefill")
        if score_dtype == cutlass.BFloat16 and 8192 <= max_pools <= 32768:
            # BF16 uses one shared candidate buffer; keep it on chip.
            self.indexer_topk_smem_input_size = max_pools
            self.enable_gmem_store = False
            self.indexer_topk_max_k = selected_pools
            self.num_threads_per_cta = 1024 if mode == "decode" and max_pools > 8192 else 512
        self.pool_size = pool_size
        self.mode = mode

    @cute.jit
    def __call__(self, scores: cute.Tensor, scratch: cute.Tensor, positions: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream):
        copy_atom, tiled_copy, tiler = self._get_tiled_copy()
        if cutlass.const_expr(self.mode == "prefill"):
            self.prefill(scores, scratch, positions, output, copy_atom, tiled_copy, tiler).launch(
                grid=(output.shape[0], 1, 1), block=(self.num_threads_per_cta, 1, 1), stream=stream
            )
        else:
            self.decode(scores, scratch, positions, output, copy_atom, tiled_copy, tiler).launch(
                grid=(output.shape[0], 1, 1), block=(self.num_threads_per_cta, 1, 1), stream=stream
            )

    @cute.kernel
    def prefill(
        self,
        scores: cute.Tensor,
        scratch: cute.Tensor,
        positions: cute.Tensor,
        output: cute.Tensor,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
        tiler: cute.Shape,
    ):
        self.select_and_expand(scores, scratch, positions, output, copy_atom, tiled_copy, tiler)

    @cute.kernel
    def decode(
        self,
        scores: cute.Tensor,
        scratch: cute.Tensor,
        positions: cute.Tensor,
        output: cute.Tensor,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
        tiler: cute.Shape,
    ):
        self.select_and_expand(scores, scratch, positions, output, copy_atom, tiled_copy, tiler)

    @cute.jit
    def select_and_expand(
        self,
        scores: cute.Tensor,
        scratch: cute.Tensor,
        positions: cute.Tensor,
        output: cute.Tensor,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
        tiler: cute.Shape,
    ):
        row = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        length = Int64(0)
        if row < positions.shape[0]:
            position = Int64(positions[row])
            if position >= 0 and position < 2147483647:
                length = position + 1
        completed = Int32(length // self.pool_size)
        available = Int32(min(completed, scores.shape[1]))
        topk_tokens = self.top_k * self.pool_size
        smem = SmemAllocator()

        def alloc(smem, dtype, count):
            return smem.allocate_tensor(dtype, cute.make_layout((count,)), byte_alignment=128)

        selected = alloc(smem, Int32, self.top_k)
        histogram = alloc(smem, Int32, self.radix + 1)
        counter = alloc(smem, Int32, 1)
        threshold = alloc(smem, Int32, 1)
        num_input = alloc(smem, Int32, 2)
        global_num_input = None
        if cutlass.const_expr(self.enable_gmem_store):
            global_num_input = alloc(smem, Int32, 2)
        indices = alloc(smem, self.index_type, self.indexer_topk_max_k)
        input_indices = smem.allocate_tensor(
            self.index_type,
            cute.make_ordered_layout((self.num_buffer_smem_input_idx, self.indexer_topk_smem_input_size), order=(1, 0)),
            byte_alignment=128,
        )
        last_remain = alloc(smem, Int32, 1)
        num_warps = cutlass.const_expr(min(self.radix, self.num_threads_per_cta) // 32)
        warp_sums = alloc(smem, Int32, num_warps)
        shared_output = cute.make_tensor(selected.iterator, cute.make_layout((scores.shape[0], self.top_k), stride=(0, 1)))
        if available > self.top_k:
            self.indexer_topk_kernel_per_row(
                scores,
                None,
                scratch,
                shared_output,
                None,
                tiler,
                copy_atom,
                tiled_copy,
                Int32(0),
                available,
                row,
                histogram,
                counter,
                threshold,
                num_input,
                global_num_input,
                indices,
                input_indices,
                last_remain,
                num_warps,
                warp_sums,
                None,
                None,
            )
            cute.arch.barrier()

        for col in cutlass.range(tid, output.shape[1], self.num_threads_per_cta):
            token = Int64(-1)
            if length <= topk_tokens:
                if col < length:
                    token = Int64(col)
            elif col < topk_tokens:
                pool = Int32(col // self.pool_size)
                if available > self.top_k:
                    pool = selected[pool]
                if pool >= 0 and pool < available:
                    token = Int64(pool) * self.pool_size + col % self.pool_size
            else:
                tail = Int64(col) - topk_tokens
                if tail < length % self.pool_size:
                    token = Int64(completed) * self.pool_size + tail
            output[row, col] = Int32(token)

    prefill.set_name_prefix("cudnn_pooled_select", remove_cutlass_symbol=True)
    decode.set_name_prefix("cudnn_pooled_select", remove_cutlass_symbol=True)
