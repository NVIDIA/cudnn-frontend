# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantize V directly into contiguous native KV128 tiles for SM120 PV."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cuda.bindings.driver as cuda


class SageFp8VQuantizerSm120Blk128:
    """Preserve Sage rounding while writing [B, H, KV-block, D, token]."""

    def __init__(self):
        self.num_threads = 256

    @cute.jit
    def __call__(
        self,
        mV: cute.Tensor,
        mVFp8: cute.Tensor,
        mVScale: cute.Tensor,
        batch_size: cutlass.Int32,
        num_heads: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        num_blocks = cute.ceil_div(seqlen_k, 128)
        mVFp8 = cute.make_tensor(
            mVFp8.iterator,
            cute.make_layout(
                (128, 128, num_blocks, num_heads, batch_size),
                stride=(128, 1, 16384, num_blocks * 16384, num_heads * num_blocks * 16384),
            ),
        )
        smem_atom = cute.nvgpu.tcgen05.make_smem_layout_atom(
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW128,
            cutlass.Float8E4M3FN,
        )
        smem_layout = cute.tile_to_shape(smem_atom, (128, 128), order=(1, 0))
        tma_atom, mVFp8_tma = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            mVFp8,
            smem_layout,
            (128, 128),
            num_multicast=1,
        )
        self.kernel(mV, mVFp8_tma, mVScale, tma_atom, smem_layout, num_heads, seqlen_k, num_blocks).launch(
            grid=(batch_size * num_heads * num_blocks, 1, 1),
            block=(self.num_threads, 1, 1),
            smem=16384,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mV: cute.Tensor,
        mVFp8: cute.Tensor,
        mVScale: cute.Tensor,
        tma_atom: cute.CopyAtom,
        smem_layout: cute.ComposedLayout,
        num_heads: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        num_blocks: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        task_idx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        kv_block = task_idx % num_blocks
        group_idx = task_idx // num_blocks
        head_idx = group_idx % num_heads
        batch_idx = group_idx // num_heads
        dim_idx = tidx % 128
        token_begin = (tidx // 128) * 64
        sV = utils.SmemAllocator().allocate_tensor(
            cutlass.Float8E4M3FN,
            smem_layout.outer,
            byte_alignment=128,
            swizzle=smem_layout.inner,
        )
        if tidx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom)
        # Reuse the exact BF16 reciprocal for 64 tokens per thread instead
        # of recomputing it in a separate CTA for every sequence row.
        reciprocal_bf16 = cute.math.div(
            cutlass.Float32(1.0),
            mVScale[head_idx * 128 + dim_idx],
            full=True,
        ).to(cutlass.BFloat16)
        for offset in cutlass.range(64, unroll=1):
            token = token_begin + offset
            seq_idx = kv_block * 128 + token
            # Zero padded tokens explicitly; the layout includes a full tile.
            scaled_bf16 = cutlass.BFloat16(0.0)
            if seq_idx < seqlen_k:
                value = mV[(group_idx * seqlen_k + seq_idx) * 128 + dim_idx]
                scaled_bf16 = (value * reciprocal_bf16).to(cutlass.BFloat16)
            sV[dim_idx, token] = scaled_bf16.to(cutlass.Float8E4M3FN)
        # One coalesced TMA store performs the final layout write. There is
        # no separate transpose/repack kernel on the public path.
        cute.arch.fence_view_async_shared()
        cute.arch.barrier()
        if warp_idx == 0:
            gV = mVFp8[None, None, kv_block, head_idx, batch_idx]
            tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
                tma_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(sV, 0, 2),
                cute.group_modes(gV, 0, 2),
            )
            cute.copy(tma_atom, tVsV, tVgV)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
