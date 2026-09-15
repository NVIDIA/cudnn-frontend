# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""FA4-style native SM120 blk128 sparse attention forward kernel."""

from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda_experimental
from cutlass.experimental import primitives as prims
import cuda.bindings.driver as cuda

from cudnn.sdpa.fwd.kernels.sm120._common import ceil_div
from cudnn.sdpa.fwd.kernels.sm120.prefill_f16 import (
    SM120FusedMultiHeadAttentionForward,
    fmul2,
)

SM120_FA4_BLK128_FWD_BLOCK_SIZE = 128


def _device_sm_count() -> int:
    """Read launch-planning metadata during JIT tracing, not cached execution."""
    result, device = cuda.cuCtxGetDevice()
    if result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError("Cannot query the current CUDA device")
    result, sm_count = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device)
    if result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError("Cannot query the CUDA SM count")
    return sm_count


class BlockSparseAttnForwardSm120Blk128Fa4(SM120FusedMultiHeadAttentionForward):
    """Native blk128 sparse attention using the SM120 FA4-style warp split.

    Full CTAs use eight compute warps, each owning 16 query rows. An
    underfilled final scheduling wave may use four-compute-warp CTAs for
    64 query rows. Both paths load complete native 128x128 K/V blocks and
    address the original blk128 sparse metadata without expansion.
    Q stays in registers; the K/V backing is reused by the O epilogue.

    This specialization intentionally covers the fixed-top-k, full-KV-block
    workload. The general native blk128 kernel remains the fallback for
    variable block counts, per-block valid sizes, and partial KV blocks.
    """

    def __init__(
        self,
        gqa_ratio: int = 1,
        head_dim: int = 128,
        value_dim: int = 128,
        blocksparse_blocksize_q: int = 128,
        blocksparse_blocksize_k: int = 128,
        dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        has_block_sizes: bool = False,
        has_block_nums: bool = False,
        block_sizes_mode: int = 0,
    ):
        """Configure FP16/BF16 native128 with fixed top-k and FP32 accumulation."""
        assert dtype in (cutlass.Float16, cutlass.BFloat16)
        assert acc_dtype == cutlass.Float32
        assert head_dim == 128 and value_dim == 128
        assert blocksparse_blocksize_q == 128 and blocksparse_blocksize_k == 128
        assert not has_block_sizes and not has_block_nums
        assert block_sizes_mode == 0

        self.gqa_ratio = gqa_ratio
        super().__init__(
            in_dtype=dtype,
            out_dtype=dtype,
            head_tile_qk=head_dim,
            head_tile_v=value_dim,
            kv_tile=SM120_FA4_BLK128_FWD_BLOCK_SIZE,
            q_tile=SM120_FA4_BLK128_FWD_BLOCK_SIZE,
            qh_per_kh=gqa_ratio,
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        blocksparse_indices_q2k: cute.Tensor,
        blocksparse_num_blocks_q2k: cute.Tensor,
        block_sparse_num: cutlass.Int32,
        blocksparse_varblk: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ) -> None:
        """Build stride-aware KV descriptors and plan full/tail CTAs during tracing."""
        del blocksparse_num_blocks_q2k, blocksparse_varblk

        assert mQ.shape[1] == mK.shape[1] == 128
        assert mV.shape[0] == mO.shape[1] == 128
        assert mQ.shape[2] == mO.shape[2]
        assert mK.shape[2] == mV.shape[2]
        assert mQ.shape[3] == mK.shape[3] == mV.shape[3] == mO.shape[3]
        assert mQ.stride[1] == mK.stride[1] == 1
        assert mV.stride[0] == mO.stride[1] == 1
        assert mK.shape[0] % self.kv_tile == 0

        def kv_tma_desc(
            tensor: cute.Tensor,
            is_v: cutlass.Constexpr[bool],
        ) -> cuda_experimental.TensorMap:
            """Describe a full native KV128 tile without packing or reordering GMEM."""
            if cutlass.const_expr(is_v):
                batch = tensor.shape[3]
                heads = tensor.shape[2]
                sequence = tensor.shape[1]
                batch_stride = tensor.stride[3]
                head_stride = tensor.stride[2]
                sequence_stride = tensor.stride[1]
                chunks = self.v_tma_swizzle_chunks
                chunk_elems = self.v_swizzle_chunk_elems
                swizzle = self.v_tma_swizzle
            else:
                batch = tensor.shape[3]
                heads = tensor.shape[2]
                sequence = tensor.shape[0]
                batch_stride = tensor.stride[3]
                head_stride = tensor.stride[2]
                sequence_stride = tensor.stride[0]
                chunks = self.k_tma_swizzle_chunks
                chunk_elems = self.k_swizzle_chunk_elems
                swizzle = self.k_tma_swizzle

            layout = cute.make_layout(
                (batch, heads, chunks, sequence, chunk_elems),
                stride=(
                    batch_stride,
                    head_stride,
                    chunk_elems,
                    sequence_stride,
                    1,
                ),
            )
            return cuda_experimental.create_tensor_map_tiled_from_view(
                cute.make_tensor(tensor.iterator, layout),
                box_dims=(1, 1, chunks, self.kv_tile, chunk_elems),
                stride_order=(4, 3, 2, 1, 0),
                swizzle=swizzle,
            )

        tma_k_desc = kv_tma_desc(mK, is_v=False)
        tma_v_desc = kv_tma_desc(mV, is_v=True)
        log2_e = 1.44269504088896340736
        num_q_tiles = ceil_div(mQ.shape[0], self.q_tile)
        total_tiles = num_q_tiles * mQ.shape[3] * mQ.shape[2]
        # Split only an underfilled final wave that still fits in one wave
        # after doubling its CTAs. K/V tiles and metadata stay native blk128.
        sm_count = _device_sm_count()
        remainder = total_tiles % sm_count
        tail_tiles = remainder if remainder <= sm_count // 2 else 0
        full_tiles = total_tiles - tail_tiles
        grid = (full_tiles + 2 * tail_tiles, 1, 1)
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            blocksparse_indices_q2k,
            block_sparse_num,
            tma_k_desc,
            tma_v_desc,
            softmax_scale * log2_e,
            full_tiles,
        ).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        blocksparse_indices_q2k: cute.Tensor,
        block_sparse_num: cutlass.Int32,
        tma_k_desc: cutlass.GridConstant[cuda_experimental.TensorMap],
        tma_v_desc: cutlass.GridConstant[cuda_experimental.TensorMap],
        softmax_scale_log2: cutlass.Float32,
        full_tiles: cutlass.Constexpr[int],
    ) -> None:
        """Initialize TMA barriers and dispatch one full-Q128 or tail-Q64 CTA.

        Tail CTAs split only query rows; both variants consume complete KV128
        tiles using the original logical sparse metadata.
        """
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % cute.arch.WARP_SIZE
        warp = cute.arch.warp_idx()

        sKV = cutlass.Array(
            mK.dtype,
            max(self.k_tile_elems + self.v_tile_elems, self.o_tile_elems),
            space=cutlass.AddressSpace.smem,
            alignment=128,
        )
        sK = sKV
        sV = sKV.subview(self.k_tile_elems)
        tma_mbar = cutlass.Array(
            cutlass.Int64,
            2,
            space=cutlass.AddressSpace.smem,
            alignment=8,
        )
        k_tma_mbar = tma_mbar
        v_tma_mbar = tma_mbar.subview(1)

        if warp == self.load_warp_id:
            if prims.elect_sync():
                prims.prefetch_tensormap(tma_k_desc.get_ptr())
                prims.prefetch_tensormap(tma_v_desc.get_ptr())
                prims.mbarrier_init(k_tma_mbar, 1)
                prims.mbarrier_init(v_tma_mbar, 1)
        prims.fence_mbarrier_init()
        prims.barrier_cta_sync(0)

        linear_idx, _, _ = cute.arch.block_idx()
        num_q_tiles = ceil_div(mQ.shape[0], self.q_tile)
        if linear_idx < full_tiles:
            q_tile_idx = linear_idx % num_q_tiles
            batch_idx = (linear_idx // num_q_tiles) % mQ.shape[3]
            head_idx = linear_idx // (num_q_tiles * mQ.shape[3])
            q_offset = cutlass.Int32(0)
            self._run_sparse_unit(
                mQ,
                mK,
                mV,
                mO,
                mLSE,
                blocksparse_indices_q2k,
                block_sparse_num,
                tma_k_desc,
                tma_v_desc,
                softmax_scale_log2,
                sKV,
                sK,
                sV,
                k_tma_mbar,
                v_tma_mbar,
                lane,
                warp,
                q_tile_idx,
                batch_idx,
                head_idx,
                q_offset,
                False,
            )

        else:
            tail_idx = linear_idx - full_tiles
            logical_idx = full_tiles + tail_idx // 2
            q_tile_idx = logical_idx % num_q_tiles
            batch_idx = (logical_idx // num_q_tiles) % mQ.shape[3]
            head_idx = logical_idx // (num_q_tiles * mQ.shape[3])
            q_offset = (tail_idx % 2) * 64
            self._run_sparse_unit(
                mQ,
                mK,
                mV,
                mO,
                mLSE,
                blocksparse_indices_q2k,
                block_sparse_num,
                tma_k_desc,
                tma_v_desc,
                softmax_scale_log2,
                sKV,
                sK,
                sV,
                k_tma_mbar,
                v_tma_mbar,
                lane,
                warp,
                q_tile_idx,
                batch_idx,
                head_idx,
                q_offset,
                True,
            )

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.jit
    def _run_sparse_unit(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        blocksparse_indices_q2k: cute.Tensor,
        block_sparse_num: cutlass.Int32,
        tma_k_desc: cutlass.GridConstant[cuda_experimental.TensorMap],
        tma_v_desc: cutlass.GridConstant[cuda_experimental.TensorMap],
        softmax_scale_log2: cutlass.Float32,
        sKV: cutlass.Array,
        sK: cutlass.Array,
        sV: cutlass.Array,
        k_tma_mbar: cutlass.Array,
        v_tma_mbar: cutlass.Array,
        lane: cutlass.Int32,
        warp: cutlass.Int32,
        q_tile_idx: cutlass.Int32,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        q_offset: cutlass.Int32,
        is_tail: cutlass.Constexpr[bool],
    ) -> None:
        """Run one sparse query unit with four or eight compute warps.

        A dedicated load warp issues native KV128 TMA transfers. Compute
        warps keep Q in registers, apply online softmax, and reuse dead KV
        shared storage for the output epilogue.
        """
        compute_warps = 4 if is_tail else 8
        pipeline_threads = compute_warps * 32 + 32
        q_seq_idx = q_tile_idx * self.q_tile + q_offset
        seqlen_q = cutlass.Int32(mQ.shape[0])
        seqlen_k = cutlass.Int32(mK.shape[0])

        q_ptr = mQ.iterator.raw_ptr()
        o_ptr = mO.iterator.raw_ptr()
        q_seq_stride, _, q_head_stride, q_batch_stride = mQ.stride
        o_seq_stride, _, o_head_stride, o_batch_stride = mO.stride
        q_head_off = batch_idx * q_batch_stride + head_idx * q_head_stride
        o_head_off = batch_idx * o_batch_stride + head_idx * o_head_stride
        kv_head_idx = head_idx // self.gqa_ratio

        # The public metadata view is (logical_k, q_block, q_head, batch).
        gIndices = blocksparse_indices_q2k[None, q_tile_idx, head_idx, batch_idx]
        num_kv_tiles = block_sparse_num

        # The compute boundary is a whole warpgroup in both specializations.
        # Every non-compute warpgroup executes the same donation instruction.
        if warp >= compute_warps:
            prims.setmaxregister(self.load_regs, prims.SetMaxRegisterAction.DECREASE)

        if warp == self.load_warp_id:
            logical_idx = num_kv_tiles - 1
            physical_idx = gIndices[logical_idx]
            self.load_one_kv_tile(
                sK,
                tma_k_desc,
                k_tma_mbar,
                batch_idx,
                kv_head_idx,
                physical_idx * self.kv_tile,
                is_v=False,
                envelope=False,
            )
            self.load_one_kv_tile(
                sV,
                tma_v_desc,
                v_tma_mbar,
                batch_idx,
                kv_head_idx,
                physical_idx * self.kv_tile,
                is_v=True,
                envelope=False,
            )

            # Four-fold unrolling reduces loop-control issue overhead while
            # keeping the fixed-top-k bound dynamic across cached launches.
            for load_offset in cutlass.range(num_kv_tiles - 1, unroll=4):
                # Resolve the next sparse address before waiting for K to be
                # consumed so the metadata load overlaps the QK work.
                logical_idx = num_kv_tiles - 2 - load_offset
                physical_idx = gIndices[logical_idx]
                prims.barrier_cta_sync(
                    self.bar_k_consumed,
                    thread_count=pipeline_threads,
                )
                self.load_one_kv_tile(
                    sK,
                    tma_k_desc,
                    k_tma_mbar,
                    batch_idx,
                    kv_head_idx,
                    physical_idx * self.kv_tile,
                    is_v=False,
                    envelope=False,
                )

                prims.barrier_cta_sync(
                    self.bar_v_consumed,
                    thread_count=pipeline_threads,
                )
                self.load_one_kv_tile(
                    sV,
                    tma_v_desc,
                    v_tma_mbar,
                    batch_idx,
                    kv_head_idx,
                    physical_idx * self.kv_tile,
                    is_v=True,
                    envelope=False,
                )

            prims.barrier_cta_sync(
                self.bar_k_consumed,
                thread_count=pipeline_threads,
            )
            prims.barrier_cta_sync(
                self.bar_v_consumed,
                thread_count=pipeline_threads,
            )
        elif warp < compute_warps:
            prims.setmaxregister(self.compute_regs, prims.SetMaxRegisterAction.INCREASE)

            compute_warp_idx = warp
            q_warp_row0 = compute_warp_idx * self.MMA_TILER[0]
            lane_div8 = lane // 8
            lane_mod8 = lane % 8
            lane_div16 = lane // 16

            row_max = cutlass.Array(cutlass.Float32, 2, alignment=16)
            row_sum = cutlass.Array(cutlass.Float32, 2, alignment=16)
            for i in cutlass.range_constexpr(2):
                row_max[i] = -cutlass.Float32.inf
                row_sum[i] = 0.0

            o_regs = cutlass.Array(
                cutlass.Float32,
                self.pv_d_frags * 4,
                alignment=16,
            )
            for i in cutlass.range_constexpr(self.pv_d_frags * 4):
                o_regs[i] = 0.0

            basic_params = SimpleNamespace(
                phase_base=cutlass.Int32(0),
                pipeline_threads=pipeline_threads,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                head_dim_qk=cutlass.Int32(self.head_tile_qk),
                q_ptr=q_ptr,
                batch_idx=batch_idx,
                head_idx=head_idx,
                q_seq_idx=q_seq_idx,
                q_head_off=q_head_off,
                q_seq_stride=q_seq_stride,
                q_head_stride=q_head_stride,
                q_warp_row0=q_warp_row0,
                lane=lane,
                lane_div8=lane_div8,
                lane_mod8=lane_mod8,
                lane_div16=lane_div16,
                tma_k_desc=tma_k_desc,
                tma_v_desc=tma_v_desc,
                k_tma_mbar=k_tma_mbar,
                v_tma_mbar=v_tma_mbar,
            )
            mma_params = SimpleNamespace(
                sK=sK,
                sV=sV,
                o_regs=o_regs,
            )
            softmax_params = SimpleNamespace(
                row_max=row_max,
                row_sum=row_sum,
                softmax_scale_log2=softmax_scale_log2,
            )

            q_regs = self.load_q_tile(basic_params)

            logical_idx = num_kv_tiles - 1
            self.compute_one_kv_tile(
                basic_params,
                mma_params,
                softmax_params,
                q_regs,
                num_kv_tiles,
                logical_idx,
                in_mask_steps=False,
                is_first_kv_tile=True,
            )
            for compute_offset in cutlass.range(num_kv_tiles - 1, unroll=4):
                logical_idx = num_kv_tiles - 2 - compute_offset
                self.compute_one_kv_tile(
                    basic_params,
                    mma_params,
                    softmax_params,
                    q_regs,
                    num_kv_tiles,
                    logical_idx,
                    in_mask_steps=False,
                    is_first_kv_tile=False,
                )

            ln2 = cutlass.Float32(0.6931471805599453)
            row_sum_inv = cutlass.Array(cutlass.Float32, 2, alignment=8)
            row_lse = cutlass.Array(cutlass.Float32, 2, alignment=8)
            for row_half in cutlass.range_constexpr(2):
                inv = cutlass.Float32(0.0)
                if row_sum[row_half] > 0.0:
                    inv = cute.math.rcp(row_sum[row_half], approx=True, ftz=True)
                row_sum_inv[row_half] = inv
                row_lse[row_half] = row_max[row_half] * softmax_scale_log2 * ln2 + cute.math.log(
                    cute.math.max(row_sum[row_half], cutlass.Float32(1e-30)),
                    fastmath=True,
                )

            if lane % 4 == 0:
                lse_arr = cutlass.make_array_view(mLSE)
                for row_half in cutlass.range_constexpr(2):
                    row_in_cta = q_warp_row0 + (lane // 4) + row_half * 8
                    lse_q_idx = q_seq_idx + row_in_cta
                    if lse_q_idx < seqlen_q:
                        lse_arr[lse_q_idx, head_idx, batch_idx] = row_lse[row_half]

            prims.barrier_cta_sync(
                self.bar_compute_sync,
                thread_count=compute_warps * 32,
            )

            # K/V are dead after the compute barrier, so their backing becomes
            # the coalesced O epilogue tile.
            sO = sKV
            row_sum_inv_vec = cutlass.Vector.from_elements(
                (
                    row_sum_inv[0],
                    row_sum_inv[0],
                    row_sum_inv[1],
                    row_sum_inv[1],
                    row_sum_inv[0],
                    row_sum_inv[0],
                    row_sum_inv[1],
                    row_sum_inv[1],
                ),
                cutlass.Float32,
            )
            for d_frag_pair in cutlass.range_constexpr(self.pv_d_frags // 2):
                o_off = (d_frag_pair * 2) * 4
                o_scaled = fmul2(o_regs[o_off:8], row_sum_inv_vec)
                o_packed = o_scaled.to(mO.dtype).bitcast(cutlass.Int32)
                sO_ptr = sO.data_ptr() + (compute_warp_idx * (self.pv_d_frags // 2) + d_frag_pair) * (16 * 16) + lane * 8
                prims.stmatrix(sO_ptr, o_packed, prims.MMALayout.ROW)

            store_row = lane_mod8 + (lane_div8 % 2) * 8
            store_col = lane_div16 * 8
            store_q_seq_idx = q_seq_idx + q_warp_row0 + store_row
            for d_frag_pair in cutlass.range_constexpr(self.pv_d_frags // 2):
                store_col_in_cta = d_frag_pair * 16 + store_col
                if store_q_seq_idx < seqlen_q:
                    gO_ptr = o_ptr + o_head_off + store_q_seq_idx * o_seq_stride + store_col_in_cta
                    sO_ptr = sO.data_ptr() + (compute_warp_idx * (self.pv_d_frags // 2) + d_frag_pair) * (16 * 16) + lane * 8
                    gO_ptr.store(sO_ptr.load(count=8, alignment=16), alignment=16)

    @cute.jit
    def compute_one_kv_tile(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        q_regs: cutlass.Array,
        num_kv_tiles: cutlass.Int32,
        kv_tile_idx: cutlass.Int32,
        in_mask_steps: cutlass.Constexpr[bool],
        is_first_kv_tile: cutlass.Constexpr[bool],
    ) -> None:
        """Compute one native KV block with the selected CTA's barrier count."""
        phase = (num_kv_tiles - 1 - kv_tile_idx) & cutlass.Int32(1)
        k_mbar = basic_params.k_tma_mbar
        v_mbar = basic_params.v_tma_mbar
        while not prims.mbarrier_try_wait_parity(k_mbar, phase):
            pass
        s_regs = self.mma_qk(basic_params, mma_params, q_regs)
        prims.barrier_cta_arrive(self.bar_k_consumed, basic_params.pipeline_threads)
        while not prims.mbarrier_try_wait_parity(v_mbar, phase):
            pass
        p_regs = self.online_softmax(
            basic_params,
            mma_params,
            softmax_params,
            s_regs,
            kv_tile_idx * self.kv_tile,
            in_mask_steps,
            is_first_kv_tile,
        )
        self.mma_pv(basic_params, mma_params, p_regs)
        prims.barrier_cta_arrive(self.bar_v_consumed, basic_params.pipeline_threads)
