# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Copyright (c) 2026, Jerry Chen

import math
import operator
from functools import partial
from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum

from cudnn.deepseek_sparse_attention.utils import copy as copy_ops
from cudnn.deepseek_sparse_attention.utils.sm90 import mma as sm90_mma
from cudnn.deepseek_sparse_attention.utils.sm90 import primitives as sm90_ops
from cudnn.deepseek_sparse_attention.utils.sm90.mma import (
    gemm,
    gemm_zero_init,
    mma_partition_fragment_AB,
)
from cudnn.deepseek_sparse_attention.utils.sm90.bwd_barriers import NamedBarrierBwd
from .pack_gqa import PackGQA
from cudnn.deepseek_sparse_attention.utils.seqlen import SeqlenInfoQK
from cudnn.deepseek_sparse_attention.utils.sm90.bwd_tile_scheduler import (
    ParamsBase,
    SingleTileScheduler,
    StaticPersistentTileScheduler,
    TileSchedulerArguments,
)


class SparseScoreRecomputeSm90:
    """Sparse KV backward score kernel for Hopper SM90.

    2-WG producer/consumer (num_threads=256): WG0 loads Q (TMA) + KV (cp.async
    scatter-gather via mTopkIdxs) + Weights/LSE; WG1 runs WGMMA + softmax/L1-norm
    + output write. Dense (full-KV) path lives in DenseScoreRecomputeSm90.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int = 128,  # index_scores: 128, attention_scores: 512
        qhead_per_kvhead: int = 64,  # index_scores: 32,  attention_scores: 64
        tile_m: int = 64,  # index_scores: 32 (post-swapAB, tile_m == qhpkv), attention_scores: 64
        tile_n: int = 64,
        KV_stage: int = 2,
        num_threads: int = 256,  # 2-WG only on sparse path: 1 producer + 1 consumer
        swap_AB: bool = True,
        topk_max: int = 512,
        is_index_scores: bool = True,  # index_scores or attention_scores
        softmax_scale: float = 1.0,
        has_topk_length: bool = True,
        num_head_tiles: int = 1,  # >1 when qhead_per_kvhead > tile_m (head tiling)
        is_sparse: bool = True,  # kept for API compat; sparse kernel always True
        output_log_probs: bool = False,
        topk_indices_global: bool = True,
    ):
        assert is_sparse, "SparseScoreRecomputeSm90 only supports sparse mode"
        assert num_threads == 256, "SparseScoreRecomputeSm90 is 2-WG only (num_threads=256); " "dense full-KV path lives in DenseScoreRecomputeSm90."
        assert qhead_per_kvhead > 1, "This version only supports MQA/GQA (qhead_per_kvhead > 1)"
        arch = 90
        self.dtype = dtype
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_index_scores = is_index_scores
        self.softmax_scale = softmax_scale
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_head_tiles = num_head_tiles
        self.tile_m_sched = tile_m * num_head_tiles  # tile scheduler uses full qhpkv
        self.num_threads = num_threads
        self.KV_stage = KV_stage
        self.swap_AB = swap_AB
        self.arch = arch
        self.topk_max = topk_max
        self.weights_or_lse_dtype = self.dtype if self.is_index_scores else cutlass.Float32
        self.has_topk_length = has_topk_length
        self.output_log_probs = output_log_probs
        self.topk_indices_global = topk_indices_global
        if output_log_probs and not self.is_index_scores:
            raise ValueError("output_log_probs is only supported for index_scores")
        # SM90 WGMMA layout: 4 warps/WG, each warp handles tile_n/4 rows,
        # 8 leader threads (lane&3==0) per warp → each thread owns tile_n/32 N-rows.
        self.num_acc_n_rows_per_thread = self.tile_n // 32
        # Sparse path is always 2-WG: WG0 = producer, WG1 = consumer.
        self.num_warp_groups = 2
        self.num_mma_warp_groups = 1  # Only WG1 (consumer) runs WGMMA

    def _check_type(self, mQ_type, mKV_type, weights_or_lse_type=None):
        if const_expr(not (mQ_type == mKV_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(self.is_index_scores):
            if const_expr(not (mQ_type == weights_or_lse_type)):
                raise TypeError("Q/KV and weights must have the same data type")
        if const_expr(self.is_index_scores):
            if const_expr(weights_or_lse_type not in [cutlass.Float16, cutlass.BFloat16]):
                raise TypeError("Weight tensor must be half precision")
        if const_expr(not self.is_index_scores):
            if const_expr(weights_or_lse_type not in [cutlass.Float32]):
                raise TypeError("LSE tensor must be float32")
        assert mQ_type == self.dtype

    def _setup_attributes(self):
        # sQ/sKV single-stage layouts
        # Note:
        # 1. for index_scores, after swapAB, 'A'=(64, 128) and 'B'=(32, 128); for attention_scores, 'A'=(64, 512) and 'B'=(64, 512)
        self.sQ_layout, self.sKV_layout = [
            sm90_mma.make_smem_layout(self.dtype, LayoutEnum.ROW_MAJOR, shape, stage)
            for shape, stage in [
                ((self.tile_m, self.tile_hdim), None),  # 32 * 128 for index_scores; 64 * 512 for attention_scores
                ((self.tile_n, self.tile_hdim), None),
            ]
        ]
        # 2-stage KV layout for WGMMA: (tile_n, tile_hdim, 2)
        # Used to create a single WGMMA fragment spanning both stages,
        # enabling stage selection via runtime index instead of if/else branch.
        self.sKV_layout_staged = sm90_mma.make_smem_layout(self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdim), stage=self.KV_stage)

    def _get_tiled_mma(self):
        # Q @ K^T GEMM
        tiled_mma_QK = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(self.tile_n, self.tile_m) if self.swap_AB else (self.tile_m, self.tile_n),
        )
        return tiled_mma_QK

    def _get_shared_storage_cls(self):
        sQ_alignment = sKV_alignment = 1024

        sQ_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(self.sQ_layout)], sQ_alignment]
        # Single allocation for 2-stage KV double-buffer (staged layout includes both stages)
        sKV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(self.sKV_layout_staged)], sKV_alignment]

        sWeights_struct = cute.struct.Align[cute.struct.MemRange[self.weights_or_lse_dtype, cute.round_up(self.tile_m, 64)], 128]

        # Sparse path: one float per topk position, consumed by softmax_l1norm_parallel.
        sTopk_struct = cute.struct.Align[cute.struct.MemRange[Float32, self.topk_max], 128]

        @cute.struct
        class SharedStorage:
            mbar_Q: cute.struct.MemRange[cutlass.Int64, 2]
            # ``mbar_KV`` is unused on the sparse path (KV uses cp.async, not TMA);
            # retained so the SMEM layout matches the M2 baseline exactly.
            mbar_KV: cute.struct.MemRange[cutlass.Int64, 2]
            sWeights: sWeights_struct
            sTopk_reduced: sTopk_struct
            sQ: sQ_struct
            sKV: sKV_struct  # 2-stage double-buffer for KV load/compute overlap

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # q_indexer tensor or q_attention tensor
        mKV: cute.Tensor,  # k_indexer tensor or k_attention tensor
        mTopkIdxs: cute.Tensor = None,  # (batch, seqlen_q, topk_max) int32; None in dense mode
        stream: cuda.CUstream = None,
        mOut: cute.Tensor = None,  # (batch, seqlen_q, topk_max)
        weights_or_lse: cute.Tensor = None,  # (batch, nheads, seqlen_q) Note: weights is half precision
        mTopkLength: cute.Tensor = None,  # (batch, seqlen_q) int32, per-q valid KV count
        mL1NormDenom: cute.Tensor = None,  # (batch, seqlen_q) float32, dense attn L1 norm denominator
    ):
        self._check_type(mQ.element_type, mKV.element_type, weights_or_lse.element_type)

        # Assume all strides are divisible by 128 bits except the last stride.
        # Skip cute.assume() for Python ints (e.g. stride=0 from broadcast dims).
        def _assume_strides(t):
            divby = 128 // t.element_type.width
            new_strides = []
            for s in t.stride[:-1]:
                if const_expr(isinstance(s, int)):
                    new_strides.append(s)
                else:
                    new_strides.append(cute.assume(s, divby=divby))
            new_strides.append(t.stride[-1])
            return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=tuple(new_strides)))

        mQ = _assume_strides(mQ)
        mKV = _assume_strides(mKV)
        mWeights = _assume_strides(weights_or_lse)
        mOut = _assume_strides(mOut)
        # mL1NormDenom is unused on the sparse path (dense-only); ignore the input.

        # Layout transpose: (b, s, h, d) --> (s, d, h, b)
        layout_transpose = [1, 3, 2, 0]
        mQ, mKV = [sm90_ops.select(t, layout_transpose) for t in (mQ, mKV)]

        mWeights_transpose_layout = [2, 1, 0]
        mWeights = sm90_ops.select(mWeights, mWeights_transpose_layout)

        # PackGQA: reshape tensor layouts for packed M dimension
        qhpkv = self.qhead_per_kvhead
        num_head_kv = mKV.shape[2]  # num_head_kv is always 1 as PackGQA
        mQ = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                ((qhpkv, mQ.shape[0]), mQ.shape[1], num_head_kv, *mQ.shape[3:]),
                stride=((mQ.stride[2], mQ.stride[0]), mQ.stride[1], mQ.stride[2] * qhpkv, *mQ.stride[3:]),
            ),
        )

        mWeights = cute.make_tensor(
            mWeights.iterator,
            cute.make_layout(
                ((qhpkv, mWeights.shape[0]), num_head_kv, *mWeights.shape[2:]),  # 32 * seqlen, 1, bs
                stride=((mWeights.stride[1], mWeights.stride[0]), mWeights.stride[1] * qhpkv, *mWeights.stride[2:]),
            ),
        )

        tiled_mma_QK = self._get_tiled_mma()

        self.num_mma_threads = 128 * self.num_mma_warp_groups  # 128 (only WG0)
        assert self.num_mma_threads <= self.num_threads
        self.num_threads_per_warp_group = 128
        # reg allocation for wg0
        self.num_mma_regs = 256

        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        self.tma_copy_bytes = {name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1])) for name, mX, layout in [("Q", mQ, self.sQ_layout)]}

        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            self.sQ_layout,
            (self.tile_m, self.tile_hdim),
        )

        # Sparse path: KV is loaded via cp.async scatter-gather indexed by
        # mTopkIdxs; no TMA KV atom is built (that's the dense path).
        mKV_for_kernel = mKV

        # TileScheduler by Q dimension (m_blocks)
        is_persistent = False
        TileScheduler = StaticPersistentTileScheduler if is_persistent else SingleTileScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m_sched),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            1,  # num_splits
            cute.size(mKV.shape[0]),
            mQ.shape[1],
            mKV.shape[1],
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m_sched, self.tile_n),
            mCuSeqlensQ=None,
            mSeqUsedQ=None,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead,
            element_size=self.dtype.width // 8,
            is_persistent=is_persistent,
            lpt=False,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        # cute.printf("grid_dim is {}", grid_dim)

        qhead_per_kvhead_divmod = FastDivmodDivisor(self.qhead_per_kvhead)

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = self.softmax_scale * LOG2_E

        self.kernel(
            tma_tensor_Q,
            tma_atom_Q,
            mKV_for_kernel,
            mOut,
            mWeights,
            mTopkIdxs,
            mTopkLength,  # sparse KV indices + per-q length
            self.sQ_layout,
            self.sKV_layout_staged,
            tiled_mma_QK,
            softmax_scale_log2,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            qhead_per_kvhead_divmod,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # tma_tensor_Q (TMA view of q)
        tma_atom_Q: cute.CopyAtom,
        mKV: cute.Tensor,
        mOut: cute.Tensor,
        mWeights: cute.Tensor,
        mTopkIdxs: cute.Tensor = None,  # (batch, seqlen_q, topk_max) int32
        mTopkLength: cute.Tensor = None,
        sQ_layout: cute.ComposedLayout = None,
        sKV_layout_staged: cute.ComposedLayout = None,
        tiled_mma_QK: cute.TiledMma = None,
        softmax_scale_log2: Float32 = None,
        tile_sched_params: ParamsBase = None,
        TileScheduler: cutlass.Constexpr[Callable] = None,
        SharedStorage: cutlass.Constexpr[Callable] = None,
        qhead_per_kvhead_divmod: FastDivmodDivisor = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        mbar_Q_ptr = storage.mbar_Q.data_ptr()
        if warp_idx == 0:
            cute.arch.mbarrier_init(mbar_Q_ptr, 1)
        cute.arch.sync_threads()

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # Single 3D staged tensor (tile_n, tile_hdim, 2) for double-buffered KV
        sKV_staged = storage.sKV.get_tensor(sKV_layout_staged.outer, swizzle=sKV_layout_staged.inner)
        # 2D views for cp.async loading
        sKV_0 = sKV_staged[None, None, 0]
        sKV_1 = sKV_staged[None, None, 1]

        sWeights = storage.sWeights.get_tensor(cute.make_layout((self.tile_m,), stride=(1,)))
        if tidx == 0:
            sWeights.fill(0.0)

        # Sparse path: one float per topk position, consumed by softmax_l1norm_parallel.
        sTopk_reduced = storage.sTopk_reduced.get_tensor(cute.make_layout((self.topk_max,), stride=(1,)))

        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0][1],
            seqlen_k_static=mKV.shape[0],
            mCuSeqlensQ=None,
            mCuSeqlensK=None,
            mSeqUsedQ=None,
            mSeqUsedK=None,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        # Sparse path: always 2-WG. WG0 = producer (cp.async KV loads),
        # WG1 = consumer (WGMMA + softmax/L1-norm + output write).
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        if warp_group_idx == 0:
            self.producer_wg0(
                mQ,
                tma_atom_Q,
                mKV,
                mWeights,
                mTopkIdxs,
                mTopkLength,
                sQ,
                sKV_0,
                sKV_1,
                sWeights,
                mbar_Q_ptr,
                tidx,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
        else:
            self.consumer_wg1(
                tiled_mma_QK,
                mOut,
                mTopkIdxs,
                mTopkLength,
                sQ,
                sKV_staged,
                sWeights,
                sTopk_reduced,
                mbar_Q_ptr,
                tidx,
                SeqlenInfoCls,
                TileSchedulerCls,
                softmax_scale_log2,
            )

    # =========================================================================
    # WG0 producer: loads Q + Weights + KV into SMEM, syncs with WG1
    # =========================================================================
    @cute.jit
    def producer_wg0(
        self,
        mQ: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        mKV: cute.Tensor,
        mWeights: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: cute.Tensor,
        sQ: cute.Tensor,
        sKV_0: cute.Tensor,
        sKV_1: cute.Tensor,
        sWeights: cute.Tensor,
        mbar_Q_ptr,
        tidx: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        """WG0 producer (tidx 0-127): loads Q (TMA) + Weights/LSE + KV (cp.async scatter-gather).
        Runs in lockstep with consumer_wg1 via named barriers.
        """
        wg_tidx = tidx % self.num_threads_per_warp_group  # 0..127
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mbar_Q_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            head_idx_kv = head_idx

            if const_expr(self.has_topk_length):
                topK = mTopkLength[batch_idx, m_block]
            else:
                topK = self.topk_max
            n_block_max = (topK + self.tile_n - 1) // self.tile_n
            topk_tail_rows = topK - (n_block_max - 1) * self.tile_n

            mKV_cur = mKV[None, None, head_idx_kv, batch_idx]
            mTopkIdxs_cur = mTopkIdxs[batch_idx, m_block, None]

            mQ_cur = mQ[None, None, head_idx, batch_idx]

            # Weights setup (shared across head tiles)
            mWeights_cur = mWeights[None, head_idx, batch_idx]
            seqlen_q_packed = mWeights_cur.shape[0][1]
            _pack_gqa = PackGQA(0, 0, False, self.qhead_per_kvhead)

            # === Head tile loop: iterate over head slices ===
            for h_tile in cutlass.range_constexpr(self.num_head_tiles):
                eff_m_block = m_block * self.num_head_tiles + h_tile

                # ---- TMA Q load (warp 0 leader) ----
                gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (eff_m_block, 0))
                load_Q, _, _ = copy_ops.tma_get_copy_fn(tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True)
                if warp_idx_in_wg == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar_Q_ptr, self.tma_copy_bytes["Q"])
                    load_Q(tma_bar_ptr=mbar_Q_ptr)

                # ---- Load Weights/LSE for this head tile ----
                if const_expr(self.is_index_scores):
                    _pack_gqa.load_Weights_packed(
                        mWeights_cur.iterator.toint(),
                        seqlen_q_packed,
                        sWeights,
                        eff_m_block,
                        self.tile_m,
                        wg_tidx,
                    )
                else:
                    _pack_gqa.load_LSE_packed(
                        mWeights_cur.iterator.toint(),
                        seqlen_q_packed,
                        sWeights,
                        eff_m_block,
                        self.tile_m,
                        wg_tidx,
                    )

                cute.arch.fence_view_async_shared()
                # ---- Sync: Q+Weights ready (cross-WG) ----
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.QueryWeightsReady),
                    number_of_threads=self.num_threads,
                )
                cute.arch.mbarrier_wait(mbar_Q_ptr, mbar_Q_phase)
                mbar_Q_phase = mbar_Q_phase ^ 1

                # ---- KV prologue: cp.async scatter-gather first block → buf 0 ----
                n_block = n_block_max - 1
                if n_block >= 0:
                    self.load_sparse_kv(mKV_cur, mTopkIdxs_cur, sKV_0, n_block, True, topk_tail_rows, m_block, batch_idx)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.KVReady),
                    number_of_threads=self.num_threads,
                )

                # ---- Main pipeline: load next KV while WG1 computes ----
                next_n_block = n_block - 1
                cur_buf = Int32(0)

                while n_block >= 0:
                    if next_n_block >= 0:
                        self.load_sparse_kv_select(mKV_cur, mTopkIdxs_cur, sKV_0, sKV_1, 1 - cur_buf, next_n_block, False, self.tile_n, m_block, batch_idx)
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    # Sync: WG0's load done + WG1's GEMM done
                    cute.arch.barrier(
                        barrier_id=int(NamedBarrierBwd.KVReady),
                        number_of_threads=self.num_threads,
                    )

                    cur_buf = 1 - cur_buf
                    n_block = next_n_block
                    next_n_block = n_block - 1
            # === End head tile loop ===

            # ---- Wait for WG1's softmax to complete ----
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.TileComplete),
                number_of_threads=self.num_threads,
            )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    # =========================================================================
    # WG1 consumer: WGMMA + postprocess + softmax + output
    # =========================================================================
    @cute.jit
    def consumer_wg1(
        self,
        tiled_mma_QK: cute.TiledMma,
        mOut: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: cute.Tensor,
        sQ: cute.Tensor,
        sKV_staged: cute.Tensor,
        sWeights: cute.Tensor,
        sTopk_reduced: cute.Tensor,
        mbar_Q_ptr,
        tidx: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        softmax_scale_log2: Float32,
    ):
        """WG1 consumer (tidx 128-255): WGMMA Q@K^T + postprocess + softmax + output.
        Runs in lockstep with producer_wg0 via named barriers.
        """
        wg_tidx = tidx % self.num_threads_per_warp_group  # 0..127
        lane = cute.arch.lane_idx()
        LOG2_E = math.log2(math.e)

        # ---- MMA partition setup (from SMEM tensors) ----
        wg_mma_QK = tiled_mma_QK.get_slice(wg_tidx)
        thr_mma_QK = tiled_mma_QK.get_slice(wg_tidx)
        tSrQ, tSrK = mma_partition_fragment_AB(wg_mma_QK, sQ, sKV_staged, self.swap_AB)
        mma_qk_fn = partial(
            gemm_zero_init,
            tiled_mma_QK,
            (self.tile_m, self.tile_n),
            tSrQ,
            tSrK,
            swap_AB=self.swap_AB,
        )
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mbar_Q_phase = Int32(0)

        # ---- Weights SMEM partition (for S→R after Q+Weights barrier) ----
        sWeights_mma = cute.make_tensor(
            sWeights.iterator,
            cute.make_layout((self.tile_m, self.tile_n), stride=(1, 0)),
        )
        if const_expr(self.swap_AB):
            sWeights_mma = sm90_ops.transpose_view(sWeights_mma)
        weights_slice = (None, 0) if const_expr(not self.swap_AB) else (0, None)
        tWsWeights = sm90_ops.make_acc_tensor_mn_view(thr_mma_QK.partition_C(sWeights_mma))[weights_slice]

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            if const_expr(self.has_topk_length):
                topK = mTopkLength[batch_idx, m_block]
            else:
                topK = self.topk_max
            n_block_max = (topK + self.tile_n - 1) // self.tile_n

            mTopkIdxs_cur = mTopkIdxs[batch_idx, m_block, None]
            mOut_cur = mOut[batch_idx, m_block, None]

            # === Head tile loop: iterate over head slices ===
            for h_tile in cutlass.range_constexpr(self.num_head_tiles):
                # ---- Sync: wait for Q+Weights from WG0 ----
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.QueryWeightsReady),
                    number_of_threads=self.num_threads,
                )
                cute.arch.mbarrier_wait(mbar_Q_ptr, mbar_Q_phase)
                mbar_Q_phase = mbar_Q_phase ^ 1

                # Weights S→R
                tWeightsrWeights = copy_ops.load_s2r(tWsWeights)

                # ---- Sync: wait for KV prologue from WG0 ----
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.KVReady),
                    number_of_threads=self.num_threads,
                )

                # ---- Main pipeline: GEMM while WG0 loads next KV ----
                n_block = n_block_max - 1
                next_n_block = n_block - 1
                cur_buf = Int32(0)
                accumulate = Boolean(h_tile > 0)
                is_last_h = Boolean(h_tile == self.num_head_tiles - 1)

                while n_block >= 0:
                    # GEMM + postprocess + head-reduce on current buffer
                    n_block_base_row = n_block * self.tile_n + warp_idx_in_wg * 16
                    acc_S = mma_qk_fn(B_idx=cur_buf, wg_wait=-1)
                    warpgroup.wait_group(0)
                    self._postprocess_and_reduce(
                        acc_S,
                        tWeightsrWeights,
                        softmax_scale_log2,
                        LOG2_E,
                        n_block_base_row,
                        lane,
                        topK,
                        sTopk_reduced,
                        m_block,
                        batch_idx,
                        n_block,
                        tidx,
                        accumulate=accumulate,
                    )

                    # Sync: WG1's GEMM done + WG0's next load done
                    cute.arch.barrier(
                        barrier_id=int(NamedBarrierBwd.KVReady),
                        number_of_threads=self.num_threads,
                    )

                    cur_buf = 1 - cur_buf
                    n_block = next_n_block
                    next_n_block = n_block - 1

            # === End head tile loop ===

            # ---- Softmax + output (WG1 only, 128 threads) ----
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.WG1_consumer_sync),
                number_of_threads=self.num_threads_per_warp_group,
            )
            self.softmax_l1norm_parallel(sTopk_reduced, topK, tidx, mTopkIdxs_cur, mOut_cur)

            # ---- Signal WG0: tile complete ----
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.TileComplete),
                number_of_threads=self.num_threads,
            )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def _postprocess_and_reduce(
        self,
        acc_S: cute.Tensor,
        tWeightsrWeights: cute.Tensor,
        softmax_scale_log2: Float32,
        LOG2_E: Float32,
        n_block_base_row: Int32,
        lane: Int32,
        topK: Int32,
        sTopk_reduced: cute.Tensor,
        m_block: Int32,
        batch_idx: Int32,
        n_block: Int32,
        tidx: Int32,
        accumulate: Boolean = Boolean(False),  # True when accumulating across head tiles
    ):
        """Post-process GEMM result (relu*weights or exp2) and head-reduce into ``sTopk_reduced``.

        For each topk position owned by this thread, sums across heads:
          indexer   (is_index_scores=True):  sum_c ReLU(acc_S[r, c]) * W[c]
          attention (is_index_scores=False): sum_c exp2(acc_S[r, c] * s_log2 - LSE[c] * log2e)
        then intra-warp (width=4) reduces across the 4 threads sharing each row and writes
        the per-position result to ``sTopk_reduced`` SMEM.

        When ``accumulate=True`` (head-tile iterations > 0), adds to the existing value in SMEM
        instead of overwriting.
        """
        # Native view: rows = N-dim (topk positions), cols = M-dim (Q/head positions)
        # tWeightsrWeights[c] gives the weight for M-position c (same indexing as transposed row)
        acc_S_mn = sm90_ops.make_acc_tensor_mn_view(acc_S)

        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            row_sum_cur_thread = Float32(0.0)
            for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                if const_expr(self.is_index_scores):
                    row_sum_cur_thread += cute.arch.fmax(acc_S_mn[r, c], Float32(0.0)) * tWeightsrWeights[c]
                else:
                    row_sum_cur_thread += cute.math.exp2(acc_S_mn[r, c] * softmax_scale_log2 - tWeightsrWeights[c] * LOG2_E, fastmath=True)
            row_sum_cur_thread = sm90_ops.warp_reduce(row_sum_cur_thread, operator.add, width=4)
            # Indexer: apply sm_scale on the fp32 head-reduced row sum (post
            # warp_reduce, pre head-tile accumulation). Preserves precision
            # vs pre-multiplying onto bf16 W on the host. Distributivity is
            # preserved across the cross-warp reduce and the head-tile
            # accumulation (`accumulate=True` branch). Attention path already
            # bakes softmax_scale into softmax_scale_log2 via the exp2 inside
            # the head loop, so it does not need an extra multiply here.
            if const_expr(self.is_index_scores and self.softmax_scale != 1.0):
                row_sum_cur_thread = row_sum_cur_thread * Float32(self.softmax_scale)
            pos = n_block_base_row + r * 8 + (lane >> 2)
            if (lane & 3) == 0 and pos < topK:
                if accumulate:
                    sTopk_reduced[pos] = sTopk_reduced[pos] + row_sum_cur_thread
                else:
                    sTopk_reduced[pos] = row_sum_cur_thread

    # cp.async one gmem row → swizzled smem
    @cute.jit
    def _copy_row(
        self,
        mKV_cur: cute.Tensor,  # (s_kv, headdim) gmem
        sKV: cute.Tensor,  # (tile_n, headdim) swizzled smem
        row: Int32,  # smem row index
        idx_in_group: Int32,  # 0..7 dim dir
        copy_atom: cute.CopyAtom,
        thr_copy: cute.TiledCopy,
        token_idx: Int32,  # pre-loaded from mTopkIdxs_cur[global_topk_row]
    ):
        gKV_row = mKV_cur[token_idx, None]
        gKV_chunks = cute.flat_divide(gKV_row, (8,))
        for tile in cutlass.range_constexpr(self.tile_hdim // 64):
            chunk_idx = tile * 8 + idx_in_group
            g_chunk = gKV_chunks[None, chunk_idx]
            sKV_row = sKV[row, None]
            sKV_row_chunks = cute.flat_divide(sKV_row, (8,))
            s_chunk = sKV_row_chunks[None, chunk_idx]
            tSg = thr_copy.partition_S(g_chunk)
            tSs = thr_copy.partition_D(s_chunk)
            cute.copy(copy_atom, tSg, tSs)

    # clear for OOB rows
    # @cute.jit
    # def _zero_row(
    #     self,
    #     sKV: cute.Tensor,
    #     row: Int32,
    # ):
    #     sKV[row, None].fill(0)

    @cute.jit
    def _zero_row(self, sK_slice: cute.Tensor, row: Int32, idx_in_group: Int32):
        """Zero-fill one K row in smem, cooperative across 8 threads in a group."""
        sK_row = sK_slice[row, None]
        sK_chunks = cute.flat_divide(sK_row, (8,))
        for tile in cutlass.range_constexpr(self.tile_hdim // 64):
            chunk_idx = tile * 8 + idx_in_group
            sK_chunks[None, chunk_idx].fill(0)

    @cute.jit
    def load_sparse_kv_select(
        self,
        mKV_cur: cute.Tensor,
        mTopkIdxs_cur: cute.Tensor,
        sKV_0: cute.Tensor,
        sKV_1: cute.Tensor,
        use_buf1: Int32,
        n_block: Int32,
        is_first: Boolean,
        num_valid_rows: Int32,
        m_block: Int32,
        batch_idx: Int32,
    ):
        """load_sparse_kv that selects target buffer via use_buf1 flag (0 or 1),
        avoiding a runtime if/else branch at the call site."""
        if use_buf1 == 0:
            self.load_sparse_kv(mKV_cur, mTopkIdxs_cur, sKV_0, n_block, is_first, num_valid_rows, m_block, batch_idx)
        else:
            self.load_sparse_kv(mKV_cur, mTopkIdxs_cur, sKV_1, n_block, is_first, num_valid_rows, m_block, batch_idx)

    @cute.jit
    def load_sparse_kv(
        self,
        mKV_cur: cute.Tensor,
        mTopkIdxs_cur: cute.Tensor,
        sKV: cute.Tensor,
        n_block: Int32,
        is_first: Boolean,
        num_valid_rows: Int32,
        m_block: Int32,
        batch_idx: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        wg_tidx = tidx % self.num_threads_per_warp_group  # 0..127
        # cp.async copy atom: 128-bit = 16B = 8 bf16, bypass L1
        async_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=128,
        )
        async_thr_copy = cute.make_tiled_copy_tv(
            async_copy_atom,
            cute.make_layout((1,)),
            cute.make_layout((8,)),
        ).get_slice(0)

        # thread organization for cp.async gather
        GROUP_SIZE = const_expr(8)
        NUM_GROUPS = const_expr(self.num_threads_per_warp_group // 8)  # 16
        ROWS_PER_GROUP = const_expr(self.tile_n // NUM_GROUPS)
        idx_in_group = wg_tidx % GROUP_SIZE  # 0..7 dim dir
        group_idx = wg_tidx // GROUP_SIZE  # 0..15 token dir
        # cp.async scatter-gather KV → sKV (all 128 WG0 threads)
        seqlen_k = cute.size(mKV_cur.shape[0])
        batch_offset = batch_idx * seqlen_k if const_expr(self.topk_indices_global) else Int32(0)
        for r in cutlass.range_constexpr(ROWS_PER_GROUP):
            row = r * NUM_GROUPS + group_idx
            global_topk_row = n_block * self.tile_n + row
            if row < num_valid_rows or not is_first:
                # Read token_idx once from GMEM; pass to _copy_row to avoid redundant read
                token_idx = mTopkIdxs_cur[global_topk_row] - batch_offset
                if const_expr(self.has_topk_length):
                    if token_idx >= 0 and token_idx < seqlen_k:
                        self._copy_row(
                            mKV_cur,
                            sKV,
                            row,
                            idx_in_group,
                            async_copy_atom,
                            async_thr_copy,
                            token_idx,
                        )
                    else:
                        self._zero_row(sKV, row, idx_in_group)
                else:
                    if token_idx >= 0 and token_idx < seqlen_k:
                        self._copy_row(
                            mKV_cur,
                            sKV,
                            row,
                            idx_in_group,
                            async_copy_atom,
                            async_thr_copy,
                            token_idx,
                        )
                    else:
                        self._zero_row(sKV, row, idx_in_group)
            else:
                # clear for OOB rows (only in first n_block)
                self._zero_row(sKV, row, idx_in_group)

    @cute.jit
    def softmax_l1norm_parallel(
        self,
        sTopk_reduced: cute.Tensor,
        cur_topk: Int32,
        tidx: Int32,
        mTopkIdxs_cur: cute.Tensor,
        mOut_cur: cute.Tensor,
    ):
        """Parallel softmax/L1-norm using 128 threads within a warp group.
        Sparse score is 2-WG-only today; this helper runs in WG1
        (consumer, tidx 128-255) and uses wg_tidx = tidx % 128 for
        warp-group-local indexing.
        """
        ELEMS_PER_THREAD = self.topk_max // self.num_threads_per_warp_group  # e.g. 512/128 = 4
        wg_tidx = tidx % self.num_threads_per_warp_group  # 0..127 for any WG
        warp_id = wg_tidx >> 5  # 0..3
        lane_id = wg_tidx & 31  # 0..31
        softmax_barrier_id = int(NamedBarrierBwd.WG1_consumer_sync)
        epsilon = Float32(1e-10)

        # Phase 1: Load elements to registers + clean invalid entries
        rVals = cute.make_rmem_tensor(ELEMS_PER_THREAD, Float32)

        if const_expr(self.is_index_scores):
            # === Softmax path ===
            # Phase 1+2 fused: load to regs + clean + find local max
            local_max = -Float32(float("inf"))
            for j in cutlass.range_constexpr(ELEMS_PER_THREAD):
                idx = j * self.num_threads_per_warp_group + wg_tidx
                val = sTopk_reduced[idx]
                if idx < cur_topk:
                    if const_expr(not self.has_topk_length):
                        token_idx = mTopkIdxs_cur[idx]
                        if token_idx < 0:
                            val = -Float32(float("inf"))
                    local_max = cute.arch.fmax(local_max, val)
                else:
                    if const_expr(self.output_log_probs):
                        val = -Float32(float("inf"))
                    else:
                        val = Float32(0.0)
                rVals[j] = val

            # Phase 3: Intra-warp max reduce (butterfly shuffle, all lanes get result)
            for s in cutlass.range_constexpr(5):  # log2(32) = 5
                local_max = cute.arch.fmax(local_max, cute.arch.shuffle_sync_bfly(local_max, offset=1 << s))

            # Phase 4: Cross-warp max reduce via sTopk_reduced[0..3] scratch
            if lane_id == 0:
                sTopk_reduced[warp_id] = local_max
            cute.arch.barrier(
                barrier_id=softmax_barrier_id,
                number_of_threads=self.num_threads_per_warp_group,
            )
            cur_max = sTopk_reduced[0]
            cur_max = cute.arch.fmax(cur_max, sTopk_reduced[1])
            cur_max = cute.arch.fmax(cur_max, sTopk_reduced[2])
            cur_max = cute.arch.fmax(cur_max, sTopk_reduced[3])

            # All entries -inf
            if cur_max == -Float32(float("inf")):
                for j in cutlass.range_constexpr(ELEMS_PER_THREAD):
                    idx = j * self.num_threads_per_warp_group + wg_tidx
                    if const_expr(self.output_log_probs):
                        mOut_cur[idx] = -Float32(float("inf"))
                    else:
                        mOut_cur[idx] = Float32(0.0)
            else:
                # Phase 5: exp + local sum (from registers, no SMEM read needed)
                local_sum = Float32(0.0)
                if const_expr(self.output_log_probs):
                    for j in cutlass.range_constexpr(ELEMS_PER_THREAD):
                        idx = j * self.num_threads_per_warp_group + wg_tidx
                        if idx < cur_topk:
                            local_sum += cute.arch.exp(rVals[j] - cur_max)
                else:
                    for j in cutlass.range_constexpr(ELEMS_PER_THREAD):
                        idx = j * self.num_threads_per_warp_group + wg_tidx
                        if idx < cur_topk:
                            exp_val = cute.arch.exp(rVals[j] - cur_max)
                            rVals[j] = exp_val
                            local_sum += exp_val

                # Phase 6: Cross-thread sum reduce
                local_sum = sm90_ops.warp_reduce(local_sum, operator.add, width=32)
                # Use [4..7] as scratch to avoid race with max scratch in [0..3]
                if lane_id == 0:
                    sTopk_reduced[warp_id + 4] = local_sum
                cute.arch.barrier(
                    barrier_id=softmax_barrier_id,
                    number_of_threads=self.num_threads_per_warp_group,
                )
                cur_sum = sTopk_reduced[4] + sTopk_reduced[5] + sTopk_reduced[6] + sTopk_reduced[7]

                # Phase 7: write prob or log-prob directly to GMEM
                if const_expr(self.output_log_probs):
                    lse = cur_max + sm90_ops.logf(cur_sum + epsilon)
                    for j in cutlass.range_constexpr(ELEMS_PER_THREAD):
                        idx = j * self.num_threads_per_warp_group + wg_tidx
                        if idx < cur_topk:
                            mOut_cur[idx] = rVals[j] - lse
                        else:
                            mOut_cur[idx] = -Float32(float("inf"))
                else:
                    inv_sum = Float32(1.0) / (cur_sum + epsilon)
                    for j in cutlass.range_constexpr(ELEMS_PER_THREAD):
                        idx = j * self.num_threads_per_warp_group + wg_tidx
                        if idx < cur_topk:
                            mOut_cur[idx] = rVals[j] * inv_sum
                        else:
                            mOut_cur[idx] = Float32(0.0)
        else:
            # === L1 norm path ===
            # Phase 1+2 fused: load to regs + clean + local sum
            local_sum = Float32(0.0)
            for j in cutlass.range_constexpr(ELEMS_PER_THREAD):
                idx = j * self.num_threads_per_warp_group + wg_tidx
                val = sTopk_reduced[idx]
                if idx < cur_topk:
                    if const_expr(not self.has_topk_length):
                        token_idx = mTopkIdxs_cur[idx]
                        if token_idx < 0:
                            val = Float32(0.0)
                    local_sum += val
                else:
                    val = Float32(0.0)
                rVals[j] = val

            # Phase 3: Cross-thread sum reduce
            local_sum = sm90_ops.warp_reduce(local_sum, operator.add, width=32)
            if lane_id == 0:
                sTopk_reduced[warp_id] = local_sum
            cute.arch.barrier(
                barrier_id=softmax_barrier_id,
                number_of_threads=self.num_threads_per_warp_group,
            )
            cur_sum = sTopk_reduced[0] + sTopk_reduced[1] + sTopk_reduced[2] + sTopk_reduced[3]

            # Phase 4: Normalize and write directly to GMEM
            inv_sum = Float32(1.0) / (cur_sum + epsilon)
            for j in cutlass.range_constexpr(ELEMS_PER_THREAD):
                idx = j * self.num_threads_per_warp_group + wg_tidx
                if idx < cur_topk:
                    mOut_cur[idx] = rVals[j] * inv_sum
                else:
                    mOut_cur[idx] = Float32(0.0)
