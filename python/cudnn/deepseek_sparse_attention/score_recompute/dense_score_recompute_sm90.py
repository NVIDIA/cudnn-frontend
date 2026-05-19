# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Copyright (c) 2026, Jerry Chen

import math
import operator
import os
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
    gemm_w_idx,
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


class DenseScoreRecomputeSm90:
    """Dense KV backward score kernel for Hopper SM90.

    Production implementation is the 3-WG (1 producer + 2 consumer)
    pingpong layout with optional warp-split TMA producer.  The legacy
    1-WG/2-WG dense implementations were removed after benchmarking showed
    the 3-WG path is uniformly faster on dense attention (+11-20%) and
    non-regressing on dense indexer (+1-5%).

    Sparse topk-indexed KV lives in ``SparseScoreRecomputeSm90``.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int = 128,
        qhead_per_kvhead: int = 64,
        tile_m: int = 64,
        tile_n: int = 64,
        KV_stage: int = 2,
        num_threads: int = 384,
        swap_AB: bool = True,
        topk_max: int = 512,
        is_index_scores: bool = True,
        softmax_scale: float = 1.0,
        has_topk_length: bool = False,
        num_head_tiles: int = 1,
        ratio: int = 1,
    ):
        # 3-WG dense score path.  Common layout / __call__ /
        # postprocess helpers live in this class below; kernel-specific
        # pieces are _get_shared_storage_cls and kernel.
        arch = 90
        assert ratio >= 1, f"ratio must be >= 1, got {ratio}"
        self.dtype = dtype
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        assert qhead_per_kvhead > 1, "MQA/GQA only"
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_index_scores = is_index_scores
        self.is_dense_attn = not is_index_scores
        self.is_dense_indexer = is_index_scores
        self.softmax_scale = softmax_scale
        self.ratio = ratio
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_head_tiles = num_head_tiles
        self.tile_m_sched = tile_m * num_head_tiles
        self.num_threads = num_threads
        # 2-stage KV pipeline: each consumer owns one sKV stage.
        self.KV_stage = 2
        self.swap_AB = swap_AB
        self.arch = arch
        self.topk_max = topk_max
        # Split the producer WG into two independent TMA pipelines:
        #   warp 0 -> stage 0 TMAs (+ Q TMA), warp 1 -> stage 1 TMAs.
        # Removes the serial "wait KVE0 -> TMA0 -> wait KVE1 -> TMA1" chain
        # so consumers' wait KV on both stages clears symmetrically.
        # Default ON: in the bit-exact A/B this gave +7pt on attn hd=512
        # (1.065x -> 1.139x) and ~+1pt / neutral on indexer hd=128, with
        # out maxdiff = 0 preserved.  Set DENSE3WG_USE_WARP_SPLIT_PROD=0
        # to force the unified producer for benchmarking / debugging.
        self.use_warp_split_producer = bool(int(os.environ.get("DENSE3WG_USE_WARP_SPLIT_PROD", "1")))
        self.weights_or_lse_dtype = self.dtype if self.is_index_scores else cutlass.Float32
        self.has_topk_length = has_topk_length
        self.num_acc_n_rows_per_thread = self.tile_n // 32
        self.num_warp_groups = self.num_threads // 128
        assert self.num_warp_groups == 3, "DenseScoreRecomputeSm90 is now 3-WG only (num_threads=384)"
        self.num_mma_warp_groups = 1
        self.num_threads_per_warp_group = 128

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
        """Override to widen sReduce so the two consumers can exchange
        their partial l1norm / lse state via SMEM, and to add the two
        per-stage KV/KVEmpty mbarriers that the 3-WG layout needs.

        sReduce layout:
          dense_attn:    [0..3] = WG0 warp-partial l1norm, [4..7] = WG1 (8 total)
          dense_indexer: [0..7] = WG0 warp-partial (max, sum_exp), [8..15] = WG1 (16 total)
        """
        sQ_alignment = 1024
        sKV_alignment = 1024

        sQ_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(self.sQ_layout)], sQ_alignment]
        sKV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(self.sKV_layout_staged)], sKV_alignment]
        sWeights_struct = cute.struct.Align[cute.struct.MemRange[self.weights_or_lse_dtype, cute.round_up(self.tile_m, 64)], 128]
        if const_expr(self.is_dense_indexer):
            sReduce_size = 16
        else:
            sReduce_size = 8
        sReduce_struct = cute.struct.Align[cute.struct.MemRange[Float32, sReduce_size], 128]

        @cute.struct
        class SharedStorage:
            mbar_Q: cute.struct.MemRange[cutlass.Int64, 2]
            # Per-stage KV mbarrier (arrive_count=1, producer single-thread).
            #   stage 0 → WG0 consumer
            #   stage 1 → WG1 consumer
            mbar_KV0: cute.struct.MemRange[cutlass.Int64, 2]
            mbar_KV1: cute.struct.MemRange[cutlass.Int64, 2]
            # KVEmpty mbarriers (arrive_count=128, consumer WG).  Consumer
            # signals producer that the stage has been consumed and is
            # safe to TMA-refill.
            mbar_KVEmpty0: cute.struct.MemRange[cutlass.Int64, 2]
            mbar_KVEmpty1: cute.struct.MemRange[cutlass.Int64, 2]
            # Pingpong between the two consumer WGs is done via two
            # NamedBarriers (PingMmaWG0_3WG / PingMmaWG1_3WG), each used
            # with bar.sync(256) / bar.arrive(256).  No dedicated
            # mbarrier SMEM needed.
            sWeights: sWeights_struct
            sReduce: sReduce_struct
            sQ: sQ_struct
            sKV: sKV_struct

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
        mL1NormDenom = _assume_strides(mL1NormDenom)  # always present on dense path

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

        # Dense path: KV is loaded via TMA (sequential full-KV iteration).
        self.tma_copy_bytes["KV"] = cute.size_in_bytes(mKV.element_type, cute.select(self.sKV_layout, mode=[0, 1]))
        tma_atom_KV, tma_tensor_KV = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mKV,
            self.sKV_layout,
            (self.tile_n, self.tile_hdim),
        )
        mKV_for_kernel = tma_tensor_KV

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
            self.sQ_layout,
            self.sKV_layout_staged,
            tiled_mma_QK,
            softmax_scale_log2,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            qhead_per_kvhead_divmod,
            mL1NormDenom,
            tma_atom_KV,
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
        mQ: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        mKV: cute.Tensor,
        mOut: cute.Tensor,
        mWeights: cute.Tensor,
        sQ_layout: cute.ComposedLayout = None,
        sKV_layout_staged: cute.ComposedLayout = None,
        tiled_mma_QK: cute.TiledMma = None,
        softmax_scale_log2: Float32 = None,
        tile_sched_params: ParamsBase = None,
        TileScheduler: cutlass.Constexpr[Callable] = None,
        SharedStorage: cutlass.Constexpr[Callable] = None,
        qhead_per_kvhead_divmod: FastDivmodDivisor = None,
        mL1NormDenom: cute.Tensor = None,
        tma_atom_KV: cute.CopyAtom = None,
    ):
        # Dense path: no topk length (always full KV iteration).
        mTopkLength = None
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_KV)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        mbar_Q_ptr = storage.mbar_Q.data_ptr()
        mbar_KV0_ptr = storage.mbar_KV0.data_ptr()
        mbar_KV1_ptr = storage.mbar_KV1.data_ptr()
        mbar_KVEmpty0_ptr = storage.mbar_KVEmpty0.data_ptr()
        mbar_KVEmpty1_ptr = storage.mbar_KVEmpty1.data_ptr()
        if warp_idx == 0:
            cute.arch.mbarrier_init(mbar_Q_ptr, 1)
            # Per-stage KV mbarriers; arrive_count=1 because the producer
            # is a single elected thread (warp 0 for stage 0, warp 1 for
            # stage 1 in warp-split mode; warp 0 for both in unified mode).
            cute.arch.mbarrier_init(mbar_KV0_ptr, 1)
            cute.arch.mbarrier_init(mbar_KV1_ptr, 1)
            # KVEmpty arrive_count=128: the consumer WG (128 threads) bumps
            # this via a single elected-thread mbarrier_arrive(count=128)
            # per iter; producer wait = 1 wait/stage/iter.
            cute.arch.mbarrier_init(mbar_KVEmpty0_ptr, 128)
            cute.arch.mbarrier_init(mbar_KVEmpty1_ptr, 128)
        cute.arch.sync_threads()

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sKV_staged = storage.sKV.get_tensor(sKV_layout_staged.outer, swizzle=sKV_layout_staged.inner)
        sKV_0 = sKV_staged[None, None, 0]
        sKV_1 = sKV_staged[None, None, 1]

        sWeights = storage.sWeights.get_tensor(cute.make_layout((self.tile_m,), stride=(1,)))
        if tidx == 0:
            sWeights.fill(0.0)
        if const_expr(self.is_dense_indexer):
            sReduce_size = 16
        else:
            sReduce_size = 8
        sReduce = storage.sReduce.get_tensor(cute.make_layout((sReduce_size,), stride=(1,)))

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

        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)

        if warp_group_idx == 0:
            self.consumer(
                0,
                tiled_mma_QK,
                mOut,
                mTopkLength,
                sQ,
                sKV_staged,
                sWeights,
                sReduce,
                mbar_Q_ptr,
                mbar_KV0_ptr,
                mbar_KVEmpty0_ptr,
                tidx,
                SeqlenInfoCls,
                TileSchedulerCls,
                softmax_scale_log2,
                mL1NormDenom,
            )
        elif warp_group_idx == 1:
            self.consumer(
                1,
                tiled_mma_QK,
                mOut,
                mTopkLength,
                sQ,
                sKV_staged,
                sWeights,
                sReduce,
                mbar_Q_ptr,
                mbar_KV1_ptr,
                mbar_KVEmpty1_ptr,
                tidx,
                SeqlenInfoCls,
                TileSchedulerCls,
                softmax_scale_log2,
                mL1NormDenom,
            )
        else:
            producer_fn = self.producer_warp_split if const_expr(self.use_warp_split_producer) else self.producer
            producer_fn(
                mQ,
                tma_atom_Q,
                mKV,
                tma_atom_KV,
                mWeights,
                mTopkLength,
                sQ,
                sKV_0,
                sKV_1,
                sWeights,
                mbar_Q_ptr,
                mbar_KV0_ptr,
                mbar_KV1_ptr,
                mbar_KVEmpty0_ptr,
                mbar_KVEmpty1_ptr,
                tidx,
                SeqlenInfoCls,
                TileSchedulerCls,
            )

    # --------------------------------------------------------------------- #
    #  Producer WG2: TMA Q (once) + TMA KV (per n_block, alternating stages) #
    #                                                                        #
    #  Unified single-warp producer: warp 0 drives everything (Q TMA,        #
    #  weights load, KV TMA for both stages).  This is the baseline that    #
    #  warp_split producer aims to improve on.                               #
    # --------------------------------------------------------------------- #

    @cute.jit
    def producer(
        self,
        mQ: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        mKV: cute.Tensor,
        tma_atom_KV: cute.CopyAtom,
        mWeights: cute.Tensor,
        mTopkLength: cute.Tensor,
        sQ: cute.Tensor,
        sKV_0: cute.Tensor,
        sKV_1: cute.Tensor,
        sWeights: cute.Tensor,
        mbar_Q_ptr,
        mbar_KV0_ptr,
        mbar_KV1_ptr,
        mbar_KVEmpty0_ptr,
        mbar_KVEmpty1_ptr,
        tidx: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        wg_tidx = tidx % self.num_threads_per_warp_group
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mbar_Q_phase = Int32(0)
        kve0_phase = Int32(0)
        kve1_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            _ = SeqlenInfoCls(batch_idx)
            head_idx_kv = head_idx

            if const_expr(self.has_topk_length):
                topK = mTopkLength[batch_idx, m_block]
            else:
                topK = self.topk_max
            n_block_max = (topK + self.tile_n - 1) // self.tile_n

            mKV_cur = mKV[None, None, head_idx_kv, batch_idx]
            mQ_cur = mQ[None, None, head_idx, batch_idx]

            mWeights_cur = mWeights[None, head_idx, batch_idx]
            seqlen_q_packed = mWeights_cur.shape[0][1]
            _pack_gqa = PackGQA(0, 0, False, self.qhead_per_kvhead)

            for h_tile in cutlass.range_constexpr(self.num_head_tiles):
                eff_m_block = m_block * self.num_head_tiles + h_tile

                # ---- Q TMA + Weights/LSE LDG ----
                gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (eff_m_block, 0))
                load_Q, _, _ = copy_ops.tma_get_copy_fn(tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True)
                if warp_idx_in_wg == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar_Q_ptr, self.tma_copy_bytes["Q"])
                    load_Q(tma_bar_ptr=mbar_Q_ptr)

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
                # QueryWeightsReady3WG: 384-thread barrier, producer →
                # both consumer WGs that Q + Weights are in SMEM.
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.QueryWeightsReady3WG),
                    number_of_threads=self.num_threads,
                )
                cute.arch.mbarrier_wait(mbar_Q_ptr, mbar_Q_phase)
                mbar_Q_phase = mbar_Q_phase ^ 1

                # ---- KV TMA pipeline (2-stage, alternating) ----
                # Iter -> stage mapping:
                #   iter 0 (n=n_block_max-1) → stage 0  (WG0 consumer)
                #   iter 1 (n=n_block_max-2) → stage 1  (WG1 consumer)
                #   iter 2 (n=n_block_max-3) → stage 0  (WG0 wraps)
                #   iter 3 (n=n_block_max-4) → stage 1  (WG1 wraps)
                # Prologue issues two TMAs upfront, so both consumers' first
                # mbarrier_wait(mbar_KV) completes immediately — after that
                # peer handoff via PingMma becomes the critical path.

                # Prologue — 2 upfront TMAs.
                if n_block_max >= 1:
                    self._issue_kv_tma(
                        mKV_cur,
                        sKV_0,
                        n_block_max - 1,
                        tma_atom_KV,
                        mbar_KV0_ptr,
                    )
                if n_block_max >= 2:
                    self._issue_kv_tma(
                        mKV_cur,
                        sKV_1,
                        n_block_max - 2,
                        tma_atom_KV,
                        mbar_KV1_ptr,
                    )

                # Steady state: refill stage 0 then stage 1, starting at
                # n = n_block_max-3 and going down by 1 each refill
                # (alternating stages).
                n_block = n_block_max - 3
                while n_block >= 0:
                    # Stage 0 (WG0).
                    cute.arch.mbarrier_wait(mbar_KVEmpty0_ptr, kve0_phase)
                    kve0_phase = kve0_phase ^ 1
                    self._issue_kv_tma(
                        mKV_cur,
                        sKV_0,
                        n_block,
                        tma_atom_KV,
                        mbar_KV0_ptr,
                    )
                    n_block = n_block - 1
                    if n_block >= 0:
                        # Stage 1 (WG1).
                        cute.arch.mbarrier_wait(mbar_KVEmpty1_ptr, kve1_phase)
                        kve1_phase = kve1_phase ^ 1
                        self._issue_kv_tma(
                            mKV_cur,
                            sKV_1,
                            n_block,
                            tma_atom_KV,
                            mbar_KV1_ptr,
                        )
                        n_block = n_block - 1

                # Drain: match consumer's final KVEmpty arrives so the
                # producer's phase stays balanced across tiles (otherwise
                # next tile's first mbarrier_wait would see a stale
                # phase bit).
                if n_block_max >= 1:
                    cute.arch.mbarrier_wait(mbar_KVEmpty0_ptr, kve0_phase)
                    kve0_phase = kve0_phase ^ 1
                if n_block_max >= 2:
                    cute.arch.mbarrier_wait(mbar_KVEmpty1_ptr, kve1_phase)
                    kve1_phase = kve1_phase ^ 1

            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.TileComplete3WG),
                number_of_threads=self.num_threads,
            )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    # --------------------------------------------------------------------- #
    #  Producer: split the KV TMA pipeline across two warps of the producer #
    #  WG so stage-0 and stage-1 TMAs issue independently.                   #
    #                                                                        #
    #  Layout:                                                               #
    #    warp 0 — Q TMA (once per h_tile) + stage-0 KV pipeline              #
    #    warp 1 — stage-1 KV pipeline                                        #
    #    warp 2, 3 — idle during KV pipeline; all 4 warps participate in    #
    #                Q+Weights load / QueryWeightsReady / TileComplete      #
    #                barriers that span the whole producer WG.               #
    #                                                                        #
    #  Why this helps (skill §2):                                             #
    #    the unified `producer` has warp 0 doing:                            #
    #      wait KVE0 → TMA stage0 → wait KVE1 → TMA stage1 → ...              #
    #    so stage-1's TMA issue is serialized behind stage-0's wait+issue,   #
    #    even when both consumer WGs have released their KVEmpty.  Perfsim   #
    #    shows WG1's wait KV is ~20-50c longer than WG0's under unified.     #
    #                                                                        #
    #    With warp split, stage-0 and stage-1 pipelines are independent —    #
    #    warp 1 waits KVE1 and issues TMA1 in parallel with warp 0's         #
    #    stage-0 work.  Both TMAs can be in-flight to the HBM subsystem      #
    #    at the same moment, balancing consumer wait-KV latencies.           #
    # --------------------------------------------------------------------- #

    @cute.jit
    def producer_warp_split(
        self,
        mQ: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        mKV: cute.Tensor,
        tma_atom_KV: cute.CopyAtom,
        mWeights: cute.Tensor,
        mTopkLength: cute.Tensor,
        sQ: cute.Tensor,
        sKV_0: cute.Tensor,
        sKV_1: cute.Tensor,
        sWeights: cute.Tensor,
        mbar_Q_ptr,
        mbar_KV0_ptr,
        mbar_KV1_ptr,
        mbar_KVEmpty0_ptr,
        mbar_KVEmpty1_ptr,
        tidx: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        wg_tidx = tidx % self.num_threads_per_warp_group
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mbar_Q_phase = Int32(0)
        kve0_phase = Int32(0)
        kve1_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            _ = SeqlenInfoCls(batch_idx)
            head_idx_kv = head_idx

            if const_expr(self.has_topk_length):
                topK = mTopkLength[batch_idx, m_block]
            else:
                topK = self.topk_max
            n_block_max = (topK + self.tile_n - 1) // self.tile_n

            mKV_cur = mKV[None, None, head_idx_kv, batch_idx]
            mQ_cur = mQ[None, None, head_idx, batch_idx]

            mWeights_cur = mWeights[None, head_idx, batch_idx]
            seqlen_q_packed = mWeights_cur.shape[0][1]
            _pack_gqa = PackGQA(0, 0, False, self.qhead_per_kvhead)

            for h_tile in cutlass.range_constexpr(self.num_head_tiles):
                eff_m_block = m_block * self.num_head_tiles + h_tile

                # ---- Q TMA (warp-0) + Weights/LSE LDG (all 128 threads) ----
                gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (eff_m_block, 0))
                load_Q, _, _ = copy_ops.tma_get_copy_fn(tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True)
                if warp_idx_in_wg == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar_Q_ptr, self.tma_copy_bytes["Q"])
                    load_Q(tma_bar_ptr=mbar_Q_ptr)

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
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.QueryWeightsReady3WG),
                    number_of_threads=self.num_threads,
                )
                cute.arch.mbarrier_wait(mbar_Q_ptr, mbar_Q_phase)
                mbar_Q_phase = mbar_Q_phase ^ 1

                # ---- KV TMA pipelines: warp 0 drives stage 0, warp 1 drives stage 1 ----
                # Each warp runs its prologue + steady state + drain
                # entirely independently.  The two pipelines never block
                # on each other because they use disjoint mbarriers
                # (mbar_KVEmpty{0,1} and mbar_KV{0,1}).
                if warp_idx_in_wg == 0:
                    # ---- Stage 0 pipeline (serves WG0 consumer) ----
                    # Prologue
                    if n_block_max >= 1:
                        self._issue_kv_tma(
                            mKV_cur,
                            sKV_0,
                            n_block_max - 1,
                            tma_atom_KV,
                            mbar_KV0_ptr,
                        )
                    # Steady state: refill stage 0 for n = n_block_max-3, -5, ...
                    n_0 = n_block_max - 3
                    while n_0 >= 0:
                        cute.arch.mbarrier_wait(mbar_KVEmpty0_ptr, kve0_phase)
                        kve0_phase = kve0_phase ^ 1
                        self._issue_kv_tma(
                            mKV_cur,
                            sKV_0,
                            n_0,
                            tma_atom_KV,
                            mbar_KV0_ptr,
                        )
                        n_0 = n_0 - 2
                    # Drain: match consumer's final KVEmpty arrive.
                    if n_block_max >= 1:
                        cute.arch.mbarrier_wait(mbar_KVEmpty0_ptr, kve0_phase)
                        kve0_phase = kve0_phase ^ 1
                elif warp_idx_in_wg == 1:
                    # ---- Stage 1 pipeline (serves WG1 consumer) ----
                    # warp 1 can't use _issue_kv_tma (it gates on
                    # warp_idx_in_wg == 0), so go through the warp-1
                    # variant below.
                    # Prologue
                    if n_block_max >= 2:
                        self._issue_kv_tma_warp1(
                            mKV_cur,
                            sKV_1,
                            n_block_max - 2,
                            tma_atom_KV,
                            mbar_KV1_ptr,
                        )
                    n_1 = n_block_max - 4
                    while n_1 >= 0:
                        cute.arch.mbarrier_wait(mbar_KVEmpty1_ptr, kve1_phase)
                        kve1_phase = kve1_phase ^ 1
                        self._issue_kv_tma_warp1(
                            mKV_cur,
                            sKV_1,
                            n_1,
                            tma_atom_KV,
                            mbar_KV1_ptr,
                        )
                        n_1 = n_1 - 2
                    if n_block_max >= 2:
                        cute.arch.mbarrier_wait(mbar_KVEmpty1_ptr, kve1_phase)
                        kve1_phase = kve1_phase ^ 1
                # warp 2, 3 fall through to the per-h_tile sync below.

                # End of h_tile: keep warp 0 (stage-0 pipeline) and warp 1
                # (stage-1 pipeline) in lock-step before moving to the next
                # h_tile.  Without this, warp 0 can race ahead and start
                # issuing h_tile=1's stage-0 prologue TMA while warp 1 is
                # still finishing h_tile=0's stage-1 drain, which (a) can
                # land h_tile=1's KV in sKV_0 before consumer WG0 has
                # consumed h_tile=0's KV (silent data corruption — only
                # visible as numeric mismatch on nh_q>=128 /
                # num_head_tiles>=2), and (b) leaves the kve0_phase /
                # kve1_phase counters out of sync between the two warps.
                #
                # The 128-thread sync gates only the producer WG and uses
                # a private named-barrier id (ProducerHTileSync_3WG = 15)
                # so it does not interfere with the consumer pingpong
                # barriers PingMmaWG0/1_3WG.  Skipped when num_head_tiles
                # == 1 (compile-time) since the outer loop runs once.
                if const_expr(self.num_head_tiles > 1):
                    cute.arch.barrier(
                        barrier_id=int(NamedBarrierBwd.ProducerHTileSync_3WG),
                        number_of_threads=self.num_threads_per_warp_group,
                    )

            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.TileComplete3WG),
                number_of_threads=self.num_threads,
            )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    # Internal helper for producer_warp_split — same body as parent's
    # _issue_kv_tma but gated on warp 1 instead of warp 0.

    @cute.jit
    def _issue_kv_tma_warp1(
        self,
        mKV_cur: cute.Tensor,
        sKV: cute.Tensor,
        n_block: Int32,
        tma_atom_KV: cute.CopyAtom,
        mbar_KV_ptr,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        gKV = cute.local_tile(mKV_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
        load_fn, _, _ = copy_ops.tma_get_copy_fn(
            tma_atom_KV,
            0,
            cute.make_layout(1),
            gKV,
            sKV,
            single_stage=True,
        )
        if warp_idx_in_wg == 1:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_KV_ptr, self.tma_copy_bytes["KV"])
            load_fn(tma_bar_ptr=mbar_KV_ptr)

    # --------------------------------------------------------------------- #
    #  Consumer WG0/WG1: WGMMA + postprocess + GMEM write                    #
    #                                                                        #
    #  Per-iter structure (FA3 / HSTU warp_scheduler_barrier pattern):       #
    #      wait mbar_KV                    # producer handoff                #
    #      bar.sync(256, self_id)          # wait for peer's pingpong token  #
    #      gemm (wg_wait=-1)               # issue MMA, don't wait           #
    #      bar.arrive(256, peer_id)        # hand peer its token             #
    #      wait_group(0)                   # let our MMA complete             #
    #      arrive mbar_KVEmpty             # producer may refill our stage   #
    #      postprocess(acc_S)              # overlaps peer's MMA on TC       #
    #                                                                        #
    #  Pingpong bookkeeping per h_tile (P = n_block iters for this WG):      #
    #      self bar.sync(256, self_id)     → +128 each call                  #
    #      peer bar.arrive(256, self_id)   → +128 each call                  #
    #      bootstrap (WG1 → WG0)           → +128 one-off at h_tile start    #
    #    WG0_id arrivals (P = WG0's iter count):                             #
    #      128 (boot) + P*128 (WG0 sync) + (P-1)*128 (WG1 arrive, guarded    #
    #                                          by "n_block > 0")             #
    #      = 2P*128 = P*256 → P releases ✓                                   #
    #    WG1_id arrivals (P' = WG1's iter count ∈ {P, P-1}):                 #
    #      P'*128 (WG1 sync) + (P-1)*128 (WG0 arrive, guarded)               #
    #      = matches P' iff the guard is correct (verified for all parities).#
    # --------------------------------------------------------------------- #

    @cute.jit
    def consumer(
        self,
        consumer_id: cutlass.Constexpr[int],
        tiled_mma_QK: cute.TiledMma,
        mOut: cute.Tensor,
        mTopkLength: cute.Tensor,
        sQ: cute.Tensor,
        sKV_staged: cute.Tensor,
        sWeights: cute.Tensor,
        sReduce: cute.Tensor,
        mbar_Q_ptr,
        mbar_KV_ptr,  # this consumer's stage mbarrier (stage 0 for WG0, stage 1 for WG1)
        mbar_KVEmpty_ptr,  # this consumer's KVEmpty mbarrier (consumer -> producer)
        tidx: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        softmax_scale_log2: Float32,
        mL1NormDenom: cute.Tensor,
    ):
        wg_tidx = tidx % self.num_threads_per_warp_group
        lane = cute.arch.lane_idx()
        LOG2_E = math.log2(math.e)

        wg_mma_QK = tiled_mma_QK.get_slice(wg_tidx)
        thr_mma_QK = tiled_mma_QK.get_slice(wg_tidx)
        tSrQ, tSrK = mma_partition_fragment_AB(wg_mma_QK, sQ, sKV_staged, self.swap_AB)
        # swap_AB flips M/N for the tiled_mma partition, matching what
        # gemm_zero_init expects.
        if const_expr(self.swap_AB):
            _acc_shape = tiled_mma_QK.partition_shape_C((self.tile_n, self.tile_m))
        else:
            _acc_shape = tiled_mma_QK.partition_shape_C((self.tile_m, self.tile_n))
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mbar_Q_phase = Int32(0)
        # Single KV stage per consumer — phase toggled each iter.
        kv_phase = Int32(0)

        # Pingpong via NamedBarriers (FA3 / HSTU warp_scheduler_barrier style).
        if const_expr(consumer_id == 0):
            self_ping_id = int(NamedBarrierBwd.PingMmaWG0_3WG)
            peer_ping_id = int(NamedBarrierBwd.PingMmaWG1_3WG)
        else:
            self_ping_id = int(NamedBarrierBwd.PingMmaWG1_3WG)
            peer_ping_id = int(NamedBarrierBwd.PingMmaWG0_3WG)
        # Which KV stage this consumer owns, as Constexpr int so WGMMA's
        # B_idx resolves statically.
        my_stage = consumer_id  # 0 (WG0) or 1 (WG1)

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

        # Per-WG slot offset into sReduce for cross-WG state combine.
        #   dense_indexer: WG0 owns [0..7], WG1 owns [8..15]
        #   dense_attn:    WG0 owns [0..3], WG1 owns [4..7]
        if const_expr(self.is_dense_indexer):
            my_sreduce_off = 0 if const_expr(consumer_id == 0) else 8
        else:
            my_sreduce_off = 0 if const_expr(consumer_id == 0) else 4

        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            _ = SeqlenInfoCls(batch_idx)

            if const_expr(self.has_topk_length):
                topK = mTopkLength[batch_idx, m_block]
            else:
                topK = self.topk_max
            n_block_max = (topK + self.tile_n - 1) // self.tile_n

            mOut_cur = mOut[batch_idx, m_block, None]

            acc_out_regs = cute.make_rmem_tensor(self.num_acc_n_rows_per_thread, Float32)

            if const_expr(self.is_dense_attn):
                l1norm_accum = Float32(0.0)
                l1norm_warp_accum = Float32(0.0)
            if const_expr(self.is_dense_indexer):
                lse_accum_max = Float32(-1e30)
                lse_accum_sum_exp = Float32(0.0)
                lse_state = cute.make_rmem_tensor(2, Float32)

            for h_tile in cutlass.range_constexpr(self.num_head_tiles):
                # Wait for Q / Weights from producer.
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.QueryWeightsReady3WG),
                    number_of_threads=self.num_threads,
                )
                cute.arch.mbarrier_wait(mbar_Q_ptr, mbar_Q_phase)
                mbar_Q_phase = mbar_Q_phase ^ 1

                tWeightsrWeights = copy_ops.load_s2r(tWsWeights)

                # WG0 handles iters n_block_max-1, n_block_max-3, ...
                # WG1 handles iters n_block_max-2, n_block_max-4, ...
                if const_expr(consumer_id == 0):
                    n_block = n_block_max - 1
                else:
                    n_block = n_block_max - 2
                accumulate = Boolean(h_tile > 0)
                is_last_h = Boolean(h_tile == self.num_head_tiles - 1)

                # Bootstrap (once per h_tile): WG1 pre-arrives 128 on WG0's
                # PingMmaWG0 barrier so WG0's very first bar.sync clears
                # immediately.  Paired with "no arrive when n_block == 0",
                # this keeps the two NamedBarriers fully balanced at
                # h_tile boundaries regardless of n_block_max parity.
                if const_expr(consumer_id == 1):
                    if n_block_max > 0:
                        cute.arch.barrier_arrive(
                            barrier_id=int(NamedBarrierBwd.PingMmaWG0_3WG),
                            number_of_threads=256,
                        )

                # Pingpong main loop.
                while n_block >= 0:
                    # Wait producer has refilled our stage.
                    cute.arch.mbarrier_wait(mbar_KV_ptr, kv_phase)
                    kv_phase = kv_phase ^ 1
                    # Wait peer's arrive token so our WGMMA is strictly
                    # interleaved with peer's on the TC issue port.
                    cute.arch.barrier(
                        barrier_id=self_ping_id,
                        number_of_threads=256,
                    )

                    n_block_base_row = n_block * self.tile_n + warp_idx_in_wg * 16
                    acc_S = cute.make_rmem_tensor(_acc_shape, Float32)
                    # gemm_w_idx: single WGMMA with zero-init, wg_wait=-1
                    # so MMA is asynchronously in flight when we return.
                    # B_idx selects our stage in the 2-stage sKV_staged
                    # at compile time (Constexpr int).
                    gemm_w_idx(
                        tiled_mma_QK,
                        acc_S,
                        tSrQ,
                        tSrK,
                        zero_init=Boolean(True),
                        B_idx=my_stage,
                        wg_wait=-1,
                        swap_AB=self.swap_AB,
                    )
                    # Wake peer: contribute 128 to peer's scheduler barrier
                    # so peer can complete its next bar.sync and issue its
                    # WGMMA back-to-back with ours.  Guard: only arrive if
                    # peer still has a next MMA (peer's next n_block =
                    # n_block - 1 ≥ 0, i.e. current n_block > 0).
                    if n_block > 0:
                        cute.arch.barrier_arrive(
                            barrier_id=peer_ping_id,
                            number_of_threads=256,
                        )

                    warpgroup.wait_group(0)

                    # Release sKV stage we just finished reading; producer
                    # may refill.  Async mbarrier_arrive (count=128) from a
                    # single elected thread in warp-0.
                    if warp_idx_in_wg == 0:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(mbar_KVEmpty_ptr, arrive_count=128)

                    warp_col_sum = self._postprocess_and_reduce(
                        acc_S,
                        tWeightsrWeights,
                        softmax_scale_log2,
                        LOG2_E,
                        n_block_base_row,
                        lane,
                        topK,
                        cute.size(mOut.shape[1]),
                        sReduce,
                        m_block,
                        batch_idx,
                        n_block,
                        tidx,
                        accumulate=accumulate,
                        mOut_cur=mOut_cur,
                        acc_out_regs=acc_out_regs,
                        lse_state=lse_state if const_expr(self.is_dense_indexer) else None,
                        compute_lse=is_last_h if const_expr(self.is_dense_indexer) else Boolean(False),
                    )

                    if const_expr(self.is_dense_attn):
                        l1norm_warp_accum += warp_col_sum

                    if const_expr(self.is_dense_indexer):
                        # Write 4 warps' per-warp (max, sum_exp) into our
                        # slot, then 128-thread sync, then read back for
                        # cross-warp reduce → accumulate into lse_accum.
                        if is_last_h:
                            if lane == 0:
                                sReduce[my_sreduce_off + warp_idx_in_wg] = lse_state[0]
                                sReduce[my_sreduce_off + 4 + warp_idx_in_wg] = lse_state[1]
                        cute.arch.fence_view_async_shared()
                        if const_expr(consumer_id == 0):
                            cute.arch.barrier(
                                barrier_id=int(NamedBarrierBwd.DenseConsumer0Sync_3WG),
                                number_of_threads=self.num_threads_per_warp_group,
                            )
                        else:
                            cute.arch.barrier(
                                barrier_id=int(NamedBarrierBwd.DenseConsumer1Sync_3WG),
                                number_of_threads=self.num_threads_per_warp_group,
                            )
                        if is_last_h:
                            global_max = cute.arch.fmax(
                                cute.arch.fmax(
                                    sReduce[my_sreduce_off + 0],
                                    sReduce[my_sreduce_off + 1],
                                ),
                                cute.arch.fmax(
                                    sReduce[my_sreduce_off + 2],
                                    sReduce[my_sreduce_off + 3],
                                ),
                            )
                            global_sum_exp = Float32(0.0)
                            for wi in cutlass.range_constexpr(4):
                                global_sum_exp += sReduce[my_sreduce_off + 4 + wi] * cute.math.exp2(
                                    (sReduce[my_sreduce_off + wi] - global_max) * LOG2_E,
                                    fastmath=True,
                                )
                            new_max = cute.arch.fmax(lse_accum_max, global_max)
                            lse_accum_sum_exp = lse_accum_sum_exp * cute.math.exp2(
                                (lse_accum_max - new_max) * LOG2_E, fastmath=True
                            ) + global_sum_exp * cute.math.exp2((global_max - new_max) * LOG2_E, fastmath=True)
                            lse_accum_max = new_max

                    n_block = n_block - 2

                # End of n_block loop for this h_tile.
                if const_expr(self.is_dense_attn):
                    # Sum 4 warps' partial col sum into a single l1norm
                    # increment for this h_tile.  Use our own slot range
                    # so we don't collide with peer.
                    if lane == 0:
                        sReduce[my_sreduce_off + warp_idx_in_wg] = l1norm_warp_accum
                    cute.arch.fence_view_async_shared()
                    if const_expr(consumer_id == 0):
                        cute.arch.barrier(
                            barrier_id=int(NamedBarrierBwd.DenseConsumer0Sync_3WG),
                            number_of_threads=self.num_threads_per_warp_group,
                        )
                    else:
                        cute.arch.barrier(
                            barrier_id=int(NamedBarrierBwd.DenseConsumer1Sync_3WG),
                            number_of_threads=self.num_threads_per_warp_group,
                        )
                    block_col_sum = sReduce[my_sreduce_off + 0] + sReduce[my_sreduce_off + 1] + sReduce[my_sreduce_off + 2] + sReduce[my_sreduce_off + 3]
                    l1norm_accum += block_col_sum
                    l1norm_warp_accum = Float32(0.0)
            # End of h_tile loop.

            # Cross-WG combine of l1norm / lse:
            #   1. Both consumers write their per-WG partial state to
            #      disjoint slots in sReduce.
            #   2. ConsumersDone3WG barrier (256-thread).
            #   3. WG0 reads both and writes the final denom to GMEM.
            if const_expr(self.is_dense_attn):
                # WG0 writes l1norm_accum to sReduce[0]; WG1 writes to [4].
                if wg_tidx == 0:
                    if const_expr(consumer_id == 0):
                        sReduce[0] = l1norm_accum
                    else:
                        sReduce[4] = l1norm_accum
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.ConsumersDone3WG),
                    number_of_threads=256,
                )
                if const_expr(consumer_id == 0):
                    final_l1norm = sReduce[0] + sReduce[4]
                    self.write_l1norm_denom(final_l1norm, mL1NormDenom, batch_idx, m_block, tidx)
            elif const_expr(self.is_dense_indexer):
                # WG0 writes (max, sum_exp) to sReduce[0..1]; WG1 → [8..9].
                if wg_tidx == 0:
                    if const_expr(consumer_id == 0):
                        sReduce[0] = lse_accum_max
                        sReduce[1] = lse_accum_sum_exp
                    else:
                        sReduce[8] = lse_accum_max
                        sReduce[9] = lse_accum_sum_exp
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.ConsumersDone3WG),
                    number_of_threads=256,
                )
                if const_expr(consumer_id == 0):
                    wg0_max = sReduce[0]
                    wg0_sum = sReduce[1]
                    wg1_max = sReduce[8]
                    wg1_sum = sReduce[9]
                    new_max = cute.arch.fmax(wg0_max, wg1_max)
                    final_sum = wg0_sum * cute.math.exp2((wg0_max - new_max) * LOG2_E, fastmath=True) + wg1_sum * cute.math.exp2(
                        (wg1_max - new_max) * LOG2_E, fastmath=True
                    )
                    self.write_lse_denom(new_max, final_sum, mL1NormDenom, batch_idx, m_block, tidx)

            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.TileComplete3WG),
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
        seqlen_q: Int32,
        sTopk_reduced: cute.Tensor,
        m_block: Int32,
        batch_idx: Int32,
        n_block: Int32,
        tidx: Int32,
        accumulate: Boolean = Boolean(False),  # True when accumulating across head tiles
        mOut_cur: cute.Tensor = None,  # dense: direct write target
        acc_out_regs: cute.Tensor = None,  # dense: caller-owned rmem tensor [num_acc_n_rows_per_thread]
        lse_state: cute.Tensor = None,  # dense indexer: rmem tensor [2] = (tile_max, tile_sum_exp)
        compute_lse: Boolean = Boolean(False),  # dense indexer: True only on last head tile
    ) -> Float32:
        """Post-process GEMM result (relu*weights or exp2) and head-reduce.

        Sparse mode: writes head-reduced results to sTopk_reduced (SMEM).
        Dense attention mode: accumulates head-reduced results into caller-supplied
            acc_out_regs (register tensor) and writes them to mOut_cur (GMEM).
            Also returns warp_col_sum for L1 norm denominator accumulation.
        Dense indexer mode: same as dense attention for per-row accumulation.
            When compute_lse=True (last head tile), computes online LSE using
            the fully accumulated acc_out_regs values and writes (tile_max,
            tile_sum_exp) to lse_state for caller to merge across warps.

        When accumulate=True, pre-loads existing GMEM values into acc_out_regs first.
        Returns warp_col_sum (Float32): meaningful only in dense attention mode.
        """
        # Native view: rows = N-dim (topk positions), cols = M-dim (Q/head positions)
        # tWeightsrWeights[c] gives the weight for M-position c (same indexing as transposed row)
        acc_S_mn = sm90_ops.make_acc_tensor_mn_view(acc_S)
        warp_col_sum = Float32(0.0)
        ratio = Int32(self.ratio)
        q_global_start = topK * ratio - Int32(seqlen_q)
        col_limit_raw = (q_global_start + m_block + Int32(1)) // ratio
        col_limit = col_limit_raw if col_limit_raw < topK else topK

        if const_expr(self.is_dense_attn):
            # === Dense attention path ===
            # acc_out_regs: caller-owned register tensor of size num_acc_n_rows_per_thread
            # (= tile_n / 32, e.g. 2 for tile_n=64). One register per N-row this thread owns.
            # This avoids GMEM read-modify-write inside the for-r loop.

            # Pre-load from GMEM when accumulating across head tiles
            if accumulate:
                for r in cutlass.range_constexpr(self.num_acc_n_rows_per_thread):
                    pos = n_block_base_row + r * 8 + (lane >> 2)
                    if (lane & 3) == 0 and pos < topK and pos < col_limit:
                        acc_out_regs[r] = mOut_cur[pos]
                    else:
                        acc_out_regs[r] = Float32(0.0)
            else:
                for r in cutlass.range_constexpr(self.num_acc_n_rows_per_thread):
                    acc_out_regs[r] = Float32(0.0)

            for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
                row_sum_cur_thread = Float32(0.0)
                for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                    row_sum_cur_thread += cute.math.exp2(acc_S_mn[r, c] * softmax_scale_log2 - tWeightsrWeights[c] * LOG2_E, fastmath=True)
                # Intra-warp row reduce: sum across 4 threads sharing the same row
                row_sum_cur_thread = sm90_ops.warp_reduce(row_sum_cur_thread, operator.add, width=4)

                # Accumulate into caller-owned register (no GMEM access inside loop)
                pos = n_block_base_row + r * 8 + (lane >> 2)
                if (lane & 3) == 0 and pos < topK and pos < col_limit:
                    acc_out_regs[r] = acc_out_regs[r] + row_sum_cur_thread

                # Column reduction: sum across 8 leader threads (T0,T4,T8,...,T28)
                # Only leader threads (lane & 3 == 0) hold valid row sums;
                # use 3-step butterfly with offsets {4, 8, 16} to reduce across
                # stride-4 positions, instead of full 32-thread warp_reduce.
                col_partial = Float32(0.0)
                if (lane & 3) == 0 and pos < topK and pos < col_limit:
                    col_partial = row_sum_cur_thread
                col_partial = col_partial + cute.arch.shuffle_sync_bfly(col_partial, offset=4)
                col_partial = col_partial + cute.arch.shuffle_sync_bfly(col_partial, offset=8)
                col_partial = col_partial + cute.arch.shuffle_sync_bfly(col_partial, offset=16)
                # Only lane 0 accumulates across r iterations
                if lane == 0:
                    warp_col_sum += col_partial

            # Write registers to GMEM once (single store per row, no read-modify-write)
            for r in cutlass.range_constexpr(self.num_acc_n_rows_per_thread):
                pos = n_block_base_row + r * 8 + (lane >> 2)
                if (lane & 3) == 0 and pos < topK:
                    if pos < col_limit:
                        mOut_cur[pos] = acc_out_regs[r]
                    else:
                        mOut_cur[pos] = Float32(0.0)

            # Broadcast accumulated warp_col_sum from lane 0 to all lanes
            warp_col_sum = sm90_ops.shuffle_sync(warp_col_sum, 0)
        elif const_expr(self.is_dense_indexer):
            # === Dense indexer path ===
            # Same register-based per-row accumulation as dense attention,
            # but uses relu*weight for head reduction.
            # LSE is computed ONLY on the last head tile (compute_lse=True)
            # using the fully accumulated acc_out_regs values.

            # Pre-init pos to Int32 so dynamic `if accumulate` doesn't change its type from None.
            pos = n_block_base_row + (lane >> 2)

            # Pre-load from GMEM when accumulating across head tiles
            if accumulate:
                for r in cutlass.range_constexpr(self.num_acc_n_rows_per_thread):
                    pos = n_block_base_row + r * 8 + (lane >> 2)
                    if (lane & 3) == 0 and pos < topK and pos < col_limit:
                        acc_out_regs[r] = mOut_cur[pos]
                    else:
                        acc_out_regs[r] = Float32(0.0)
            else:
                for r in cutlass.range_constexpr(self.num_acc_n_rows_per_thread):
                    acc_out_regs[r] = Float32(0.0)

            # Head reduction: relu(score) * weight, then accumulate across head tiles.
            # sm_scale is applied on the fp32 head-reduced row sum (post
            # warp_reduce, pre n-block/head-tile accumulation). Applying it
            # here — rather than pre-multiplying onto bf16 W on the host —
            # preserves precision. Distributivity is preserved across the
            # cross-warp reduce and the n-block / head-tile accumulation.
            for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
                row_sum_cur_thread = Float32(0.0)
                for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                    row_sum_cur_thread += cute.arch.fmax(acc_S_mn[r, c], Float32(0.0)) * tWeightsrWeights[c]
                # Intra-warp row reduce: sum across 4 threads sharing the same row
                row_sum_cur_thread = sm90_ops.warp_reduce(row_sum_cur_thread, operator.add, width=4)
                if const_expr(self.softmax_scale != 1.0):
                    row_sum_cur_thread = row_sum_cur_thread * Float32(self.softmax_scale)

                # Accumulate into caller-owned register
                pos = n_block_base_row + r * 8 + (lane >> 2)
                if (lane & 3) == 0 and pos < topK and pos < col_limit:
                    acc_out_regs[r] = acc_out_regs[r] + row_sum_cur_thread

            # Write accumulated scores to GMEM
            for r in cutlass.range_constexpr(self.num_acc_n_rows_per_thread):
                pos = n_block_base_row + r * 8 + (lane >> 2)
                if (lane & 3) == 0 and pos < topK:
                    if pos < col_limit:
                        mOut_cur[pos] = acc_out_regs[r]
                    else:
                        mOut_cur[pos] = Float32(0.0)

            # LSE computation: only on last head tile when acc_out_regs has final scores.
            # On earlier head tiles acc_out_regs only has partial head sums, so
            # log(sum exp(partial)) != log(sum exp(final)).
            #
            # Batched approach: collect local max across all r values first,
            # then one butterfly reduce for max, then one pass for sum_exp.
            # Saves ~half the shuffles vs per-r online merge.
            if compute_lse:
                # Step 1: Each leader thread finds local max across its r values
                local_max = Float32(-1e30)
                for r in cutlass.range_constexpr(self.num_acc_n_rows_per_thread):
                    pos = n_block_base_row + r * 8 + (lane >> 2)
                    if (lane & 3) == 0 and pos < topK and pos < col_limit:
                        local_max = cute.arch.fmax(local_max, acc_out_regs[r])

                # Step 2: Butterfly max reduction across 8 leader threads
                tile_max = local_max
                tile_max = cute.arch.fmax(tile_max, cute.arch.shuffle_sync_bfly(tile_max, offset=4))
                tile_max = cute.arch.fmax(tile_max, cute.arch.shuffle_sync_bfly(tile_max, offset=8))
                tile_max = cute.arch.fmax(tile_max, cute.arch.shuffle_sync_bfly(tile_max, offset=16))

                # Step 3: Each leader thread sums exp(val - tile_max) for its r values
                local_sum_exp = Float32(0.0)
                for r in cutlass.range_constexpr(self.num_acc_n_rows_per_thread):
                    pos = n_block_base_row + r * 8 + (lane >> 2)
                    if (lane & 3) == 0 and pos < topK and pos < col_limit:
                        local_sum_exp += cute.math.exp2((acc_out_regs[r] - tile_max) * LOG2_E, fastmath=True)

                # Step 4: Butterfly sum reduction across 8 leader threads
                tile_sum_exp = local_sum_exp
                tile_sum_exp = tile_sum_exp + cute.arch.shuffle_sync_bfly(tile_sum_exp, offset=4)
                tile_sum_exp = tile_sum_exp + cute.arch.shuffle_sync_bfly(tile_sum_exp, offset=8)
                tile_sum_exp = tile_sum_exp + cute.arch.shuffle_sync_bfly(tile_sum_exp, offset=16)

                # Write to lse_state for caller to do cross-warp merge.
                # After butterfly, all leader threads (incl. lane 0) have correct values.
                # Caller only reads lse_state from lane 0, so no broadcast needed.
                lse_state[0] = tile_max
                lse_state[1] = tile_sum_exp

        return warp_col_sum

    @cute.jit
    def write_l1norm_denom(
        self,
        l1norm_accum: Float32,
        mL1NormDenom: cute.Tensor,  # (batch, seqlen_q) float32
        batch_idx: Int32,
        m_block: Int32,
        tidx: Int32,
    ):
        """Write accumulated L1 norm denominator to global memory.
        Only the first thread of the warp group writes to avoid conflicts.
        """
        wg_tidx = tidx % self.num_threads_per_warp_group
        if wg_tidx == 0:
            mL1NormDenom[batch_idx, m_block] = l1norm_accum

    @cute.jit
    def write_lse_denom(
        self,
        lse_accum_max: Float32,
        lse_accum_sum_exp: Float32,
        mLseDenom: cute.Tensor,  # (batch, seqlen_q) float32
        batch_idx: Int32,
        m_block: Int32,
        tidx: Int32,
    ):
        """Write accumulated LSE denominator to global memory.
        LSE = log(sum_exp) + max.
        Only the first thread of the warp group writes to avoid conflicts.
        """
        wg_tidx = tidx % self.num_threads_per_warp_group
        if wg_tidx == 0:
            mLseDenom[batch_idx, m_block] = sm90_ops.logf(lse_accum_sum_exp) + lse_accum_max

    # =========================================================================
    # Dense TMA KV loading helpers (single-thread TMA + mbarrier)
    # =========================================================================

    @cute.jit
    def _issue_kv_tma(
        self,
        mKV_cur: cute.Tensor,  # 2D GMEM (seqlen_k, headdim) — TMA tensor view
        sKV: cute.Tensor,  # 2D SMEM target (tile_n, tile_hdim)
        n_block: Int32,
        tma_atom_KV: cute.CopyAtom,
        mbar_KV_ptr,
    ):
        """Issue one TMA bulk-copy of a KV tile from GMEM to SMEM (non-blocking).
        Only warp-0 leader thread drives the TMA; mbarrier tracks completion.
        Caller is responsible for mbarrier_wait afterwards.
        """
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        gKV = cute.local_tile(mKV_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
        load_fn, _, _ = copy_ops.tma_get_copy_fn(
            tma_atom_KV,
            0,
            cute.make_layout(1),
            gKV,
            sKV,
            single_stage=True,
        )
        if warp_idx_in_wg == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_KV_ptr, self.tma_copy_bytes["KV"])
            load_fn(tma_bar_ptr=mbar_KV_ptr)
