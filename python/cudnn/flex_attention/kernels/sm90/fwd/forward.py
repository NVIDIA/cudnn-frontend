# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM90 (Hopper) forward pass for FlexAttention.

from functools import partial
from types import SimpleNamespace
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass import pipeline
from cutlass.base_dsl.arch import Arch
from cutlass.cute.nvgpu import cpasync, warpgroup
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum

import cuda.bindings.driver as cuda

from cudnn.flex_attention._compat import copy_utils
from cudnn.flex_attention._compat import layout_utils
from cudnn.flex_attention._compat import sm90_utils
from cudnn.flex_attention._compat.cute_dsl_utils import ParamsBase

from cudnn.flex_attention.kernels.common import device_utils as utils
from cudnn.flex_attention.kernels.common import pipeline as pipeline_custom
from cudnn.flex_attention.kernels.common.pack_gqa import PackGQA, make_packgqa_tiled_tma_atom, pack_gqa_layout
from cudnn.flex_attention.kernels.common.seqlen_info import SeqlenInfoQK
from cudnn.flex_attention.kernels.common.softmax import Softmax
from cudnn.flex_attention.kernels.common.tile_scheduler import (
    PlanDynamicPersistentTileSchedulerSm90,
)
from cudnn.flex_attention.kernels.sm90.fwd.forward_base import FlexAttentionForwardBase
from cudnn.flex_attention.kernels.sm90.fwd.forward_config import make_sm90_fwd_tiled_mma
from cudnn.flex_attention.kernels.sm90.fwd.named_barrier import NamedBarrierFwd
from cudnn.flex_attention.plan.kernels import BlockSparseTensors
from cudnn.flex_attention.plan.kernels.packed_mask import (
    consume_block_sparse_loads,
    produce_block_sparse_loads,
)
from cudnn.flex_attention.runtime.dsl_utils import assume_tensor_aligned


class FlexAttentionForwardSm90(FlexAttentionForwardBase):
    def __init__(
        self,
        *args,
        intra_wg_overlap: bool = True,
        mma_pv_is_rs: bool = True,
        use_smem_mask_pipeline: bool = True,
        num_mask_payload_groups: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if type(use_smem_mask_pipeline) is not bool:
            raise TypeError("use_smem_mask_pipeline must be a bool")
        self.intra_wg_overlap = intra_wg_overlap
        self.mma_pv_is_rs = mma_pv_is_rs
        self.use_smem_mask_pipeline = use_smem_mask_pipeline
        self.num_mask_payload_groups = num_mask_payload_groups
        self.mask_stages = 2
        self.buffer_align_bytes = 1024
        self.use_tma_KV = True
        assert self.use_tma_KV
        self.cluster_shape_mn = (1, 1)
        assert self.arch >= Arch.sm_90 and self.arch <= Arch.sm_90a, "Only SM 9.x is supported"

    def _get_smem_layout_atom(self):
        sQ_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdim),
            self.dtype,
        )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdimv),
            self.dtype,
        )
        sO_layout_atom = sV_layout_atom
        if not self.mma_pv_is_rs:
            sP_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, self.dtype, self.tile_n),
                self.dtype,
            )
        else:
            sP_layout_atom = None
        return sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom

    def _get_tiled_mma(self):
        return make_sm90_fwd_tiled_mma(
            self.dtype,
            self.tile_m,
            self.tile_n,
            self.tile_hdimv,
            self.mma_pv_is_rs,
        )

    def _get_shared_storage_cls(self):
        sK_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sK_layout)],
            self.buffer_align_bytes,
        ]
        cosize_sVO = max(cute.cosize(self.sV_layout), cute.cosize(self.sO_layout))
        sVO_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sVO], self.buffer_align_bytes]
        sQ_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(self.sQ_layout)], 1024]
        cosize_sQVO = max(cute.cosize(self.sQ_layout), cosize_sVO)
        sQVO_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQVO], 1024]
        cosize_sP = cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
        # 1 stage * 2 for Q pipeline (full + empty), self.num_stages*2 for K, self.num_stages*2 for V,
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mask_mbar_size = self.mask_stages * 2 if self.use_smem_mask_pipeline else 0
        mbar_ptr_Mask_struct = cute.struct.MemRange[cutlass.Int64, mask_mbar_size]
        mbar_ptr_O_empty_struct = cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 1], 16]
        scheduler_work_struct = cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]
        sMask_size = self.num_mask_payload_groups * self.mask_payload_words * self.mask_stages if self.use_smem_mask_pipeline else 0
        sMask_struct = cute.struct.Align[cute.struct.MemRange[Uint32, sMask_size], 16]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            mbar_ptr_Mask: mbar_ptr_Mask_struct
            mbar_ptr_O_empty: mbar_ptr_O_empty_struct
            scheduler_work: scheduler_work_struct
            sMask: sMask_struct
            sV: sVO_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            mbar_ptr_Mask: mbar_ptr_Mask_struct
            mbar_ptr_O_empty: mbar_ptr_O_empty_struct
            scheduler_work: scheduler_work_struct
            sMask: sMask_struct
            sQ: sQVO_struct
            sK: sK_struct
            sP: sP_struct

        return SharedStorageQKV if const_expr(not self.Q_in_regs) else SharedStorageSharedQV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,  # (b, s_k, h_k, d) or (total_k, h_k, d) for varlen
        mV: cute.Tensor,  # (b, s_k, h_k, dv) or (total_k, h_k, dv) for varlen
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        blocksparse_tensors: BlockSparseTensors = None,
        scheduler_tile_counter: Optional[cute.Tensor] = None,
        scheduler_num_sms: Int32 = 132,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Configure and launch the FlexAttention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """

        self._check_type(*(t.element_type if t is not None else None for t in (mQ, mK, mV, mO, mLSE, mCuSeqlensQ, mCuSeqlensK)))

        assert blocksparse_tensors is not None
        self.varlen_q = mCuSeqlensQ is not None
        assert blocksparse_tensors.fwd_work_desc is not None
        assert scheduler_tile_counter is not None

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ, mO = [layout_utils.select(t, QO_layout_transpose) for t in (mQ, mO)]
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [layout_utils.select(t, KV_layout_transpose) for t in (mK, mV)]
        LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE = layout_utils.select(mLSE, LSE_layout_transpose) if const_expr(mLSE is not None) else None

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        assert self.num_wg_mma in [1, 2, 3]
        self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_threads_per_warp_group  # If not TMA_Q
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs, self.num_producer_regs = {1: (256, 56), 2: (240, 24), 3: (160, 32)}[self.num_wg_mma]
        self.use_scheduler_barrier = False
        self.use_tma_Q = self.arch >= Arch.sm_90 and not (self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0)
        self.use_tma_O = self.use_tma_Q
        if const_expr(not self.use_tma_Q or not self.use_tma_KV):
            raise NotImplementedError("SM90 plan-owned persistent scheduling requires TMA Q/K/V")
        # Producer needs more registers when doing cp.async Q or KV loads
        if const_expr(self.num_wg_mma == 2 and (not self.use_tma_Q or not self.use_tma_KV)):
            self.num_mma_regs, self.num_producer_regs = 224, 40
        self.rescale_O_before_gemm = self.tile_hdimv > 128 and self.intra_wg_overlap
        self._setup_attributes()
        # TODO: we prob don't need most of what's in _setup_attributes
        self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
            sm90_utils.make_smem_layout(mX.element_type, LayoutEnum.ROW_MAJOR, shape, stage)
            for mX, shape, stage in [
                (mQ, (self.tile_m, self.tile_hdim), None),
                (mK, (self.tile_n, self.tile_hdim), self.num_stages),
                (mV, (self.tile_n, self.tile_hdimv), self.num_stages),
                (mO, (self.tile_m, self.tile_hdimv), None),
            ]
        ]
        self.sP_layout = None
        if const_expr(not self.mma_pv_is_rs):
            self.sP_layout = sm90_utils.make_smem_layout(mV.element_type, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n))

        self.mask_payload_words = (self.tile_m * self.tile_n // self.num_mma_threads + 31) // 32
        assert blocksparse_tensors.mask_block_masks is not None
        SharedStorage = self._get_shared_storage_cls()

        mQ_og, mO_og = mQ, mO
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)

        # TMA
        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()  # Might multicast
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
            ]
        }
        make_tiled_tma_atom_fn = (
            partial(make_packgqa_tiled_tma_atom, qhead_per_kvhead=self.qhead_per_kvhead, head_idx=2)
            if const_expr(self.pack_gqa)
            else cpasync.make_tiled_tma_atom
        )
        tma_atom_Q, tma_tensor_Q = None, None
        if const_expr(self.use_tma_Q):
            tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
                gmem_tiled_copy_Q,
                mQ_og if const_expr(self.pack_gqa) else mQ,
                self.sQ_layout,
                (self.tile_m, self.tile_hdim),  # No mcast
            )
        tma_atom_K, tma_tensor_K = None, None
        tma_atom_V, tma_tensor_V = None, None
        if const_expr(self.use_tma_KV):
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mK,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
                1,  # No mcast for now
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mV,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
                1,  # No mcast for now
            )
        tma_atom_O, tma_tensor_O = None, None
        if const_expr(self.use_tma_O):
            mO_tma = mO_og if const_expr(self.pack_gqa) else mO
            if const_expr(self.varlen_q):
                mO_tma = copy_utils.create_ragged_tensor_for_tma(mO_tma, ragged_dim=0, ptr_shift=True)
            tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
                gmem_tiled_copy_O,
                mO_tma,
                self.sO_layout,
                (self.tile_m, self.tile_hdimv),  # No mcast
            )
        TileScheduler = PlanDynamicPersistentTileSchedulerSm90
        tile_sched_params = TileScheduler.to_underlying_arguments(
            blocksparse_tensors.fwd_work_desc,
            scheduler_tile_counter,
            num_sm=Int32(scheduler_num_sms),
            num_mma_threads=self.num_mma_threads,
        )
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale)
        self.kernel(
            tma_tensor_Q if const_expr(self.use_tma_Q) else mQ,
            tma_tensor_K if const_expr(self.use_tma_KV) else mK,
            tma_tensor_V if const_expr(self.use_tma_KV) else mV,
            tma_tensor_O if const_expr(self.use_tma_O) else mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            blocksparse_tensors,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
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
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        blocksparse_tensors: BlockSparseTensors,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_O,
            ):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        sO_empty_mbar_ptr = storage.mbar_ptr_O_empty.data_ptr()
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(sO_empty_mbar_ptr, 1)

        # Mbarrier / pipeline init
        mbar_ptr_Q = storage.mbar_ptr_Q.data_ptr()

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
        if const_expr(self.use_tma_Q):
            pipeline_q = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["Q"],
                defer_sync=True,
            )
        else:
            pipeline_q = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        if const_expr(self.use_tma_KV):
            pipeline_k = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["K"],
                defer_sync=True,
            )
            pipeline_v = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["V"],
                defer_sync=True,
            )
        else:
            pipeline_k = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )
            pipeline_v = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        pipeline_mask = None
        if const_expr(self.use_smem_mask_pipeline):
            assert blocksparse_tensors.mask_block_masks is not None
            mask_tx_count = self.num_mask_payload_groups * self.mask_payload_words * 4
            pipeline_mask = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_Mask.data_ptr(),
                num_stages=self.mask_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=mask_tx_count,
                defer_sync=True,
            )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
            sO = storage.sV.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)
        else:
            sV = storage.sQ.get_tensor(sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type)
            sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)
        # Transpose view of V to tensor with layout (head_dim_v, tile_n) for tiled mma
        sVt = layout_utils.transpose_view(sV)
        sP = None
        if const_expr(sP_layout is not None):
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        sMask = None
        if const_expr(self.use_smem_mask_pipeline):
            assert blocksparse_tensors.mask_block_masks is not None
            sMask = storage.sMask.get_tensor(
                cute.make_layout(
                    (self.num_mask_payload_groups, self.mask_payload_words, self.mask_stages),
                    stride=(
                        self.mask_payload_words,
                        1,
                        self.num_mask_payload_groups * self.mask_payload_words,
                    ),
                )
            )
        # Match the Hopper C++ layout: epilogue O overlaps V, while Q stays independent.
        sSchedulerWork = storage.scheduler_work.get_tensor(cute.make_layout((4,)))

        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mCuTotalMBlocks=blocksparse_tensors.cu_total_m_blocks,
            mCuBlockIdxOffsets=blocksparse_tensors.cu_block_idx_offsets,
            # Don't need to pass in tile_mn because we won't access offset_padded
        )
        ProducerTileSchedulerCls = partial(TileScheduler.create, tile_sched_params, sSchedulerWork, True)
        ConsumerTileSchedulerCls = partial(TileScheduler.create, tile_sched_params, sSchedulerWork, False)

        # Cluster wait before starting
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx < 4:  # Producer
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                pipeline_mask,
                sMask,
                gmem_tiled_copy_Q,
                blocksparse_tensors,
                SeqlenInfoCls,
                ProducerTileSchedulerCls,
                sO_empty_mbar_ptr,
            )

        else:  # Consumer
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                mO,
                mLSE,
                sQ,
                sK,
                sVt,
                sP,
                sO,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                pipeline_mask,
                sMask,
                gmem_tiled_copy_O,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                softmax_scale,
                SeqlenInfoCls,
                ConsumerTileSchedulerCls,
                blocksparse_tensors,
                sO_empty_mbar_ptr,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_mask: Optional[pipeline.PipelineAsync],
        sMask: Optional[cute.Tensor],
        gmem_tiled_copy_Q: cute.TiledCopy,
        blocksparse_tensors: Optional[BlockSparseTensors],
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        sO_empty_mbar_ptr: Optional[cute.Pointer],
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx, _, _ = cute.arch.thread_idx()

        # TMA: only warp 0 loads. cp_async: all warps load.
        # When not use_tma_Q, all 128 producer threads participate in Q loading.
        is_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_KV or not self.use_tma_Q)
        # KV loading restricted to warp 0 for TMA, all warps for non-TMA KV
        is_kv_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_KV)

        if is_load_warp:
            q_producer_phase = Int32(1)
            kv_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
            mask_producer_state = None
            if const_expr(self.use_smem_mask_pipeline):
                mask_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mask_stages)
            o_empty_phase = Int32(1)
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                # if work_tile.is_valid_tile:
                m_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                head_idx_kv = head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx

                load_Q = None
                if const_expr(self.use_tma_Q):
                    gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
                    load_Q, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True)

                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
                mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
                tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K, 0, cute.make_layout(1), gK, sK)
                tma_load_K_fn = copy_utils.tma_producer_copy_fn(tma_load_K_fn, pipeline_k)
                tma_load_V_fn, _, _ = copy_utils.tma_get_copy_fn(tma_atom_V, 0, cute.make_layout(1), gV, sV)
                tma_load_V_fn = copy_utils.tma_producer_copy_fn(tma_load_V_fn, pipeline_v)

                pack_gqa = None
                if const_expr(not self.use_tma_Q):
                    pack_gqa = PackGQA(self.tile_m, self.tile_hdim, self.check_hdim_oob, self.qhead_per_kvhead)

                if const_expr(self.use_tma_Q):
                    if warp_idx_in_wg == 0:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                        q_producer_phase ^= 1
                else:
                    pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                    pack_gqa.load_Q(mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q)
                    cute.arch.cp_async_commit_group()
                    pipeline_q.producer_commit_w_index(0)
                    q_producer_phase ^= 1
                tile_scheduler.prefetch_next_work()
                if is_kv_load_warp:
                    if const_expr(self.use_smem_mask_pipeline):
                        kv_producer_state, mask_producer_state = produce_block_sparse_loads(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            m_block,
                            seqlen,
                            kv_producer_state,
                            tma_load_K_fn,
                            tma_load_V_fn,
                            pipeline_k,
                            pipeline_v,
                            self.intra_wg_overlap,
                            self.tile_m,
                            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                            o_empty_mbar_ptr=sO_empty_mbar_ptr,
                            o_empty_phase=o_empty_phase,
                            pipeline_mask=pipeline_mask,
                            mask_producer_state=mask_producer_state,
                            sMask=sMask,
                            payload_groups=self.num_mask_payload_groups,
                            payload_words=self.mask_payload_words,
                        )
                    else:
                        kv_producer_state = produce_block_sparse_loads(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            m_block,
                            seqlen,
                            kv_producer_state,
                            tma_load_K_fn,
                            tma_load_V_fn,
                            pipeline_k,
                            pipeline_v,
                            self.intra_wg_overlap,
                            self.tile_m,
                            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                            o_empty_mbar_ptr=sO_empty_mbar_ptr,
                            o_empty_phase=o_empty_phase,
                        )
                    if const_expr(sO_empty_mbar_ptr is not None):
                        o_empty_phase ^= 1

                work_tile = tile_scheduler.advance_to_next_work()
                # End of persistent scheduler loop

            # Producer tail is only useful for cluster to avoid early exit of blocks.
            # We only need producer_tail on V since that's the last that's loaded, we don't
            # need it for Q (no cluster) and K.
            if is_kv_load_warp:
                pipeline_v.producer_tail(kv_producer_state)
                if const_expr(self.use_smem_mask_pipeline):
                    pipeline_mask.producer_tail(mask_producer_state)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sP: Optional[cute.Tensor],
        sO: cute.Tensor,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_mask: Optional[pipeline.PipelineAsync],
        sMask: Optional[cute.Tensor],
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: BlockSparseTensors,
        sO_empty_mbar_ptr: Optional[cute.Pointer] = None,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(self.num_wg_mma, stride=self.num_threads_per_warp_group)
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ, sK)
        mma_qk_fn = partial(sm90_utils.gemm_zero_init, tiled_mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK)
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt)
        mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_P = utils.get_smem_store_atom(self.arch.major * 10 + self.arch.minor, self.dtype)
        smem_thr_copy_P = cute.make_tiled_copy_C(smem_copy_atom_P, tiled_mma_qk).get_slice(tidx)
        tPsP = smem_thr_copy_P.partition_D(sP) if const_expr(sP is not None) else None
        smem_copy_params = SimpleNamespace(smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP)

        self.mma_init()

        q_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
        mask_consumer_state = None
        if const_expr(self.use_smem_mask_pipeline):
            mask_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mask_stages)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            arch=90,
            softmax_scale=softmax_scale,
        )

        # For RescaleOBeforeGemm: persistent scores_scale across iterations
        scores_scale = None
        if const_expr(self.rescale_O_before_gemm):
            scores_scale = cute.make_rmem_tensor_like(softmax.row_max, Float32)

        mma_one_n_block_all = partial(
            self.mma_one_n_block_intrawg_overlap if const_expr(self.intra_wg_overlap) else self.mma_one_n_block,
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            acc_O=acc_O,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            check_inf=True,
            scores_scale=scores_scale,
        )

        process_first_half_block = partial(
            self.first_half_block_overlap,
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            scores_scale=scores_scale,
            softmax=softmax,
            acc_O=acc_O,
        )
        process_last_half_block = partial(
            self.last_half_block_overlap,
            pipeline_v=pipeline_v,
            mma_pv_fn=mma_pv_fn,
            scores_scale=scores_scale,
            softmax=softmax,
            acc_O=acc_O,
        )
        while work_tile.is_valid_tile:
            # if work_tile.is_valid_tile:

            # shape: (atom_v_m * rest_m)
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            mma_one_n_block = partial(mma_one_n_block_all, seqlen=seqlen, softmax=softmax)
            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
            if const_expr(self.use_smem_mask_pipeline):
                kv_consumer_state, _, processed_any, mask_consumer_state = consume_block_sparse_loads(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    seqlen,
                    kv_consumer_state,
                    mma_pv_fn,
                    mma_one_n_block,
                    process_first_half_block,
                    process_last_half_block,
                    self.intra_wg_overlap,
                    self.warp_scheduler_barrier_sync,
                    self.warp_scheduler_barrier_arrive,
                    self.tile_m,
                    tidx,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    payload_words=self.mask_payload_words,
                    pipeline_mask=pipeline_mask,
                    mask_consumer_state=mask_consumer_state,
                    sMask=sMask,
                )
            else:
                kv_consumer_state, _, processed_any = consume_block_sparse_loads(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    seqlen,
                    kv_consumer_state,
                    mma_pv_fn,
                    mma_one_n_block,
                    process_first_half_block,
                    process_last_half_block,
                    self.intra_wg_overlap,
                    self.warp_scheduler_barrier_sync,
                    self.warp_scheduler_barrier_arrive,
                    self.tile_m,
                    tidx,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    payload_words=self.mask_payload_words,
                )

            pipeline_q.consumer_release_w_index(0)

            if not processed_any:
                softmax.reset()
                acc_O.fill(0.0)

            q_consumer_phase ^= 1

            # normalize acc_O by row_sum and calculate the lse
            row_scale = softmax.finalize()
            softmax.rescale_O(acc_O, row_scale)

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue
            # ///////////////////////////////////////////////////////////////////////////////
            next_work_tile = tile_scheduler.advance_to_next_work()
            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
                sO_empty_mbar_ptr,
            )
            work_tile = next_work_tile

    @cute.jit
    def first_half_block_overlap(
        self,
        n_block: Int32,
        mma_qk_fn: Callable,
        kv_consumer_state,
        pipeline_k,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        acc_O: Optional[cute.Tensor] = None,
        mask_fn: Callable = None,
        is_first_block: bool = False,
        mask_prefetch_fn: Optional[Callable] = None,
    ):
        """Processes the first half block when using intra-warpgroup-overlap"""

        r_bitmask = None
        if const_expr(mask_prefetch_fn is not None):
            r_bitmask = mask_prefetch_fn()
        pipeline_k.consumer_wait(kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state))
        acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
        pipeline_k.consumer_release(kv_consumer_state)

        # Apply mask; mask_seqlen always True for first block
        # Caveat: if full block further right than mask block, seqlen masking is redundant;
        # however, masking is being applied anyway, so essentially no perf hit
        if const_expr(r_bitmask is not None):
            mask_fn(
                acc_S,
                n_block=n_block,
                mask_seqlen=True,
                r_bitmask=r_bitmask,
            )
        elif const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block, mask_seqlen=True)

        row_scale = softmax.online_softmax(acc_S, is_first=is_first_block)
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = tOrP if const_expr(self.mma_pv_is_rs) else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        utils.cvt_f16(tOrP_acc, tOrP_cur)

        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
            # Fence and barrier to make smem store visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()

        # For RescaleOBeforeGemm: initialize acc_O
        if const_expr(self.rescale_O_before_gemm):
            acc_O.fill(0.0)
            scores_scale.store(row_scale.load())

        return kv_consumer_state

    @cute.jit
    def last_half_block_overlap(
        self,
        kv_consumer_state,
        pipeline_v,
        mma_pv_fn: Callable,
        zero_init: bool,
        scores_scale: Optional[cute.Tensor] = None,
        softmax: Optional[Softmax] = None,
        acc_O: Optional[cute.Tensor] = None,
    ):
        """Processes the final PV GEMM when using intra-warpgroup-overlap"""

        # For RescaleOBeforeGemm: rescale O before the final PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)

        pipeline_v.consumer_wait(kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state))
        mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=zero_init, wg_wait=0)
        pipeline_v.consumer_release(kv_consumer_state)
        kv_consumer_state.advance()
        return kv_consumer_state

    @cute.jit
    def mma_one_n_block(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,  # not used
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
        mask_prefetch_fn: Optional[Callable] = None,
    ):
        r_bitmask = None
        if const_expr(mask_prefetch_fn is not None):
            r_bitmask = mask_prefetch_fn()
        pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
        # S = Q @ K.T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)

        if const_expr(mask_fn is not None):
            if const_expr(mask_prefetch_fn is not None):
                mask_fn(acc_S=acc_S, n_block=n_block, r_bitmask=r_bitmask)
            else:
                mask_fn(acc_S=acc_S, n_block=n_block)

        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = tOrP if const_expr(self.mma_pv_is_rs) else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        # tOrP.store(tOrP_acc.load().to(self.dtype))
        # the "to(self.dtype)" conversion fails to vectorize for block sizes other
        # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
        # 2 elements. So we just call ptx directly.
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        softmax.rescale_O(acc_O, row_scale)
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
        pipeline_v.consumer_wait(smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read))
        self.warp_scheduler_barrier_sync()
        # O += P @ V
        mma_pv_fn(B_idx=smem_pipe_read.index, wg_wait=0)
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    @cute.jit
    def mma_one_n_block_intrawg_overlap(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        mask_fn: Optional[Callable] = None,
        check_inf: cutlass.Constexpr = True,
        mask_prefetch_fn: Optional[Callable] = None,
    ):
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        r_bitmask = None
        if const_expr(mask_prefetch_fn is not None):
            r_bitmask = mask_prefetch_fn()
        pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
        self.warp_scheduler_barrier_sync()
        # S = Q @ K.T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        # RescaleOBeforeGemm: rescale O while QK GEMM is in flight, before PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)
        pipeline_v.consumer_wait(smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v))
        # O += P @ V
        mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(1)
        pipeline_k.consumer_release(smem_pipe_read)

        if const_expr(mask_fn is not None):
            if const_expr(mask_prefetch_fn is not None):
                mask_fn(acc_S=acc_S, n_block=n_block, r_bitmask=r_bitmask)
            else:
                mask_fn(acc_S=acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        warpgroup.wait_group(0)
        pipeline_v.consumer_release(smem_pipe_read_v)
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = tOrP if const_expr(self.mma_pv_is_rs) else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        # tOrP_cur.store(tOrP_acc.load().to(self.dtype))
        # the "to(self.dtype)" conversion fails to vectorize for block sizes other
        # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
        # 2 elements. So we just call ptx directly.
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        if const_expr(not self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, row_scale)
        if const_expr(self.rescale_O_before_gemm):
            scores_scale.store(row_scale.load())
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
        return smem_pipe_read

    @cute.jit
    def mma_init(self):
        warp_group_idx = utils.canonical_warp_group_idx(sync=False)
        if const_expr(self.use_scheduler_barrier):
            if warp_group_idx == 1:
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                    number_of_threads=2 * self.num_threads_per_warp_group,
                )

    def warp_scheduler_barrier_sync(self):
        if const_expr(self.use_scheduler_barrier):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) - 1 + utils.canonical_warp_group_idx(sync=False),
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    def warp_scheduler_barrier_arrive(self):
        if const_expr(self.use_scheduler_barrier):
            assert self.num_wg_mma in [2, 3]
            cur_wg = utils.canonical_warp_group_idx(sync=False) - 1
            if const_expr(self.num_wg_mma == 2):
                next_wg = 1 - cur_wg
            else:
                t = cur_wg + 1
                next_wg = t % self.num_wg_mma
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
                number_of_threads=2 * self.num_threads_per_warp_group,
            )
