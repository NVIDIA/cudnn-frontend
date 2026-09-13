# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

import math
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Float32, Int32, Int64, Uint32
from cutlass.utils import ClcDynamicPersistentTileScheduler
from cudnn.flex_attention._compat import copy_utils

import cuda.bindings.driver as cuda

from cudnn.flex_attention.kernels.common.tile_scheduler import (
    SM100_TMEM_CAPACITY_COLUMNS,
    ClcState,
    PlanClcPersistentTileSchedulerSm100,
    make_sm100_thread_cooperative_group as make_thread_cooperative_group,
)
from cudnn.flex_attention.plan.kernels import BlockSparseTensors
from cudnn.flex_attention.plan.kernels.packed_mask import (
    apply_loaded_arbitrary_mask,
    load_mask_payload,
)
from cudnn.flex_attention.runtime.dsl_utils import (
    as_bshkrd_tensor,
    assume_tensor_aligned,
    struct_scalar_ptr,
)


@cute.jit
def _hd256_bs_block_info(
    blocksparse_tensors,
    batch_idx: Int32,
    head_idx: Int32,
    m_block: Int32,
):
    """Return one hd256 Q2K plan row and its partial/full payload bases."""
    mask_block_cnt = blocksparse_tensors.mask_block_cnt
    mask_block_idx = blocksparse_tensors.mask_block_idx
    full_block_cnt = blocksparse_tensors.full_block_cnt
    full_block_idx = blocksparse_tensors.full_block_idx
    mask_block_offset = blocksparse_tensors.mask_block_offset
    full_block_offset = blocksparse_tensors.full_block_offset
    sequence_desc = blocksparse_tensors.sequence_desc
    assert len(mask_block_cnt.shape) == 2
    assert mask_block_offset is not None
    assert full_block_cnt is not None
    assert full_block_idx is not None
    assert full_block_offset is not None

    total_m_blocks = mask_block_cnt.shape[1]
    plan_head = Int32(0)
    if mask_block_cnt.shape[0] != 1:
        plan_head = head_idx
    # The plan owns the compact Q-row prefix for both fixed and varlen inputs.
    # The attention kernel must not reconstruct sample tile counts from q_len.
    outer_row = sequence_desc[batch_idx, Int32(4)] + m_block
    plan_row = plan_head * total_m_blocks + outer_row
    mask_cnt = mask_block_cnt[plan_head, outer_row]
    mask_off = mask_block_offset[plan_row]
    full_cnt = full_block_cnt[plan_head, outer_row]
    full_off = full_block_offset[plan_row]
    mask_idx = mask_block_idx
    full_idx = full_block_idx
    total = mask_cnt + full_cnt
    return (total, mask_cnt, mask_off, mask_idx, full_cnt, full_off, full_idx)


@cute.jit
def _hd256_work_coord(work_tile, cta_rank: Int32, cta_group_size: Int32):
    """Adapt the shared plan descriptor to the kernel's local coordinate shape."""

    m_block, head_idx, batch_idx, _ = work_tile.tile_idx
    physical_m_block = m_block * cta_group_size + cta_rank
    return (physical_m_block, Int32(0), (head_idx, batch_idx))


@cute.jit
def _hd256_bs_nblock(i: Int32, bs_info):
    """Map the i-th processed block (0 <= i < max(total, 1)) to its KV block index.

    Iterates the mask list first, then the full list. For an empty m_block
    (total == 0) the kernel still runs one iteration on block 0; the arbitrary
    mask masks every entry of such tiles (an empty CSR row is, by construction,
    a fully-masked Q tile), so the dummy block contributes nothing to the output.
    """
    total, mask_cnt, mask_off, mask_idx, _, full_off, full_idx = bs_info
    n_block = Int32(0)
    if total > 0:
        if i < mask_cnt:
            n_block = mask_idx[mask_off + i]
        else:
            n_block = full_idx[full_off + i - mask_cnt]
    return n_block


class BlackwellFusedMultiHeadAttentionForward:
    def __init__(
        self,
        use_2cta_instrs: bool,
    ):
        if type(use_2cta_instrs) is not bool:
            raise TypeError("use_2cta_instrs must be a bool")

        qk_acc_dtype = cutlass.Float32
        pv_acc_dtype = cutlass.Float32
        mma_tiler = (128, 128, 256)
        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        self.cta_tiler = (
            mma_tiler[0],
            mma_tiler[1],
            mma_tiler[2],
        )
        self.use_2cta_instrs = use_2cta_instrs
        self.cta_group_size = 2 if use_2cta_instrs else 1
        self.qk_mma_tiler = (
            self.cta_group_size * mma_tiler[0],
            mma_tiler[1],
            min(self.cta_tiler[2], 128),
        )
        self.pv_mma_tiler = self.qk_mma_tiler
        self.pv_block_tiler = (
            self.pv_mma_tiler[0] // self.cta_group_size,
            self.pv_mma_tiler[1],
            self.pv_mma_tiler[2],
        )
        self.iterations_qk = self.cta_tiler[2] // self.qk_mma_tiler[2]
        self.iterations_pv = self.cta_tiler[2] // self.pv_mma_tiler[1]
        self.cluster_shape_mn = (self.cta_group_size, 1)
        self.tmem_warp_shape_mn = (4, 1)
        self.mask_payload_valid_words = 4
        self.mask_payload_padded_words = 4
        self.tile_scheduler_cls = PlanClcPersistentTileSchedulerSm100

        self.softmax_warp_ids = (0, 1, 2, 3)
        self.correction_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_ids = (9, 11)
        self.load_warp_id = self.load_warp_ids[0]
        self.v_load_warp_id = self.load_warp_ids[1]
        self.empty_warp_id = (10,)
        self.sched_warp_id = self.empty_warp_id[0]
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.softmax_warp_ids,  # this is to get a round num threads
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.empty_warp_id,
            )
        )

        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_warp * (1 + len(self.softmax_warp_ids) + len(self.correction_warp_ids)),
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * len(self.correction_warp_ids),
        )

        self.tmem_s_offset = 0
        self.tmem_o_offset = 256

        self.num_regs_softmax = 240
        self.num_regs_correction = 128
        self.num_regs_other = 72
        # Each CTA stages its physical M128 x Dv128 PV accumulator slice.
        self.o_tma_tile = self.pv_block_tiler[:2]
        self.o_store_stages = 1

    def _setup_attributes(self):
        self.qk_load_stages = self.iterations_qk
        self.k_stage = 2
        self.v_stage = 2
        self.qk_acc_stage = 2
        self.mma_corr_stage = 1
        self.num_clc_stage = 1
        self.num_clc_response_bytes = 16

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        blocksparse_tensors: BlockSparseTensors = None,
        stream: cuda.CUstream = None,
    ):
        assert blocksparse_tensors is not None, "SM100 hd256 arbitrary forward requires a compact Q2K plan"
        assert (mCuSeqlensQ is None) == (mCuSeqlensK is None), "SM100 hd256 arbitrary varlen forward requires both cu_seqlens tensors"
        if cutlass.const_expr(mCuSeqlensQ is not None):
            assert blocksparse_tensors.sequence_desc is not None, "SM100 hd256 arbitrary varlen forward requires sequence descriptors"
        assert len(blocksparse_tensors.mask_block_cnt.shape) == 2, "SM100 hd256 arbitrary forward requires rank-2 compact counts"
        assert len(blocksparse_tensors.mask_block_idx.shape) == 1
        assert blocksparse_tensors.full_block_cnt is not None
        assert len(blocksparse_tensors.full_block_cnt.shape) == 2
        assert blocksparse_tensors.mask_block_offset is not None
        assert len(blocksparse_tensors.mask_block_offset.shape) == 1
        assert blocksparse_tensors.full_block_offset is not None
        assert len(blocksparse_tensors.full_block_offset.shape) == 1
        assert blocksparse_tensors.full_block_idx is not None
        assert len(blocksparse_tensors.full_block_idx.shape) == 1
        assert blocksparse_tensors.mask_block_masks is not None, "SM100 hd256 arbitrary forward requires mask_block_masks"
        assert len(blocksparse_tensors.mask_block_masks.shape) == 4, "SM100 hd256 arbitrary forward payload must be rank 4"
        q_tensor, k_tensor, v_tensor, o_tensor = mQ, mK, mV, mO
        lse_tensor = mLSE
        cum_seqlen_q = mCuSeqlensQ
        cum_seqlen_k = mCuSeqlensK

        if cutlass.const_expr(cum_seqlen_q is not None):
            assert len(mQ.shape) == 3
            s_q = mQ.shape[0]
            h_q = mQ.shape[1]
            d = mQ.shape[2]
        else:
            assert len(mQ.shape) == 4
            s_q = mQ.shape[1]
            h_q = mQ.shape[2]
            d = mQ.shape[3]

        if cutlass.const_expr(cum_seqlen_k is not None):
            assert len(mK.shape) == 3
            h_k = mK.shape[1]
        else:
            assert len(mK.shape) == 4
            h_k = mK.shape[2]
        if cutlass.const_expr(cum_seqlen_q is not None):
            b = mCuSeqlensQ.shape[0] - 1
        else:
            b = mQ.shape[0]
        scale_softmax = softmax_scale
        scale_softmax_log2 = softmax_scale * math.log2(math.exp(1.0))
        scale_output = 1.0
        s_lse = s_q
        h_r = h_q // h_k
        s_lse64 = Int64(s_lse)
        h_r64 = Int64(h_r)
        h_k64 = Int64(h_k)
        b64 = Int64(b)
        b_lse = b64 if cum_seqlen_q is None else 1
        stride_b_lse = h_r64 * h_k64 * s_lse64 if cum_seqlen_q is None else 0

        varlen_q = cum_seqlen_q is not None
        varlen_k = cum_seqlen_k is not None
        q_norm = as_bshkrd_tensor(q_tensor, h_k, h_r, varlen_q)
        o_norm = as_bshkrd_tensor(o_tensor, h_k, h_r, varlen_q)
        k_norm = as_bshkrd_tensor(k_tensor, h_k, 1, varlen_k)
        v_norm = as_bshkrd_tensor(v_tensor, h_k, 1, varlen_k)
        q_norm, k_norm, v_norm, o_norm = [assume_tensor_aligned(tensor) for tensor in (q_norm, k_norm, v_norm, o_norm)]
        s_q_total = q_norm.shape[1]
        s_k_total = k_norm.shape[1]

        # (s, d, ((h_r, h_k), b))
        q = cute.make_tensor(
            q_norm.iterator,
            cute.make_layout(
                (s_q_total, d, ((h_r, h_k), b)),
                stride=(
                    q_norm.stride[1],
                    q_norm.stride[4],
                    ((q_norm.stride[3], q_norm.stride[2]), q_norm.stride[0]),
                ),
            ),
        )
        # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        k = cute.make_tensor(
            k_norm.iterator,
            cute.make_layout(
                (s_k_total, d, ((h_r, h_k), b)),
                stride=(
                    k_norm.stride[1],
                    k_norm.stride[4],
                    ((0, k_norm.stride[2]), k_norm.stride[0]),
                ),
            ),
        )
        # (d, s, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        v = cute.make_tensor(
            v_norm.iterator,
            cute.make_layout(
                (d, s_k_total, ((h_r, h_k), b)),
                stride=(
                    v_norm.stride[4],
                    v_norm.stride[1],
                    ((0, v_norm.stride[2]), v_norm.stride[0]),
                ),
            ),
        )
        # (s, d, ((h_r, h_k), b))
        o = cute.make_tensor(
            o_norm.iterator,
            cute.make_layout(
                (s_q_total, d, ((h_r, h_k), b)),
                stride=(
                    o_norm.stride[1],
                    o_norm.stride[4],
                    ((o_norm.stride[3], o_norm.stride[2]), o_norm.stride[0]),
                ),
            ),
        )
        if cutlass.const_expr(lse_tensor is not None):
            # (s, ((h_r, h_k), b))
            lse_layout = cute.make_layout(
                (s_lse64, ((h_r, h_k), b_lse)),
                stride=(1, ((s_lse64, h_r64 * s_lse64), stride_b_lse)),
            )
            lse = cute.make_tensor(lse_tensor.iterator, lse_layout)
        else:
            lse = None

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type
        self.tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.q_dtype.width

        assert blocksparse_tensors.fwd_work_desc is not None
        self.tile_sched_params = self.tile_scheduler_cls.to_underlying_arguments(
            blocksparse_tensors.fwd_work_desc,
            self.cta_group_size,
        )
        grid = self.tile_scheduler_cls.get_grid_shape(self.tile_sched_params)

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        if cutlass.const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of v is not supported")

        # check type consistency
        if cutlass.const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if cutlass.const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        # the intermediate tensor p is from tmem & k-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.pv_mma_tiler[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        self.epi_tile = self.pv_block_tiler[:2]

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.qk_load_stages,
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.k_stage,
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.q_dtype,
            self.qk_acc_stage,
        )
        p_tmem_layout = cute.select(p_tmem_layout_staged, mode=[0, 1, 2])
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.v_dtype,
            self.v_stage,
        )
        o_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.o_tma_tile,
            self.o_store_stages,
        )
        # TMA load for Q
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load for K
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for V
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            v_smem_layout,
            self.pv_mma_tiler,
            pv_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Fixed-length O can use a regular TMA descriptor: descriptor bounds
        # handle a short final M tile.  True-varlen keeps the predicated R2G
        # path because each sample has a runtime base offset and extent.
        tma_atom_o = None
        tma_tensor_o = o
        if cutlass.const_expr(cum_seqlen_q is None):
            tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
                cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
                o,
                cute.select(o_smem_layout_staged, mode=[0, 1]),
                self.o_tma_tile,
            )

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        v_copy_size = cute.size_in_bytes(self.v_dtype, v_smem_layout)
        self.tma_copy_q_bytes = q_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_k_bytes = k_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_v_bytes = v_copy_size * cute.size(pv_tiled_mma.thr_id.shape)

        @cute.struct
        class SharedStorage:
            # TMA G2S load barriers: LOAD warp (producer) -> MMA warp (consumer)
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.qk_load_stages * 2]  # load_q_{producer,consumer}
            load_k_mbar_ptr: cute.struct.MemRange[Int64, self.k_stage * 2]  # load_k_{producer,consumer}
            load_v_mbar_ptr: cute.struct.MemRange[Int64, self.v_stage * 2]  # load_v_{producer,consumer}
            mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            p_mma_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            # Softmax -> Correction signaling barriers (row_max/row_sum vec ready)
            s_corr_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]  # s_corr_{producer,consumer}
            sum_mbar_ptr: cute.struct.MemRange[Int64, 2]
            # MMA -> Correction ownership barriers for O_partial tokens (online rescale/finalize)
            mma_corr_mbar_ptr: cute.struct.MemRange[Int64, self.mma_corr_stage * 2]  # mma_corr_{producer,consumer}
            # A CTA-wide "TMEM lifetime" barrier used to safely deallocate TMEM after all users finish.
            tmem_dealloc_mbar: Int64
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # CLC pipeline barriers and response buffer
            clc_mbar_ptr: cute.struct.MemRange[Int64, 2]
            clc_response: cute.struct.Align[cute.struct.MemRange[Int32, 4], 16]

        self.shared_storage = SharedStorage

        grid = cute.round_up(grid, self.cluster_shape_mnk)
        sequence_desc_qk = blocksparse_tensors.sequence_desc if cutlass.const_expr(mCuSeqlensQ is not None) else None
        # Launch the kernel synchronously
        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_o,
            tma_tensor_o,
            sequence_desc_qk,
            sequence_desc_qk,
            lse,
            scale_softmax_log2,
            scale_softmax,
            scale_output,
            self.cluster_layout_vmnk,
            q_smem_layout_staged,
            k_smem_layout_staged,
            p_tmem_layout,
            v_smem_layout_staged,
            o_smem_layout_staged,
            self.tile_sched_params,
            blocksparse_tensors,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(self.cluster_shape_mnk if cute.size(self.cluster_shape_mnk) > 1 else None),
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        tma_atom_o: Optional[cute.CopyAtom],
        mO_qdl: cute.Tensor,
        mSequenceDescQ: Optional[cute.Tensor],
        mSequenceDescK: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        scale_softmax_log2: Float32,
        scale_softmax: Float32,
        scale_output: Float32,
        cluster_layout_vmnk: cute.Layout,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: PlanClcPersistentTileSchedulerSm100.Params,
        blocksparse_tensors,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        #
        # Prefetch tma desc
        #
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
        if warp_idx == self.v_load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
        if cutlass.const_expr(tma_atom_o is not None):
            if warp_idx == self.correction_warp_ids[0]:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)

        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = Int32(0)
        if cutlass.const_expr(self.use_2cta_instrs):
            mma_tile_coord_v = bidx % Int32(self.cta_group_size)
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(mma_tile_coord_v)
        is_leader_cta = mma_tile_coord_v == Int32(0)

        # Alloc
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.qk_load_stages,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_k_producer, load_k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_k_bytes,
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.v_stage,
            producer_group=make_thread_cooperative_group(len([self.v_load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_v_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_s_producer, mma_s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.softmax_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        p_mma_producer, p_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(
                len(self.softmax_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.p_mma_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        s_corr_producer, s_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.softmax_warp_ids)),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.correction_warp_ids)),
            barrier_storage=storage.s_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        sum_producer, sum_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.softmax_warp_ids)),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.correction_warp_ids)),
            barrier_storage=storage.sum_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        mma_corr_producer, mma_corr_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_corr_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.correction_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_corr_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        o_store_pipeline = None
        if cutlass.const_expr(tma_atom_o is not None):
            o_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.o_store_stages,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.threads_per_warp,
                ),
            )
        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            struct_scalar_ptr(storage.tmem_holding_buf),
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=struct_scalar_ptr(storage.tmem_dealloc_mbar),
        )
        clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_clc_consumer_threads = (
            self.cta_group_size
            * self.threads_per_warp
            * (1 + len(self.softmax_warp_ids) + len(self.correction_warp_ids) + 1 + len(self.load_warp_ids))  # scheduler warp  # MMA warp
        )
        clc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_clc_consumer_threads)
        clc_response_ptr = storage.clc_response.data_ptr()
        clc = ClcState.create(
            hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                self.tile_scheduler_cls.clc_problem_shape(tile_sched_params),
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            ),
            pipeline=pipeline.PipelineClcFetchAsync.create(
                barrier_storage=storage.clc_mbar_ptr.data_ptr(),
                num_stages=self.num_clc_stage,
                producer_group=clc_pipeline_producer_group,
                consumer_group=clc_pipeline_consumer_group,
                tx_count=self.num_clc_response_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            ),
            consumer_state=pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_clc_stage),
            producer_state=pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_clc_stage),
        )

        # Cluster arrive after barrier init
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sQ = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=q_smem_layout_staged.outer,
            swizzle=q_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sK = smem.allocate_tensor(
            element_type=self.k_dtype,
            layout=k_smem_layout_staged.outer,
            swizzle=k_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sV = smem.allocate_tensor(
            element_type=self.v_dtype,
            layout=v_smem_layout_staged.outer,
            swizzle=v_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sO = None
        if cutlass.const_expr(tma_atom_o is not None):
            sO = smem.allocate_tensor(
                element_type=self.o_dtype,
                layout=o_smem_layout_staged.outer,
                swizzle=o_smem_layout_staged.inner,
                byte_alignment=128,
            )

        sSum = smem.allocate_tensor(
            element_type=self.qk_acc_dtype,
            layout=cute.make_layout(len(self.softmax_warp_ids) * self.threads_per_warp),
            byte_alignment=128,
        )
        qk_thr_mma = qk_tiled_mma.get_slice(mma_tile_coord_v)
        pv_thr_mma = pv_tiled_mma.get_slice(mma_tile_coord_v)
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrV = pv_thr_mma.make_fragment_B(sV)
        qk_acc_shape = qk_thr_mma.partition_shape_C((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tStS = qk_thr_mma.make_fragment_C(cute.append(qk_acc_shape, self.qk_acc_stage))
        pv_acc_shape = pv_thr_mma.partition_shape_C((self.pv_mma_tiler[0], self.pv_mma_tiler[1]))
        tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                self.iterations_pv,
                stride=self.pv_mma_tiler[1] // self.tmem_warp_shape_mn[1],
            ),
        )
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
        tOtO_staged = cute.make_tensor(tOtO.iterator + self.tmem_o_offset, tOtO_layout)

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        for _i in cutlass.range_constexpr(len(self.empty_warp_id)):
            if warp_idx == self.empty_warp_id[_i]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

        # Cluster wait
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD Q/K
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            tile_sched = self.tile_scheduler_cls.create(tile_sched_params, clc)
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                curr_block_coord = _hd256_work_coord(work_tile, mma_tile_coord_v, Int32(self.cta_group_size))
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                continue_cond = False
                batch_coord = curr_block_coord[2][1]
                seqlen_q = mQ_qdl.shape[0]
                cuseqlen_q = Int32(0)
                cuseqlen_k = Int32(0)
                block_offset = (
                    Int32(0),
                    Int32(0),
                    Int32(0),
                    ((Int32(0), Int32(0)), Int32(0)),
                )
                if cutlass.const_expr(mSequenceDescQ is not None):
                    cuseqlen_q = mSequenceDescQ[batch_coord, Int32(0)]
                    seqlen_q = mSequenceDescQ[batch_coord, Int32(2)]
                    if cutlass.const_expr(mSequenceDescK is not None):
                        cuseqlen_k = mSequenceDescK[batch_coord, Int32(1)]
                    block_offset = (
                        cuseqlen_q,
                        cuseqlen_k,
                        Int32(0),
                        ((Int32(0), Int32(0)), Int32(0)),
                    )
                    continue_cond = mma_block_coord[0] * self.qk_mma_tiler[0] >= seqlen_q
                if not continue_cond:
                    mQ_qdl_ = cute.domain_offset(cute.select(block_offset, mode=[0, 2, 3]), mQ_qdl)
                    # Local tile partition global tensors
                    q_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
                    # (bM, bK, loopM, loopK, loopL)
                    gQ_qdl = cute.flat_divide(mQ_qdl_, cute.select(self.qk_mma_tiler, mode=[0, 2]))
                    tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
                    tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_q,
                        block_in_cluster_coord_vmnk[2],
                        q_cta_layout,
                        cute.group_modes(sQ, 0, 3),
                        cute.group_modes(tSgQ_qdl, 0, 3),
                    )
                    kv_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
                    mK_kdl_ = cute.domain_offset(cute.select(block_offset, mode=[1, 2, 3]), mK_kdl)
                    gK_kdl = cute.flat_divide(mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2]))
                    tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                    tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_k,
                        block_in_cluster_coord_vmnk[1],
                        kv_cta_layout,
                        cute.group_modes(sK, 0, 3),
                        cute.group_modes(tSgK_kdl, 0, 3),
                    )
                    tKgK = tKgK_kdl[None, None, None, mma_block_coord[2]]
                    # ((atom_v, rest_v), RestK)
                    tQgQ = tQgQ_qdl[None, mma_block_coord[0], None, mma_block_coord[2]]

                    bs_info = _hd256_bs_block_info(
                        blocksparse_tensors,
                        batch_coord,
                        curr_block_coord[2][0],
                        mma_block_coord[0],
                    )
                    bs_total = bs_info[0]
                    seqlen_kv_loop_steps = bs_total
                    if bs_total == 0:
                        seqlen_kv_loop_steps = Int32(1)

                    # Q
                    for iter in cutlass.range(self.iterations_qk, unroll=1):
                        q_handle = load_q_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_q,
                            tQgQ[None, iter],
                            tQsQ[None, q_handle.index],
                            tma_bar_ptr=q_handle.barrier,
                        )

                    # K0
                    kv_coord = Int32(0)
                    k_blk = _hd256_bs_nblock(kv_coord, bs_info)
                    for iter in cutlass.range(self.iterations_qk, unroll=1):
                        k_handle = load_k_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_k,
                            tKgK[None, k_blk, iter],
                            tKsK[None, k_handle.index],
                            tma_bar_ptr=k_handle.barrier,
                        )
                    kv_coord += 1
                    for _ in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        k_blk = _hd256_bs_nblock(kv_coord, bs_info)
                        for iter in cutlass.range(self.iterations_qk, unroll=1):
                            k_handle = load_k_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_k,
                                tKgK[None, k_blk, iter],
                                tKsK[None, k_handle.index],
                                tma_bar_ptr=k_handle.barrier,
                            )
                        kv_coord += 1

                work_tile = tile_sched.advance_to_next_work()
                # End of persistent scheduler loop
            load_k_producer.tail()
            load_q_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD V
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.v_load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            v_tile_sched = self.tile_scheduler_cls.create(tile_sched_params, clc)
            v_work_tile = v_tile_sched.initial_work_tile_info()
            while v_work_tile.is_valid_tile:
                v_block_coord = _hd256_work_coord(v_work_tile, mma_tile_coord_v, Int32(self.cta_group_size))
                v_mma_block_coord = (
                    v_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    v_block_coord[1],
                    v_block_coord[2],
                )
                v_batch_coord = v_block_coord[2][1]
                v_seqlen_q = mQ_qdl.shape[0]
                v_cuseqlen_k = Int32(0)
                v_continue_cond = False
                if cutlass.const_expr(mSequenceDescQ is not None):
                    v_seqlen_q = mSequenceDescQ[v_batch_coord, Int32(2)]
                    v_continue_cond = v_mma_block_coord[0] * self.qk_mma_tiler[0] >= v_seqlen_q
                    if cutlass.const_expr(mSequenceDescK is not None):
                        v_cuseqlen_k = mSequenceDescK[v_batch_coord, Int32(1)]
                if not v_continue_cond:
                    v_block_offset = (
                        Int32(0),
                        v_cuseqlen_k,
                        ((Int32(0), Int32(0)), Int32(0)),
                    )
                    mV_dkl_ = cute.domain_offset(v_block_offset, mV_dkl)
                    v_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
                    gV_dkl = cute.flat_divide(mV_dkl_, cute.select(self.pv_mma_tiler, mode=[1, 2]))
                    tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
                    tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v,
                        block_in_cluster_coord_vmnk[1],
                        v_cta_layout,
                        cute.group_modes(sV, 0, 3),
                        cute.group_modes(tSgV_dkl, 0, 3),
                    )
                    tVgV = tVgV_dkl[None, None, None, v_mma_block_coord[2]]
                    v_bs_info = _hd256_bs_block_info(
                        blocksparse_tensors,
                        v_batch_coord,
                        v_block_coord[2][0],
                        v_mma_block_coord[0],
                    )
                    v_loop_steps = v_bs_info[0]
                    if v_loop_steps == 0:
                        v_loop_steps = Int32(1)
                    for v_coord in cutlass.range(0, v_loop_steps, 1, unroll=1):
                        v_blk = _hd256_bs_nblock(v_coord, v_bs_info)
                        for iter in cutlass.range(self.iterations_pv, unroll=1):
                            v_handle = load_v_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_v,
                                tVgV[None, iter, v_blk],
                                tVsV[None, v_handle.index],
                                tma_bar_ptr=v_handle.barrier,
                            )
                v_work_tile = v_tile_sched.advance_to_next_work()
            load_v_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            tile_sched = self.tile_scheduler_cls.create(tile_sched_params, clc)
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile and is_leader_cta:
                curr_block_coord = _hd256_work_coord(work_tile, mma_tile_coord_v, Int32(self.cta_group_size))
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                continue_cond = False
                seqlen_q = mQ_qdl.shape[0]
                batch_coord = curr_block_coord[2][1]
                if cutlass.const_expr(mSequenceDescQ is not None):
                    cuseqlen_q = mSequenceDescQ[batch_coord, Int32(0)]
                    seqlen_q = mSequenceDescQ[batch_coord, Int32(2)]
                    continue_cond = mma_block_coord[0] * self.qk_mma_tiler[0] >= seqlen_q

                if not continue_cond:
                    if cutlass.const_expr(mSequenceDescK is not None):
                        cuseqlen_k = mSequenceDescK[batch_coord, Int32(1)]

                    bs_total = _hd256_bs_block_info(
                        blocksparse_tensors,
                        batch_coord,
                        curr_block_coord[2][0],
                        mma_block_coord[0],
                    )[0]
                    seqlen_kv_loop_steps = bs_total
                    if bs_total == 0:
                        seqlen_kv_loop_steps = Int32(1)

                    load_q_releaser = load_q_consumer.clone()
                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    # QK0
                    s_handle = mma_s_producer.acquire_and_advance()
                    tStS_slice = tStS[None, None, None, s_handle.index]
                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for iter in cutlass.range(self.iterations_qk, unroll=1):
                        load_q_consumer.wait_and_advance()
                        tSrQ_slice = tSrQ[None, None, None, iter]
                        k_handle = load_k_consumer.wait_and_advance()
                        tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                        num_kphases = cute.size(tSrQ_slice, mode=[2])
                        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                            kphase_coord = (None, None, kphase_idx)
                            cute.gemm(
                                qk_tiled_mma,
                                tStS_slice,
                                tSrQ_slice[kphase_coord],
                                tSrK_trans_slice[kphase_coord],
                                tStS_slice,
                            )
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        k_handle.release()
                    s_handle.commit()

                    for _step in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        # QKi
                        s_handle = mma_s_producer.acquire_and_advance()
                        tStS_slice = tStS[None, None, None, s_handle.index]
                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for iter in cutlass.range(self.iterations_qk, unroll=1):
                            tSrQ_slice = tSrQ[None, None, None, iter]
                            k_handle = load_k_consumer.wait_and_advance()
                            tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                            num_kphases = cute.size(tSrQ_slice, mode=[2])
                            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                kphase_coord = (None, None, kphase_idx)
                                cute.gemm(
                                    qk_tiled_mma,
                                    tStS_slice,
                                    tSrQ_slice[kphase_coord],
                                    tSrK_trans_slice[kphase_coord],
                                    tStS_slice,
                                )
                                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            k_handle.release()
                        s_handle.commit()

                        # PVi-1
                        p_handle = p_mma_consumer.wait_and_advance()
                        o_handle = mma_corr_producer.acquire_and_advance()
                        pv_whether_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                        for iter in cutlass.range(self.iterations_pv, unroll=1):
                            v_handle = load_v_consumer.wait_and_advance()
                            pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                            tOtO_slice = tOtO_staged[None, None, None, iter]
                            tStS_slice = tStS[None, None, None, p_handle.index]
                            tP = cute.make_tensor(tStS_slice.iterator, p_tmem_layout_staged.outer)
                            tOrP = pv_thr_mma.make_fragment_A(tP)
                            tOrP_slice = cute.make_tensor(
                                cute.recast_ptr(tStS_slice.iterator, dtype=self.q_dtype),
                                tOrP.layout,
                            )
                            tOrV_slice = tOrV[None, None, None, v_handle.index]
                            num_kphases = cute.size(tOrV_slice, mode=[2])
                            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                kphase_coord = (None, None, kphase_idx)
                                cute.gemm(
                                    pv_tiled_mma,
                                    tOtO_slice,
                                    tOrP_slice[kphase_coord],
                                    tOrV_slice[kphase_coord],
                                    tOtO_slice,
                                )
                                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            v_handle.release()
                        o_handle.commit()
                        p_handle.release()

                    for _iter in cutlass.range(self.iterations_qk, unroll=1):
                        load_q_releaser.release()
                        load_q_releaser.advance()

                    # PVend
                    p_handle = p_mma_consumer.wait_and_advance()
                    o_handle = mma_corr_producer.acquire_and_advance()
                    pv_whether_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                    for iter in cutlass.range(self.iterations_pv, unroll=1):
                        v_handle = load_v_consumer.wait_and_advance()
                        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                        tOtO_slice = tOtO_staged[None, None, None, iter]
                        tStS_slice = tStS[None, None, None, p_handle.index]
                        tP = cute.make_tensor(tStS_slice.iterator, p_tmem_layout_staged.outer)
                        tOrP = pv_thr_mma.make_fragment_A(tP)
                        tOrP_slice = cute.make_tensor(
                            cute.recast_ptr(tStS_slice.iterator, dtype=self.q_dtype),
                            tOrP.layout,
                        )
                        tOrV_slice = tOrV[None, None, None, v_handle.index]
                        num_kphases = cute.size(tOrV_slice, mode=[2])
                        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                            kphase_coord = (None, None, kphase_idx)
                            cute.gemm(
                                pv_tiled_mma,
                                tOtO_slice,
                                tOrP_slice[kphase_coord],
                                tOrV_slice[kphase_coord],
                                tOtO_slice,
                            )
                            pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        v_handle.release()
                    o_handle.commit()
                    p_handle.release()
                work_tile = tile_sched.advance_to_next_work()
            if cutlass.const_expr(self.use_2cta_instrs):
                if not is_leader_cta:
                    while work_tile.is_valid_tile:
                        work_tile = tile_sched.advance_to_next_work()
            # End of persistent scheduler loop
            mma_s_producer.tail()
            mma_corr_producer.tail()
            tmem.relinquish_alloc_permit()
            self.tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax_warp_ids[0]:
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            tile_sched = self.tile_scheduler_cls.create(tile_sched_params, clc)
            work_tile = tile_sched.initial_work_tile_info()
            assert blocksparse_tensors.mask_block_masks is not None
            mask_payloads = blocksparse_tensors.mask_block_masks

            while work_tile.is_valid_tile:
                curr_block_coord = _hd256_work_coord(work_tile, mma_tile_coord_v, Int32(self.cta_group_size))
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                continue_cond = False
                seqlen_q = mQ_qdl.shape[0]
                cuseqlen_q = Int32(0)
                if cutlass.const_expr(mSequenceDescQ is not None):
                    cuseqlen_q = mSequenceDescQ[batch_coord, Int32(0)]
                    seqlen_q = mSequenceDescQ[batch_coord, Int32(2)]
                    continue_cond = mma_block_coord[0] * self.qk_mma_tiler[0] >= seqlen_q
                if not continue_cond:
                    if cutlass.const_expr(mSequenceDescK is not None):
                        cuseqlen_k = mSequenceDescK[batch_coord, Int32(1)]

                    row_max = -Float32.inf
                    row_max_prev = -Float32.inf
                    row_sum = 0.0

                    bs_info = _hd256_bs_block_info(
                        blocksparse_tensors,
                        batch_coord,
                        curr_block_coord[2][0],
                        mma_block_coord[0],
                    )
                    bs_total = bs_info[0]
                    # Keep one zero-masked iteration for the pipeline protocol.
                    start_count = Int32(0)
                    end_count = bs_total
                    if bs_total == 0:
                        end_count = Int32(1)
                    cS_base = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
                    cS = cute.domain_offset((mma_block_coord[0] * self.qk_mma_tiler[0], 0), cS_base)

                    if bs_total == 0:
                        step = Int32(0)
                        col_block = _hd256_bs_nblock(step, bs_info)
                        cS_iter = cute.domain_offset((0, col_block * self.qk_mma_tiler[1]), cS)
                        tScS_iter = qk_thr_mma.partition_C(cS_iter)
                        (
                            row_max,
                            row_sum,
                            mma_s_consumer,
                            p_mma_producer,
                            s_corr_producer,
                        ) = self.softmax_step(
                            (
                                True,
                                None,
                                Int32(0),
                                True,
                                mma_tile_coord_v,
                            ),
                            (
                                row_max_prev,
                                row_sum,
                                scale_softmax_log2,
                            ),
                            (tStS, tScS_iter),
                            (mma_s_consumer, p_mma_producer, s_corr_producer),
                        )
                        row_max_prev = row_max
                    else:
                        # Keep the mask choice static so full-block SASS does not
                        # issue predicated-off packed-mask instructions.
                        for step in cutlass.range(start_count, bs_info[1], 1, unroll=1):
                            col_block = _hd256_bs_nblock(step, bs_info)
                            cS_iter = cute.domain_offset((0, col_block * self.qk_mma_tiler[1]), cS)
                            tScS_iter = qk_thr_mma.partition_C(cS_iter)
                            (
                                row_max,
                                row_sum,
                                mma_s_consumer,
                                p_mma_producer,
                                s_corr_producer,
                            ) = self.softmax_step(
                                (
                                    True,
                                    mask_payloads,
                                    bs_info[2] + step,
                                    False,
                                    mma_tile_coord_v,
                                ),
                                (
                                    row_max_prev,
                                    row_sum,
                                    scale_softmax_log2,
                                ),
                                (tStS, tScS_iter),
                                (mma_s_consumer, p_mma_producer, s_corr_producer),
                            )
                            row_max_prev = row_max
                        for step in cutlass.range(bs_info[1], end_count, 1, unroll=1):
                            col_block = _hd256_bs_nblock(step, bs_info)
                            cS_iter = cute.domain_offset((0, col_block * self.qk_mma_tiler[1]), cS)
                            tScS_iter = qk_thr_mma.partition_C(cS_iter)
                            (
                                row_max,
                                row_sum,
                                mma_s_consumer,
                                p_mma_producer,
                                s_corr_producer,
                            ) = self.softmax_step(
                                (False, None, Int32(0), False, Int32(0)),
                                (
                                    row_max_prev,
                                    row_sum,
                                    scale_softmax_log2,
                                ),
                                (tStS, tScS_iter),
                                (mma_s_consumer, p_mma_producer, s_corr_producer),
                            )
                            row_max_prev = row_max
                    sum_producer = self.store_sum_max(
                        row_max,
                        mLSE,
                        row_sum,
                        sSum,
                        sum_producer,
                        curr_block_coord,
                        seqlen_q,
                        mSequenceDescQ,
                        cuseqlen_q,
                        scale_softmax,
                    )
                work_tile = tile_sched.advance_to_next_work()
            p_mma_producer.tail()
            s_corr_producer.tail()
            self.tmem_alloc_barrier.arrive()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_correction)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            tile_sched = self.tile_scheduler_cls.create(tile_sched_params, clc)
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                curr_block_coord = _hd256_work_coord(work_tile, mma_tile_coord_v, Int32(self.cta_group_size))
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                batch_coord = curr_block_coord[2][1]
                seqlen_q = mQ_qdl.shape[0]
                continue_cond = False
                cuseqlen_q = Int32(0)
                if cutlass.const_expr(mSequenceDescQ is not None):
                    cuseqlen_q = mSequenceDescQ[batch_coord, Int32(0)]
                    seqlen_q = mSequenceDescQ[batch_coord, Int32(2)]
                    continue_cond = mma_block_coord[0] * self.qk_mma_tiler[0] >= seqlen_q

                if not continue_cond:
                    if cutlass.const_expr(mSequenceDescK is not None):
                        cuseqlen_k = mSequenceDescK[batch_coord, Int32(1)]

                    mO_qdl_eff = mO_qdl
                    if cutlass.const_expr(mSequenceDescQ is not None):
                        block_offset_o = (
                            cuseqlen_q,
                            Int32(0),
                            Int32(0),
                            ((Int32(0), Int32(0)), Int32(0)),
                        )
                        mO_qdl_eff = cute.domain_offset(cute.select(block_offset_o, mode=[0, 2, 3]), mO_qdl)

                    # (bM, bN, loopM, loopN, loopL)
                    gO_qdl = cute.flat_divide(mO_qdl_eff, cute.select(self.pv_block_tiler, mode=[0, 1]))
                    cO_qdl = cute.flat_divide(
                        cute.make_identity_tensor(mO_qdl_eff.shape),
                        cute.select(self.pv_block_tiler, mode=[0, 1]),
                    )

                    gO_tma = None
                    if cutlass.const_expr(tma_atom_o is not None):
                        gO_tma = cute.local_tile(
                            mO_qdl[None, None, curr_block_coord[2]],
                            self.o_tma_tile,
                            (curr_block_coord[0], None),
                        )

                    bs_total = _hd256_bs_block_info(
                        blocksparse_tensors,
                        batch_coord,
                        curr_block_coord[2][0],
                        mma_block_coord[0],
                    )[0]
                    seqlen_kv_loop_steps = bs_total
                    if bs_total == 0:
                        seqlen_kv_loop_steps = Int32(1)
                    gO_staged = gO_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]
                    cO_staged = cO_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]
                    cS = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
                    tScS = qk_thr_mma.partition_C(cS)

                    # Empty step as the first step is no need for correction
                    stats_handle = s_corr_consumer.wait_and_advance()
                    stats_handle.release()
                    for step in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                        # Oi-1 -> Oi
                        mma_corr_consumer, s_corr_consumer = self.correction_rescale(
                            scale_softmax_log2,
                            (s_corr_consumer, tStS, tScS),
                            (mma_corr_consumer, tOtO_staged, cO_staged),
                            self.epi_tile,
                        )
                    # O_partial -> O_final
                    if cutlass.const_expr(tma_atom_o is not None):
                        mma_corr_consumer, sum_consumer = self.correction_epilog_tma(
                            scale_output,
                            (sum_consumer, sSum),
                            (
                                mma_corr_consumer,
                                gO_tma,
                                sO,
                                tma_atom_o,
                                tOtO_staged,
                                pv_thr_mma,
                                o_store_pipeline,
                            ),
                        )
                    else:
                        mma_corr_consumer, sum_consumer = self.correction_epilog(
                            (seqlen_q, scale_output),
                            (sum_consumer, sSum),
                            (mma_corr_consumer, gO_staged, cO_staged, tOtO_staged),
                            self.epi_tile,
                        )
                work_tile = tile_sched.advance_to_next_work()
            if cutlass.const_expr(tma_atom_o is not None):
                if warp_idx == self.correction_warp_ids[0]:
                    o_store_pipeline.producer_tail()
            self.tmem_alloc_barrier.arrive()

        # ///////////////////////////////////////////////////////////////////////////////
        #  CLC scheduler warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.sched_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            tile_sched = self.tile_scheduler_cls.create(tile_sched_params, clc)
            work_tile = tile_sched.initial_work_tile_info()
            if is_leader_cta:
                while work_tile.is_valid_tile:
                    tile_sched.prefetch_next_work()
                    work_tile = tile_sched.advance_to_next_work()
                tile_sched.producer_tail()
            else:
                while work_tile.is_valid_tile:
                    work_tile = tile_sched.advance_to_next_work()

        return

    @cute.jit
    def softmax_step(
        self,
        mask_args: Tuple,
        value_args: Tuple,
        tensor_args: Tuple,
        pipeline_args: Tuple,
    ) -> Tuple[Float32, Float32, pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        (
            apply_arbitrary_mask,
            mask_payloads,
            payload_idx,
            empty_dummy,
            subtile_idx,
        ) = mask_args
        row_max, row_sum, scale_softmax_log2 = value_args
        tStS, tScS = tensor_args
        mma_s_consumer, p_mma_producer, s_corr_producer = pipeline_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        s_handle = mma_s_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, s_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        tmem_load_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype)
        tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_slice)
        thr_load = tmem_tiled_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_load.partition_S(tStS_slice)
        tTMEM_LOADcS = thr_load.partition_D(tScS_slice)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.qk_acc_dtype)
        cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)

        cute.arch.fence_view_async_tmem_load()
        s_handle.release()
        if cutlass.const_expr(apply_arbitrary_mask):
            assert cutlass.const_expr(
                cute.size(tTMEM_LOADrS) == self.mask_payload_valid_words * 32
            ), "SM100 hd256 arbitrary payload does not cover the score fragment"
            if cutlass.const_expr(empty_dummy):
                # Preserve the empty-row pipeline iteration without a global
                # payload read from an empty partial list.
                r_bitmask = cute.make_rmem_tensor(
                    (self.mask_payload_padded_words,),
                    Uint32,
                )
                r_bitmask.fill(Uint32(0))
                apply_loaded_arbitrary_mask(
                    tTMEM_LOADrS,
                    r_bitmask,
                    self.mask_payload_valid_words,
                )
            else:
                assert mask_payloads is not None
                # Load only after the score wait to reduce register pressure.
                r_bitmask = load_mask_payload(
                    mask_payloads,
                    payload_idx,
                    thread_idx,
                    subtile_idx=subtile_idx,
                    payload_words=self.mask_payload_padded_words,
                )
                apply_loaded_arbitrary_mask(
                    tTMEM_LOADrS,
                    r_bitmask,
                    self.mask_payload_valid_words,
                )
        old_row_max = row_max
        row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
        row_max_safe = row_max
        if row_max == -cutlass.Float32.inf:
            row_max_safe = 0.0

        stats_handle = s_corr_producer.acquire_and_advance()
        stats_layout = cute.composition(tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], 2)))
        stats_c_layout = cute.composition(tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], 2)))
        tOtStats = cute.make_tensor(tStS_slice.iterator + self.tilePlikeFP32, stats_layout)
        tOcStats = cute.make_tensor(tScS_slice.iterator, stats_c_layout)
        tmem_store_stats_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )
        tiled_tmem_store_stats = tcgen05.make_tmem_copy(tmem_store_stats_atom, tOtStats)
        thr_tmem_store_stats = tiled_tmem_store_stats.get_slice(thread_idx)
        tTMEM_STOREcStats = thr_tmem_store_stats.partition_S(tOcStats)
        tTMEM_STORErStats = cute.make_rmem_tensor(tTMEM_STOREcStats.shape, self.qk_acc_dtype)
        tTMEM_STORErStats[0] = old_row_max
        tTMEM_STORErStats[1] = row_max_safe
        tTMEM_STOREtStats = thr_tmem_store_stats.partition_D(tOtStats)
        cute.copy(tiled_tmem_store_stats, tTMEM_STORErStats, tTMEM_STOREtStats)
        cute.arch.fence_view_async_tmem_store()
        stats_handle.commit()

        scale = scale_softmax_log2
        minus_row_max_scale = (0.0 - row_max_safe) * scale
        # Acquire P write slot early — overlaps any pipeline stall with exp2 compute
        p_handle = p_mma_producer.acquire_and_advance()
        # Fragment-based FMA + exp2 + bf16 conversion
        # Trades SFU for FMA via polynomial emulation on a fraction of elements
        ex2_frg_tile = 32
        ex2_frg_cnt = cute.size(tTMEM_LOADrS) // ex2_frg_tile
        tTMEM_LOADrS_ex2 = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(ex2_frg_tile))
        tTMEM_STORErP = cute.make_rmem_tensor(tTMEM_LOADrS.shape, self.q_dtype)
        tTMEM_STORErP_ex2 = cute.logical_divide(tTMEM_STORErP, cute.make_layout(ex2_frg_tile))
        for j in cutlass.range_constexpr(ex2_frg_cnt):
            for k in cutlass.range_constexpr(0, ex2_frg_tile, 2):
                tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j] = cute.arch.fma_packed_f32x2(
                    (tTMEM_LOADrS_ex2[k, j], tTMEM_LOADrS_ex2[k + 1, j]),
                    (scale, scale),
                    (minus_row_max_scale, minus_row_max_scale),
                )
                tTMEM_LOADrS_ex2[k, j] = cute.math.exp2(tTMEM_LOADrS_ex2[k, j], fastmath=True)
                tTMEM_LOADrS_ex2[k + 1, j] = cute.math.exp2(tTMEM_LOADrS_ex2[k + 1, j], fastmath=True)
            tTMEM_STORErP_ex2[None, j].store(tTMEM_LOADrS_ex2[None, j].load().to(self.q_dtype))
        tmem_store_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype)
        tilePlikeFP32 = tStS_slice.shape[1] // Float32.width * self.q_dtype.width
        tStS_P_layout = cute.composition(tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], tilePlikeFP32)))
        tStS_P = cute.make_tensor(tStS_slice.iterator, tStS_P_layout)
        tScS_P_layout = cute.composition(tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], tilePlikeFP32)))
        tScS_P = cute.make_tensor(tScS_slice.iterator, tScS_P_layout)
        tmem_tiled_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
        thr_store = tmem_tiled_store.get_slice(thread_idx)
        tTMEM_STOREtP = thr_store.partition_D(tStS_P)
        tTMEM_STOREcS = thr_store.partition_S(tScS_P)
        tTMEM_STORErP_ = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErP.iterator, dtype=self.qk_acc_dtype),
            tTMEM_STOREcS.shape,
        )
        cute.copy(tmem_tiled_store, tTMEM_STORErP_, tTMEM_STOREtP)
        cute.arch.fence_view_async_tmem_store()

        p_handle.commit()
        acc_scale_ = scale * (old_row_max - row_max_safe)
        acc_scale = cute.math.exp2(acc_scale_, fastmath=True) * 0.5
        # TODO: calc row sum with TensorSSA
        row_sum *= acc_scale
        local_row_sum_0 = (row_sum, row_sum)
        local_row_sum_1 = (0.0, 0.0)
        local_row_sum_2 = (0.0, 0.0)
        local_row_sum_3 = (0.0, 0.0)
        reduction_unroll = 4
        frg_tile = cute.size(tTMEM_LOADrS) // reduction_unroll
        tTMEM_LOADrS_frg = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(frg_tile))
        for j in cutlass.range_constexpr(0, cute.size(tTMEM_LOADrS_frg, mode=[0]), 2):
            local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, (tTMEM_LOADrS_frg[j, 0], tTMEM_LOADrS_frg[j + 1, 0]))
            local_row_sum_1 = cute.arch.add_packed_f32x2(local_row_sum_1, (tTMEM_LOADrS_frg[j, 1], tTMEM_LOADrS_frg[j + 1, 1]))
            local_row_sum_2 = cute.arch.add_packed_f32x2(local_row_sum_2, (tTMEM_LOADrS_frg[j, 2], tTMEM_LOADrS_frg[j + 1, 2]))
            local_row_sum_3 = cute.arch.add_packed_f32x2(local_row_sum_3, (tTMEM_LOADrS_frg[j, 3], tTMEM_LOADrS_frg[j + 1, 3]))
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_1)
        local_row_sum_2 = cute.arch.add_packed_f32x2(local_row_sum_2, local_row_sum_3)
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_2)
        row_sum = local_row_sum_0[0] + local_row_sum_0[1]
        return row_max, row_sum, mma_s_consumer, p_mma_producer, s_corr_producer

    @cute.jit
    def correction_rescale(
        self,
        scale_softmax_log2: Float32,
        stats_args: tuple,
        o_args: tuple,
        epi_tile: cute.Tile,
    ) -> pipeline.PipelineConsumer:
        s_corr_consumer, tStS, tScS = stats_args
        mma_o_consumer, tOtO_staged, cO_staged = o_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))

        stats_handle = s_corr_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, stats_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        stats_layout = cute.composition(tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], 2)))
        stats_c_layout = cute.composition(tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], 2)))
        tOtStats = cute.make_tensor(tStS_slice.iterator + self.tilePlikeFP32, stats_layout)
        tOcStats = cute.make_tensor(tScS_slice.iterator, stats_c_layout)
        tmem_load_stats_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )
        tiled_tmem_load_stats = tcgen05.make_tmem_copy(tmem_load_stats_atom, tOtStats)
        thr_tmem_load_stats = tiled_tmem_load_stats.get_slice(thread_idx)
        tTMEM_LOADtStats = thr_tmem_load_stats.partition_S(tOtStats)
        tTMEM_LOADcStats = thr_tmem_load_stats.partition_D(tOcStats)
        tTMEM_LOADrStats = cute.make_rmem_tensor(tTMEM_LOADcStats.shape, self.qk_acc_dtype)
        cute.copy(tiled_tmem_load_stats, tTMEM_LOADtStats, tTMEM_LOADrStats)
        cute.arch.fence_view_async_tmem_load()

        scale = scale_softmax_log2 * (tTMEM_LOADrStats[0] - tTMEM_LOADrStats[1])
        scale = cute.math.exp2(scale, fastmath=True)
        stats_handle.release()
        o_handle = mma_o_consumer.wait_and_advance()
        for iter in cutlass.range(self.iterations_pv, unroll_full=True):
            tOtO = tOtO_staged[(None, None), 0, 0, iter]
            cO = cO_staged[None, None, iter]
            tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
            cO_epi = cute.zipped_divide(cO, epi_tile)
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(16)),
                self.pv_acc_dtype,
            )
            tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_epi)
            thr_load = tmem_tiled_load.get_slice(thread_idx)
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition(16)),
                self.pv_acc_dtype,
            )
            tmem_store_atom = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_epi)
            thr_store = tmem_store_atom.get_slice(thread_idx)
            tTMEM_LOADtO = thr_load.partition_S(tOtO_epi)
            tTMEM_LOADcO = thr_load.partition_D(cO_epi)
            tTMEM_STOREtO = thr_store.partition_D(tOtO_epi)
            tTMrO = cute.make_rmem_tensor_like(
                cute.append(
                    cute.make_layout(tTMEM_LOADcO[None, 0, 0].shape),
                    cute.make_layout(2, stride=cute.size(tTMEM_LOADcO[None, 0, 0].shape)),
                ),
                self.pv_acc_dtype,
            )
            tTMEM_LOADtO_0 = tTMEM_LOADtO[None, 0, 0]
            cute.copy(tmem_tiled_load, tTMEM_LOADtO_0, tTMrO[None, 0])
            cute.arch.fence_view_async_tmem_load()
            iter_num = cute.size(tTMEM_LOADtO, mode=[1])
            for i in cutlass.range(1, iter_num, unroll_full=True):
                tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
                cute.copy(tmem_tiled_load, tTMEM_LOADtO_i, tTMrO[None, i % 2])
                for j in cutlass.range(0, cute.size(tTMrO, mode=[0]), 2, unroll_full=True):
                    tTMrO[j, (i - 1) % 2], tTMrO[j + 1, (i - 1) % 2] = cute.arch.mul_packed_f32x2(
                        (tTMrO[j, (i - 1) % 2], tTMrO[j + 1, (i - 1) % 2]),
                        (scale, scale),
                    )
                tTMEM_STOREtO_prev_i = tTMEM_STOREtO[None, i - 1, 0]
                cute.copy(tmem_store_atom, tTMrO[None, (i - 1) % 2], tTMEM_STOREtO_prev_i)
                cute.arch.fence_view_async_tmem_load()

            for j in cutlass.range(0, cute.size(tTMrO, mode=[0]), 2, unroll_full=True):
                tTMrO[j, (iter_num - 1) % 2], tTMrO[j + 1, (iter_num - 1) % 2] = cute.arch.mul_packed_f32x2(
                    (
                        tTMrO[j, (iter_num - 1) % 2],
                        tTMrO[j + 1, (iter_num - 1) % 2],
                    ),
                    (scale, scale),
                )
            cute.copy(
                tmem_store_atom,
                tTMrO[None, (iter_num - 1) % 2],
                tTMEM_STOREtO[None, iter_num - 1, 0],
            )
        cute.arch.fence_view_async_tmem_store()
        o_handle.release()
        return mma_o_consumer, s_corr_consumer

    @cute.jit
    def correction_epilog(
        self,
        value_args: Tuple,
        sum_args: Tuple,
        o_args: Tuple,
        epi_tile: cute.Tile,
    ) -> Tuple[pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        seqlen_q, scale_output = value_args
        sum_consumer, sSum = sum_args
        mma_o_consumer, gO_staged, cO_staged, tOtO_staged = o_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        sum_handle = sum_consumer.wait_and_advance()
        row_sum = sSum[thread_idx]
        cute.arch.fence_view_async_shared()
        sum_handle.release()
        row_sum_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
        scale = scale_output / row_sum if not row_sum_is_zero_or_nan else 0.0
        o_handle = mma_o_consumer.wait_and_advance()
        for iter in cutlass.range(self.iterations_pv):
            gO = gO_staged[None, None, iter]
            cO = cO_staged[None, None, iter]
            tOtO = tOtO_staged[(None, None), 0, 0, iter]
            tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
            cO_epi = cute.zipped_divide(cO, epi_tile)
            gO_epi = cute.zipped_divide(gO, epi_tile)
            tidx, _, _ = cute.arch.thread_idx()
            thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
            tmem_copy_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.pv_acc_dtype)
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_epi)
            thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
            tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_epi)
            tTMEM_LOADgO = thr_tmem_load.partition_D(gO_epi)
            tTMEM_LOADcO = thr_tmem_load.partition_D(cO_epi)
            for i in cutlass.range(cute.size(tTMEM_LOADtO, mode=[1]), unroll_full=True):
                tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
                tTMEM_LOADgO_i = tTMEM_LOADgO[None, i, 0]
                tTMEM_LOADcO_i = tTMEM_LOADcO[None, i, 0]
                tTMrO = cute.make_rmem_tensor(tTMEM_LOADcO[None, 0, i].shape, self.pv_acc_dtype)
                cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
                cute.arch.fence_view_async_tmem_load()
                for j in cutlass.range(0, cute.size(tTMrO), 2, unroll_full=True):
                    tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                        (tTMrO[j], tTMrO[j + 1]),
                        (scale, scale),
                    )
                tSMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
                o_vec = tTMrO.load()
                tSMrO.store(o_vec.to(self.o_dtype))
                if cute.elem_less(tTMEM_LOADcO_i[0][0], seqlen_q):
                    cute.autovec_copy(tSMrO, tTMEM_LOADgO_i)
        o_handle.release()
        return mma_o_consumer, sum_consumer

    @cute.jit
    def correction_epilog_tma(
        self,
        scale_output: Float32,
        sum_args: Tuple,
        o_args: Tuple,
    ) -> Tuple[pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        """Finalize one fixed-length O tile through its staged TMA store path."""

        sum_consumer, sSum = sum_args
        (
            mma_o_consumer,
            gO_tma,
            sO,
            tma_atom_o,
            tOtO_staged,
            pv_thr_mma,
            o_store_pipeline,
        ) = o_args
        tidx, _, _ = cute.arch.thread_idx()
        epilogue_threads: cutlass.Constexpr[int] = self.threads_per_warp * len(self.correction_warp_ids)
        thread_idx = tidx % epilogue_threads

        sum_handle = sum_consumer.wait_and_advance()
        row_sum = sSum[thread_idx]
        cute.arch.fence_view_async_shared()
        sum_handle.release()
        row_sum_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
        scale = scale_output / row_sum if not row_sum_is_zero_or_nan else 0.0

        o_handle = mma_o_consumer.wait_and_advance()
        cO = pv_thr_mma.partition_C(cute.make_identity_tensor(self.pv_mma_tiler[:2]))
        store_o, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_o,
            0,
            cute.make_layout(1),
            sO,
            gO_tma,
        )
        tOtO_first = tOtO_staged[(None, None), 0, 0, 0]
        tOtO_first_epi = cute.zipped_divide(tOtO_first, self.o_tma_tile)
        tmem_copy_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.pv_acc_dtype,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(
            tmem_copy_atom,
            tOtO_first_epi,
        )
        smem_copy_atom = sm100_utils.get_smem_store_op(
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            tiled_tmem_load,
        )
        tiled_smem_store = cute.make_tiled_copy_D(
            smem_copy_atom,
            tiled_tmem_load,
        )
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)

        for pv_iter in cutlass.range_constexpr(self.iterations_pv):
            stage: cutlass.Constexpr[int] = pv_iter % self.o_store_stages
            sO_stage = sO[None, None, stage]
            tOtO = tOtO_staged[(None, None), 0, 0, pv_iter]
            tOtO_epi = cute.zipped_divide(tOtO, self.o_tma_tile)
            cO_epi = cute.zipped_divide(cO, self.o_tma_tile)
            sO_epi = cute.zipped_divide(sO_stage, self.o_tma_tile)
            tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_epi)
            tTMEM_LOADcO = thr_tmem_load.partition_D(cO_epi)
            tTMEM_LOADsO = thr_tmem_load.partition_D(sO_epi)
            for i in cutlass.range(
                cute.size(tTMEM_LOADtO, mode=[1]),
                unroll_full=True,
            ):
                tTMrO = cute.make_rmem_tensor(
                    tTMEM_LOADcO[None, 0, i].shape,
                    self.pv_acc_dtype,
                )
                cute.copy(
                    tiled_tmem_load,
                    tTMEM_LOADtO[None, i, 0],
                    tTMrO,
                )
                cute.arch.fence_view_async_tmem_load()
                for j in cutlass.range(0, cute.size(tTMrO), 2, unroll_full=True):
                    tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                        (tTMrO[j], tTMrO[j + 1]),
                        (scale, scale),
                    )
                copy_utils.cvt_copy(
                    tiled_smem_store,
                    tTMrO,
                    tTMEM_LOADsO[None, i, 0],
                )

            cute.arch.fence_proxy("async.shared", space="cta")
            self.epilog_sync_barrier.arrive_and_wait()
            if thread_idx < Int32(self.threads_per_warp):
                store_o(src_idx=stage, dst_idx=pv_iter)
                o_store_pipeline.producer_commit()
                o_store_pipeline.producer_acquire()
            self.epilog_sync_barrier.arrive_and_wait()

        o_handle.release()
        return mma_o_consumer, sum_consumer

    @cute.jit
    def store_sum_max(
        self,
        row_max,
        mLSE,
        row_sum,
        sSum,
        sum_producer,
        current_block_coord,
        seqlen_q,
        mSequenceDescQ,
        cuseqlen_q,
        scale_softmax,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        sum_handle = sum_producer.acquire_and_advance()
        sSum[thread_idx] = row_sum
        cute.arch.fence_view_async_shared()
        sum_handle.commit()
        row_sum_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum

        if cutlass.const_expr(mLSE is not None):
            q_idx = current_block_coord[0] * self.cta_tiler[0] + tidx
            hb_idx = (current_block_coord[2][0], Int32(0)) if cutlass.const_expr(mSequenceDescQ is not None) else current_block_coord[2]
            lse_value = scale_softmax * row_max + cute.math.log(row_sum, fastmath=True) if not row_sum_is_zero_or_nan else -Float32.inf
            if cute.elem_less(q_idx, seqlen_q):
                global_q_idx = q_idx + cuseqlen_q if cutlass.const_expr(mSequenceDescQ is not None) else q_idx
                mLSE[global_q_idx, hb_idx] = lse_value
        return sum_producer
