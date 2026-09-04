# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

"""Fused multi-head attention (FMHA) backward for the SM100 architecture using CUTE DSL.

Constraints:
* Supported head dimensions: 256 only
* cta_tiler_mn must be 64,128
* Batch size must be the same for Q, K, and V tensors
"""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Int32
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

import cuda.bindings.driver as cuda
from cudnn.flex_attention.kernels.common.tile_scheduler import (
    SM100_TMEM_CAPACITY_COLUMNS,
    make_sm100_thread_cooperative_group as make_thread_cooperative_group,
)

import cudnn.flex_attention.kernels.common.copy_utils as fa_copy_utils
from cudnn.flex_attention.plan.kernels import BlockSparseTensors
from cudnn.flex_attention.plan.kernels.packed_mask import (
    apply_loaded_arbitrary_mask,
    get_block_sparse_iteration_info_bwd,
    get_physical_subtile_count_bwd_sm90,
    load_mask_payload,
)
from cudnn.flex_attention.runtime.dsl_utils import struct_scalar_ptr

LAYOUT_RANK_CONSTANT = 3


@cute.jit
def split_wg(
    t: cute.Tensor,
    num_warp_groups: Int32,
    wg_idx: Int32,
) -> cute.Tensor:
    """Split warp group."""
    # Split the staged mode so each warp group owns one contiguous slice.
    ret = None
    if cutlass.const_expr(cute.rank(t.layout) == LAYOUT_RANK_CONSTANT):
        p = cute.composition(
            t,
            cute.make_layout(
                (
                    t.shape[0],
                    t.shape[1],
                    (num_warp_groups, cute.size(t, mode=[2]) // num_warp_groups),
                )
            ),
        )
        ret = p[None, None, (wg_idx, None)]
    else:
        p = cute.composition(
            t,
            cute.make_layout(
                (
                    t.shape[0],
                    t.shape[1],
                    t.shape[2],
                    (num_warp_groups, cute.size(t, mode=[3]) // num_warp_groups),
                )
            ),
        )
        ret = p[None, None, None, (wg_idx, None)]
    return ret


@cute.jit
def _hd256_bwd_sparse_head_state(
    block_sparse_tensors: BlockSparseTensors,
    batch_idx: Int32,
    head_idx: Int32,
    n_block: Int32,
    subtile_factor: cutlass.Constexpr[int],
    m_block_max: Int32,
    n_blocks_per_sample: Int32,
):
    """Load one K2Q row state once when entering a Q head."""

    curr_q_cnt, curr_q_idx, curr_full_cnt, curr_full_idx, partial_base, full_base, _ = get_block_sparse_iteration_info_bwd(
        block_sparse_tensors,
        batch_idx,
        head_idx,
        n_block,
        subtile_factor=subtile_factor,
        n_blocks_per_sample=n_blocks_per_sample,
    )
    partial_count = get_physical_subtile_count_bwd_sm90(
        curr_q_cnt,
        curr_q_idx,
        subtile_factor,
        m_block_max,
    )
    full_count = get_physical_subtile_count_bwd_sm90(
        curr_full_cnt,
        curr_full_idx,
        subtile_factor,
        m_block_max,
    )

    return partial_base, full_base, partial_count, full_count, partial_count + full_count


@cute.jit
def _hd256_bwd_sparse_m_block(
    block_sparse_tensors: BlockSparseTensors,
    iter_idx: Int32,
    partial_base: Int32,
    full_base: Int32,
    partial_count: Int32,
    full_count: Int32,
    subtile_factor: cutlass.Constexpr[int],
    m_block_max: Int32,
):
    """Map a head-local physical iteration using cached K2Q row metadata."""

    mask_block_idx = block_sparse_tensors.mask_block_idx
    full_block_idx = block_sparse_tensors.full_block_idx
    assert full_block_idx is not None
    m_block = Int32(0)
    is_full_block = False
    is_partial_block = iter_idx < partial_count
    payload_idx = partial_base
    q_subtile = Int32(0)
    if is_partial_block:
        sparse_ordinal = iter_idx // subtile_factor
        q_subtile = iter_idx % subtile_factor
        m_block = mask_block_idx[partial_base + sparse_ordinal] * subtile_factor + q_subtile
        payload_idx += sparse_ordinal
    else:
        full_iter_idx = iter_idx - partial_count
        if full_iter_idx < full_count:
            sparse_ordinal = full_iter_idx // subtile_factor
            q_subtile = full_iter_idx % subtile_factor
            m_block = full_block_idx[full_base + sparse_ordinal] * subtile_factor + q_subtile
            is_full_block = True

    m_block_safe = cutlass.min(m_block, m_block_max - 1)
    return (
        m_block,
        m_block_safe,
        is_full_block,
        is_partial_block,
        payload_idx,
        q_subtile,
    )


@cute.jit
def _hd256_bwd_sparse_head_loop_count(
    block_sparse_tensors: BlockSparseTensors,
    batch_idx: Int32,
    head_idx: Int32,
    n_block: Int32,
    subtile_factor: cutlass.Constexpr[int],
    m_block_max: Int32,
    n_blocks_per_sample: Int32,
):
    """Return the physical Q-subtile count for one Q head and K2Q row."""

    curr_q_cnt, curr_q_idx, curr_full_cnt, curr_full_idx, _, _, _ = get_block_sparse_iteration_info_bwd(
        block_sparse_tensors,
        batch_idx,
        head_idx,
        n_block,
        subtile_factor=subtile_factor,
        n_blocks_per_sample=n_blocks_per_sample,
    )
    total_count = get_physical_subtile_count_bwd_sm90(
        curr_q_cnt,
        curr_q_idx,
        subtile_factor,
        m_block_max,
    )
    total_count += get_physical_subtile_count_bwd_sm90(
        curr_full_cnt,
        curr_full_idx,
        subtile_factor,
        m_block_max,
    )
    return total_count


@cute.jit
def _hd256_bwd_sparse_group_loop_count(
    block_sparse_tensors: BlockSparseTensors,
    batch_idx: Int32,
    kv_head_idx: Int32,
    n_block: Int32,
    qhead_per_kvhead: cutlass.Constexpr[int],
    subtile_factor: cutlass.Constexpr[int],
    m_block_max: Int32,
    n_blocks_per_sample: Int32,
):
    """Sum the K2Q work owned by every Q head in one KV group."""

    total_count = Int32(0)
    for qhead_offset in cutlass.range_constexpr(qhead_per_kvhead):
        head_idx = kv_head_idx * qhead_per_kvhead + qhead_offset
        total_count += _hd256_bwd_sparse_head_loop_count(
            block_sparse_tensors,
            batch_idx,
            head_idx,
            n_block,
            subtile_factor,
            m_block_max,
            n_blocks_per_sample,
        )
    return total_count


@cute.jit
def _hd256_bwd_sparse_group_first_work_item(
    block_sparse_tensors: BlockSparseTensors,
    batch_idx: Int32,
    kv_head_idx: Int32,
    n_block: Int32,
    qhead_per_kvhead: cutlass.Constexpr[int],
    subtile_factor: cutlass.Constexpr[int],
    m_block_max: Int32,
    n_blocks_per_sample: Int32,
):
    """Return the first nonempty Q-head work item in one KV group."""

    qhead_offset = Int32(0)
    head_idx = kv_head_idx * qhead_per_kvhead
    partial_base, full_base, partial_count, full_count, head_count = _hd256_bwd_sparse_head_state(
        block_sparse_tensors,
        batch_idx,
        head_idx,
        n_block,
        subtile_factor,
        m_block_max,
        n_blocks_per_sample,
    )
    while head_count <= Int32(0) and qhead_offset + Int32(1) < qhead_per_kvhead:
        qhead_offset += Int32(1)
        head_idx = kv_head_idx * qhead_per_kvhead + qhead_offset
        partial_base, full_base, partial_count, full_count, head_count = _hd256_bwd_sparse_head_state(
            block_sparse_tensors,
            batch_idx,
            head_idx,
            n_block,
            subtile_factor,
            m_block_max,
            n_blocks_per_sample,
        )
    return (
        qhead_offset,
        Int32(0),
        head_count,
        partial_base,
        full_base,
        partial_count,
        full_count,
    )


@cute.jit
def _hd256_bwd_sparse_group_next_work_item(
    block_sparse_tensors: BlockSparseTensors,
    batch_idx: Int32,
    kv_head_idx: Int32,
    n_block: Int32,
    qhead_offset: Int32,
    head_iter_idx: Int32,
    head_count: Int32,
    partial_base: Int32,
    full_base: Int32,
    partial_count: Int32,
    full_count: Int32,
    qhead_per_kvhead: cutlass.Constexpr[int],
    subtile_factor: cutlass.Constexpr[int],
    m_block_max: Int32,
    n_blocks_per_sample: Int32,
):
    """Advance sequentially without rescanning every Q head on each iteration."""

    head_iter_idx += Int32(1)
    if head_iter_idx >= head_count:
        qhead_offset += Int32(1)
        head_iter_idx = Int32(0)
        head_count = Int32(0)
        if qhead_offset < qhead_per_kvhead:
            head_idx = kv_head_idx * qhead_per_kvhead + qhead_offset
            partial_base, full_base, partial_count, full_count, head_count = _hd256_bwd_sparse_head_state(
                block_sparse_tensors,
                batch_idx,
                head_idx,
                n_block,
                subtile_factor,
                m_block_max,
                n_blocks_per_sample,
            )
        while head_count <= Int32(0) and qhead_offset + Int32(1) < qhead_per_kvhead:
            qhead_offset += Int32(1)
            head_idx = kv_head_idx * qhead_per_kvhead + qhead_offset
            partial_base, full_base, partial_count, full_count, head_count = _hd256_bwd_sparse_head_state(
                block_sparse_tensors,
                batch_idx,
                head_idx,
                n_block,
                subtile_factor,
                m_block_max,
                n_blocks_per_sample,
            )
    return (
        qhead_offset,
        head_iter_idx,
        head_count,
        partial_base,
        full_base,
        partial_count,
        full_count,
    )


class BlackwellFusedMultiHeadAttentionBackwardDKDVKernel:
    """FMHA backward class for executing CuTeDSL kernel."""

    def __init__(
        self,
        qhead_per_kvhead: int,
    ):
        """Initialization."""
        self.acc_dtype = cutlass.Float32
        self.cta_tiler = (128, 64, 256)
        self.tile_shape_Q = self.cta_tiler[0]
        self.tile_shape_K = self.cta_tiler[1]
        self.tile_shape_dQ_K = self.cta_tiler[2]
        self.tile_shape_dV_dO = self.cta_tiler[2]
        # For S
        self.KQ_mma_tiler = (
            self.cta_tiler[1] * 2,
            self.cta_tiler[0],
            self.cta_tiler[2],
        )
        # For dP
        self.VdO_mma_tiler = (
            self.cta_tiler[1] * 2,
            self.cta_tiler[0],
            self.cta_tiler[2],
        )
        # For dV
        self.PdO_mma_tiler = (
            self.cta_tiler[1] * 2,
            self.cta_tiler[2],
            self.cta_tiler[0],
        )
        # For dK
        self.dSQ_mma_tiler = (
            self.cta_tiler[1] * 2,
            self.cta_tiler[2],
            self.cta_tiler[0],
        )
        self.cluster_shape_mn = (2, 1)
        self.qhead_per_kvhead = qhead_per_kvhead
        self.mask_payload_valid_words = 1
        self.mask_payload_padded_words = 1
        self.subtile_factor = 2
        assert self.qhead_per_kvhead >= 1

        self.compute_warp_id = (0, 1, 2, 3, 4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_id = 9

        self.num_compute_warps = 8

        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * (self.num_compute_warps + 4)

        self.cta_sync_bar_id = 0
        self.tmem_alloc_sync_bar_id = 1
        self.compute_sync_bar_id = 2
        self.epilogue_sync_bar_id = 3

        self.tmem_dK_offset = 0
        self.tmem_dV_offset = self.cta_tiler[2] // 2
        self.tmem_dP_offset = self.cta_tiler[2] + self.cta_tiler[0] // 2
        self.tmem_S_offset = self.cta_tiler[2]

        self.num_regs_compute = 128
        self.num_regs_mma = 128
        self.num_regs_empty = 96
        self.num_regs_load = 96

        self.buffer_align_bytes = 128

    def _setup_attributes(self):
        """Settings for pipeline stage."""
        self.load_mma_Q_stage = 1
        self.load_mma_K_stage = 1
        self.load_mma_V_stage = 1
        self.load_mma_QT_stage = 1
        self.load_mma_dO_stage = 1
        self.load_compute_LSE_stage = 1
        self.load_compute_sum_OdO_stage = 1
        self.mma_compute_S_stage = 1
        self.mma_compute_dP_stage = 1
        self.compute_mma_P_stage = 1
        self.compute_mma_dS_stage = 1
        self.mma_compute_dKdV_stage = 2

    @cute.jit
    def __call__(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        dO: cute.Tensor,
        scaled_LSE: cute.Tensor,
        sum_OdO: cute.Tensor,
        cumulative_s_q: cute.Tensor | None,
        cumulative_s_k: cute.Tensor | None,
        scale_softmax: cutlass.Float32,
        block_sparse_tensors: BlockSparseTensors,
        max_seqlen_k_runtime: Int32,
        stream: cuda.CUstream,
    ):
        """Host function to launch CuTeDSL kernel."""
        assert block_sparse_tensors.mask_block_offset is not None
        assert (cumulative_s_q is None) == (cumulative_s_k is None), "SM100 hd256 arbitrary dKdV varlen requires both cu_seqlens tensors"
        varlen = cumulative_s_q is not None
        if cutlass.const_expr(cumulative_s_k is not None):
            assert block_sparse_tensors.cu_total_m_blocks is not None, "SM100 hd256 arbitrary varlen dKdV requires compact K128 row prefixes"
        assert len(block_sparse_tensors.mask_block_cnt.shape) == 2, "SM100 hd256 arbitrary dKdV requires compact rank-2 K2Q counts"
        assert block_sparse_tensors.mask_block_masks is not None
        assert len(block_sparse_tensors.mask_block_masks.shape) == 4, "SM100 hd256 arbitrary dKdV payload must be rank 4"
        # Infer shape metadata from normalized 5D tensors (B, S, H_k, H_r, D).
        h_r = Q.shape[3]
        h_k = Q.shape[2]
        if cutlass.const_expr(cumulative_s_q is not None):
            b = cumulative_s_q.shape[0] - 1
        else:
            b = Q.shape[0]
        problem_shape = (
            Q.shape[1],
            K.shape[1],
            Q.shape[4],
            ((h_r, h_k), b),
        )
        hb = ((h_r, h_k), b)
        # (b, s, h_k, h_r, d) -> (s, d, ((h_r, h_k), b))
        Q = cute.make_tensor(
            Q.iterator,
            cute.make_layout(
                (Q.shape[1], Q.shape[4], hb),
                stride=(
                    cute.assume(Q.stride[1], divby=64),
                    Q.stride[4],
                    (
                        (Q.stride[3], Q.stride[2]),
                        0 if cumulative_s_q is not None else cute.assume(Q.stride[0], divby=64),
                    ),
                ),
            ),
        )
        # (b, s, h_k, 1, d) -> (s, d, ((1, h_k), b))
        K = cute.make_tensor(
            K.iterator,
            cute.make_layout(
                (K.shape[1], K.shape[4], hb),
                stride=(
                    cute.assume(K.stride[1], divby=64),
                    K.stride[4],
                    (
                        (0, K.stride[2]),
                        0 if cumulative_s_k is not None else cute.assume(K.stride[0], divby=64),
                    ),
                ),
            ),
        )
        # (b, s, h_k, 1, d) -> (s, d, ((1, h_k), b))
        V = cute.make_tensor(
            V.iterator,
            cute.make_layout(
                (V.shape[1], V.shape[4], hb),
                stride=(
                    cute.assume(V.stride[1], divby=64),
                    V.stride[4],
                    (
                        (0, V.stride[2]),
                        0 if cumulative_s_k is not None else cute.assume(V.stride[0], divby=64),
                    ),
                ),
            ),
        )
        # (s, d, ((h_r, h_k), b)) -> (d, s, ((h_r, h_k), b))
        QT = cute.make_tensor(
            Q.iterator,
            cute.make_layout(
                (Q.shape[1], Q.shape[0], Q.shape[2]),
                stride=(
                    Q.stride[1],
                    Q.stride[0],
                    Q.stride[2],
                ),
            ),
        )
        dK = cute.make_tensor(
            dK.iterator,
            cute.make_layout(
                (dK.shape[1], dK.shape[4], hb),
                stride=(
                    cute.assume(dK.stride[1], divby=64),
                    dK.stride[4],
                    (
                        (0, dK.stride[2]),
                        0 if cumulative_s_k is not None else cute.assume(dK.stride[0], divby=64),
                    ),
                ),
            ),
        )
        dV = cute.make_tensor(
            dV.iterator,
            cute.make_layout(
                (dV.shape[1], dV.shape[4], hb),
                stride=(
                    cute.assume(dV.stride[1], divby=64),
                    dV.stride[4],
                    (
                        (0, dV.stride[2]),
                        0 if cumulative_s_k is not None else cute.assume(dV.stride[0], divby=64),
                    ),
                ),
            ),
        )
        # (s, d, ((h_r, h_k), b))
        dO = cute.make_tensor(
            dO.iterator,
            cute.make_layout(
                (dO.shape[1], dO.shape[4], hb),
                stride=(
                    cute.assume(dO.stride[1], divby=64),
                    dO.stride[4],
                    (
                        (dO.stride[3], dO.stride[2]),
                        0 if cumulative_s_q is not None else cute.assume(dO.stride[0], divby=64),
                    ),
                ),
            ),
        )

        # (s, d, ((h_r, h_k), b)) -> (d, s, ((h_r, h_k), b))
        dOT = cute.make_tensor(
            dO.iterator,
            cute.make_layout(
                (dO.shape[1], dO.shape[0], dO.shape[2]),
                stride=(
                    dO.stride[1],
                    dO.stride[0],
                    dO.stride[2],
                ),
            ),
        )

        self.Q_major_mode = utils.LayoutEnum.from_tensor(Q).mma_major_mode()
        self.K_major_mode = utils.LayoutEnum.from_tensor(K).mma_major_mode()
        self.dK_major_mode = utils.LayoutEnum.from_tensor(dK).mma_major_mode()
        self.V_major_mode = utils.LayoutEnum.from_tensor(V).mma_major_mode()
        self.dV_major_mode = utils.LayoutEnum.from_tensor(dV).mma_major_mode()

        if cutlass.const_expr(self.Q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError(f"The layout of q is not supported: {self.Q_major_mode}")
        if cutlass.const_expr(self.K_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.dK_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of dk is not supported")
        if cutlass.const_expr(self.V_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of v is not supported")
        if cutlass.const_expr(self.dV_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of dv is not supported")

        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO
        PT_source = tcgen05.OperandSource.SMEM

        # compute S
        KQ_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            K.element_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.KQ_mma_tiler[:2],
        )
        # compute dP
        VdO_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            V.element_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.VdO_mma_tiler[:2],
        )
        # compute dV
        PdO_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            dO.element_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.PdO_mma_tiler[:2],
            PT_source,
        )
        # compute dK
        dSQ_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            Q.element_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.dSQ_mma_tiler[:2],
        )
        atom_thr_size = cute.size(KQ_tiled_mma.thr_id.shape)
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)  # type: ignore[assignment]
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (atom_thr_size,),
        )

        K_smem_layout_staged = sm100_utils.make_smem_layout_a(
            KQ_tiled_mma,
            self.KQ_mma_tiler,
            K.element_type,
            1,
        )
        Q_smem_layout_staged = sm100_utils.make_smem_layout_b(
            KQ_tiled_mma,
            self.KQ_mma_tiler,
            Q.element_type,
            self.load_mma_Q_stage,
        )
        V_smem_layout_staged = sm100_utils.make_smem_layout_a(
            VdO_tiled_mma,
            self.VdO_mma_tiler,
            V.element_type,
            1,
        )
        dO_smem_layout_staged = sm100_utils.make_smem_layout_b(
            VdO_tiled_mma,
            self.VdO_mma_tiler,
            dO.element_type,
            self.load_mma_dO_stage,
        )
        dST_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dSQ_tiled_mma,
            self.dSQ_mma_tiler,
            Q.element_type,
            self.compute_mma_dS_stage,
        )
        QT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dSQ_tiled_mma,
            self.dSQ_mma_tiler,
            Q.element_type,
            self.load_mma_QT_stage,
        )
        P_smem_layout_staged = sm100_utils.make_smem_layout_a(
            PdO_tiled_mma,
            self.PdO_mma_tiler,
            Q.element_type,
            self.compute_mma_P_stage,
        )
        dOT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            PdO_tiled_mma,
            self.PdO_mma_tiler,
            dO.element_type,
            self.load_mma_dO_stage,
        )
        LSE_smem_layout = cute.make_layout((self.cta_tiler[0], self.load_compute_LSE_stage))
        sum_OdO_smem_layout = cute.make_layout((self.cta_tiler[0], self.load_compute_sum_OdO_stage))

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)

        K_smem_layout = cute.select(K_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            K,
            K_smem_layout,
            self.KQ_mma_tiler,
            KQ_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        V_smem_layout = cute.select(V_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            V,
            V_smem_layout,
            self.VdO_mma_tiler,
            VdO_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        Q_smem_layout = cute.select(Q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            Q,
            Q_smem_layout,
            self.KQ_mma_tiler,
            KQ_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        QT_smem_layout = cute.select(QT_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_QT, tma_tensor_QT = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            QT,
            QT_smem_layout,
            self.dSQ_mma_tiler,
            dSQ_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        dO_smem_layout = cute.select(dO_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            dO,
            dO_smem_layout,
            self.VdO_mma_tiler,
            VdO_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        dOT_smem_layout = cute.select(dOT_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dOT, tma_tensor_dOT = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            dOT,
            dOT_smem_layout,
            self.PdO_mma_tiler,
            PdO_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # for 2cta, tma_copy_QT_bytes is same as the tma_copy_Q_bytes
        self.tma_copy_Q_bytes = cute.size_in_bytes(Q.element_type, Q_smem_layout) * atom_thr_size
        self.tma_copy_K_bytes = cute.size_in_bytes(K.element_type, K_smem_layout) * atom_thr_size
        self.tma_copy_V_bytes = cute.size_in_bytes(V.element_type, V_smem_layout) * atom_thr_size
        self.tma_copy_dO_bytes = cute.size_in_bytes(dO.element_type, dO_smem_layout) * atom_thr_size

        # Each compute warp group owns half the dK/dV output. Both groups stage
        # their slices in CTA-shared memory before the TMA stores.
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        num_compute_wgs = self.num_compute_warps // 4
        # For BF16 D256 this forms four (64, 64) stages, covering a logical
        # (64, 256) CTA buffer with one TMA store per stage.
        epi_cols_dKV = math.gcd(128 // (dK.element_type.width // 8), self.cta_tiler[2] // num_compute_wgs)
        num_epi_stages_dKV = (self.cta_tiler[2] // num_compute_wgs) // epi_cols_dKV
        epi_tile_dKV = (self.cta_tiler[1], epi_cols_dKV)
        total_epi_stages = num_compute_wgs * num_epi_stages_dKV
        dK_layout_enum = utils.LayoutEnum.from_tensor(dK)
        dV_layout_enum = utils.LayoutEnum.from_tensor(dV)
        sdK_epi_layout = sm100_utils.make_smem_layout_epi(
            dK.element_type,
            dK_layout_enum,
            epi_tile_dKV,
            total_epi_stages,
        )
        sdV_epi_layout = sm100_utils.make_smem_layout_epi(
            dV.element_type,
            dV_layout_enum,
            epi_tile_dKV,
            total_epi_stages,
        )
        tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
            tma_store_op,
            dK,
            cute.select(sdK_epi_layout, mode=[0, 1]),
            epi_tile_dKV,
        )
        tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
            tma_store_op,
            dV,
            cute.select(sdV_epi_layout, mode=[0, 1]),
            epi_tile_dKV,
        )

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            load_mma_Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_Q_stage * 2]
            load_mma_K_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_K_stage * 2]
            load_mma_V_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_V_stage * 2]
            load_mma_QT_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_QT_stage * 2]
            load_mma_dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_dO_stage * 2]
            load_mma_dOT_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_dO_stage * 2]
            load_compute_lse_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_compute_LSE_stage * 2]
            load_compute_sum_OdO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_compute_sum_OdO_stage * 2]
            mma_compute_S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_S_stage * 2]
            mma_compute_dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_dP_stage * 2]
            compute_mma_P_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.compute_mma_P_stage * 2]
            compute_mma_dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.compute_mma_dS_stage * 2]
            mma_compute_dKdV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_dKdV_stage * 2]
            tmem_holding_buf: cutlass.Int32
            tmem_dealloc_mbar: cutlass.Int64
            # Smem tensors
            sK: cute.struct.Align[
                cute.struct.MemRange[K.element_type, cute.cosize(K_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # only used in 2cta
            sV: cute.struct.Align[
                cute.struct.MemRange[V.element_type, cute.cosize(V_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[Q.element_type, cute.cosize(Q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sQT: cute.struct.Align[
                cute.struct.MemRange[Q.element_type, cute.cosize(QT_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[dO.element_type, cute.cosize(dO_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sdOT: cute.struct.Align[
                cute.struct.MemRange[dO.element_type, cute.cosize(dOT_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # Used by the 2CTA transpose path.
            sP: cute.struct.Align[
                cute.struct.MemRange[Q.element_type, cute.cosize(P_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sdST: cute.struct.Align[
                cute.struct.MemRange[Q.element_type, cute.cosize(dST_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(LSE_smem_layout)],
                self.buffer_align_bytes,
            ]
            sSum_OdO: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(sum_OdO_smem_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # =============================== bwd ===============================
        K_val = problem_shape[1]
        if cutlass.const_expr(varlen):
            K_val = max_seqlen_k_runtime
        grid_problem_shape = (
            problem_shape[0],
            K_val,
            problem_shape[2],
            problem_shape[3],
        )
        bwd_grid = self._compute_bwd_grid(grid_problem_shape, self.cta_tiler[1])
        bwd_grid = cute.round_up(bwd_grid, self.cluster_shape_mnk)

        self.dkdv_bwd(
            KQ_tiled_mma,
            VdO_tiled_mma,
            PdO_tiled_mma,
            dSQ_tiled_mma,
            tma_atom_K,
            tma_tensor_K,
            K,
            tma_atom_V,
            tma_tensor_V,
            tma_atom_Q,
            tma_tensor_Q,
            Q,
            tma_atom_QT,
            tma_tensor_QT,
            tma_atom_dO,
            tma_tensor_dO,
            tma_atom_dOT,
            tma_tensor_dOT,
            dK,
            dV,
            tma_atom_dK,
            tma_tensor_dK,
            tma_atom_dV,
            tma_tensor_dV,
            scaled_LSE,
            scale_softmax,
            sum_OdO,
            problem_shape,
            cumulative_s_q,
            cumulative_s_k,
            self.cluster_layout_vmnk,
            K_smem_layout_staged,
            Q_smem_layout_staged,
            V_smem_layout_staged,
            dO_smem_layout_staged,
            dST_smem_layout_staged,
            QT_smem_layout_staged,
            dOT_smem_layout_staged,
            P_smem_layout_staged,
            LSE_smem_layout,
            sum_OdO_smem_layout,
            sdK_epi_layout,
            sdV_epi_layout,
            block_sparse_tensors,
        ).launch(
            grid=bwd_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),  # type: ignore [attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def dkdv_bwd(
        self,
        KQ_tiled_mma: cute.TiledMma,
        VdO_tiled_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        K_in: cute.Tensor,
        K_ref: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        V_in: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        Q_in: cute.Tensor,
        Q_ref: cute.Tensor,
        tma_atom_QT: cute.CopyAtom,
        QT_in: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        dO_in: cute.Tensor,
        tma_atom_dOT: cute.CopyAtom,
        dOT_in: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        tma_atom_dK: cute.CopyAtom,
        dK_tma: cute.Tensor,
        tma_atom_dV: cute.CopyAtom,
        dV_tma: cute.Tensor,
        LSE: cute.Tensor,
        scale_softmax: cutlass.Float32,
        sum_OdO: cute.Tensor,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        cumulative_s_q: cute.Tensor | None,
        cumulative_s_k: cute.Tensor | None,
        cluster_layout_vmnk: cute.Layout,
        K_smem_layout_staged: cute.ComposedLayout,
        Q_smem_layout_staged: cute.ComposedLayout,
        V_smem_layout_staged: cute.ComposedLayout,
        dO_smem_layout_staged: cute.ComposedLayout,
        dST_smem_layout_staged: cute.ComposedLayout,
        QT_smem_layout_staged: cute.ComposedLayout,
        dOT_smem_layout_staged: cute.ComposedLayout,
        P_smem_layout_staged: cute.ComposedLayout,
        LSE_smem_layout: cute.Layout,
        sum_OdO_smem_layout: cute.Layout,
        sdK_epi_layout: cute.ComposedLayout,
        sdV_epi_layout: cute.ComposedLayout,
        block_sparse_tensors: Optional[BlockSparseTensors],
    ):
        """Core CuTeDSL backward kernel."""
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        varlen = cumulative_s_q is not None

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_QT)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_dO)
            cpasync.prefetch_descriptor(tma_atom_dOT)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_mma_Q_producer, load_mma_Q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_Q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_Q_bytes,
            barrier_storage=storage.load_mma_Q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_K_producer, load_mma_K_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_K_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_K_bytes,
            barrier_storage=storage.load_mma_K_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_V_producer, load_mma_V_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_V_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_V_bytes,
            barrier_storage=storage.load_mma_V_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_QT_producer, load_mma_QT_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_QT_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_Q_bytes,
            barrier_storage=storage.load_mma_QT_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_dO_producer, load_mma_dO_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_dO_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_dO_bytes,
            barrier_storage=storage.load_mma_dO_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_dOT_producer, load_mma_dOT_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_dO_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_dO_bytes,
            barrier_storage=storage.load_mma_dOT_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_compute_LSE_producer, load_compute_LSE_consumer = pipeline.PipelineCpAsync.create(
            num_stages=self.load_compute_LSE_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * self.num_compute_warps),
            barrier_storage=storage.load_compute_lse_mbar_ptr.data_ptr(),
        ).make_participants()
        load_compute_sum_OdO_producer, load_compute_sum_OdO_consumer = pipeline.PipelineCpAsync.create(
            num_stages=self.load_compute_sum_OdO_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * self.num_compute_warps),
            barrier_storage=storage.load_compute_sum_OdO_mbar_ptr.data_ptr(),
        ).make_participants()
        mma_compute_S_producer, mma_compute_S_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_compute_S_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]),
            barrier_storage=storage.mma_compute_S_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_compute_dP_producer, mma_compute_dP_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_compute_dP_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]),
            barrier_storage=storage.mma_compute_dP_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        compute_mma_P_producer, compute_mma_P_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.compute_mma_P_stage,
            producer_group=make_thread_cooperative_group(self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.compute_mma_P_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        compute_mma_dS_producer, compute_mma_dS_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.compute_mma_dS_stage,
            producer_group=make_thread_cooperative_group(self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.compute_mma_dS_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_compute_dKdV_producer, mma_compute_dKdV_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_compute_dKdV_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]),
            barrier_storage=storage.mma_compute_dKdV_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        cute.arch.barrier(barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta)

        # setup mma
        sQ = storage.sQ.get_tensor(Q_smem_layout_staged.outer, swizzle=Q_smem_layout_staged.inner)
        sK = storage.sK.get_tensor(K_smem_layout_staged.outer, swizzle=K_smem_layout_staged.inner)
        sV = storage.sV.get_tensor(V_smem_layout_staged.outer, swizzle=V_smem_layout_staged.inner)
        sdO = storage.sdO.get_tensor(dO_smem_layout_staged.outer, swizzle=dO_smem_layout_staged.inner)
        sLSE = storage.sLSE.get_tensor(LSE_smem_layout)
        sSum_OdO = storage.sSum_OdO.get_tensor(sum_OdO_smem_layout)
        # for 2cta, QT use different mem from Q

        sQT = storage.sQT.get_tensor(QT_smem_layout_staged.outer, swizzle=QT_smem_layout_staged.inner)
        sdST = storage.sdST.get_tensor(dST_smem_layout_staged.outer, swizzle=dST_smem_layout_staged.inner)

        sP = storage.sP.get_tensor(P_smem_layout_staged.outer, swizzle=P_smem_layout_staged.inner)

        sdOT = storage.sdOT.get_tensor(dOT_smem_layout_staged.outer, swizzle=dOT_smem_layout_staged.inner)

        # tSTrK shape : (MMA, MMA_M, MMA_K, STAGE)
        tSTrK = KQ_tiled_mma.make_fragment_A(sK)
        # tSTrQ shape : (MMA, MMA_N, MMA_K, STAGE)
        tSTrQ = KQ_tiled_mma.make_fragment_B(sQ)

        # tdPTrV shape : (MMA, MMA_M, MMA_K, STAGE)
        tdPTrV = VdO_tiled_mma.make_fragment_A(sV)
        # tdPTrdO shape : (MMA, MMA_N, MMA_K, STAGE)
        tdPTrdO = VdO_tiled_mma.make_fragment_B(sdO)

        # tdKrdST shape: (MMA, MMA_M, MMA_K, STAGE)
        tdKrdST = dSQ_tiled_mma.make_fragment_A(sdST)
        # tdKrQT shape : (MMA, MMA_N, MMA_K, STAGE)
        tdKrQT = dSQ_tiled_mma.make_fragment_B(sQT)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=self.threads_per_cta,
        )

        tmem = utils.TmemAllocator(
            struct_scalar_ptr(storage.tmem_holding_buf),
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.load_warp_id,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=struct_scalar_ptr(storage.tmem_dealloc_mbar),
        )

        tmem.allocate(self.tmem_alloc_cols)

        # wait for tmem allocation and retrieve the pointer
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        # Cluster arrive after barrier init
        # is_relaxed=False has memory consistency guarantee
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=False)

        tSTtST_shape = KQ_tiled_mma.partition_shape_C(cute.select(self.KQ_mma_tiler, mode=[0, 1]))
        tSTtST = KQ_tiled_mma.make_fragment_C(tSTtST_shape)
        # tSTtST shape : (MMA, MMA_M, MMA_N)
        tSTtST = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tSTtST.layout)

        # tdVrP shape : (MMA, MMA_M, MMA_K, STAGE)
        tdVrP = PdO_tiled_mma.make_fragment_A(sP)
        # tdVrdOT shape : (MMA, MMA_N, MMA_K, STAGE)
        tdVrdOT = PdO_tiled_mma.make_fragment_B(sdOT)

        tdPTtdPT_shape = VdO_tiled_mma.partition_shape_C(cute.select(self.VdO_mma_tiler, mode=[0, 1]))
        tdPTtdPT = VdO_tiled_mma.make_fragment_C(tdPTtdPT_shape)
        # tdPTtdPT shape : (MMA, MMA_M, MMA_N)
        tdPTtdPT = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPTtdPT.layout)

        tdKtdK_shape = dSQ_tiled_mma.partition_shape_C(cute.select(self.dSQ_mma_tiler, mode=[0, 1]))
        tdKtdK = dSQ_tiled_mma.make_fragment_C(tdKtdK_shape)
        # tdKtdK shape : (MMA, MMA_M, MMA_N)
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)

        tdVtdV_shape = PdO_tiled_mma.partition_shape_C(cute.select(self.PdO_mma_tiler, mode=[0, 1]))
        tdVtdV = PdO_tiled_mma.make_fragment_C(tdVtdV_shape)
        # tdVtdV shape : (MMA, MMA_M, MMA_N)
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)

        # Get the current batch problem shape.
        blk_coord = (Int32(0), bidx, Int32(0), ((Int32(0), bidy), bidz))
        seqlen_q_cur_batch = Q_ref.shape[0]
        seqlen_k_cur_batch = K_ref.shape[0]
        blk_offset = (Int32(0), Int32(0), Int32(0), ((Int32(0), Int32(0)), Int32(0)))
        if cutlass.const_expr(varlen):
            assert isinstance(cumulative_s_q, cute.Tensor)
            assert isinstance(cumulative_s_k, cute.Tensor)
            seqlen_q_cur_batch = cumulative_s_q[bidz + 1] - cumulative_s_q[bidz]
            seqlen_k_cur_batch = cumulative_s_k[bidz + 1] - cumulative_s_k[bidz]
            blk_offset = (
                cumulative_s_q[bidz],
                cumulative_s_k[bidz],
                Int32(0),
                ((Int32(0), Int32(0)), Int32(0)),
            )

        iter_start, iter_end = self.get_Q_block_min_max(
            seqlen_q_cur_batch,
        )
        m_block_max = cute.ceil_div(seqlen_q_cur_batch, self.tile_shape_Q)
        assert block_sparse_tensors is not None
        n_block_sparse = blk_coord[1] // cute.size(KQ_tiled_mma.thr_id.shape)
        sparse_loop_count = Int32(0)
        valid_sparse_row = n_block_sparse * self.KQ_mma_tiler[0] < seqlen_k_cur_batch
        if valid_sparse_row:
            sparse_loop_count = _hd256_bwd_sparse_group_loop_count(
                block_sparse_tensors,
                bidz,
                bidy,
                n_block_sparse,
                self.qhead_per_kvhead,
                self.subtile_factor,
                m_block_max,
                cute.ceil_div(problem_shape[1], self.KQ_mma_tiler[0]),
            )
        iter_start = Int32(0)
        iter_end = sparse_loop_count

        # Cluster wait
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        iter_count = iter_end - iter_start
        problem_shape_cur_batch = (
            seqlen_q_cur_batch,
            seqlen_k_cur_batch,
            problem_shape[2],
            problem_shape[3],
        )
        if iter_count <= 0:
            if bidx * self.tile_shape_K < seqlen_k_cur_batch:
                self.epilogue_clear(
                    blk_coord,
                    blk_offset,
                    problem_shape_cur_batch,
                    dK,
                    dV,
                )
        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_load)

            self.load(
                K_in,
                V_in,
                Q_in,
                QT_in,
                dO_in,
                dOT_in,
                LSE,
                sum_OdO,
                sK,
                sQ,
                sQT,
                sV,
                sdO,
                sdOT,
                sLSE,
                sSum_OdO,
                KQ_tiled_mma,
                VdO_tiled_mma,
                PdO_tiled_mma,
                dSQ_tiled_mma,
                tma_atom_K,
                tma_atom_Q,
                tma_atom_QT,
                tma_atom_V,
                tma_atom_dO,
                tma_atom_dOT,
                blk_offset,
                problem_shape_cur_batch,
                varlen,
                iter_count,
                iter_start,
                load_mma_Q_producer,
                load_mma_K_producer,
                load_mma_V_producer,
                load_compute_LSE_producer,
                load_mma_dO_producer,
                load_mma_dOT_producer,
                load_compute_sum_OdO_producer,
                load_mma_QT_producer,
                block_sparse_tensors,
                m_block_max,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_alloc(self.num_regs_mma)

            self.mma_2cta(
                KQ_tiled_mma,
                VdO_tiled_mma,
                PdO_tiled_mma,
                dSQ_tiled_mma,
                tSTtST,
                tSTrQ,
                tSTrK,
                tdPTtdPT,
                tdPTrV,
                tdPTrdO,
                tdVtdV,
                tdVrP,
                tdVrdOT,
                tdKrdST,
                tdKtdK,
                tdKrQT,
                iter_count,
                load_mma_Q_consumer,
                load_mma_K_consumer,
                load_mma_V_consumer,
                mma_compute_S_producer,
                load_mma_dO_consumer,
                mma_compute_dP_producer,
                load_mma_dOT_consumer,
                compute_mma_P_consumer,
                compute_mma_dS_consumer,
                load_mma_QT_consumer,
                mma_compute_dKdV_producer,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Compute
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_idx >= self.compute_warp_id[0] and warp_idx <= self.compute_warp_id[-1]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_compute)

            self.compute(
                tSTtST,
                tdPTtdPT,
                sP,
                sLSE,
                sdST,
                sdOT,
                sSum_OdO,
                dK,
                dV,
                tdKtdK,
                tdVtdV,
                blk_coord,
                blk_offset,
                problem_shape_cur_batch,
                iter_count,
                scale_softmax,
                mma_compute_S_consumer,
                compute_mma_P_producer,
                load_compute_LSE_consumer,
                load_compute_sum_OdO_consumer,
                mma_compute_dP_consumer,
                compute_mma_dS_producer,
                mma_compute_dKdV_consumer,
                varlen,
                seqlen_k_cur_batch,
                tma_atom_dK,
                dK_tma,
                tma_atom_dV,
                dV_tma,
                sdK_epi_layout,
                sdV_epi_layout,
                block_sparse_tensors,
                m_block_max,
            )

            cute.arch.barrier(
                barrier_id=self.epilogue_sync_bar_id,
                number_of_threads=self.num_compute_warps * self.threads_per_warp,
            )

        else:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        # Deallocate TMEM before the empty-work early exit.
        # Dealloc the tensor memory
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    @cute.jit
    def get_Q_block_min_max(
        self,
        seq_Q: Int32,
    ):
        """Get Q tiles range."""
        return cutlass.Int32(0), cute.ceil_div(seq_Q, self.tile_shape_Q)

    @cute.jit
    def load(
        self,
        K_in: cute.Tensor,
        V_in: cute.Tensor,
        Q_in: cute.Tensor,
        QT_in: cute.Tensor,
        dO_in: cute.Tensor,
        dOT_in: cute.Tensor,
        LSE_in: cute.Tensor,
        sum_OdO_in: cute.Tensor,
        sK: cute.Tensor,
        sQ: cute.Tensor,
        sQT: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdOT: cute.Tensor,
        sLSE: cute.Tensor,
        sSum_OdO: cute.Tensor,
        KQ_tiled_mma: cute.TiledMma,
        VdO_tiled_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_QT: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dOT: cute.CopyAtom,
        blk_offset: cute.Shape,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        varlen: bool,
        iter_count: Int32,
        iter_start: Int32,
        load_mma_Q_producer,
        load_mma_K_producer,
        load_mma_V_producer,
        load_compute_LSE_producer,
        load_mma_dO_producer,
        load_mma_dOT_producer,
        load_compute_sum_OdO_producer,
        load_mma_QT_producer,
        block_sparse_tensors: Optional[BlockSparseTensors] = None,
        m_block_max: Int32 = Int32(0),
    ):
        """TMA load."""
        tidx, _, _ = cute.arch.thread_idx()
        blk_coord_k, blk_coord_h_k, blk_coord_b = cute.arch.block_idx()
        blk_coord_h_r = Int32(0)
        blk_coord_h = (blk_coord_h_r, blk_coord_h_k)
        iter_index = iter_start
        mma_tile_coord_v = blk_coord_k % cute.size(KQ_tiled_mma.thr_id.shape)
        mma_tile_coord_m = blk_coord_k // cute.size(KQ_tiled_mma.thr_id.shape)

        K = cute.domain_offset(cute.select(blk_offset, mode=[1, 2, 3]), K_in)
        V = cute.domain_offset(cute.select(blk_offset, mode=[1, 2, 3]), V_in)
        Q = cute.domain_offset(cute.select(blk_offset, mode=[0, 2, 3]), Q_in)
        QT = cute.domain_offset(cute.select(blk_offset, mode=[2, 0, 3]), QT_in)
        dO = cute.domain_offset(cute.select(blk_offset, mode=[0, 2, 3]), dO_in)
        dOT = cute.domain_offset(cute.select(blk_offset, mode=[2, 0, 3]), dOT_in)
        blk_offset_stats = blk_offset
        if cutlass.const_expr(varlen):
            cuseqlen_q_stats = cute.assume(
                (blk_offset[0] + blk_coord_b * self.tile_shape_Q) // self.tile_shape_Q * self.tile_shape_Q,
                divby=self.tile_shape_Q,
            )
            blk_offset_stats = (
                cuseqlen_q_stats,
                blk_offset[1],
                blk_offset[2],
                blk_offset[3],
            )
        LSE = cute.domain_offset(cute.select(blk_offset_stats, mode=[0, 3]), LSE_in)
        sum_OdO = cute.domain_offset(cute.select(blk_offset_stats, mode=[0, 3]), sum_OdO_in)

        gK = cute.local_tile(K, cute.select(self.KQ_mma_tiler, mode=[0, 2]), (None, None, None))
        gQ = cute.local_tile(Q, cute.select(self.KQ_mma_tiler, mode=[1, 2]), (None, None, None))
        gQT = cute.local_tile(QT, cute.select(self.dSQ_mma_tiler, mode=[1, 2]), (None, None, None))
        gV = cute.local_tile(V, cute.select(self.VdO_mma_tiler, mode=[0, 2]), (None, None, None))
        gdO = cute.local_tile(dO, cute.select(self.VdO_mma_tiler, mode=[1, 2]), (None, None, None))
        gdOT = cute.local_tile(dOT, cute.select(self.PdO_mma_tiler, mode=[1, 2]), (None, None, None))

        KQ_thr_mma = KQ_tiled_mma.get_slice(mma_tile_coord_v)
        VdO_thr_mma = VdO_tiled_mma.get_slice(mma_tile_coord_v)
        PdO_thr_mma = PdO_tiled_mma.get_slice(mma_tile_coord_v)
        dSQ_thr_mma = dSQ_tiled_mma.get_slice(mma_tile_coord_v)

        tSTgK = KQ_thr_mma.partition_A(gK)
        tSTgQ = KQ_thr_mma.partition_B(gQ)
        tdK_gQT = dSQ_thr_mma.partition_B(gQT)
        tdPTgV = VdO_thr_mma.partition_A(gV)
        tdPTgdO = VdO_thr_mma.partition_B(gdO)
        tdVgdOT = PdO_thr_mma.partition_B(gdOT)

        cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (KQ_tiled_mma.thr_id,))
        cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cute.arch.block_idx_in_cluster())

        tKsK, tKgK_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_K,
            cta_in_cluster_coord_vmnk[2],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tSTgK, 0, 3),
        )
        tQsQ, tQgQ_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_Q,
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSTgQ, 0, 3),
        )
        tQTsQT, tQTgQT_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_QT,
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sQT, 0, 3),
            cute.group_modes(tdK_gQT, 0, 3),
        )
        tVsV, tVgV_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_V,
            cta_in_cluster_coord_vmnk[2],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tdPTgV, 0, 3),
        )
        tdOsdO, tdOgdO_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dO,
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sdO, 0, 3),
            cute.group_modes(tdPTgdO, 0, 3),
        )
        tdOTsdOT, tdOTgdOT_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dOT,
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sdOT, 0, 3),
            cute.group_modes(tdVgdOT, 0, 3),
        )

        k_handle = load_mma_K_producer.acquire_and_advance()
        cute.copy(
            tma_atom_K,
            tKgK_mkl[(None, mma_tile_coord_m, 0, (blk_coord_h, blk_coord_b))],
            tKsK[None, 0],
            tma_bar_ptr=k_handle.barrier,
        )

        m_block_for_load = iter_index
        assert block_sparse_tensors is not None
        (
            blk_coord_h_r,
            iter_index,
            sparse_head_count,
            sparse_partial_base,
            sparse_full_base,
            sparse_partial_count,
            sparse_full_count,
        ) = _hd256_bwd_sparse_group_first_work_item(
            block_sparse_tensors,
            blk_coord_b,
            blk_coord_h_k,
            mma_tile_coord_m,
            self.qhead_per_kvhead,
            self.subtile_factor,
            m_block_max,
            cute.ceil_div(problem_shape[1], self.KQ_mma_tiler[0]),
        )
        blk_coord_h = (blk_coord_h_r, blk_coord_h_k)
        _, m_block_for_load, _, _, _, _ = _hd256_bwd_sparse_m_block(
            block_sparse_tensors,
            iter_index,
            sparse_partial_base,
            sparse_full_base,
            sparse_partial_count,
            sparse_full_count,
            self.subtile_factor,
            m_block_max,
        )

        q_handle = load_mma_Q_producer.acquire_and_advance()
        cute.copy(
            tma_atom_Q,
            tQgQ_mkl[(None, m_block_for_load, 0, (blk_coord_h, blk_coord_b))],
            tQsQ[None, q_handle.index],
            tma_bar_ptr=q_handle.barrier,
        )

        lse_handle = load_compute_LSE_producer.acquire_and_advance()
        thread_idx = tidx % self.threads_per_warp
        async_copy_num_elts = sLSE.shape[0] // self.threads_per_warp
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            self.acc_dtype,
            num_bits_per_copy=self.acc_dtype.width,
        )
        sLSE_for_copy = cute.flat_divide(sLSE, (1,))
        LSE_for_copy = cute.flat_divide(LSE, (1,))
        # Warp-coalesced: at each i, lane T accesses index `T + i*W` (stride-1
        # across the warp) instead of `T*N + i` (stride-N across the warp).
        for i in cutlass.range_constexpr(async_copy_num_elts):
            LSE_idx = self.tile_shape_Q * m_block_for_load + thread_idx + i * self.threads_per_warp
            sLSE_idx = thread_idx + i * self.threads_per_warp
            if cute.elem_less(LSE_idx, problem_shape[0]):
                cute.copy(
                    atom_async_copy,
                    LSE_for_copy[None, LSE_idx, (blk_coord_h, blk_coord_b)],
                    sLSE_for_copy[None, sLSE_idx, lse_handle.index],
                )
            else:
                sLSE_for_copy[None, sLSE_idx, lse_handle.index].fill(0.0)
        lse_handle.commit()

        v_handle = load_mma_V_producer.acquire_and_advance()
        cute.copy(
            tma_atom_V,
            tVgV_mkl[(None, mma_tile_coord_m, 0, (blk_coord_h, blk_coord_b))],
            tVsV[(None, 0)],
            tma_bar_ptr=v_handle.barrier,
        )

        do_handle = load_mma_dO_producer.acquire_and_advance()
        cute.copy(
            tma_atom_dO,
            tdOgdO_mkl[(None, m_block_for_load, 0, (blk_coord_h, blk_coord_b))],
            tdOsdO[(None, do_handle.index)],
            tma_bar_ptr=do_handle.barrier,
        )

        sum_odo_handle = load_compute_sum_OdO_producer.acquire_and_advance()
        sSum_OdO_for_copy = cute.flat_divide(sSum_OdO, (1,))
        sum_OdO_for_copy = cute.flat_divide(sum_OdO, (1,))
        for i in cutlass.range_constexpr(async_copy_num_elts):
            sum_OdO_idx = self.tile_shape_Q * m_block_for_load + thread_idx + i * self.threads_per_warp
            sSum_OdO_idx = thread_idx + i * self.threads_per_warp
            if cute.elem_less(sum_OdO_idx, problem_shape[0]):
                cute.copy(
                    atom_async_copy,
                    sum_OdO_for_copy[None, sum_OdO_idx, (blk_coord_h, blk_coord_b)],
                    sSum_OdO_for_copy[None, sSum_OdO_idx, sum_odo_handle.index],
                )
            else:
                sSum_OdO_for_copy[None, sSum_OdO_idx, sum_odo_handle.index].fill(0.0)
        sum_odo_handle.commit()

        dot_handle = load_mma_dOT_producer.acquire_and_advance()
        cute.copy(
            tma_atom_dOT,
            tdOTgdOT_mkl[(None, 0, m_block_for_load, (blk_coord_h, blk_coord_b))],
            tdOTsdOT[None, dot_handle.index],
            tma_bar_ptr=dot_handle.barrier,
        )

        qt_handle = load_mma_QT_producer.acquire_and_advance()
        cute.copy(
            tma_atom_QT,
            tQTgQT_mkl[(None, 0, m_block_for_load, (blk_coord_h, blk_coord_b))],
            tQTsQT[None, qt_handle.index],
            tma_bar_ptr=qt_handle.barrier,
        )

        iter_count -= 1
        if iter_count > 0:
            (
                blk_coord_h_r,
                iter_index,
                sparse_head_count,
                sparse_partial_base,
                sparse_full_base,
                sparse_partial_count,
                sparse_full_count,
            ) = _hd256_bwd_sparse_group_next_work_item(
                block_sparse_tensors,
                blk_coord_b,
                blk_coord_h_k,
                mma_tile_coord_m,
                blk_coord_h_r,
                iter_index,
                sparse_head_count,
                sparse_partial_base,
                sparse_full_base,
                sparse_partial_count,
                sparse_full_count,
                self.qhead_per_kvhead,
                self.subtile_factor,
                m_block_max,
                cute.ceil_div(problem_shape[1], self.KQ_mma_tiler[0]),
            )

        while iter_count > 0:
            assert block_sparse_tensors is not None
            blk_coord_h = (blk_coord_h_r, blk_coord_h_k)

            m_block_for_load = iter_index
            assert block_sparse_tensors is not None
            _, m_block_for_load, _, _, _, _ = _hd256_bwd_sparse_m_block(
                block_sparse_tensors,
                iter_index,
                sparse_partial_base,
                sparse_full_base,
                sparse_partial_count,
                sparse_full_count,
                self.subtile_factor,
                m_block_max,
            )

            q_handle = load_mma_Q_producer.acquire_and_advance()
            cute.copy(
                tma_atom_Q,
                tQgQ_mkl[(None, m_block_for_load, 0, (blk_coord_h, blk_coord_b))],
                tQsQ[None, q_handle.index],
                tma_bar_ptr=q_handle.barrier,
            )

            lse_handle = load_compute_LSE_producer.acquire_and_advance()
            sLSE_for_copy = cute.flat_divide(sLSE, (1,))
            LSE_for_copy = cute.flat_divide(LSE, (1,))
            for i in cutlass.range_constexpr(async_copy_num_elts):
                LSE_idx = self.tile_shape_Q * m_block_for_load + thread_idx + i * self.threads_per_warp
                sLSE_idx = thread_idx + i * self.threads_per_warp
                if cute.elem_less(LSE_idx, problem_shape[0]):
                    cute.copy(
                        atom_async_copy,
                        LSE_for_copy[None, LSE_idx, (blk_coord_h, blk_coord_b)],
                        sLSE_for_copy[None, sLSE_idx, lse_handle.index],
                    )
                else:
                    sLSE_for_copy[None, sLSE_idx, lse_handle.index].fill(0.0)
            lse_handle.commit()

            do_handle = load_mma_dO_producer.acquire_and_advance()
            cute.copy(
                tma_atom_dO,
                tdOgdO_mkl[(None, m_block_for_load, 0, (blk_coord_h, blk_coord_b))],
                tdOsdO[None, do_handle.index],
                tma_bar_ptr=do_handle.barrier,
            )

            sum_odo_handle = load_compute_sum_OdO_producer.acquire_and_advance()
            sSum_OdO_for_copy = cute.flat_divide(sSum_OdO, (1,))
            sum_OdO_for_copy = cute.flat_divide(sum_OdO, (1,))
            for i in cutlass.range_constexpr(async_copy_num_elts):
                sum_OdO_idx = self.tile_shape_Q * m_block_for_load + thread_idx + i * self.threads_per_warp
                sSum_OdO_idx = thread_idx + i * self.threads_per_warp
                if cute.elem_less(sum_OdO_idx, problem_shape[0]):
                    cute.copy(
                        atom_async_copy,
                        sum_OdO_for_copy[None, sum_OdO_idx, (blk_coord_h, blk_coord_b)],
                        sSum_OdO_for_copy[None, sSum_OdO_idx, sum_odo_handle.index],
                    )
                else:
                    sSum_OdO_for_copy[None, sSum_OdO_idx, sum_odo_handle.index].fill(0.0)
            sum_odo_handle.commit()

            dot_handle = load_mma_dOT_producer.acquire_and_advance()
            cute.copy(
                tma_atom_dOT,
                tdOTgdOT_mkl[(None, 0, m_block_for_load, (blk_coord_h, blk_coord_b))],
                tdOTsdOT[None, dot_handle.index],
                tma_bar_ptr=dot_handle.barrier,
            )

            qt_handle = load_mma_QT_producer.acquire_and_advance()
            cute.copy(
                tma_atom_QT,
                tQTgQT_mkl[(None, 0, m_block_for_load, (blk_coord_h, blk_coord_b))],
                tQTsQT[None, qt_handle.index],
                tma_bar_ptr=qt_handle.barrier,
            )

            iter_count -= 1
            if iter_count > 0:
                (
                    blk_coord_h_r,
                    iter_index,
                    sparse_head_count,
                    sparse_partial_base,
                    sparse_full_base,
                    sparse_partial_count,
                    sparse_full_count,
                ) = _hd256_bwd_sparse_group_next_work_item(
                    block_sparse_tensors,
                    blk_coord_b,
                    blk_coord_h_k,
                    mma_tile_coord_m,
                    blk_coord_h_r,
                    iter_index,
                    sparse_head_count,
                    sparse_partial_base,
                    sparse_full_base,
                    sparse_partial_count,
                    sparse_full_count,
                    self.qhead_per_kvhead,
                    self.subtile_factor,
                    m_block_max,
                    cute.ceil_div(problem_shape[1], self.KQ_mma_tiler[0]),
                )

        load_mma_K_producer.tail()
        load_mma_V_producer.tail()
        load_mma_Q_producer.tail()
        load_compute_LSE_producer.tail()
        load_mma_dO_producer.tail()
        load_mma_dOT_producer.tail()
        load_compute_sum_OdO_producer.tail()
        load_mma_QT_producer.tail()

    @cute.jit
    def mma_2cta(
        self,
        KQ_tiled_mma: cute.TiledMma,
        VdO_tiled_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        tSTtST: cute.Tensor,
        tSTrQ: cute.Tensor,
        tSTrK: cute.Tensor,
        tdPTtdPT: cute.Tensor,
        tdPTrV: cute.Tensor,
        tdPTrdO: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdVrP: cute.Tensor,
        tdVrdOT: cute.Tensor,
        tdKrdST: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdKrQT: cute.Tensor,
        iter_count: Int32,
        load_mma_Q_consumer,
        load_mma_K_consumer,
        load_mma_V_consumer,
        mma_compute_S_producer,
        load_mma_dO_consumer,
        mma_compute_dP_producer,
        load_mma_dOT_consumer,
        compute_mma_P_consumer,
        compute_mma_dS_consumer,
        load_mma_QT_consumer,
        mma_compute_dKdV_producer,
    ):
        """CuTeDSL kernel for mma pipeline."""
        load_mma_K_releaser = load_mma_K_consumer.clone()
        load_mma_V_releaser = load_mma_V_consumer.clone()

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_leader_cta = cta_rank_in_cluster % 2 == 0

        if is_leader_cta:
            s_handle = mma_compute_S_producer.acquire_and_advance()
            load_mma_K_consumer.wait_and_advance()
            q_handle = load_mma_Q_consumer.wait_and_advance()

            # Compute S = K * Q
            for k_block in cutlass.range(0, cute.size(tSTrQ, mode=[2]), unroll_full=True):
                KQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, k_block != 0)
                cute.gemm(
                    KQ_tiled_mma,
                    tSTtST,
                    tSTrK[None, None, k_block, 0],
                    tSTrQ[None, None, k_block, q_handle.index],
                    tSTtST,
                )
            q_handle.release()

            cute.arch.fence_view_async_tmem_store()
            s_handle.commit()

            do_handle = load_mma_dO_consumer.wait_and_advance()
            load_mma_V_consumer.wait_and_advance()

            dp_handle = mma_compute_dP_producer.acquire_and_advance()

            # Compute dP = V * dO
            VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdPTrV, mode=[2]), unroll_full=True):
                cute.gemm(
                    VdO_tiled_mma,
                    tdPTtdPT,
                    tdPTrV[None, None, k_block, 0],
                    tdPTrdO[None, None, k_block, do_handle.index],
                    tdPTtdPT,
                )
                VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            dp_handle.commit()
            do_handle.release()
            # V is produced once by load(); hold its stage until the end via the cloned releaser.

            p_handle = compute_mma_P_consumer.wait_and_advance()
            dot_handle = load_mma_dOT_consumer.wait_and_advance()

            # Compute dV = P * dO (First iteration)
            PdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdVrP, mode=[2]), unroll_full=True):
                cute.gemm(
                    PdO_tiled_mma,
                    tdVtdV,
                    tdVrP[None, None, k_block, 0],
                    tdVrdOT[None, None, k_block, dot_handle.index],
                    tdVtdV,
                )
                PdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            dot_handle.release()
            p_handle.release()

        iter_count -= 1

        dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        while iter_count > 0:
            if is_leader_cta:
                q_handle = load_mma_Q_consumer.wait_and_advance()
                s_handle = mma_compute_S_producer.acquire_and_advance()

                # Compute S = K * Q
                for k_block in cutlass.range(0, cute.size(tSTrQ, mode=[2]), unroll_full=True):
                    KQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, k_block != 0)
                    cute.gemm(
                        KQ_tiled_mma,
                        tSTtST,
                        tSTrK[None, None, k_block, 0],
                        tSTrQ[None, None, k_block, q_handle.index],
                        tSTtST,
                    )
                q_handle.release()
                s_handle.commit()

            if is_leader_cta:
                qt_handle = load_mma_QT_consumer.wait_and_advance()
                ds_handle = compute_mma_dS_consumer.wait_and_advance()

                # Compute dK = dS * QT
                for k_block in cutlass.range(0, cute.size(tdKrdST, mode=[2]), unroll_full=True):
                    cute.gemm(
                        dSQ_tiled_mma,
                        tdKtdK,
                        tdKrdST[
                            None,
                            None,
                            k_block,
                            ds_handle.index,
                        ],
                        tdKrQT[None, None, k_block, qt_handle.index],
                        tdKtdK,
                    )
                    dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                qt_handle.release()
                ds_handle.release()

            if is_leader_cta:
                dp_handle = mma_compute_dP_producer.acquire_and_advance()
                do_handle = load_mma_dO_consumer.wait_and_advance()
                # V only produced once by load(); reuse same V (index 0) for all loop iterations
                VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(0, cute.size(tdPTrV, mode=[2]), unroll_full=True):
                    cute.gemm(
                        VdO_tiled_mma,
                        tdPTtdPT,
                        tdPTrV[None, None, k_block, 0],
                        tdPTrdO[None, None, k_block, do_handle.index],
                        tdPTtdPT,
                    )
                    VdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                dp_handle.commit()
                do_handle.release()

            if is_leader_cta:
                p_handle = compute_mma_P_consumer.wait_and_advance()
                dot_handle = load_mma_dOT_consumer.wait_and_advance()

                # Compute dV = P * dO (Loop iterations)
                for k_block in cutlass.range(0, cute.size(tdVrP, mode=[2]), unroll_full=True):
                    cute.gemm(
                        PdO_tiled_mma,
                        tdVtdV,
                        tdVrP[None, None, k_block, 0],
                        tdVrdOT[None, None, k_block, dot_handle.index],
                        tdVtdV,
                    )
                    PdO_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                p_handle.release()
                dot_handle.release()

            iter_count -= 1

        if is_leader_cta:
            dkdv_handle = mma_compute_dKdV_producer.acquire_and_advance()
            dkdv_handle.commit()

            load_mma_K_releaser.release()
            load_mma_K_releaser.advance()
            load_mma_V_releaser.release()
            load_mma_V_releaser.advance()

        if is_leader_cta:
            dkdv_handle = mma_compute_dKdV_producer.acquire_and_advance()

            ds_handle = compute_mma_dS_consumer.wait_and_advance()
            qt_handle = load_mma_QT_consumer.wait_and_advance()

            # Compute dK = dS * Q
            for k_block in cutlass.range(0, cute.size(tdKrdST, mode=[2]), unroll_full=True):
                cute.gemm(
                    dSQ_tiled_mma,
                    tdKtdK,
                    tdKrdST[None, None, k_block, ds_handle.index],
                    tdKrQT[None, None, k_block, qt_handle.index],
                    tdKtdK,
                )
                dSQ_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            dkdv_handle.commit()
            qt_handle.release()
            ds_handle.release()

        mma_compute_S_producer.tail()
        mma_compute_dP_producer.tail()
        mma_compute_dKdV_producer.tail()

    @cute.jit
    def reg_to_smem_mma128x128_2cta(
        self,
        regs: cute.Tensor,
        smem: cute.Tensor,
        index: Int32,
        tiler_mn: tuple[Int32, Int32],
        dp_idx: Int32,
        wg_idx: Int32,
    ):
        smem_slice = smem[None, None, None, index]
        # K>> smem_slice:  tensor<ptr<f16, smem, align<1024>, S<3,4,3>> o ((64,16),1,(4,2)):((64,1),0,(16,4096))>
        thread_layout = cute.make_ordered_layout(
            # (tileN, tileM)
            tiler_mn,
            (0, 1),
        )
        # K>> thread_layout:  (64,128):(128,1)
        smem_slice_tmp = cute.composition(smem_slice, thread_layout)

        # NOTE: hardcode for tcgen05.ld.32x32b.x16 & mma128x64+2cta
        tmp_shape = ((32, 2), (16, 2, 2, 2))  # for 128x64 tile
        tmp_stride = ((64, 32 * 64), (1, 16, 32, 64 * 64))
        smem_copy = cute.make_tensor(smem_slice_tmp.iterator, cute.make_layout(tmp_shape, stride=tmp_stride))

        warp_idx = dp_idx // 32
        warp_row_idx = warp_idx % 2
        warp_col_idx = warp_idx // 2  # corresponding to the second 64 cols in smem
        lane_idx = dp_idx % 32
        reg_shape = regs.shape  # ((8,1),1,2):((1,0),0,8) for 64x64, ((16,1),1,2):((1,0),0,16) for 128x64
        block_loops = reg_shape[2]

        # TODO: maybe can use cp.async for optimization
        for ib in cutlass.range(block_loops):
            regs_copy = regs[(None, 0), 0, ib]
            smem_copy_slice = smem_copy[(lane_idx, warp_row_idx), (None, wg_idx, ib, warp_col_idx)]
            cute.autovec_copy(regs_copy, smem_copy_slice)

    @cute.jit
    def compute(
        self,
        tSTtST: cute.Tensor,
        tdPTtdPT: cute.Tensor,
        sP: cute.Tensor,
        sLSE: cute.Tensor,
        sdST: cute.Tensor,
        sdOT: cute.Tensor,
        sSum_OdO: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdVtdV: cute.Tensor,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        iter_count: Int32,
        scale_softmax: cutlass.Float32,
        mma_compute_S_consumer,
        compute_mma_P_producer,
        load_compute_LSE_consumer,
        load_compute_sum_OdO_consumer,
        mma_compute_dP_consumer,
        compute_mma_dS_producer,
        mma_compute_dKdV_consumer,
        varlen: bool,
        problem_shape_k_cur_batch: Int32,
        tma_atom_dK: cute.CopyAtom,
        dK_tma: cute.Tensor,
        tma_atom_dV: cute.CopyAtom,
        dV_tma: cute.Tensor,
        sdK_epi_layout: cute.ComposedLayout,
        sdV_epi_layout: cute.ComposedLayout,
        block_sparse_tensors: Optional[BlockSparseTensors] = None,
        m_block_max: Int32 = Int32(0),
    ):
        """CuTeDSL kernel for recomputing softmax and producing dk and dv."""
        tidx, _, _ = cute.arch.thread_idx()
        Q, _, _, _ = problem_shape
        _, blk_coord_k, _, _ = blk_coord
        assert block_sparse_tensors is not None
        (
            sparse_qhead_offset,
            sparse_head_iter_idx,
            sparse_head_count,
            sparse_partial_base,
            sparse_full_base,
            sparse_partial_count,
            sparse_full_count,
        ) = _hd256_bwd_sparse_group_first_work_item(
            block_sparse_tensors,
            blk_coord[3][1],
            blk_coord[3][0][1],
            blk_coord_k // 2,
            self.qhead_per_kvhead,
            self.subtile_factor,
            m_block_max,
            cute.ceil_div(problem_shape[1], self.KQ_mma_tiler[0]),
        )

        # adi: TMEM_ST, TMEM_DPT
        tmem_load_op = tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16))
        tmem_load_atom = cute.make_copy_atom(
            tmem_load_op,
            self.acc_dtype,
        )

        tSTtST = tSTtST[(None, None), 0, 0]
        tdPTtdPT = tdPTtdPT[(None, None), 0, 0]

        cST = cute.make_identity_tensor(cute.select(self.cta_tiler, mode=[1, 0]))
        cdPT = cute.make_identity_tensor(cute.select(self.cta_tiler, mode=[1, 0]))

        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128
        tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tSTtST)
        thr_t2r = tiled_t2r.get_slice(dp_idx)

        tTR_cST = thr_t2r.partition_D(cST)
        tTR_cST = split_wg(tTR_cST, num_warp_groups, wg_idx)
        tTR_rST = cute.make_rmem_tensor(tTR_cST.shape, self.acc_dtype)

        tTR_tST = thr_t2r.partition_S(tSTtST)
        tTR_tST = split_wg(tTR_tST, num_warp_groups, wg_idx)

        tTR_cdPT_p = thr_t2r.partition_D(cdPT)
        tTR_cdPT = split_wg(tTR_cdPT_p, num_warp_groups, wg_idx)
        tTR_rdPT = cute.make_rmem_tensor(tTR_cdPT.shape, self.acc_dtype)

        tTR_tdPT = thr_t2r.partition_S(tdPTtdPT)
        tTR_tdPT = split_wg(tTR_tdPT, num_warp_groups, wg_idx)

        while iter_count > 0:
            is_partial_block = False
            payload_idx = Int32(0)
            q_subtile = Int32(0)
            assert block_sparse_tensors is not None
            iter_index = sparse_head_iter_idx
            (
                _,
                _,
                _,
                is_partial_block,
                payload_idx,
                q_subtile,
            ) = _hd256_bwd_sparse_m_block(
                block_sparse_tensors,
                iter_index,
                sparse_partial_base,
                sparse_full_base,
                sparse_partial_count,
                sparse_full_count,
                self.subtile_factor,
                m_block_max,
            )
            s_handle = mma_compute_S_consumer.wait_and_advance()
            p_handle = compute_mma_P_producer.acquire_and_advance()
            lse_handle = load_compute_LSE_consumer.wait_and_advance()

            # Compute P = softmax(S, LSE)
            cute.copy(tiled_t2r, tTR_tST, tTR_rST)

            if is_partial_block:
                assert block_sparse_tensors is not None
                assert block_sparse_tensors.mask_block_masks is not None
                assert cutlass.const_expr(
                    cute.size(tTR_rST) == self.mask_payload_valid_words * 32
                ), "SM100 hd256 arbitrary dKdV payload does not cover the score fragment"
                payload_subtile_idx = q_subtile * 2 + blk_coord_k % 2
                payload_group_idx = wg_idx * 128 + dp_idx
                r_bitmask = load_mask_payload(
                    block_sparse_tensors.mask_block_masks,
                    payload_idx,
                    payload_group_idx,
                    subtile_idx=payload_subtile_idx,
                    payload_words=self.mask_payload_padded_words,
                )
                apply_loaded_arbitrary_mask(
                    tTR_rST,
                    r_bitmask,
                    self.mask_payload_valid_words,
                )

            log2_e = cutlass.Float32(math.log2(math.e))
            softmax_scale_log2_e = scale_softmax * log2_e

            for i in cutlass.range(0, cute.size(tTR_rST), 2, unroll_full=True):
                lse = (
                    -sLSE[
                        cute.get(tTR_cST[i], mode=[1]),
                        lse_handle.index,
                    ],
                    -sLSE[
                        cute.get(tTR_cST[i + 1], mode=[1]),
                        lse_handle.index,
                    ],
                )
                tTR_rST[i], tTR_rST[i + 1] = cute.arch.fma_packed_f32x2(
                    (tTR_rST[i], tTR_rST[i + 1]),
                    (softmax_scale_log2_e, softmax_scale_log2_e),
                    lse,
                )
                tTR_rST[i] = cute.math.exp2(tTR_rST[i], fastmath=True)
                tTR_rST[i + 1] = cute.math.exp2(tTR_rST[i + 1], fastmath=True)

            # convert fp32 P to fp16 P which will be used in the PdO
            tTR_rPT = self.quantize(tTR_rST, dV.element_type)  # tTR_rST is ST in fp32 in RF.
            self.reg_to_smem_mma128x128_2cta(
                tTR_rPT,
                sP,
                p_handle.index,
                (self.tile_shape_K, self.tile_shape_Q),
                dp_idx,
                wg_idx,
            )
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=self.compute_sync_bar_id,
                number_of_threads=self.num_compute_warps * self.threads_per_warp,
            )

            p_handle.commit()

            s_handle.release()
            lse_handle.release()

            sum_odo_handle = load_compute_sum_OdO_consumer.wait_and_advance()
            dp_handle = mma_compute_dP_consumer.wait_and_advance()
            ds_handle = compute_mma_dS_producer.acquire_and_advance()

            # Compute dS = dsoftmax(P, dP, sum_OdO)
            cute.copy(tiled_t2r, tTR_tdPT, tTR_rdPT)

            for i in cutlass.range(0, cute.size(tTR_rdPT), 2, unroll_full=True):
                dpsum_0 = -sSum_OdO[
                    cute.get(tTR_cdPT[i], mode=[1]),
                    sum_odo_handle.index,
                ]
                dpsum_1 = -sSum_OdO[
                    cute.get(tTR_cdPT[i + 1], mode=[1]),
                    sum_odo_handle.index,
                ]
                if cutlass.const_expr(varlen):
                    if not cute.elem_less(cute.get(tTR_cdPT[i], mode=[1]), Q):
                        dpsum_0 = 0.0
                    if not cute.elem_less(cute.get(tTR_cdPT[i + 1], mode=[1]), Q):
                        dpsum_1 = 0.0
                tTR_rdPT[i], tTR_rdPT[i + 1] = cute.arch.add_packed_f32x2(
                    (tTR_rdPT[i], tTR_rdPT[i + 1]),
                    (dpsum_0, dpsum_1),
                )
                tTR_rdPT[i], tTR_rdPT[i + 1] = cute.arch.mul_packed_f32x2((tTR_rdPT[i], tTR_rdPT[i + 1]), (tTR_rST[i], tTR_rST[i + 1]))
            # convert fp32 dS to fp16 dS which will be used in the computation of dK and DQ
            tTR_rdST = self.quantize(tTR_rdPT, dV.element_type)

            cute.arch.fence_view_async_tmem_load()
            dp_handle.release()

            self.reg_to_smem_mma128x128_2cta(
                tTR_rdST,
                sdST,
                ds_handle.index,
                (self.tile_shape_K, self.tile_shape_Q),
                dp_idx,
                wg_idx,
            )
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=self.compute_sync_bar_id,
                number_of_threads=self.num_compute_warps * self.threads_per_warp,
            )

            ds_handle.commit()
            sum_odo_handle.release()

            iter_count -= 1
            if iter_count > 0:
                (
                    sparse_qhead_offset,
                    sparse_head_iter_idx,
                    sparse_head_count,
                    sparse_partial_base,
                    sparse_full_base,
                    sparse_partial_count,
                    sparse_full_count,
                ) = _hd256_bwd_sparse_group_next_work_item(
                    block_sparse_tensors,
                    blk_coord[3][1],
                    blk_coord[3][0][1],
                    blk_coord_k // 2,
                    sparse_qhead_offset,
                    sparse_head_iter_idx,
                    sparse_head_count,
                    sparse_partial_base,
                    sparse_full_base,
                    sparse_partial_count,
                    sparse_full_count,
                    self.qhead_per_kvhead,
                    self.subtile_factor,
                    m_block_max,
                    cute.ceil_div(problem_shape[1], self.KQ_mma_tiler[0]),
                )

        # Epilogue
        mma_compute_dKdV_consumer = self.epilogue(
            blk_coord,
            blk_offset,
            problem_shape,
            dK,
            dV,
            tdKtdK,
            tdVtdV,
            scale_softmax,
            mma_compute_dKdV_consumer,
            problem_shape_k_cur_batch,
            tma_atom_dK,
            dK_tma,
            tma_atom_dV,
            dV_tma,
            sdK_epi_layout,
            sdV_epi_layout,
            varlen,
            sdOT,
            sP,
        )

        compute_mma_P_producer.tail()
        compute_mma_dS_producer.tail()

    @cute.jit
    def quantize(
        self,
        input_t: cute.Tensor,
        element_dtype: type[cutlass.Numeric],
    ) -> cute.Tensor:
        """Convert Float32 to element dtype."""
        output = cute.make_rmem_tensor(input_t.shape, element_dtype)
        output.store(input_t.load().to(element_dtype))
        return output

    @cute.jit
    def store(
        self,
        gmem: cute.Tensor,
        regs: cute.Tensor,
        coord: cute.Tensor,
        tensor_shape: cute.Shape,
    ):
        for i in cutlass.range(cute.size(coord, mode=[2]), unroll_full=True):
            coord_i = coord[None, 0, i]
            gmem_i = gmem[None, 0, i]
            regs_i = regs[None, 0, i]
            if cute.elem_less(coord_i[0], tensor_shape):
                gmem_i.store(regs_i.load().to(gmem.element_type))

    @cute.jit
    def epilogue_clear(
        self,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        dK: cute.Tensor,
        dV: cute.Tensor,
    ):
        """Early stopping needs to clear dK and dV."""
        tidx, _, _ = cute.arch.thread_idx()
        _, K, _, HB = problem_shape
        _, blk_coord_k, _, blk_coord_batch = blk_coord

        mdK_offset = cute.assume(blk_offset[1] * dK.stride[0], divby=64)
        mdK = cute.make_tensor(
            dK.iterator + mdK_offset,
            cute.make_layout((K, self.tile_shape_dQ_K, HB), stride=dK.stride),
        )
        gdK = cute.local_tile(mdK, (self.cta_tiler[1], self.cta_tiler[2]), (None, None, None))
        gdK = gdK[None, None, blk_coord_k, 0, blk_coord_batch]
        cdK = cute.domain_offset(
            (blk_coord_k * self.tile_shape_K, 0),
            cute.make_identity_tensor((self.cta_tiler[1], self.cta_tiler[2])),
        )

        mdV_offset = cute.assume(blk_offset[1] * dV.stride[0], divby=64)
        mdV = cute.make_tensor(
            dV.iterator + mdV_offset,
            cute.make_layout((K, self.tile_shape_dV_dO, HB), stride=dV.stride),
        )
        gdV = cute.local_tile(mdV, (self.cta_tiler[1], self.cta_tiler[2]), (None, None, None))
        gdV = gdV[None, None, blk_coord_k, 0, blk_coord_batch]
        cdV = cute.domain_offset(
            (blk_coord_k * self.tile_shape_K, 0),
            cute.make_identity_tensor((self.cta_tiler[1], self.cta_tiler[2])),
        )

        num_zero_epi_threads = 256

        tiled_copy_r2g = fa_copy_utils.tiled_copy_2d(dK.element_type, self.cta_tiler[2], num_zero_epi_threads)

        thr_copy_r2g = tiled_copy_r2g.get_slice(tidx)

        tRG_gdK = thr_copy_r2g.partition_D(gdK)
        tRG_cdK = thr_copy_r2g.partition_D(cdK)
        tRG_gdV = thr_copy_r2g.partition_D(gdV)
        tRG_cdV = thr_copy_r2g.partition_D(cdV)

        zero_frg = cute.make_rmem_tensor_like(tRG_gdK[None, 0, None])
        zero_frg.fill(dK.element_type(0.0))

        # check we don't need zero fragment duplication
        V_frg_size = cute.size(tRG_gdV[None, 0, None])
        assert cute.size(zero_frg) == V_frg_size

        if tidx < num_zero_epi_threads:
            for n in cutlass.range(cute.size(tRG_gdK.shape[1]), unroll_full=True):
                if cute.elem_less(tRG_cdK[0, n, 0][0], problem_shape[1]):
                    cute.copy(tiled_copy_r2g, zero_frg, tRG_gdK[None, n, None])

            for n in cutlass.range(cute.size(tRG_gdV.shape[1]), unroll_full=True):
                if cute.elem_less(tRG_cdV[0, n, 0][0], problem_shape[1]):
                    cute.copy(tiled_copy_r2g, zero_frg, tRG_gdV[None, n, None])

    @cute.jit
    def epilogue(
        self,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        dK: cute.Tensor,
        dV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdVtdV: cute.Tensor,
        scale_softmax: cutlass.Float32,
        mma_compute_dKdV_consumer,
        problem_shape_k_cur_batch: Int32,
        tma_atom_dK: cute.CopyAtom,
        dK_tma: cute.Tensor,
        tma_atom_dV: cute.CopyAtom,
        dV_tma: cute.Tensor,
        sdK_epi_layout: cute.ComposedLayout,
        sdV_epi_layout: cute.ComposedLayout,
        varlen: bool,
        sdOT: cute.Tensor,
        sP: cute.Tensor,
    ):
        """Variant 3a (5/5) Path 2: CTA-shared SMEM with cooperative WG writes + TMA bulk store.

        Both warp-groups cooperatively populate a per-CTA (64, 256) virtual SMEM
        buffer (4 stages of (64, 64) aliased onto sP+sdST). Per-thread t2r N
        coverage is interleaved across the full hd=256, so per-WG TMA is not
        viable — instead we treat SMEM as one shared per-CTA buffer and let
        each thread's `self.store`-equivalent write into it via a (64, 256)
        virtual tensor whose N axis maps (n%64, n//64) → (N_within, stage).
        After an inter-WG barrier (256 threads), the leader warp fires 4 TMA
        bulk stores, one per stage, to the corresponding (64, 64) GMEM slice.
        Varlen falls back to per-thread self.store as in flash_bwd_sm100.py.
        """
        tidx, _, _ = cute.arch.thread_idx()
        _, K, D, HB = problem_shape
        _, blk_coord_k, _, blk_coord_batch = blk_coord

        # adi: TMEM_DK, TMEM_DV
        tmem_copy_op = tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32))
        load_op = cute.make_copy_atom(
            tmem_copy_op,
            self.acc_dtype,
        )

        tdKtdK = tdKtdK[(None, None), 0, 0]
        mdK_offset = cute.assume(blk_offset[1] * dK.stride[0], divby=64)
        mdK = cute.make_tensor(
            dK.iterator + mdK_offset,
            cute.make_layout((K, self.tile_shape_dQ_K, HB), stride=dK.stride),
        )
        gdK = cute.local_tile(mdK, (self.cta_tiler[1], self.cta_tiler[2]), (None, None, None))
        gdK = gdK[None, None, blk_coord_k, 0, blk_coord_batch]
        cdK = cute.domain_offset(
            (blk_coord_k * self.tile_shape_K, 0),
            cute.make_identity_tensor((self.cta_tiler[1], self.cta_tiler[2])),
        )

        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0

        # Path 2 SMEM staging. dV stages through sdOT (already-consumed by the
        # dV MMA before the dV epilogue begins). dK stages through sP+sdST
        # (dead after dK MMA completes, before dK epilogue runs).
        s_epi_dK = cute.make_tensor(
            cute.recast_ptr(sP.iterator, sdK_epi_layout.inner, dK.element_type),
            sdK_epi_layout.outer,
        )
        s_epi_dV = cute.make_tensor(
            cute.recast_ptr(sdOT.iterator, sdV_epi_layout.inner, dV.element_type),
            sdV_epi_layout.outer,
        )

        # Compile-time: stage tile shape and number of stages.
        epi_cols_dKV = math.gcd(128 // (dV.element_type.width // 8), self.cta_tiler[2] // num_warp_groups)
        num_epi_stages_dKV = (self.cta_tiler[2] // num_warp_groups) // epi_cols_dKV
        total_epi_stages = num_warp_groups * num_epi_stages_dKV
        epi_tile_dKV = (self.cta_tiler[1], epi_cols_dKV)

        # Local (M, N) coord tensor for SMEM indexing (no global domain offset
        # — cdK/cdV are domain-offset by blk_coord_k * tile_shape_K to match
        # the GMEM destination, but the SMEM indexing must be per-CTA-local).
        cdV_local = cute.make_identity_tensor((self.cta_tiler[1], self.cta_tiler[2]))
        cdK_local = cdV_local

        tiled_t2r_dK = tcgen05.make_tmem_copy(load_op, tdKtdK)
        thread_t2r_dK = tiled_t2r_dK.get_slice(dp_idx)

        tTR_cdK = thread_t2r_dK.partition_D(cdK)
        tTR_cdK = split_wg(tTR_cdK, num_warp_groups, wg_idx)
        tTR_cdK_local = thread_t2r_dK.partition_D(cdK_local)
        tTR_cdK_local = split_wg(tTR_cdK_local, num_warp_groups, wg_idx)
        tTR_gdK = thread_t2r_dK.partition_D(gdK)
        tTR_gdK = split_wg(tTR_gdK, num_warp_groups, wg_idx)
        tTR_rdK = cute.make_rmem_tensor(tTR_cdK.shape, self.acc_dtype)
        tTR_tdK = thread_t2r_dK.partition_S(tdKtdK)
        tTR_tdK = split_wg(tTR_tdK, num_warp_groups, wg_idx)

        mdV_in = cute.make_tensor(dV.iterator, cute.make_layout((K, self.cta_tiler[2], HB), stride=dV.stride))
        offset_mdV = cute.assume(blk_offset[1] * mdV_in.stride[0], divby=64)
        mdV = cute.make_tensor(mdV_in.iterator + offset_mdV, mdV_in.layout)
        gdV = cute.local_tile(mdV, (self.cta_tiler[1], self.cta_tiler[2]), (None, None, None))
        gdV = gdV[None, None, blk_coord_k, 0, blk_coord_batch]

        cdV = cute.domain_offset(
            (blk_coord_k * self.cta_tiler[1], 0),
            cute.make_identity_tensor((self.cta_tiler[1], self.cta_tiler[2])),
        )

        tdVtdV = tdVtdV[(None, None), 0, 0]

        tiled_t2r_dV = tcgen05.make_tmem_copy(load_op, tdVtdV)
        thread_t2r_dV = tiled_t2r_dV.get_slice(dp_idx)

        tTR_cdV = thread_t2r_dV.partition_D(cdV)
        tTR_cdV = split_wg(tTR_cdV, num_warp_groups, wg_idx)
        tTR_cdV_local = thread_t2r_dV.partition_D(cdV_local)
        tTR_cdV_local = split_wg(tTR_cdV_local, num_warp_groups, wg_idx)
        tTR_gdV = thread_t2r_dV.partition_D(gdV)
        tTR_gdV = split_wg(tTR_gdV, num_warp_groups, wg_idx)
        tTR_rdV = cute.make_rmem_tensor(tTR_cdV.shape, self.acc_dtype)
        tTR_tdV = thread_t2r_dV.partition_S(tdVtdV)
        tTR_tdV = split_wg(tTR_tdV, num_warp_groups, wg_idx)

        # GMEM destinations for the multi-stage TMA path (gated on not-varlen).
        if cutlass.const_expr(not varlen):
            mdV_tma_3d = cute.make_tensor(
                dV_tma.iterator,
                cute.make_layout((K, self.cta_tiler[2], HB), stride=dV_tma.stride),
            )
            mdV_tma_cur = mdV_tma_3d[None, None, blk_coord_batch]
            gdV_tma = cute.local_tile(mdV_tma_cur, (self.cta_tiler[1], self.cta_tiler[2]), (blk_coord_k, 0))
            gdV_tma_epi = cute.local_tile(gdV_tma, epi_tile_dKV, (0, None))

            mdK_tma_3d = cute.make_tensor(
                dK_tma.iterator,
                cute.make_layout((K, self.cta_tiler[2], HB), stride=dK_tma.stride),
            )
            mdK_tma_cur = mdK_tma_3d[None, None, blk_coord_batch]
            gdK_tma = cute.local_tile(mdK_tma_cur, (self.cta_tiler[1], self.cta_tiler[2]), (blk_coord_k, 0))
            gdK_tma_epi = cute.local_tile(gdK_tma, epi_tile_dKV, (0, None))

        cta_threads = self.num_compute_warps * self.threads_per_warp

        dkdv_handle = mma_compute_dKdV_consumer.wait_and_advance()

        if blk_coord_k * self.tile_shape_K < problem_shape_k_cur_batch:
            cute.copy(tiled_t2r_dV, tTR_tdV, tTR_rdV)
            tTR_rdV_cast = cute.make_rmem_tensor(tTR_rdV.shape, dV.element_type)
            tTR_rdV_cast.store(tTR_rdV.load().to(dV.element_type))

            if cutlass.const_expr(not varlen):
                # reg -> SMEM via per-element indexed stores using tTR_cdV's
                # per-thread (M, N) coords. (M, N) is per-CTA cdV space (M=0..63,
                # N=0..255). We map N=(n%epi_cols, n//epi_cols) → (N_within, stage)
                # of the 3D s_epi_dV tensor.
                for _i in cutlass.range_constexpr(cute.size(tTR_cdV_local, mode=[2])):
                    for _j in cutlass.range_constexpr(cute.size(tTR_cdV_local[None, 0, _i])):
                        c = tTR_cdV_local[None, 0, _i][_j]
                        m_pos = c[0]
                        n_pos = c[1]
                        stage_pos = n_pos // epi_cols_dKV
                        n_within_pos = n_pos % epi_cols_dKV
                        v = tTR_rdV_cast[None, 0, _i][_j]
                        s_epi_dV[m_pos, n_within_pos, stage_pos] = v
                cute.arch.fence_view_async_shared()
                # Inter-WG barrier — both warp-groups must finish their writes
                # before the leader warp reads SMEM via TMA.
                cute.arch.barrier(barrier_id=5, number_of_threads=cta_threads)
                # TMA bulk store, one (64, 64) box per stage.
                if leader_warp and wg_idx == 0:
                    for _stage in cutlass.range_constexpr(total_epi_stages):
                        sdV_stage = s_epi_dV[None, None, _stage]
                        gdV_stage = gdV_tma_epi[None, None, _stage]
                        td_sdV, td_gdV = cpasync.tma_partition(
                            tma_atom_dV,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sdV_stage, 0, 2),
                            cute.group_modes(gdV_stage, 0, 2),
                        )
                        cute.copy(tma_atom_dV, td_sdV, td_gdV)
                        cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            else:
                self.store(tTR_gdV, tTR_rdV, tTR_cdV, (K, D))

        cute.arch.fence_view_async_tmem_load()
        dkdv_handle.release()

        dkdv_handle = mma_compute_dKdV_consumer.wait_and_advance()

        if blk_coord_k * self.tile_shape_K < problem_shape_k_cur_batch:
            cute.copy(tiled_t2r_dK, tTR_tdK, tTR_rdK)

            for i in cutlass.range(cute.size(tTR_rdK), unroll_full=True):
                tTR_rdK[i] = scale_softmax * tTR_rdK[i]

            tTR_rdK_cast = cute.make_rmem_tensor(tTR_rdK.shape, dK.element_type)
            tTR_rdK_cast.store(tTR_rdK.load().to(dK.element_type))

            if cutlass.const_expr(not varlen):
                for _i in cutlass.range_constexpr(cute.size(tTR_cdK_local, mode=[2])):
                    for _j in cutlass.range_constexpr(cute.size(tTR_cdK_local[None, 0, _i])):
                        c = tTR_cdK_local[None, 0, _i][_j]
                        m_pos = c[0]
                        n_pos = c[1]
                        stage_pos = n_pos // epi_cols_dKV
                        n_within_pos = n_pos % epi_cols_dKV
                        v = tTR_rdK_cast[None, 0, _i][_j]
                        s_epi_dK[m_pos, n_within_pos, stage_pos] = v
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(barrier_id=6, number_of_threads=cta_threads)
                if leader_warp and wg_idx == 0:
                    for _stage in cutlass.range_constexpr(total_epi_stages):
                        sdK_stage = s_epi_dK[None, None, _stage]
                        gdK_stage = gdK_tma_epi[None, None, _stage]
                        td_sdK, td_gdK = cpasync.tma_partition(
                            tma_atom_dK,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sdK_stage, 0, 2),
                            cute.group_modes(gdK_stage, 0, 2),
                        )
                        cute.copy(tma_atom_dK, td_sdK, td_gdK)
                        cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            else:
                self.store(tTR_gdK, tTR_rdK, tTR_cdK, (K, D))

        cute.arch.fence_view_async_tmem_load()
        dkdv_handle.release()

    @staticmethod
    def _compute_bwd_grid(
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        block_k: int,
    ) -> tuple[Int32, Int32, Int32]:
        """Compute grid shape for bwd kernel."""
        K = problem_shape[1]
        _, H_K = problem_shape[3][0]
        B = problem_shape[3][1]
        return (cute.ceil_div(K, block_k), cute.size(H_K), cute.size(B))

    #  Barrier to between dP = v * dO and consume of dP in compute()
