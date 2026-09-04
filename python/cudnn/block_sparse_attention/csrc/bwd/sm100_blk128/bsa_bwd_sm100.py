# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
# SPDX-License-Identifier: MIT
import math
from typing import Callable, NamedTuple, Optional, Tuple
from functools import partial

import cuda.bindings.driver as cuda

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Int64, const_expr
from cutlass.utils import LayoutEnum
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass.pipeline import PipelineAsync

from cudnn.block_sparse_attention.csrc.utils import layout_utils
from cudnn.block_sparse_attention.csrc.utils import kernel_utils as utils
from cudnn.block_sparse_attention.csrc.utils.cute_dsl_utils import (
    assume_tensor_aligned,
    get_broadcast_dims,
    to_cute_tensor,
    torch2cute_dtype_map,
)
from cudnn.block_sparse_attention.csrc.utils import copy_utils
from cudnn.block_sparse_attention.csrc.utils import pipeline
from cudnn.block_sparse_attention.csrc.utils.tcgen05_mma_helpers import gemm_w_idx, gemm_ptx_w_idx
from cudnn.block_sparse_attention.csrc.utils.seqlen_info import SeqlenInfoQK
from cudnn.block_sparse_attention.csrc.utils.block_info import BlockInfo
from cudnn.block_sparse_attention.csrc.bwd.bsa_bwd_prepost import (
    _bwd_postprocess_convert,
    _bwd_preprocess,
    _get_device_arch,
)
from cudnn.block_sparse_attention.csrc.utils.cute_dsl_utils import ParamsBase, sub_packed_f32x2
from cudnn.block_sparse_attention.csrc.utils.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
)

from cudnn.block_sparse_attention.csrc.utils.named_barrier import NamedBarrierBwdSm100


class BsaK2qCsrTensors(NamedTuple):
    """BSA bucketed k2q CSR tensors for the blk128 backward path."""

    bucketed_k2q_offsets: cute.Tensor
    bucketed_k2q_indices: cute.Tensor

    def __new_from_mlir_values__(self, values):
        return BsaK2qCsrTensors(*values)


@cute.jit
def get_total_q_block_count_bsa_k2q_csr(
    bsa_k2q_csr_tensors: BsaK2qCsrTensors,
    batch_idx,
    head_idx,
    q_group,
    n_block,
):
    """Return the number of Q tiles contributing to one KV tile."""
    bucketed_k2q_offsets, _ = bsa_k2q_csr_tensors
    begin = bucketed_k2q_offsets[batch_idx, head_idx, q_group, n_block]
    end = bucketed_k2q_offsets[batch_idx, head_idx, q_group, n_block + 1]
    return end - begin


@cute.jit
def get_bsa_k2q_csr_tile_coord(
    sched_n_block,
    bsa_k2q_csr_tensors: BsaK2qCsrTensors,
):
    """Map scheduler N tile to (q_group, kv_block) for BSA k2q CSR."""
    bucketed_k2q_offsets, _ = bsa_k2q_csr_tensors
    num_kv_blocks = cute.size(bucketed_k2q_offsets.shape[3]) - 1
    q_group = sched_n_block // num_kv_blocks
    n_block = sched_n_block - q_group * num_kv_blocks
    return q_group, n_block


@cute.jit
def get_bsa_k2q_csr_iteration_info_bwd(
    bsa_k2q_csr_tensors: BsaK2qCsrTensors,
    batch_idx,
    head_idx,
    q_group,
    n_block,
):
    """Return CSR state consumed by the BSA k2q CSR loop call sites.

    For BSA CSR, curr_q_cnt carries the CSR begin offset and curr_q_idx is the
    full per-head edge vector.
    """
    bucketed_k2q_offsets, bucketed_k2q_indices = bsa_k2q_csr_tensors
    begin = bucketed_k2q_offsets[batch_idx, head_idx, q_group, n_block]
    end = bucketed_k2q_offsets[batch_idx, head_idx, q_group, n_block + 1]
    curr_q_idx = bucketed_k2q_indices[batch_idx, head_idx, None]
    return begin, curr_q_idx, end - begin


@cute.jit
def get_m_block_from_iter_bwd(
    iter_idx,
    curr_q_cnt,
    curr_q_idx: cute.Tensor,
):
    """Map an iteration in one CSR row to the actual Q tile index."""
    return curr_q_idx[curr_q_cnt + iter_idx]


@cute.jit
def produce_bsa_k2q_csr_q_loads_bwd_sm100(
    bsa_k2q_csr_tensors: BsaK2qCsrTensors,
    batch_idx,
    head_idx,
    q_group,
    n_block,
    producer_state_Q_LSE,
    producer_state_dO_dPsum,
    pipeline_Q,
    pipeline_LSE,
    pipeline_dO,
    pipeline_dPsum,
    load_K,
    load_V,
    load_Q,
    load_dO,
    copy_stats,
    gLSE,
    sLSE,
    gdPsum,
    sdPsum,
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    m_block_max: int = 0,
):
    """Load Q/dO/LSE/dPsum tiles in BSA bucketed k2q CSR order."""
    (
        curr_q_cnt,
        curr_q_idx,
        loop_count,
    ) = get_bsa_k2q_csr_iteration_info_bwd(
        bsa_k2q_csr_tensors,
        batch_idx,
        head_idx,
        q_group,
        n_block,
    )

    for iter_idx in cutlass.range(loop_count, unroll=1):
        m_block = get_m_block_from_iter_bwd(iter_idx, curr_q_cnt, curr_q_idx)
        m_block_safe = m_block
        if m_block_max > 0:
            m_block_safe = cutlass.min(m_block, m_block_max - 1)

        if iter_idx == 0:
            pipeline_Q.producer_acquire(producer_state_Q_LSE, extra_tx_count=tma_copy_bytes_K)
            load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE))
            load_Q(m_block_safe, producer_state=producer_state_Q_LSE)
            pipeline_Q.producer_commit(producer_state_Q_LSE)
            pipeline_LSE.producer_acquire(producer_state_Q_LSE)
            with copy_utils.bulk_copy_elect_one():
                copy_stats(
                    gLSE[None, m_block_safe],
                    sLSE[None, producer_state_Q_LSE.index],
                    mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                )
            producer_state_Q_LSE.advance()

            pipeline_dO.producer_acquire(producer_state_dO_dPsum, extra_tx_count=tma_copy_bytes_V)
            load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_dPsum))
            load_dO(m_block_safe, producer_state=producer_state_dO_dPsum)
            pipeline_dO.producer_commit(producer_state_dO_dPsum)
            pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
            with copy_utils.bulk_copy_elect_one():
                copy_stats(
                    gdPsum[None, m_block_safe],
                    sdPsum[None, producer_state_dO_dPsum.index],
                    mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                )
            producer_state_dO_dPsum.advance()
        else:
            pipeline_Q.producer_acquire(producer_state_Q_LSE)
            load_Q(m_block_safe, producer_state=producer_state_Q_LSE)
            pipeline_Q.producer_commit(producer_state_Q_LSE)
            pipeline_LSE.producer_acquire(producer_state_Q_LSE)
            with copy_utils.bulk_copy_elect_one():
                copy_stats(
                    gLSE[None, m_block_safe],
                    sLSE[None, producer_state_Q_LSE.index],
                    mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                )
            producer_state_Q_LSE.advance()

            pipeline_dO.producer_acquire(producer_state_dO_dPsum)
            load_dO(m_block_safe, producer_state=producer_state_dO_dPsum)
            pipeline_dO.producer_commit(producer_state_dO_dPsum)
            pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
            with copy_utils.bulk_copy_elect_one():
                copy_stats(
                    gdPsum[None, m_block_safe],
                    sdPsum[None, producer_state_dO_dPsum.index],
                    mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                )
            producer_state_dO_dPsum.advance()

    return producer_state_Q_LSE, producer_state_dO_dPsum


class BlockSparseAttnBackwardSm100Blk128:
    arch = 100

    def __init__(
        self,
        head_dim: int,
        force_dkv_postprocess: cutlass.Constexpr = False,
    ):
        assert head_dim in (64, 128), f"SM100 blk128 bwd supports head_dim in {{64, 128}}, got {head_dim}"
        # tile_hdim drives the K reduction of S/P/dV/dK mma's and the dQ accumulator
        # cols-per-stage; tile_hdimv drives the N of PV-style mma's. Both follow
        # head_dim directly (no head_dim_v split for now).
        self.tile_hdim = head_dim
        self.tile_hdimv = head_dim

        self.tile_m = 128
        self.tile_n = 128

        # CTA tiler
        self.cta_tiler = (self.tile_n, self.tile_m, self.tile_hdim)
        # S = K @ Q.T
        self.mma_tiler_kq = (self.tile_n, self.tile_m, self.tile_hdim)
        # dP = V @ dO.T
        self.mma_tiler_vdo = (self.tile_n, self.tile_m, self.tile_hdimv)
        # dV = P.T @ dO
        self.mma_tiler_pdo = (self.tile_n, self.tile_hdimv, self.tile_m)
        # dK = dS.T @ Q
        self.mma_tiler_dsq = (self.tile_n, self.tile_hdim, self.tile_m)
        # dQ = dS @ K
        self.mma_tiler_dsk = (self.tile_m, self.tile_hdim, self.tile_n)

        self.acc_dtype = Float32

        self.force_dkv_postprocess = force_dkv_postprocess

        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.idle_warp_id = 14
        self.empty_warp_id = 15

        # 16 warps -> 512 threads
        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.reduce_warp_ids,
                *self.compute_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.idle_warp_id,
                self.empty_warp_id,
            )
        )
        self.compute_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.Compute),
            num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )
        self.reduce_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.dQaccReduce),
            num_threads=len(self.reduce_warp_ids) * cute.arch.WARP_SIZE,
        )
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.tmem_S_offset = 0
        self.tmem_P_offset = 0  # overlap with S
        self.tmem_dV_offset = self.tmem_S_offset + self.tile_n
        self.tmem_dP_offset = self.tmem_dV_offset + self.tile_hdimv
        self.tmem_dQ_offset = self.tmem_dP_offset
        self.tmem_dK_offset = self.tmem_dP_offset + self.tile_m
        self.tmem_dS_offset = self.tmem_dP_offset  # overlap with dP

        self.num_regs_reduce = 152
        self.num_regs_compute = 136
        self.num_regs_load = 96 - 8
        self.num_regs_mma = self.num_regs_load
        self.num_regs_empty = 24

        assert self.num_regs_reduce + self.num_regs_compute * 2 + max(self.num_regs_load, self.num_regs_mma) <= 512
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.Q_stage = 2
        self.dO_stage = 1
        self.single_stage = 1
        # LSE_stage = Q_stage and dPsum_stage = dO_stage
        self.sdKVaccum_stage = 2
        self.dQ_reduce_ncol = 32
        self.sdQaccum_stage = 64 // self.dQ_reduce_ncol
        self.dQ_reduce_ncol_t2r = self.dQ_reduce_ncol
        assert self.tile_hdim % self.dQ_reduce_ncol == 0
        self.dQaccum_reduce_stage = self.tile_hdim // self.dQ_reduce_ncol
        self.dQaccum_reduce_stage_t2r = self.tile_hdim // self.dQ_reduce_ncol_t2r
        self.dK_reduce_ncol = math.gcd(32, self.tile_hdim // 2)

    def _get_tiled_mma(self):
        # S.T = K @ Q.T
        tiled_mma_S = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_kq[:2],
        )
        # dP.T = V @ dO.T
        tiled_mma_dP = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            self.do_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_vdo[:2],
        )
        # dV += P.T @ dO --> (K, MN) major
        tiled_mma_dV = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            self.do_dtype,
            OperandMajorMode.K,  # P_major_mode
            OperandMajorMode.MN,  # dO_major_mode
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_pdo[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        # dK += dS.T @ Q
        tiled_mma_dK = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            self.do_dtype,
            OperandMajorMode.K,  # dS_major_mode
            OperandMajorMode.MN,  # Q_major_mode
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_dsq[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        # dQ = dS @ K
        tiled_mma_dQ = sm100_utils_basic.make_trivial_tiled_mma(
            self.k_dtype,
            self.k_dtype,
            OperandMajorMode.MN,  # dS_major_mode
            OperandMajorMode.MN,  # Kt_major_mode
            self.acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_dsk[:2],
        )
        return tiled_mma_S, tiled_mma_dP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _setup_smem_layout(self):
        # S.T = K @ Q.T
        sK_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.k_dtype,
            1,
        )
        self.sK_layout = cute.slice_(sK_layout, (None, None, None, 0))
        self.sQ_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.q_dtype,
            self.Q_stage,
        )
        # dP.T = V @ dO.T
        sV_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.v_dtype,
            1,
        )
        self.sV_layout = cute.slice_(sV_layout, (None, None, None, 0))
        self.sdOt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dV += P.T @ dO
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            1,
        )
        self.tP_layout = cute.slice_(tP_layout, (None, None, None, 0))
        self.sdO_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dK += dS.T @ Q
        sdSt_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.sdSt_layout = cute.slice_(sdSt_layout, (None, None, None, 0))
        tdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.tdS_layout = cute.slice_(tdS_layout, (None, None, None, 0))
        self.sQt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.q_dtype,
            self.Q_stage,
        )
        # dQ = dS @ K
        sdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.ds_dtype,
            1,
        )
        self.sdS_layout = cute.slice_(sdS_layout, (None, None, None, 0))
        sKt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.k_dtype,
            1,
        )
        self.sKt_layout = cute.slice_(sKt_layout, (None, None, None, 0))
        self.sdQaccum_layout = cute.make_layout((self.tile_m * self.dQ_reduce_ncol, self.sdQaccum_stage))
        self.sLSE_layout = cute.make_layout(shape=(self.tile_m, self.Q_stage), stride=(1, cute.round_up(self.tile_m, 64)))
        self.sdPsum_layout = cute.make_layout(
            shape=(self.tile_m, self.dO_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )
        self.sdK_epi_tile = (
            self.tile_n,
            math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdim // 2),  # 64 or 32
        )  # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        self.sdV_epi_tile = (
            self.tile_n,
            math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdimv // 2),  # 64 or 32
        )  # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        # headdim_64 gets 1 stage
        self.num_epi_stages = max(1, (self.tile_hdim // 2) // self.sdK_epi_tile[1])
        self.num_epi_stages_v = max(1, (self.tile_hdimv // 2) // self.sdV_epi_tile[1])
        self.sdK_flat_epi_tile = self.tile_n * (self.tile_hdim // 2) // self.num_epi_stages
        self.sdV_flat_epi_tile = self.tile_n * (self.tile_hdimv // 2) // self.num_epi_stages_v
        if const_expr(not self.dKV_postprocess):
            self.sdK_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dk_dtype,
                LayoutEnum.ROW_MAJOR,
                self.sdK_epi_tile,
                2,  # num compute wgs
            )
            self.sdV_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dv_dtype,
                LayoutEnum.ROW_MAJOR,
                self.sdV_epi_tile,
                2,  # num compute wgs
            )
        else:
            self.sdK_layout = cute.make_layout((self.tile_n * self.dK_reduce_ncol, 2))
            # self.dK_reduce_ncol same for dV
            self.sdV_layout = cute.make_layout((self.tile_n * self.dK_reduce_ncol, 2))

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        bsa_k2q_csr_tensors: BsaK2qCsrTensors,
        stream: cuda.CUstream = None,
    ):
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.do_dtype = mdO.element_type
        self.lse_dtype = mLSE.element_type
        self.dpsum_dtype = mdPsum.element_type
        self.dqaccum_dtype = mdQaccum.element_type
        self.dk_dtype = mdK.element_type
        self.dv_dtype = mdV.element_type
        self.ds_dtype = self.q_dtype

        self.dKV_postprocess = self.force_dkv_postprocess

        if const_expr(self.dKV_postprocess):
            assert self.dk_dtype.width == 32
            assert self.dv_dtype.width == 32

        mdQaccum, mdK, mdV = [assume_tensor_aligned(t) for t in (mdQaccum, mdK, mdV)]

        QO_layout_transpose = [1, 3, 2, 0]
        mQ, mdO = [layout_utils.select(t, mode=QO_layout_transpose) for t in (mQ, mdO)]

        KV_layout_transpose = [1, 3, 2, 0]
        mK, mV = [layout_utils.select(t, mode=KV_layout_transpose) for t in (mK, mV)]

        LSE_dPsum_dQaccum_transpose = [2, 1, 0]
        mLSE, mdPsum, mdQaccum = [layout_utils.select(t, mode=LSE_dPsum_dQaccum_transpose) for t in (mLSE, mdPsum, mdQaccum)]

        if const_expr(not self.dKV_postprocess):
            layout_dKV_transpose = KV_layout_transpose
        else:
            layout_dKV_transpose = [2, 1, 0]
        mdK, mdV = [layout_utils.select(t, mode=layout_dKV_transpose) for t in (mdK, mdV)]
        dO_transpose = [1, 0, 2, 3]
        mdO = layout_utils.select(mdO, mode=dO_transpose)

        self._setup_attributes()
        (
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dK,
            self.tiled_mma_dV,
            self.tiled_mma_dQ,
        ) = self._get_tiled_mma()
        self._setup_smem_layout()

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)),
            (self.tiled_mma_S.thr_id.shape,),
        )

        if const_expr(not self.dKV_postprocess):
            self.mdK_layout_enum = LayoutEnum.from_tensor(mdK)
            self.mdV_layout_enum = LayoutEnum.from_tensor(mdV)
            dK_major_mode = self.mdK_layout_enum.mma_major_mode()
            dV_major_mode = self.mdV_layout_enum.mma_major_mode()
            if const_expr(dK_major_mode != OperandMajorMode.K):
                raise RuntimeError("The layout of mdK is wrong")
            if const_expr(dV_major_mode != OperandMajorMode.K):
                raise RuntimeError("The layout of mdV is wrong")

        if const_expr(not self.dKV_postprocess):
            tma_copy_op_dKV = cpasync.CopyBulkTensorTileS2GOp()
            tma_atom_dK, mdK_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdK,
                cute.select(self.sdK_layout, mode=[0, 1]),
                self.sdK_epi_tile,
                1,  # no mcast
            )
            tma_atom_dV, mdV_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdV,
                cute.select(self.sdV_layout, mode=[0, 1]),
                self.sdV_epi_tile,
                1,  # no mcast
            )
        else:
            mdV_tma_tensor = mdV
            mdK_tma_tensor = mdK
            tma_atom_dV = None
            tma_atom_dK = None

        if const_expr(not self.dKV_postprocess):
            thr_layout_r2s_dKV = cute.make_ordered_layout((128, 1), order=(1, 0))  # 128 threads
            val_layout_r2s_dKV = cute.make_ordered_layout((1, 128 // self.dk_dtype.width), order=(1, 0))  # 4 or 8 vals for 16 byte store
            copy_atom_r2s_dKV = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dk_dtype,
                num_bits_per_copy=128,
            )
            tiled_copy_r2s_dKV = cute.make_tiled_copy_tv(copy_atom_r2s_dKV, thr_layout_r2s_dKV, val_layout_r2s_dKV)
        else:
            tiled_copy_r2s_dKV = copy_utils.tiled_copy_1d(Float32, 128, num_copy_elems=128 // Float32.width)

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        # S.T = K @ Q.T
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK,
            cute.select(self.sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            cluster_layout_vmnk.shape,
        )
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            cluster_layout_vmnk.shape,
        )
        # dP.T = V @ dO.T
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV,
            cute.select(self.sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            self.tiled_mma_dP,
            cluster_layout_vmnk.shape,
        )
        # dV = P.T @ dO
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_pdo,
            self.tiled_mma_dV,
            cluster_layout_vmnk.shape,
        )
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = self.tile_m * self.dQ_reduce_ncol * Float32.width // 8
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.dK_reduce_ncol * Float32.width // 8

        TileScheduler = SingleTileScheduler
        bucketed_k2q_offsets, _ = bsa_k2q_csr_tensors
        num_sched_n_blocks = (cute.size(bucketed_k2q_offsets.shape[3]) - 1) * cute.size(bucketed_k2q_offsets.shape[2])
        tile_sched_args = TileSchedulerArguments(
            num_sched_n_blocks,  # num_blocks
            cute.size(mQ.shape[2]),  # num_heads = num_query_heads
            cute.size(mK.shape[3]),  # num_batches
            1,  # num_splits
            cute.size(mQ.shape[0]),  # pass seqlen_q or total_q for seqlen_k
            mQ.shape[1],  # headdim
            mV.shape[1],  # headdim_v
            total_q=cute.size(mK.shape[0]) * cute.size(mK.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],  # (tile_n, tile_m)
            qhead_per_kvhead_packgqa=1,
            element_size=self.k_dtype.width // 8,
            lpt=False,
            head_swizzle=False,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # Compute allocation sizes for shared buffers that are reused
        # sQ is reused for sdK, sdO is reused for sdV
        sQ_alloc_bytes = max(
            cute.size_in_bytes(self.q_dtype, self.sQ_layout),
            cute.size_in_bytes(self.dk_dtype, self.sdK_layout),
        )
        sdO_alloc_bytes = max(
            cute.size_in_bytes(self.dv_dtype, self.sdV_layout),
            cute.size_in_bytes(self.do_dtype, self.sdO_layout),
        )

        sdK_bytes = cute.size_in_bytes(self.dk_dtype, self.sdK_layout)
        sdV_bytes = cute.size_in_bytes(self.dv_dtype, self.sdV_layout)
        assert sdV_bytes <= sdO_alloc_bytes, "sdV doesn't fit in sdO storage allocation"
        assert sdK_bytes <= sQ_alloc_bytes, "sdK doesn't fit in sQ storage allocation"

        @cute.struct
        class SharedStorage:
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
            dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
            LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
            dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
            S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
            dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
            dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
            dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.sdKVaccum_stage]
            dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: Int32

            sQ: cute.struct.Align[
                cute.struct.MemRange[cute.Uint8, sQ_alloc_bytes],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(self.sK_layout)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(self.sV_layout)],
                self.buffer_align_bytes,
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[cute.Uint8, sdO_alloc_bytes],
                self.buffer_align_bytes,
            ]
            sdS: cute.struct.Align[
                cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                128,
            ]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                128,
            ]
            sdPsum: cute.struct.Align[
                cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
                128,
            ]
            sdQaccum: cute.struct.Align[
                cute.struct.MemRange[self.dqaccum_dtype, cute.cosize(self.sdQaccum_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            mLSE,
            mdPsum,
            tma_tensor_dO,
            mdV,
            mdK,
            mdQaccum,
            mdV_tma_tensor,
            mdK_tma_tensor,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dV,
            tma_atom_dK,
            self.sQ_layout,
            self.sQt_layout,
            self.sK_layout,
            self.sKt_layout,
            self.sV_layout,
            self.sLSE_layout,
            self.sdPsum_layout,
            self.sdO_layout,
            self.sdOt_layout,
            self.sdSt_layout,
            self.sdS_layout,
            self.sdQaccum_layout,
            self.sdK_layout,
            self.sdV_layout,
            self.tP_layout,
            self.tdS_layout,
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dV,
            self.tiled_mma_dK,
            self.tiled_mma_dQ,
            tiled_copy_r2s_dKV,
            softmax_scale,
            softmax_scale_log2,
            tile_sched_params,
            bsa_k2q_csr_tensors,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        sQ_layout: cute.ComposedLayout,
        sQt_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sdPsum_layout: cute.Layout,
        sdO_layout: cute.ComposedLayout,
        sdOt_layout: cute.ComposedLayout,
        sdSt_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        sdK_layout: cute.ComposedLayout | cute.Layout,
        sdV_layout: cute.ComposedLayout | cute.Layout,
        tP_layout: cute.ComposedLayout,
        tdS_layout: cute.ComposedLayout,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_copy_r2s_dKV: cute.TiledCopy,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        tile_sched_params: ParamsBase,
        bsa_k2q_csr_tensors: Optional[BsaK2qCsrTensors] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_Q)
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V)
                cpasync.prefetch_descriptor(tma_atom_dO)
                if const_expr(tma_atom_dV is not None):
                    cpasync.prefetch_descriptor(tma_atom_dV)
                if const_expr(tma_atom_dK is not None):
                    cpasync.prefetch_descriptor(tma_atom_dK)

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)),
            (tiled_mma_S.thr_id.shape,),
        )

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE * len((self.mma_warp_id, *self.compute_warp_ids, *self.reduce_warp_ids)),
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
        )

        # UMMA producers and AsyncThread consumers
        pipeline_umma_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len([self.mma_warp_id]))
        pipeline_async_consumer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids))
        pipeline_S_P = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_umma_producer_group,
            consumer_group=pipeline_async_consumer_group,
            barrier_storage=storage.S_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dP = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_umma_producer_group,
            consumer_group=pipeline_async_consumer_group,
            barrier_storage=storage.dP_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dKV = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            producer_group=pipeline_umma_producer_group,
            consumer_group=pipeline_async_consumer_group,
            barrier_storage=storage.dKV_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        pipeline_dQ_async_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.reduce_warp_ids),
        )  # Compute
        pipeline_dQ = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_umma_producer_group,
            consumer_group=pipeline_dQ_async_consumer_group,
            barrier_storage=storage.dQ_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # AsyncThread producers and UMMA consumers
        # Only 1 thread per warp will signal
        pipeline_PdS_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids),
        )  # Compute
        pipeline_PdS_consumer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len([self.mma_warp_id]))  # MMA
        pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline_PdS_producer_group,
            consumer_group=pipeline_PdS_consumer_group,
            barrier_storage=storage.dS_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # TMA producer and UMMA consumers
        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len([self.load_warp_id]))
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len([self.mma_warp_id]))
        pipeline_consumer_group_compute = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * 1,
        )
        pipeline_LSE = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.LSE_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["LSE"],
            defer_sync=True,
        )
        pipeline_dPsum = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.dPsum_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["dPsum"],
            defer_sync=True,
        )
        pipeline_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        pipeline_dO = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.dO_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=False,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype)
        sQt = cute.make_tensor(cute.recast_ptr(sQ.iterator, sQt_layout.inner, dtype=self.q_dtype), sQt_layout.outer)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sKt = cute.make_tensor(cute.recast_ptr(sK.iterator, sKt_layout.inner), sKt_layout.outer)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sdSt = storage.sdS.get_tensor(sdSt_layout.outer, swizzle=sdSt_layout.inner)
        sdS = cute.make_tensor(cute.recast_ptr(sdSt.iterator, sdS_layout.inner), sdS_layout.outer)

        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner, dtype=self.do_dtype)
        sdOt = cute.make_tensor(
            cute.recast_ptr(sdO.iterator, sdOt_layout.inner, dtype=self.do_dtype),
            sdOt_layout.outer,
        )

        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)
        if const_expr(not self.dKV_postprocess):
            sdV = storage.sdO.get_tensor(sdV_layout.outer, swizzle=sdV_layout.inner, dtype=self.dv_dtype)
            sdK = storage.sQ.get_tensor(sdK_layout.outer, swizzle=sdK_layout.inner, dtype=self.dk_dtype)
        else:
            sdV = storage.sdO.get_tensor(sdV_layout, dtype=self.dv_dtype)
            sdK = storage.sQ.get_tensor(sdK_layout, dtype=self.dk_dtype)

        # Buffer sizing is guaranteed by max(...) in SharedStorage declarations
        # for both sQ (reused as sdK) and sdO (reused as sdV)
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        # TMEM
        # This is a fake tensor, by right need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        # S
        thr_mma_S = tiled_mma_S.get_slice(0)
        Sacc_shape = thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2])  # (M, N)
        tStS = thr_mma_S.make_fragment_C(Sacc_shape)
        # (MMA, MMA_M, MMA_N)
        tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS.layout)
        # dP
        thr_mma_dP = tiled_mma_dP.get_slice(0)
        dPacc_shape = thr_mma_dP.partition_shape_C(self.mma_tiler_vdo[:2])
        tdPtdP = thr_mma_dP.make_fragment_C(dPacc_shape)
        tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPtdP.layout)
        # dV
        thr_mma_dV = tiled_mma_dV.get_slice(0)
        dvacc_shape = thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2])
        tdVtdV = thr_mma_dV.make_fragment_C(dvacc_shape)
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)
        tP = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_P_offset, dtype=self.do_dtype), tP_layout.outer)
        # dK
        thr_mma_dK = tiled_mma_dK.get_slice(0)
        dkacc_shape = thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2])
        tdKtdK = thr_mma_dK.make_fragment_C(dkacc_shape)
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)
        tdS = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_dS_offset, dtype=self.ds_dtype), tdS_layout.outer)
        # dQ
        thr_mma_dQ = tiled_mma_dQ.get_slice(0)
        dQacc_shape = thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2])
        tdQtdQ = thr_mma_dQ.make_fragment_C(dQacc_shape)
        tdQtdQ = cute.make_tensor(tmem_ptr + self.tmem_dQ_offset, tdQtdQ.layout)

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            False,
            False,
            False,  # is_split_kv
            None,
            None,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        #  EMPTY
        # (15)
        if warp_idx == self.empty_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)

        #  IDLE
        # (14)
        if warp_idx == self.idle_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)

        #  LOAD
        # (13)
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                mQ,
                mK,
                mV,
                mdO,
                mLSE,
                mdPsum,
                sQ,
                sK,
                sV,
                sdO,
                sLSE,
                sdPsum,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_dO,
                pipeline_Q,
                pipeline_dO,
                pipeline_LSE,
                pipeline_dPsum,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                bsa_k2q_csr_tensors,
            )

        #  MMA
        # (12)
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)

            # Alloc tmem buffer
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)

            self.mma(
                tiled_mma_S,
                tiled_mma_dP,
                tiled_mma_dV,
                tiled_mma_dK,
                tiled_mma_dQ,
                sQ,
                sQt,
                sK,
                sKt,
                sV,
                sdO,
                sdOt,
                tP,
                sdS,
                tdS,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                tdQtdQ,
                pipeline_Q,
                pipeline_dO,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                pipeline_dQ,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                bsa_k2q_csr_tensors,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # Compute
        # (4, 5, 6, 7, 8, 9, 10, 11) --> 8 warps
        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_compute)  # 8 warps
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.compute_loop(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                sLSE,
                sdPsum,
                mdV,
                mdK,
                sdS,
                pipeline_LSE,
                pipeline_dPsum,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                softmax_scale,
                softmax_scale_log2,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                sdV,
                sdK,
                mdV_tma_tensor,
                mdK_tma_tensor,
                tma_atom_dV,
                tma_atom_dK,
                tiled_copy_r2s_dKV,
                bsa_k2q_csr_tensors,
            )
            tmem_alloc_barrier.arrive()

        # Reduce
        # (0, 1, 2, 3) - dQ
        if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_reduce)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.dQacc_reduce(
                mdQaccum,
                sdQaccum,
                thr_mma_dQ,
                tdQtdQ,
                pipeline_dQ,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                bsa_k2q_csr_tensors,
            )
            tmem_alloc_barrier.arrive()

        return

    @cute.jit
    def load(
        self,
        thr_mma_S: cute.ThrMma,
        thr_mma_dP: cute.ThrMma,
        thr_mma_dV: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        bsa_k2q_csr_tensors: Optional[BsaK2qCsrTensors] = None,
    ):
        producer_state_Q_LSE = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.Q_stage)
        producer_state_dO_dPsum = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.dO_stage)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            q_group, n_block = get_bsa_k2q_csr_tile_coord(n_block, bsa_k2q_csr_tensors)
            seqlen = SeqlenInfoCls(batch_idx)
            _, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx]
            mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx]
            mdO_cur = mdO[None, None, head_idx, batch_idx]
            mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2, padded=True)[None, head_idx]
            mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2, padded=True)[None, head_idx]

            # (1) S.T = K @ Q.T
            gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block, 0))
            tSgK = thr_mma_S.partition_A(gK)

            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0))
            tSgQ = thr_mma_S.partition_B(gQ)
            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
            gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))
            gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
            tdPgdO = thr_mma_dV.partition_B(gdO)

            load_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K,
                0,
                cute.make_layout(1),
                tSgK,
                sK,
                single_stage=True,
            )

            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                src_tensor=tSgQ,
                dst_tensor=sQ,
            )
            load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)

            # (2) dP = V @ dO.T
            gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (n_block, 0))
            tdPgV = thr_mma_dP.partition_A(gV)

            load_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V,
                0,
                cute.make_layout(1),
                tdPgV,
                sV,
                single_stage=True,
            )

            # (3) dV += P.T @ dO
            gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
            tdVgdO = thr_mma_dV.partition_B(gdO)
            load_dO, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                src_tensor=tdVgdO,
                dst_tensor=sdO,
            )
            load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)

            copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
            copy_stats = partial(cute.copy, copy_atom_stats)

            total_m_block_cnt = get_total_q_block_count_bsa_k2q_csr(
                bsa_k2q_csr_tensors,
                batch_idx,
                head_idx,
                q_group,
                n_block,
            )
            process_tile = total_m_block_cnt > Int32(0)

            if process_tile:
                producer_state_Q_LSE, producer_state_dO_dPsum = produce_bsa_k2q_csr_q_loads_bwd_sm100(
                    bsa_k2q_csr_tensors,
                    batch_idx,
                    head_idx,
                    q_group,
                    n_block,
                    producer_state_Q_LSE,
                    producer_state_dO_dPsum,
                    pipeline_Q,
                    pipeline_LSE,
                    pipeline_dO,
                    pipeline_dPsum,
                    load_K,
                    load_V,
                    load_Q,
                    load_dO,
                    copy_stats,
                    gLSE,
                    sLSE,
                    gdPsum,
                    sdPsum,
                    self.tma_copy_bytes["K"],
                    self.tma_copy_bytes["V"],
                    m_block_max=m_block_max,
                )

                pipeline_Q.producer_tail(producer_state_Q_LSE.clone())
                pipeline_LSE.producer_tail(producer_state_Q_LSE)
                pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
                pipeline_dPsum.producer_tail(producer_state_dO_dPsum)

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma(
        self,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        sQ: cute.Tensor,
        sQt: cute.Tensor,
        sK: cute.Tensor,
        sKt: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdOt: cute.Tensor,
        tP: cute.Tensor,
        sdS: cute.Tensor,
        tdS: cute.Tensor,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdQtdQ: cute.Tensor,
        pipeline_Q: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        pipeline_dQ: PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        bsa_k2q_csr_tensors: Optional[BsaK2qCsrTensors] = None,
    ):
        # S = K @ Q.T
        tSrK = tiled_mma_S.make_fragment_A(sK)
        tSrQ = tiled_mma_S.make_fragment_B(sQ)
        # dP = V @ dOt.T
        tdPrV = tiled_mma_dP.make_fragment_A(sV)
        tdPrdOt = tiled_mma_dP.make_fragment_B(sdOt)
        # dK = dS.T @ Q
        tdKrdS = tiled_mma_dK.make_fragment_A(tdS)

        tdKrQ = tiled_mma_dK.make_fragment_B(sQt)
        # dQ = dS @ K
        tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)
        tdQrK = tiled_mma_dQ.make_fragment_B(sKt)
        # dV = P @ dO.T
        tdVrdO = tiled_mma_dV.make_fragment_B(sdO)
        tdVrP = tiled_mma_dV.make_fragment_A(tP)

        mma_qk_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_S,
            tStS,
            tSrK,
            tSrQ,
            sA=sK,
            sB=sQ,
            zero_init=True,
        )
        mma_dov_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dP,
            tdPtdP,
            tdPrV,
            tdPrdOt,
            sA=sV,
            sB=sdOt,
            zero_init=True,
        )
        mma_pdo_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dV,
            tdVtdV,
            tdVrP,
            tdVrdO,
            sA=None,
            sB=sdO,
            tA_addr=self.tmem_P_offset,
        )
        mma_dsk_fn = partial(
            gemm_w_idx,
            tiled_mma_dQ,
            tdQtdQ,
            tdQrdS,
            tdQrK,
            zero_init=True,
            num_unroll_groups=1,
        )
        mma_dsq_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dK,
            tdKtdK,
            tdKrdS,
            tdKrQ,
            sA=None,
            sB=sQt,
            tA_addr=self.tmem_dS_offset,
        )

        pipeline_Q_consumer = pipeline_Q.make_consumer()

        consumer_state_Q = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage)
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage)
        producer_phase_acc = Int32(1)  # For S & P, dP, dQ
        consumer_state_dS = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
        producer_phase_dKV = Int32(1)
        cta_group = pipeline_S_P.cta_group

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            q_group, n_block = get_bsa_k2q_csr_tile_coord(n_block, bsa_k2q_csr_tensors)
            seqlen = SeqlenInfoCls(batch_idx)  # must be seqlen_k
            _, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

            block_iter_count = get_total_q_block_count_bsa_k2q_csr(
                bsa_k2q_csr_tensors,
                batch_idx,
                head_idx,
                q_group,
                n_block,
            )
            process_tile = block_iter_count > Int32(0)

            if process_tile:
                # Loop-carried runtime flag: Python `not` on it would bake a constant
                # accumulate predicate at trace time, so carry the zero_init value
                # itself as a dynamic Boolean and pass it straight to the MMA helper.
                zero_init_dK = Boolean(True)

                # 1) S = K @ Q
                handle_Q = pipeline_Q_consumer.wait_and_advance()
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                mma_qk_fn(B_idx=handle_Q.index)
                pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                # 2) dP = V @ dO.T
                pipeline_dO.consumer_wait(consumer_state_dO)
                pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
                mma_dov_fn(B_idx=consumer_state_dO.index)
                pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                producer_phase_acc ^= 1
                # 3) dV = P.T @ dO
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
                pipeline_dO.consumer_release(consumer_state_dO)
                consumer_state_dO.advance()

                main_loop_iters = block_iter_count - 1

                handle_Q_next = handle_Q
                for _ in cutlass.range(main_loop_iters, unroll=1):
                    # (1) S.T = K @ Q.T
                    handle_Q_next = pipeline_Q_consumer.wait_and_advance()
                    mma_qk_fn(B_idx=handle_Q_next.index)
                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                    # (2) dK += dS.T @ Q
                    pipeline_dS.consumer_wait(consumer_state_dS)
                    mma_dsq_fn(B_idx=handle_Q.index, zero_init=zero_init_dK)
                    zero_init_dK = Boolean(False)
                    handle_Q.release()

                    # (3) dQ = dS @ K
                    mma_dsk_fn()
                    pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                    pipeline_dS.consumer_release(consumer_state_dS)
                    consumer_state_dS.advance()

                    # (4) dP = V @ dO.T
                    pipeline_dO.consumer_wait(consumer_state_dO)
                    pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
                    mma_dov_fn(B_idx=consumer_state_dO.index)
                    pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                    # (5) dV += P.T @ dO
                    producer_phase_acc ^= 1
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()

                    handle_Q = handle_Q_next

                pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)

                # Tail: remaining dK and dQ.
                pipeline_dS.consumer_wait(consumer_state_dS)
                mma_dsq_fn(B_idx=handle_Q.index, zero_init=zero_init_dK)
                pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                producer_phase_dKV ^= 1

                mma_dsk_fn()
                pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                handle_Q.release()
                pipeline_dS.consumer_release(consumer_state_dS)
                consumer_state_dS.advance()

                producer_phase_acc ^= 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def split_wg(
        self,
        t: cute.Tensor,
        wg_idx: cutlass.Int32,
        num_wg: cutlass.Constexpr[int],
    ):
        reduced_shape = cute.product_each(t.shape)
        rank = len(reduced_shape)
        if const_expr(reduced_shape[1] > 1):
            assert rank >= 2, "Need rank >= 2 for t in split_wg"
            t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1] // num_wg))
            coord = (None, (None, wg_idx)) + (None,) * (rank - 2)
        else:
            assert rank >= 3, "Need rank >= 3 for t in split_wg"
            if const_expr(rank == 3):
                t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1], reduced_shape[2] // num_wg))
                coord = (
                    None,
                    None,
                    (None, wg_idx),
                ) + (
                    None,
                ) * (rank - 3)
            else:
                t = cute.logical_divide(
                    t,
                    (
                        reduced_shape[0],
                        reduced_shape[1],
                        reduced_shape[2],
                        reduced_shape[3] // num_wg,
                    ),
                )
                coord = (
                    None,
                    None,
                    None,
                    (None, wg_idx),
                ) + (
                    None,
                ) * (rank - 4)
        return t[coord]

    @cute.jit
    def apply_seqlen_k_mask(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        n_block: Int32,
        seqlen,
    ):
        col = 0
        thr_col_offset = tScS_t2r[0][col]
        seqlenk_col_limit = seqlen.seqlen_k - n_block * self.tile_n - thr_col_offset
        if seqlenk_col_limit <= 0:
            for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                acc_S[i] = -cutlass.Float32.inf

    @cute.jit
    def compute_loop(
        self,
        thr_mma_S: cute.ThrMma,
        thr_mma_dP: cute.ThrMma,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        sdS: cute.Tensor,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        sdV: Optional[cute.Tensor],
        sdK: Optional[cute.Tensor],
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        tiled_copy_r2s_dKV: Optional[cute.TiledCopy],
        bsa_k2q_csr_tensors: Optional[BsaK2qCsrTensors] = None,
    ):
        sLSE_2D = cute.make_tensor(
            sLSE.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.Q_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sdPsum_2D = cute.make_tensor(
            sdPsum.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.dO_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sLSE_2D = layout_utils.transpose_view(sLSE_2D)
        sdPsum_2D = layout_utils.transpose_view(sdPsum_2D)

        # tix: [128...384]  8 warps
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())  # 4-11
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        dp_idx = tidx % 128
        num_wg = len(self.compute_warp_ids) // 4  # 2
        # wg_idx:
        # 0: [256...384]
        # 1: [128...256]

        tileP_f32_like = self.cta_tiler[1] // 32 * self.v_dtype.width
        # tStS has shape ((128, 128), 1, 1), tStP has shape ((128, 64), 1, 1)
        # tP overlap with tS
        tStP = cute.composition(tStS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        tStP = cute.make_tensor(tStS.iterator, tStP.layout)  # Otherwise the tmem address is wrong
        tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
        tScP = cute.composition(tScS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        # tdS overlap with tdP
        tdPtdS = cute.composition(tdPtdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        tdPcdP = thr_mma_dP.partition_C(cute.make_identity_tensor(self.mma_tiler_vdo[:2]))
        tdPcdS = cute.composition(tdPcdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))

        # Fixed repetition for the blk128 path.
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32)
        tmem_store_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32)

        # tmem -> rmem
        thr_copy_t2r = copy_utils.make_tmem_copy(tmem_load_atom, num_wg).get_slice(tidx)
        tStS_t2r = thr_copy_t2r.partition_S(tStS)  # (((32, 32), 1), 2, 1, 1)
        tdPtdP_t2r = thr_copy_t2r.partition_S(tdPtdP)
        tScS_t2r = thr_copy_t2r.partition_D(tScS)  # ((32, 1), 2, 1, 1)
        # ((32, 1), 2, 1, 1, STAGE)
        tSsLSE = thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))
        tSsdPsum = thr_copy_t2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D))
        # rmem -> tmem
        thr_copy_r2t = copy_utils.make_tmem_copy(tmem_store_atom, num_wg).get_slice(tidx)
        tScP_r2t = thr_copy_r2t.partition_S(tScP)
        tStP_r2t = thr_copy_r2t.partition_D(tStP)
        tdPcdS_r2t = thr_copy_r2t.partition_S(tdPcdS)
        tdPtdS_r2t = thr_copy_r2t.partition_D(tdPtdS)
        # rmem -> smem
        # This part is a bit iffy, we might be making a lot of assumptions here
        copy_atom_r2s = sm100_utils_basic.get_smem_store_op(LayoutEnum.ROW_MAJOR, self.ds_dtype, Float32, thr_copy_t2r)
        thr_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, thr_copy_t2r).get_slice(tidx)

        # We assume the swizzle (i.e. layout.inner) stays the same
        sdS_epi_layout = sm100_utils_basic.make_smem_layout_epi(self.ds_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_m), 1)
        sdS_layout = cute.slice_(sdS_epi_layout.outer, (None, None, 0))  # ((8,16), (64,2))
        # Need to group into 1 mode to be compatible w thr_copy_r2s
        sdS_layout = cute.make_layout((sdS_layout.shape,), stride=(sdS_layout.stride,))
        sdS_epi = cute.make_tensor(sdS.iterator, sdS_layout)
        tRS_sdS = thr_copy_r2s.partition_D(sdS_epi)

        consumer_state_S_P_dP = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
        producer_state_dS = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)
        consumer_state_dKV = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 2)
        consumer_state_LSE = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage)
        consumer_state_dPsum = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            q_group, n_block = get_bsa_k2q_csr_tile_coord(n_block, bsa_k2q_csr_tensors)
            seqlen = SeqlenInfoCls(batch_idx)
            _, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

            (
                curr_q_cnt,
                curr_q_idx,
                loop_count,
            ) = get_bsa_k2q_csr_iteration_info_bwd(
                bsa_k2q_csr_tensors,
                batch_idx,
                head_idx,
                q_group,
                n_block,
            )
            process_tile = loop_count > Int32(0)

            for iter_idx in cutlass.range(loop_count, unroll=1):
                m_block = get_m_block_from_iter_bwd(iter_idx, curr_q_cnt, curr_q_idx)
                pipeline_LSE.consumer_wait(consumer_state_LSE)
                tSrLSE_s2r = cute.make_rmem_tensor(tScS_t2r[None, 0, 0, 0].shape, Float32)

                pipeline_S_P.consumer_wait(consumer_state_S_P_dP)
                tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, Float32)
                cute.copy(thr_copy_t2r, tStS_t2r, tSrS_t2r)

                self.apply_seqlen_k_mask(tSrS_t2r, tScS_t2r, n_block, seqlen)
                num_stages = cute.size(tScS_t2r, mode=[1])
                tSrP_r2t_f32 = cute.make_rmem_tensor(tScP_r2t.shape, Float32)  # 64
                tSrP_r2t = cute.recast_tensor(tSrP_r2t_f32, self.q_dtype)
                for stage in cutlass.range_constexpr(num_stages):
                    tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                    tSsLSE_cur = tSsLSE[None, stage, 0, 0, consumer_state_LSE.index]
                    cute.autovec_copy(tSsLSE_cur, tSrLSE_s2r)
                    tSrLSE = tSrLSE_s2r
                    for v in cutlass.range_constexpr(cute.size(tSrS_t2r, mode=[0]) // 2):
                        lse_pair = (tSrLSE[2 * v], tSrLSE[2 * v + 1])
                        tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = cute.arch.fma_packed_f32x2(
                            ((tSrS_cur[2 * v], tSrS_cur[2 * v + 1])),
                            (softmax_scale_log2, softmax_scale_log2),
                            (-lse_pair[0], -lse_pair[1]),
                        )
                        tSrS_cur[2 * v] = cute.math.exp2(tSrS_cur[2 * v], fastmath=True)
                        tSrS_cur[2 * v + 1] = cute.math.exp2(tSrS_cur[2 * v + 1], fastmath=True)
                    utils.cvt_f16(tSrS_cur, tSrP_r2t[None, stage, 0, 0])
                    if const_expr(stage == 0):
                        cute.arch.fence_view_async_tmem_load()
                        # Without this barrier, we could have 1 warp writing to P in tmem while
                        # another warp is still reading S from tmem.
                        self.compute_sync_barrier.arrive_and_wait()
                    cute.copy(
                        thr_copy_r2t,
                        tSrP_r2t_f32[None, stage, None, None],
                        tStP_r2t[None, stage, None, None],
                    )

                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_shared()
                self.compute_sync_barrier.arrive_and_wait()
                with cute.arch.elect_one():
                    pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                # Normally we'd need syncwarp here since only 1 thread will signal in
                # consumer_release, but we already have the self.compute_sync_barrier before this
                pipeline_LSE.consumer_release(consumer_state_LSE)
                consumer_state_LSE.advance()
                pipeline_dPsum.consumer_wait(consumer_state_dPsum)
                pipeline_dP.consumer_wait(consumer_state_S_P_dP)

                for stage in cutlass.range_constexpr(num_stages):
                    tdPrdP_t2r = cute.make_rmem_tensor(tScS_t2r[None, 0, None, None].shape, Float32)
                    cute.copy(thr_copy_t2r, tdPtdP_t2r[None, stage, None, None], tdPrdP_t2r)
                    cute.arch.fence_view_async_tmem_load()
                    self.compute_sync_barrier.arrive_and_wait()
                    tdPrdP_cur = tdPrdP_t2r[None, 0, 0]
                    tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                    tSsdPsum_cur = tSsdPsum[None, stage, 0, 0, consumer_state_dPsum.index]
                    tSrdPsum = cute.make_rmem_tensor_like(tSsdPsum_cur, Float32)
                    cute.autovec_copy(tSsdPsum_cur, tSrdPsum)
                    for v in cutlass.range_constexpr(cute.size(tdPrdP_t2r, mode=[0]) // 2):
                        dPsum_pair = (tSrdPsum[2 * v], tSrdPsum[2 * v + 1])
                        tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = sub_packed_f32x2((tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), dPsum_pair)
                        tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.mul_packed_f32x2(
                            (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                            (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                        )

                    tdPrdS_cvt = cute.make_rmem_tensor_like(tdPrdP_cur, self.ds_dtype)
                    utils.cvt_f16(tdPrdP_cur, tdPrdS_cvt)
                    if const_expr(stage == 0):
                        pipeline_dS.producer_acquire(producer_state_dS)

                    tdPrdS_r2t_f32 = cute.recast_tensor(tdPrdS_cvt, Float32)
                    cute.copy(thr_copy_r2t, tdPrdS_r2t_f32, tdPtdS_r2t[None, stage, 0, 0])

                    cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])

                cute.arch.fence_view_async_tmem_store()

                consumer_state_S_P_dP.advance()

                cute.arch.fence_view_async_shared()
                self.compute_sync_barrier.arrive_and_wait()
                # Normally we'd need syncwarp here since only 1 thread will signal in
                # consumer_release, but we already have the self.compute_sync_barrier before this
                pipeline_dPsum.consumer_release(consumer_state_dPsum)
                consumer_state_dPsum.advance()
                with cute.arch.elect_one():
                    pipeline_dS.producer_commit(producer_state_dS)
                producer_state_dS.advance()

            if process_tile:
                thr_copy_r2s_dKV = tiled_copy_r2s_dKV.get_slice(dp_idx)
                consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                    dp_idx,
                    batch_idx,
                    head_idx,
                    n_block,
                    thr_mma_dV,
                    tdVtdV,
                    mdV_tma_tensor,
                    sdV,
                    tma_atom_dV,
                    thr_copy_r2s_dKV,
                    pipeline_dKV,
                    consumer_state_dKV,
                    None,
                    int(NamedBarrierBwdSm100.EpilogueWG1),
                    "V",
                )
                consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                    dp_idx,
                    batch_idx,
                    head_idx,
                    n_block,
                    thr_mma_dK,
                    tdKtdK,
                    mdK_tma_tensor,
                    sdK,
                    tma_atom_dK,
                    thr_copy_r2s_dKV,
                    pipeline_dKV,
                    consumer_state_dKV,
                    softmax_scale if const_expr(not self.dKV_postprocess) else None,
                    int(NamedBarrierBwdSm100.EpilogueWG1),
                    "K",
                )
            if const_expr(not self.dKV_postprocess):
                should_zero_dKV = not process_tile

                if should_zero_dKV:
                    zero_dk_major = math.gcd(64, self.tile_hdim)
                    zero_dv_major = math.gcd(64, self.tile_hdimv)
                    zero_dk_copy_elems = math.gcd(zero_dk_major, 128 // self.dk_dtype.width)
                    zero_dv_copy_elems = math.gcd(zero_dv_major, 128 // self.dv_dtype.width)
                    gmem_tiled_copy_zero_dK = copy_utils.tiled_copy_2d(
                        self.dk_dtype,
                        zero_dk_major // zero_dk_copy_elems,
                        128,  # num_threads
                        zero_dk_copy_elems,
                    )
                    gmem_tiled_copy_zero_dV = copy_utils.tiled_copy_2d(
                        self.dv_dtype,
                        zero_dv_major // zero_dv_copy_elems,
                        128,  # num_threads
                        zero_dv_copy_elems,
                    )
                    gmem_thr_copy_zero_dK = gmem_tiled_copy_zero_dK.get_slice(dp_idx)
                    gmem_thr_copy_zero_dV = gmem_tiled_copy_zero_dV.get_slice(dp_idx)
                    mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3)[None, None, head_idx]
                    mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3)[None, None, head_idx]
                    gdK = cute.local_tile(mdK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
                    gdV = cute.local_tile(mdV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0))
                    tdKgdK = gmem_thr_copy_zero_dK.partition_D(gdK)
                    tdVgdV = gmem_thr_copy_zero_dV.partition_D(gdV)
                    cdK = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
                    cdV = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
                    tdKcdK = gmem_thr_copy_zero_dK.partition_D(cdK)
                    tdVcdV = gmem_thr_copy_zero_dV.partition_D(cdV)
                    assert cute.size(tdKgdK[None, 0, 0]) == cute.size(tdVgdV[None, 0, 0])
                    zero = cute.make_rmem_tensor_like(tdKgdK[None, 0, 0])
                    zero.fill(0.0)
                    if tidx < 128:
                        for i in cutlass.range_constexpr(tdKgdK.shape[1]):
                            row_idx = tdKcdK[0, i, 0][0]
                            if row_idx < seqlen.seqlen_k - self.tile_n * n_block:
                                for j in cutlass.range_constexpr(tdKgdK.shape[2]):
                                    cute.copy(gmem_tiled_copy_zero_dK, zero, tdKgdK[None, i, j])
                    else:
                        for i in cutlass.range_constexpr(tdVgdV.shape[1]):
                            row_idx = tdVcdV[0, i, 0][0]
                            if row_idx < seqlen.seqlen_k - self.tile_n * n_block:
                                for j in cutlass.range_constexpr(tdVgdV.shape[2]):
                                    cute.copy(gmem_tiled_copy_zero_dV, zero, tdVgdV[None, i, j])

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def dQacc_reduce(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        thr_mma_dQ: cute.ThrMma,
        tdQtdQ: cute.Tensor,
        pipeline_dQ: PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        bsa_k2q_csr_tensors: Optional[BsaK2qCsrTensors] = None,
    ):
        num_reduce_threads = cute.arch.WARP_SIZE * len(self.reduce_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_reduce_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx() % len(self.reduce_warp_ids))
        is_tma_warp = warp_idx == 0
        # TMEM -> RMEM
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol_t2r)), Float32)
        thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ).get_slice(tidx)
        tdQtdQ_t2r = thr_copy_t2r.partition_S(tdQtdQ)
        tdQcdQ = thr_mma_dQ.partition_C(cute.make_identity_tensor(self.mma_tiler_dsk[:2]))
        tdQrdQ_t2r_shape = thr_copy_t2r.partition_D(tdQcdQ).shape
        expected_reduce_stages_t2r = self.dQaccum_reduce_stage_t2r
        assert cute.size(tdQrdQ_t2r_shape, mode=[1]) == expected_reduce_stages_t2r, "dQaccum t2r reduce stage mismatch"

        thr_copy_dQaccum_r2s = copy_utils.tiled_copy_1d(self.dqaccum_dtype, num_reduce_threads, num_copy_elems=128 // self.dqaccum_dtype.width).get_slice(tidx)
        tdQsdQ = thr_copy_dQaccum_r2s.partition_D(sdQaccum)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        dQ_consumer_state = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
        dQ_tma_store_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.sdQaccum_stage)
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            q_group, n_block = get_bsa_k2q_csr_tile_coord(n_block, bsa_k2q_csr_tensors)
            seqlen = SeqlenInfoCls(batch_idx)
            _, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
            gdQaccum_ = cute.local_tile(mdQaccum_cur, (self.tile_m * self.tile_hdim,), (None,))
            # (M * K / STAGE, STAGE, _)
            gdQaccum = cute.flat_divide(gdQaccum_, (self.tile_m * self.tile_hdim // self.dQaccum_reduce_stage,))

            (
                curr_q_cnt,
                curr_q_idx,
                loop_count,
            ) = get_bsa_k2q_csr_iteration_info_bwd(
                bsa_k2q_csr_tensors,
                batch_idx,
                head_idx,
                q_group,
                n_block,
            )
            process_tile = loop_count > Int32(0)

            for iter_idx in cutlass.range(loop_count, unroll=1):
                m_block = get_m_block_from_iter_bwd(iter_idx, curr_q_cnt, curr_q_idx)
                if m_block_max > 0:
                    m_block = cutlass.min(m_block, m_block_max - 1)
                pipeline_dQ.consumer_wait(dQ_consumer_state)
                # TMEM -> RMEM
                tdQrdQ_t2r = cute.make_rmem_tensor(tdQrdQ_t2r_shape, Float32)
                cute.copy(thr_copy_t2r, tdQtdQ_t2r, tdQrdQ_t2r)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_dQ.consumer_release(dQ_consumer_state)
                dQ_consumer_state.advance()

                gdQaccum_cur = gdQaccum[None, None, m_block]

                tdQrdQ_shape = (
                    self.dQ_reduce_ncol,
                    self.tile_hdim // self.dQ_reduce_ncol,
                )
                tdQrdQ = cute.make_tensor(tdQrdQ_t2r.iterator, tdQrdQ_shape)

                for stage in cutlass.range_constexpr(cute.size(tdQrdQ, mode=[1])):
                    smem_idx = dQ_tma_store_producer_state.index
                    tdQsdQ_r2s = tdQsdQ[None, None, smem_idx]
                    tdQrdQ_r2s = cute.make_tensor(tdQrdQ[None, stage].iterator, tdQsdQ_r2s.shape)
                    cute.copy(thr_copy_dQaccum_r2s, tdQrdQ_r2s, tdQsdQ_r2s)
                    cute.arch.fence_view_async_shared()
                    self.reduce_sync_barrier.arrive_and_wait()
                    if is_tma_warp:
                        with cute.arch.elect_one():
                            copy_utils.cpasync_reduce_bulk_add_f32(
                                sdQaccum[None, smem_idx].iterator,
                                gdQaccum_cur[None, stage].iterator,
                                self.tma_copy_bytes["dQ"] // 1,
                            )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(self.sdQaccum_stage - 1, read=True)
                    self.reduce_sync_barrier.arrive_and_wait()
                    dQ_tma_store_producer_state.advance()
            if process_tile:
                if is_tma_warp:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                self.reduce_sync_barrier.arrive_and_wait()

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def epilogue_dK_or_dV_tma(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        thr_mma: cute.ThrMma,
        tdKVtdKV: cute.Tensor,
        mdKV: cute.Tensor,
        sdKV: cute.Tensor,
        tma_atom_dKV: cute.CopyAtom,
        thr_copy_r2s_dKV: cute.TiledCopy,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        scale: Optional[Float32],
        barrier_id: Int32,
        K_or_V: cutlass.Constexpr[str],
    ) -> cutlass.pipeline.PipelineState:
        assert K_or_V in ("K", "V")
        tile_hdim = self.tile_hdim if const_expr(K_or_V == "K") else self.tile_hdimv
        dtype = self.dk_dtype if const_expr(K_or_V == "K") else self.dv_dtype
        epi_tile = self.sdK_epi_tile if const_expr(K_or_V == "K") else self.sdV_epi_tile
        flat_epi_tile = self.sdK_flat_epi_tile if const_expr(K_or_V == "K") else self.sdV_flat_epi_tile
        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        num_wg = num_compute_threads // 128
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0

        if const_expr(not self.dKV_postprocess):
            sdKV = sdKV[None, None, wg_idx]  # (tile_n, 64) for bf16
        else:
            sdKV = sdKV[None, wg_idx]  # (tile_n * 32) for fp32

        # (8, tile_n / 128, 64 / 8) = (8, 1, 8) or (4, tile_n * 32 / (128 * 4)) = (4, 8)
        tdKVsdKV_r2s = thr_copy_r2s_dKV.partition_D(sdKV)

        if const_expr(not self.dKV_postprocess):
            mdKV_cur = mdKV[None, None, head_idx, batch_idx]  # (seqlen, hdim)
            gdKV_p = cute.local_tile(mdKV_cur, (self.tile_n, tile_hdim), (n_block, 0))  # (tile_n, hdim) - per CTA
            gdKV = self.split_wg(gdKV_p, wg_idx, num_wg)  # (tile_n, hdim / 2)
            gdKV_epi = cute.local_tile(gdKV, epi_tile, (0, None))  # (tile_n, 64, epi_stage = (hdim / 2) / 64)
        else:
            mdKV_cur = mdKV[None, head_idx, batch_idx]  # (seqlen * hdim)
            gdKV_p = cute.local_tile(mdKV_cur, (self.tile_n * tile_hdim,), (n_block,))  # (tile_n * hdim)
            gdKV = cute.logical_divide(gdKV_p, (self.tile_n * tile_hdim // num_wg,))[((None, wg_idx),)]  # (tile_n * hdim / 2)
            gdKV_epi = cute.flat_divide(gdKV, (flat_epi_tile,))  # (tile_n * hdim / 2 / epi_stage, epi_stage)

        if const_expr(not self.dKV_postprocess):
            tdKVsdKV, tdKVgdKV = cpasync.tma_partition(
                tma_atom_dKV,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sdKV, 0, 2),
                cute.group_modes(gdKV_epi, 0, 2),
            )  # (TMA) and (TMA, EPI_STAGE)
            assert len(tdKVsdKV.shape) == 1, "Wrong rank for SMEM fragment tdKVsdKV"
            assert len(tdKVgdKV.shape) == 2, "Wrong rank for GMEM fragment tdKVgdKV"
            num_epi_stages = cute.size(tdKVgdKV.shape[1])
            if const_expr(K_or_V == "K"):
                assert num_epi_stages == self.num_epi_stages, "Epi stage calculation is wrong (K)"
            else:
                assert num_epi_stages == self.num_epi_stages_v, "Epi stage calculation is wrong (V)"
        else:
            num_epi_stages = self.num_epi_stages if const_expr(K_or_V == "K") else self.num_epi_stages_v

        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dK_reduce_ncol)), Float32)

        pipeline_dKV.consumer_wait(consumer_state_dKV)

        for epi_stage in cutlass.range_constexpr(num_epi_stages):
            # TMEM -> RMEM -- setup
            thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(tidx)
            tdKVtdKV_t2r_p = thr_copy_t2r.partition_S(tdKVtdKV)
            tdKVtdKV_t2r = self.split_wg(tdKVtdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVtdKV_t2r = tdKVtdKV_t2r[None, epi_stage]

            cdKV = cute.make_identity_tensor((self.tile_n, tile_hdim))
            tdKVcdKV = thr_mma.partition_C(cdKV)
            tdKVcdKV_t2r_p = thr_copy_t2r.partition_D(tdKVcdKV)
            tdKVcdKV_t2r = self.split_wg(tdKVcdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVcdKV_t2r = tdKVcdKV_t2r[None, epi_stage]

            tdKVrdKV_t2r = cute.make_rmem_tensor(tdKVcdKV_t2r.shape, Float32)

            assert cute.size(tdKVrdKV_t2r) == cute.size(tdKVtdKV_t2r) // cute.arch.WARP_SIZE, "RMEM<->TMEM fragment size mismatch"

            # TMEM -> RMEM -- copy and fence
            cute.copy(thr_copy_t2r, tdKVtdKV_t2r, tdKVrdKV_t2r)
            cute.arch.fence_view_async_tmem_load()

            # RMEM -- scale and convert
            if const_expr(scale is not None):
                for i in cutlass.range(cute.size(tdKVrdKV_t2r.shape) // 2, unroll_full=True):
                    tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1] = cute.arch.mul_packed_f32x2((tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1]), (scale, scale))
            tdKVrdKV = cute.make_rmem_tensor(tdKVrdKV_t2r.shape, dtype)  # (32 columns)
            tdKVrdKV.store(tdKVrdKV_t2r.load().to(dtype))

            # RMEM -> SMEM -- copy, fence and barrier
            tdKVrdKV_r2s = cute.make_tensor(tdKVrdKV.iterator, tdKVsdKV_r2s.shape)
            cute.copy(thr_copy_r2s_dKV, tdKVrdKV_r2s, tdKVsdKV_r2s)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

            # SMEM -> GMEM
            if leader_warp:
                if const_expr(not self.dKV_postprocess):
                    cute.copy(tma_atom_dKV, tdKVsdKV, tdKVgdKV[None, epi_stage])
                else:
                    with cute.arch.elect_one():
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdKV.iterator,
                            gdKV_epi[None, epi_stage].iterator,
                            self.tma_copy_bytes["dKacc"],
                        )
                if const_expr(epi_stage < num_epi_stages - 1):
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier_arrive(barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE)

            # Barrier since all warps need to wait for SMEM to be freed
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV


SM100_BLK128_BWD_SPARSE_BLOCK_SIZE = 128
SM100_BWD_HEAD_DIM = 128


def sm100_blk128_bwd_default_bucketed_k2q_size_blocks(
    num_q_blocks: int,
    num_heads: int,
) -> int:
    """Default qbucket size tuned for the blk128 customer benchmark cases."""
    if num_q_blocks >= 4096 and num_heads <= 1:
        return 256
    if num_q_blocks >= 2048:
        return 512
    return 384


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def bsa_sm100_blk128_bwd_bucketed_k2q_csr(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    bucketed_k2q_offsets: torch.Tensor,
    bucketed_k2q_indices: torch.Tensor,
    softmax_scale: Optional[float] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SM100/SM110 blk128 bwd entry receiving BSA bucketed k2q CSR."""
    assert q.dtype == torch.bfloat16, "SM100 blk128 bwd only supports bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype
    assert lse.dtype == torch.float32
    assert _get_device_arch() // 10 in [10, 11]
    assert bucketed_k2q_offsets.dtype == torch.int32
    assert bucketed_k2q_indices.dtype == torch.int32

    batch_size, num_heads, seqlen_q, head_dim = q.shape
    seqlen_k = k.shape[2]
    assert head_dim in (64, SM100_BWD_HEAD_DIM), f"SM100 blk128 bwd supports head_dim in {{64, {SM100_BWD_HEAD_DIM}}}, got {head_dim}"
    assert k.shape == v.shape == (batch_size, num_heads, seqlen_k, head_dim)
    assert out.shape == dout.shape == q.shape
    assert lse.shape == (batch_size, num_heads, seqlen_q)

    num_q_blocks = _ceil_div(seqlen_q, SM100_BLK128_BWD_SPARSE_BLOCK_SIZE)
    num_kv_blocks = _ceil_div(seqlen_k, SM100_BLK128_BWD_SPARSE_BLOCK_SIZE)
    assert bucketed_k2q_offsets.shape[:2] == (batch_size, num_heads)
    assert bucketed_k2q_offsets.ndim == 4
    assert bucketed_k2q_offsets.shape[-1] == num_kv_blocks + 1
    assert bucketed_k2q_indices.shape[:2] == (batch_size, num_heads)
    num_q_groups = bucketed_k2q_offsets.shape[2]
    use_dkv_postprocess = num_q_groups > 1

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)

    q_bshd = q.transpose(1, 2)
    k_bshd = k.transpose(1, 2)
    v_bshd = v.transpose(1, 2)
    out_bshd = out.transpose(1, 2)
    dout_bshd = dout.transpose(1, 2)
    dq_bshd = dq.transpose(1, 2)
    dk_bshd = dk.transpose(1, 2)
    dv_bshd = dv.transpose(1, 2)

    seqlen_q_rounded = num_q_blocks * SM100_BLK128_BWD_SPARSE_BLOCK_SIZE
    seqlen_k_rounded = num_kv_blocks * SM100_BLK128_BWD_SPARSE_BLOCK_SIZE
    head_dim_rounded = _ceil_div(head_dim, 32) * 32
    dq_accum = torch.empty(
        batch_size,
        num_heads,
        seqlen_q_rounded * head_dim_rounded,
        dtype=torch.float32,
        device=q.device,
    )
    dpsum = torch.empty(
        batch_size,
        num_heads,
        seqlen_q_rounded,
        dtype=torch.float32,
        device=q.device,
    )
    lse_log2 = torch.empty_like(dpsum)
    if use_dkv_postprocess:
        dk_accum = torch.zeros(
            batch_size,
            num_heads,
            seqlen_k_rounded * head_dim_rounded,
            dtype=torch.float32,
            device=q.device,
        )
        dv_accum = torch.zeros(
            batch_size,
            num_heads,
            seqlen_k_rounded * head_dim_rounded,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        dk_accum = None
        dv_accum = None

    dtype = torch2cute_dtype_map[q.dtype]
    _bwd_preprocess(
        out_bshd,
        dout_bshd,
        dpsum,
        lse,
        lse_log2,
        dq_accum,
        None,
        None,
        None,
        dtype,
        head_dim,
        head_dim,
        SM100_BLK128_BWD_SPARSE_BLOCK_SIZE,
    )

    compile_key = (
        "bsa_sm100_blk128_bucketed_k2q",
        _get_device_arch(),
        dtype,
        head_dim,
        get_broadcast_dims(q_bshd),
        get_broadcast_dims(k_bshd),
        get_broadcast_dims(v_bshd),
        get_broadcast_dims(dout_bshd),
        use_dkv_postprocess,
    )
    cache = bsa_sm100_blk128_bwd_bucketed_k2q_csr.compile_cache
    if compile_key not in cache:
        q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor = [to_cute_tensor(t) for t in (q_bshd, k_bshd, v_bshd, dout_bshd, dq_bshd)]
        dk_runtime = dk_accum if use_dkv_postprocess else dk_bshd
        dv_runtime = dv_accum if use_dkv_postprocess else dv_bshd
        dk_tensor, dv_tensor = [to_cute_tensor(t) for t in (dk_runtime, dv_runtime)]
        dq_accum_tensor, dpsum_tensor, lse_log2_tensor = [to_cute_tensor(t) for t in (dq_accum, dpsum, lse_log2)]
        bsa_csr_tensors = BsaK2qCsrTensors(
            to_cute_tensor(bucketed_k2q_offsets, assumed_align=4),
            to_cute_tensor(bucketed_k2q_indices, assumed_align=4),
        )
        bwd_kernel = BlockSparseAttnBackwardSm100Blk128(
            head_dim,
            force_dkv_postprocess=use_dkv_postprocess,
        )
        current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        cache[compile_key] = cute.compile(
            bwd_kernel,
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_accum_tensor,
            dk_tensor,
            dv_tensor,
            float(softmax_scale),
            bsa_csr_tensors,
            current_stream,
            options="--enable-tvm-ffi",
        )

    cache[compile_key](
        q_bshd.detach(),
        k_bshd.detach(),
        v_bshd.detach(),
        dout_bshd,
        lse_log2,
        dpsum,
        dq_accum,
        dk_accum if use_dkv_postprocess else dk_bshd,
        dv_accum if use_dkv_postprocess else dv_bshd,
        float(softmax_scale),
        (bucketed_k2q_offsets, bucketed_k2q_indices),
    )

    _bwd_postprocess_convert(
        dq_accum,
        dq_bshd,
        float(softmax_scale),
        None,
        None,
        _get_device_arch(),
        dtype,
        head_dim,
        SM100_BLK128_BWD_SPARSE_BLOCK_SIZE,
        False,
    )
    if use_dkv_postprocess:
        _bwd_postprocess_convert(
            dk_accum,
            dk_bshd,
            float(softmax_scale),
            None,
            None,
            _get_device_arch(),
            dtype,
            head_dim,
            SM100_BLK128_BWD_SPARSE_BLOCK_SIZE,
            False,
        )
        _bwd_postprocess_convert(
            dv_accum,
            dv_bshd,
            1.0,
            None,
            None,
            _get_device_arch(),
            dtype,
            head_dim,
            SM100_BLK128_BWD_SPARSE_BLOCK_SIZE,
            False,
        )
    return dq, dk, dv


bsa_sm100_blk128_bwd_bucketed_k2q_csr.compile_cache = {}
