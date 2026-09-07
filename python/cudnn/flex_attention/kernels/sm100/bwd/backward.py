# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
import math
from functools import partial
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import PipelineAsync
from cutlass.utils import LayoutEnum

import cuda.bindings.driver as cuda
from cudnn.flex_attention.kernels.common import barrier, copy_utils, pipeline
from cudnn.flex_attention.kernels.common import device_utils as utils
from cudnn.flex_attention.kernels.sm100.blackwell_helpers import gemm_ptx_w_idx, gemm_w_idx  # noqa
from cudnn.flex_attention.kernels.common.block_info import BlockInfo
from cudnn.flex_attention.plan.kernels.packed_mask import (
    apply_loaded_arbitrary_mask,
    get_block_sparse_iteration_info_bwd,
    get_curr_dq_write_order_bwd,
    get_m_block_from_iter_bwd,
    get_total_q_block_count_bwd,
    load_mask_payload,
    produce_block_sparse_q_loads_bwd_sm100,
)
from cudnn.flex_attention.plan.kernels import BlockSparseTensors
from cudnn.flex_attention.runtime.dsl_utils import bulk_copy, assume_tensor_aligned, struct_scalar_ptr
from cudnn.flex_attention.kernels.sm100.bwd.named_barrier import NamedBarrierBwdSm100
from cudnn.flex_attention.kernels.sm100.bwd.backward_config import SM100_BWD_MASK_PAYLOAD_WORDS
from cudnn.flex_attention.kernels.common.seqlen_info import SeqlenInfoQK
from cudnn.flex_attention.kernels.common.tile_scheduler import (
    SingleTileLPTBwdScheduler,  # noqa
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)
from cudnn.flex_attention._compat import layout_utils
from cudnn.flex_attention._compat.cute_dsl_utils import ParamsBase


class FlexAttentionBackwardSm100:
    arch = 100

    def __init__(
        self,
        head_dim: int,
        head_dim_v: int,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        deterministic: bool = False,
        spt: Optional[bool] = None,
        use_2cta_instrs: bool = False,
    ):
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim = head_dim
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.head_dim_v = head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv

        self.tile_m = 128
        self.tile_n = 128

        assert self.tile_hdim <= 128 or (self.tile_hdim == 192 and self.tile_hdimv == 128)
        assert self.tile_hdimv <= 128

        self.use_2cta_instrs = use_2cta_instrs
        self.cta_group_size = 2 if self.use_2cta_instrs else 1

        assert self.tile_hdim != 192 or self.use_2cta_instrs, "Must use 2CTA for hdim 192"

        # CTA tiler
        self.cta_tiler = (self.tile_n, self.tile_m, self.tile_hdim)
        # S = K @ Q.T
        self.mma_tiler_kq = (self.cta_group_size * self.tile_n, self.tile_m, self.tile_hdim)
        # dP = V @ dO.T
        self.mma_tiler_vdo = (self.cta_group_size * self.tile_n, self.tile_m, self.tile_hdimv)
        # dV = P.T @ dO
        self.mma_tiler_pdo = (self.cta_group_size * self.tile_n, self.tile_hdimv, self.tile_m)
        # dK = dS.T @ Q
        self.mma_tiler_dsq = (self.cta_group_size * self.tile_n, self.tile_hdim, self.tile_m)
        # dQ = dS @ K
        # 2-CTA: reduction dim is cluster-wide (tile_n * cta_group_size).
        self.mma_tiler_dsk = (self.tile_m, self.tile_hdim, self.tile_n * self.cta_group_size)

        self.acc_dtype = Float32

        self.cluster_shape_mn = (self.cta_group_size, 1)
        if self.use_2cta_instrs:
            assert (head_dim, head_dim_v) in ((128, 128), (192, 128)), "arbitrary cooperative backward requires " "D/Dv=128/128 or 192/128"
        self.qhead_per_kvhead = qhead_per_kvhead
        self.deterministic = deterministic
        self.spt_override = spt

        self.subtile_factor = 1

        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.relay_warp_id = 14
        self.empty_warp_id = 15

        # 16 warps -> 512 threads
        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.reduce_warp_ids,
                *self.compute_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.relay_warp_id,
                self.empty_warp_id,
            )
        )
        # NamedBarrier
        self.compute_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.Compute),
            num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )
        self.reduce_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.dQaccReduce),
            num_threads=len(self.reduce_warp_ids) * cute.arch.WARP_SIZE,
        )
        # TMEM setup
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        if self.use_2cta_instrs and self.tile_hdim == 192 and self.tile_hdimv == 128:
            self.tmem_dV_offset = 0
            self.tmem_dK_offset = self.tmem_dV_offset + self.tile_hdimv
            self.tmem_S_offset = self.tmem_dK_offset + self.tile_hdim
            self.tmem_P_offset = self.tmem_S_offset  # overlap with S
            self.tmem_dP_offset = 512 - self.tile_m
            self.tmem_dS_offset = self.tmem_dP_offset  # overlaps with dP
            self.tmem_dQ_offset = 512 - self.tile_hdim // 2
        else:
            self.tmem_S_offset = 0
            self.tmem_P_offset = 0  # overlap with S
            self.tmem_dV_offset = self.tmem_S_offset + self.tile_n
            self.tmem_dP_offset = self.tmem_dV_offset + self.tile_hdimv
            self.tmem_dQ_offset = (self.tmem_S_offset + (self.tile_hdim // 2)) if self.use_2cta_instrs else self.tmem_dP_offset
            self.tmem_dK_offset = self.tmem_dP_offset + self.tile_m
            self.tmem_dS_offset = self.tmem_dP_offset  # overlap with dP

        self.num_regs_reduce = 136 if self.use_2cta_instrs else 152
        self.num_regs_compute = 136
        self.num_regs_load = 104 if self.use_2cta_instrs else 96 - 8
        self.num_regs_mma = 104 if self.use_2cta_instrs else self.num_regs_load
        self.num_regs_empty = 24

        if const_expr(self.tile_hdim == 192):
            self.num_regs_reduce = 128 + 8
            self.num_regs_compute = 128 + 8
            self.num_regs_load = 128 - 24
            self.num_regs_mma = self.num_regs_load

        assert self.num_regs_reduce + self.num_regs_compute * 2 + max(self.num_regs_load, self.num_regs_mma) <= 512
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.Q_stage = 1 if self.use_2cta_instrs else 2
        self.dO_stage = 1
        self.single_stage = 1
        # LSE_stage = Q_stage and dPsum_stage = dO_stage
        self.sdKVaccum_stage = 2
        # number of tma reduce adds per dQacc mma
        # todo: try 32/1 or 48/2 for 2cta d=192 dv=128
        if self.use_2cta_instrs and self.tile_hdim == 192:
            self.dQ_reduce_ncol_t2r = 32
            self.dQ_reduce_ncol = 24
            self.sdQaccum_stage = 2
        else:
            if self.use_2cta_instrs:
                self.dQ_reduce_ncol = 16 if self.deterministic else 8
                self.sdQaccum_stage = 2 if self.deterministic else 4
                self.dQ_reduce_ncol_t2r = 32
            else:
                # Preserve the 64-column staging budget while allowing padded
                # head dimensions that are not divisible by 32.
                self.dQ_reduce_ncol = math.gcd(32, self.tile_hdim)
                self.sdQaccum_stage = 64 // self.dQ_reduce_ncol
                self.dQ_reduce_ncol_t2r = self.dQ_reduce_ncol
        assert (self.tile_hdim // self.cta_group_size) % self.dQ_reduce_ncol == 0
        self.dQaccum_reduce_stage = self.tile_hdim // self.dQ_reduce_ncol
        self.dQaccum_reduce_stage_t2r = self.tile_hdim // self.dQ_reduce_ncol_t2r
        # Number of TMA reduce-add columns per dK/dV epilogue.  D and Dv are
        # independent, so their per-WG widths must not share one reduction shape.
        self.dK_reduce_ncol = math.gcd(32, self.tile_hdim // 2)
        self.dV_reduce_ncol = math.gcd(32, self.tile_hdimv // 2)
        # CTA group for MMA operations
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

    def _get_tiled_mma(self):
        # S.T = K @ Q.T
        tiled_mma_S = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_kq[:2],
        )
        # dP.T = V @ dO.T
        tiled_mma_dP = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_vdo[:2],
        )
        # dV += P.T @ dO --> (K, MN) major
        tiled_mma_dV = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # P_major_mode
            tcgen05.OperandMajorMode.MN,  # dO_major_mode
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_pdo[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        # dK += dS.T @ Q
        tiled_mma_dK = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # dS_major_mode
            tcgen05.OperandMajorMode.MN,  # Q_major_mode
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dsq[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        # dQ = dS @ K
        tiled_mma_dQ = sm100_utils_basic.make_trivial_tiled_mma(
            self.k_dtype,
            tcgen05.OperandMajorMode.MN,  # dS_major_mode
            tcgen05.OperandMajorMode.MN,  # Kt_major_mode
            self.acc_dtype,
            self.cta_group,
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
        self.sdS_xchg_layout = cute.make_layout(shape=(self.tile_n, self.tile_m // 2))

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
            self.sdV_layout = cute.make_layout((self.tile_n * self.dV_reduce_ncol, 2))

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
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        # Block-sparse tensors (Q direction - for iterating m_blocks per n_block):
        blocksparse_tensors: BlockSparseTensors = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
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

        self.is_varlen_k = mCuSeqlensK is not None
        self.is_varlen_q = mCuSeqlensQ is not None
        self.dKV_postprocess = self.qhead_per_kvhead > 1 or self.check_hdim_oob or self.check_hdim_v_oob
        self.use_tma_store = self.dKV_postprocess or not (self.qhead_per_kvhead == 1 and mCuSeqlensK is not None)

        if const_expr(self.dKV_postprocess):
            assert self.dk_dtype.width == 32, "Must accumulate dK in float precision"
            assert self.dv_dtype.width == 32, "Must accumulate dV in float precision"

        mdQaccum, mdK, mdV, mLSE, mdPsum = [assume_tensor_aligned(t) for t in (mdQaccum, mdK, mdV, mLSE, mdPsum)]

        # (b, s, n, h) --> (s, h, n, b) or (t, n, h) -> (t, h, n)
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ, mdO = [layout_utils.select(t, mode=QO_layout_transpose) for t in (mQ, mdO)]

        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [layout_utils.select(t, mode=KV_layout_transpose) for t in (mK, mV)]

        # (b, n, s) --> (s, n, b) or (n, t) --> (t, n)
        LSE_dPsum_dQaccum_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE, mdPsum, mdQaccum = [layout_utils.select(t, mode=LSE_dPsum_dQaccum_transpose) for t in (mLSE, mdPsum, mdQaccum)]

        if const_expr(not self.dKV_postprocess):
            layout_dKV_transpose = KV_layout_transpose
        else:
            layout_dKV_transpose = [2, 1, 0] if const_expr(mCuSeqlensK is None) else [1, 0]
        mdK, mdV = [layout_utils.select(t, mode=layout_dKV_transpose) for t in (mdK, mdV)]
        # (s, h, n, b) --> (h, s, n, b) or (t, h, n) -> (h, t, b)
        dO_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensQ is None) else [1, 0, 2]
        mdO = layout_utils.select(mdO, mode=dO_transpose)

        # Transposes for 2-CTA K/Q paths (Q follows Q seqlens, K follows K seqlens)
        transpose_sh_q = dO_transpose
        transpose_sh_k = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]

        # (b, n, block, stage) -> (block, stage, n, b)
        semaphore_transpose = [2, 3, 1, 0]
        if const_expr(self.deterministic):
            assert mdQ_semaphore is not None
            mdQ_semaphore = layout_utils.select(mdQ_semaphore, mode=semaphore_transpose)

        if const_expr(self.deterministic and self.qhead_per_kvhead > 1):
            assert mdK_semaphore is not None
            assert mdV_semaphore is not None
            mdK_semaphore, mdV_semaphore = [layout_utils.select(t, mode=semaphore_transpose) for t in (mdK_semaphore, mdV_semaphore)]
        else:
            mdK_semaphore = None
            mdV_semaphore = None

        self._setup_attributes()
        (
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dK,
            self.tiled_mma_dV,
            self.tiled_mma_dQ,
        ) = self._get_tiled_mma()
        self._setup_smem_layout()

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma_S.thr_id.shape,),
        )
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_q_do_mcast = self.num_mcast_ctas_b > 1

        if const_expr(not self.dKV_postprocess):
            self.mdK_layout_enum = LayoutEnum.from_tensor(mdK)
            self.mdV_layout_enum = LayoutEnum.from_tensor(mdV)
            dK_major_mode = self.mdK_layout_enum.mma_major_mode()
            dV_major_mode = self.mdV_layout_enum.mma_major_mode()
            if const_expr(dK_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mdK is wrong")
            if const_expr(dV_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mdV is wrong")

        if const_expr(self.use_tma_store and not self.dKV_postprocess):
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

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        # S.T = K @ Q.T
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK,
            cute.select(self.sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        Q_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(self.cluster_shape_mnk, self.tiled_mma_S.thr_id)
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            Q_tma_op,
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        # dP.T = V @ dO.T
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV,
            cute.select(self.sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            self.tiled_mma_dP,
            self.cluster_layout_vmnk.shape,
        )
        # dV = P.T @ dO
        dO_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(self.cluster_shape_mnk, self.tiled_mma_dV.thr_id)
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            dO_tma_op,
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_pdo,
            self.tiled_mma_dV,
            self.cluster_layout_vmnk.shape,
        )
        # ------------------------------------------------------------
        # 2-CTA
        # ------------------------------------------------------------
        tma_atom_dOt = tma_tensor_dOt = None
        if const_expr(self.use_2cta_instrs):
            tma_atom_dOt, tma_tensor_dOt = cute.nvgpu.make_tiled_tma_atom_B(
                dO_tma_op,
                layout_utils.select(mdO, mode=transpose_sh_q),
                cute.select(self.sdOt_layout, mode=[0, 1, 2]),
                self.mma_tiler_vdo,
                self.tiled_mma_dP,
                self.cluster_layout_vmnk.shape,
            )
        tma_atom_Qt = tma_tensor_Qt = None
        if const_expr(self.use_2cta_instrs):
            tma_atom_Qt, tma_tensor_Qt = cute.nvgpu.make_tiled_tma_atom_B(
                Q_tma_op,
                layout_utils.select(mQ, mode=transpose_sh_q),
                cute.select(self.sQt_layout, mode=[0, 1, 2]),
                self.mma_tiler_dsq,
                self.tiled_mma_dK,
                self.cluster_layout_vmnk.shape,
            )
        tma_atom_Kt = tma_tensor_Kt = None
        if const_expr(self.use_2cta_instrs):
            Kt_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(self.cluster_shape_mnk, self.tiled_mma_dQ.thr_id)
            tma_atom_Kt, tma_tensor_Kt = cute.nvgpu.make_tiled_tma_atom_B(
                Kt_tma_op,
                layout_utils.select(mK, mode=transpose_sh_k),
                cute.select(self.sKt_layout, mode=[0, 1, 2]),
                self.mma_tiler_dsk,
                self.tiled_mma_dQ,
                self.cluster_layout_vmnk.shape,
            )

        self.tma_copy_bytes = {
            name: self.cta_group_size * cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
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
        self.tma_copy_bytes["dVacc"] = self.tile_n * self.dV_reduce_ncol * Float32.width // 8
        self.tma_copy_bytes["dS"] = cute.size_in_bytes(self.ds_dtype, self.sdS_layout)
        self.tma_copy_bytes["sdS_xchg"] = self.tma_copy_bytes["dS"] // 2  # Half of dS for exchange

        if const_expr(self.is_varlen_k):
            TileScheduler = SingleTileVarlenScheduler
        elif const_expr(self.deterministic):
            TileScheduler = SingleTileLPTBwdScheduler
        else:
            TileScheduler = SingleTileScheduler
        if const_expr(self.spt_override is None):
            self.spt = False
        else:
            assert self.spt_override is not None
            self.spt = self.spt_override and self.deterministic
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.cta_tiler[0]),  # num_blocks
            cute.size(mQ.shape[2]),  # num_heads = num_query_heads
            cute.size(mK.shape[3]) if const_expr(mCuSeqlensK is None) else cute.size(mCuSeqlensK.shape[0] - 1),  # num_batches
            cute.size(mQ.shape[0]),  # pass seqlen_q or total_q for seqlen_k
            mQ.shape[1],  # headdim
            mV.shape[1],  # headdim_v
            total_q=(
                cute.size(mK.shape[0]) if const_expr(mCuSeqlensK is not None) else cute.size(mK.shape[0]) * cute.size(mK.shape[3])  # pass total_k for total_q
            ),
            tile_shape_mn=self.cta_tiler[:2],  # (tile_n, tile_m)
            cluster_shape_mn=self.cluster_shape_mnk[:2],
            mCuSeqlensQ=mCuSeqlensK,
            qhead_per_kvhead_packgqa=1,  # pack_gqa disabled for bwd
            element_size=self.k_dtype.width // 8,
            lpt=self.spt,
            head_swizzle=self.deterministic,
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
        # 2-CTA: sdV reuses sV, sdK reuses sK
        sV_bytes = cute.size_in_bytes(self.v_dtype, self.sV_layout)
        sK_bytes = cute.size_in_bytes(self.k_dtype, self.sK_layout)
        sV_alloc_bytes = max(sV_bytes, sdV_bytes)
        if const_expr(self.use_2cta_instrs):
            assert sV_bytes <= sV_alloc_bytes and sdV_bytes <= sV_alloc_bytes
            assert sdK_bytes <= sK_bytes, "sdK doesn't fit in sK storage allocation (2-CTA)"

        if const_expr(self.use_2cta_instrs):
            sQt_size = cute.cosize(self.sQt_layout) if const_expr(self.tile_hdim <= 128) else 0
            sdOt_size = cute.cosize(self.sdOt_layout) if const_expr(self.tile_hdim <= 128) else 0
            sdS_xchg_size = cute.cosize(self.sdS_xchg_layout) if const_expr(self.tile_hdim <= 128) else 0

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
                tmem_dealloc_mbar: cutlass.Int64

                # 2-CTA
                Qt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                Kt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_cluster_empty_mbar_ptr: cutlass.Int64
                dS_cluster_full_mbar_ptr: cutlass.Int64
                dS_cluster_leader_mbar_ptr: cutlass.Int64
                dQaccum_empty_mbar_ptr: cutlass.Int64

                sQ: cute.struct.Align[
                    cute.struct.MemRange[self.q_dtype, cute.cosize(self.sQ_layout)],
                    self.buffer_align_bytes,
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[self.k_dtype, cute.cosize(self.sK_layout)],
                    self.buffer_align_bytes,
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sV_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sdO: cute.struct.Align[
                    cute.struct.MemRange[self.do_dtype, cute.cosize(self.sdO_layout)],
                    self.buffer_align_bytes,
                ]
                sQt: cute.struct.Align[
                    cute.struct.MemRange[self.q_dtype, sQt_size],
                    self.buffer_align_bytes,
                ]
                sdOt: cute.struct.Align[
                    cute.struct.MemRange[self.do_dtype, sdOt_size],
                    self.buffer_align_bytes,
                ]
                sdS_xchg: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, sdS_xchg_size],
                    self.buffer_align_bytes,
                ]
                sKt: cute.struct.Align[
                    cute.struct.MemRange[self.k_dtype, cute.cosize(self.sKt_layout)],
                    self.buffer_align_bytes,
                ]
                sdS: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                    self.buffer_align_bytes,
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
                    self.buffer_align_bytes if sdS_xchg_size == 0 else 128,
                ]

        else:

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
                tmem_dealloc_mbar: Int64

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

        softmax_scale_log2 = softmax_scale * math.log2(math.e)

        assert blocksparse_tensors is not None

        if const_expr(self.use_2cta_instrs):
            assert blocksparse_tensors is not None, "SM100 arbitrary 2-CTA backward requires a K2Q plan."
            assert (mCuSeqlensQ is None) == (mCuSeqlensK is None), "SM100 arbitrary 2-CTA backward requires both cu_seqlens tensors."
            if const_expr(mCuSeqlensK is None):
                assert blocksparse_tensors.cu_total_m_blocks is None, "SM100 arbitrary fixed 2-CTA backward requires fixed-row K2Q metadata."
            else:
                assert blocksparse_tensors.cu_total_m_blocks is not None, "SM100 arbitrary varlen 2-CTA backward requires compact K256 prefixes."
            assert blocksparse_tensors.mask_block_masks is not None

        if const_expr(self.use_2cta_instrs):
            assert (self.tile_hdim == 128 and self.tile_hdimv == 128) or (self.tile_hdim == 192 and self.tile_hdimv == 128), (
                "2-CTA block sparse backward is only supported for arbitrary " "SM100/SM103 D=128, Dv=128 or D=192, Dv=128."
            )
            assert blocksparse_tensors.mask_block_offset is not None, "2-CTA block sparse backward only supports linear CSR block sparsity."
            # Cooperative arbitrary CSR is register-limited: MHA needs more
            # compute budget, while GQA benefits from larger load/MMA budgets.
            if const_expr(self.qhead_per_kvhead == 1):
                self.num_regs_reduce = 128
                self.num_regs_load = 96
                if const_expr(self.deterministic):
                    # The LPT/semaphore specialization raises descriptor pressure
                    # in the MMA warp.  Moving 16 registers from the two compute
                    # warp groups to MMA removes its descriptor stack slots while
                    # preserving the 512-register warp-group budget.
                    self.num_regs_compute = 136
                    self.num_regs_mma = 112
                else:
                    self.num_regs_compute = 144
                    self.num_regs_mma = self.num_regs_load
            else:
                self.num_regs_reduce = 128
                self.num_regs_compute = 136
                self.num_regs_load = 112
                self.num_regs_mma = self.num_regs_load
        self.kernel(
            tma_tensor_Q,
            tma_tensor_Qt,
            tma_tensor_K,
            tma_tensor_Kt,
            tma_tensor_V,
            mLSE,
            mdPsum,
            tma_tensor_dO,
            tma_tensor_dOt,
            mdV,
            mdK,
            mdQaccum,
            mdV_tma_tensor,
            mdK_tma_tensor,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            mCuSeqlensQ,
            mCuSeqlensK,
            tma_atom_Q,
            tma_atom_Qt,
            tma_atom_K,
            tma_atom_Kt,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dOt,
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
            self.sdS_xchg_layout,
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
            blocksparse_tensors,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk if cute.size(self.cluster_shape_mnk) > 1 else None,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mQt: Optional[cute.Tensor],
        mK: cute.Tensor,
        mKt: Optional[cute.Tensor],
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        mdOt: Optional[cute.Tensor],
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        mdQ_semaphore: Optional[cute.Tensor],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_Qt: Optional[cute.CopyAtom],
        tma_atom_K: cute.CopyAtom,
        tma_atom_Kt: Optional[cute.CopyAtom],
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dOt: Optional[cute.CopyAtom],
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
        sdS_xchg_layout: cute.Layout,
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
        blocksparse_tensors: BlockSparseTensors = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % self.cta_group_size
        is_leader_cta = mma_tile_coord_v == 0

        # Prefetch tma descriptor
        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_Q)
                if const_expr(tma_atom_Qt is not None):
                    cpasync.prefetch_descriptor(tma_atom_Qt)
                cpasync.prefetch_descriptor(tma_atom_K)
                if const_expr(tma_atom_Kt is not None):
                    cpasync.prefetch_descriptor(tma_atom_Kt)
                cpasync.prefetch_descriptor(tma_atom_V)
                if const_expr(tma_atom_dOt is not None):
                    cpasync.prefetch_descriptor(tma_atom_dOt)
                cpasync.prefetch_descriptor(tma_atom_dO)
                if const_expr(tma_atom_dV is not None):
                    cpasync.prefetch_descriptor(tma_atom_dV)
                if const_expr(tma_atom_dK is not None):
                    cpasync.prefetch_descriptor(tma_atom_dK)

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_S.thr_id.shape,),
        )

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        if const_expr(self.use_2cta_instrs):
            dS_cluster_full_mbar_ptr = storage.dS_cluster_full_mbar_ptr
            dS_cluster_empty_mbar_ptr = storage.dS_cluster_empty_mbar_ptr
            dS_cluster_leader_mbar_ptr = storage.dS_cluster_leader_mbar_ptr
            dQaccum_empty_mbar_ptr = storage.dQaccum_empty_mbar_ptr
        else:
            dS_cluster_full_mbar_ptr = None
            dS_cluster_empty_mbar_ptr = None
            dS_cluster_leader_mbar_ptr = None
            dQaccum_empty_mbar_ptr = None

        # Barrier initialization
        if const_expr(self.use_2cta_instrs):
            if const_expr(self.tile_hdim == 192):
                if warp_idx == 2:
                    cute.arch.mbarrier_init(
                        dQaccum_empty_mbar_ptr,
                        len(self.reduce_warp_ids),
                    )
            if warp_idx == 4:
                cute.arch.mbarrier_init(dS_cluster_full_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_empty_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_leader_mbar_ptr, 2)

        tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE * len((self.mma_warp_id, *self.compute_warp_ids, *self.reduce_warp_ids)),
        )
        use_unaligned_tmem_barrier = const_expr(self.use_2cta_instrs)
        tmem = cutlass.utils.TmemAllocator(
            struct_scalar_ptr(storage.tmem_holding_buf),
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=struct_scalar_ptr(storage.tmem_dealloc_mbar),
        )

        # UMMA producers and AsyncThread consumers
        pipeline_producer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len([self.mma_warp_id]))
        pipeline_consumer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids) * self.cta_group_size
        )
        # The only supported 2-CTA cluster has ranks {0, 1}, whose UMMA pair
        # base is always zero. Keeping the dynamically computed consumer and
        # dS-producer pair base alive across role register budgets can spill.
        umma_pair_base_override = 0 if const_expr(self.use_2cta_instrs) else None
        pipeline_S_P = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.S_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            consumer_mask_override=umma_pair_base_override,
        )
        pipeline_dP = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dP_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            consumer_mask_override=umma_pair_base_override,
        )
        pipeline_dKV = pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dKV_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            consumer_mask_override=umma_pair_base_override,
        )
        pipeline_consumer_group_MMA_AsyncThread_dQ = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.reduce_warp_ids) * self.cta_group_size,
        )  # Compute
        pipeline_dQ = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread_dQ,
            barrier_storage=storage.dQ_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            consumer_mask_override=umma_pair_base_override,
        )

        # AsyncThread producers and UMMA consumers
        # Only 1 thread per warp will signal
        pipeline_PdS_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * self.cta_group_size,
        )  # Compute
        pipeline_PdS_consumer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len([self.mma_warp_id]))  # MMA
        pipeline_dS = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline_PdS_producer_group,
            consumer_group=pipeline_PdS_consumer_group,
            barrier_storage=storage.dS_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            producer_mask_override=umma_pair_base_override,
        )

        # TMA producer and UMMA consumers
        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len([self.load_warp_id]))
        # The arrive count is the number of mcast size
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len([self.mma_warp_id]) * self.num_mcast_ctas_b)
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

        if const_expr(self.use_2cta_instrs):
            if const_expr(self.tile_hdim == 192):
                pipeline_Qt = pipeline_Q
            else:
                pipeline_Qt = pipeline.PipelineTmaUmma.create(
                    barrier_storage=storage.Qt_mbar_ptr.data_ptr(),
                    num_stages=self.Q_stage,
                    producer_group=pipeline_producer_group,
                    consumer_group=pipeline_consumer_group,
                    tx_count=self.tma_copy_bytes["Q"],
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                )
            pipeline_Kt = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.Kt_mbar_ptr.data_ptr(),
                num_stages=self.single_stage,
                producer_group=pipeline_producer_group,
                consumer_group=pipeline_consumer_group,
                tx_count=self.tma_copy_bytes["K"],
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_Qt = pipeline_Kt = pipeline_Q

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
        if const_expr(self.use_2cta_instrs and self.tile_hdim <= 128):
            sQt = storage.sQt.get_tensor(sQt_layout.outer, swizzle=sQt_layout.inner, dtype=self.q_dtype)
        else:
            sQt = cute.make_tensor(cute.recast_ptr(sQ.iterator, sQt_layout.inner, dtype=self.q_dtype), sQt_layout.outer)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(self.use_2cta_instrs):
            sKt = storage.sKt.get_tensor(sKt_layout.outer, swizzle=sKt_layout.inner)
        else:
            sKt = cute.make_tensor(cute.recast_ptr(sK.iterator, sKt_layout.inner), sKt_layout.outer)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner, dtype=self.v_dtype)
        sdSt = storage.sdS.get_tensor(sdSt_layout.outer, swizzle=sdSt_layout.inner)
        sdS = cute.make_tensor(cute.recast_ptr(sdSt.iterator, sdS_layout.inner), sdS_layout.outer)
        if const_expr(self.use_2cta_instrs):
            if const_expr(self.tile_hdim <= 128):
                sdS_xchg = storage.sdS_xchg.get_tensor(sdS_xchg_layout)
            else:
                sdS_xchg = storage.sdQaccum.get_tensor(sdS_xchg_layout, dtype=self.ds_dtype)
        else:
            sdS_xchg = None

        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner, dtype=self.do_dtype)
        if const_expr(self.use_2cta_instrs and self.tile_hdim <= 128):
            sdOt = storage.sdOt.get_tensor(sdOt_layout.outer, swizzle=sdOt_layout.inner, dtype=self.do_dtype)
        else:
            sdOt = cute.make_tensor(
                cute.recast_ptr(sdO.iterator, sdOt_layout.inner, dtype=self.do_dtype),
                sdOt_layout.outer,
            )

        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)
        if const_expr(self.use_2cta_instrs):
            if const_expr(not self.dKV_postprocess):
                sdV = storage.sV.get_tensor(sdV_layout.outer, swizzle=sdV_layout.inner, dtype=self.dv_dtype)
                sdK = storage.sK.get_tensor(sdK_layout.outer, swizzle=sdK_layout.inner, dtype=self.dk_dtype)
            else:
                sdV = storage.sV.get_tensor(sdV_layout, dtype=self.dv_dtype)
                sdK = storage.sK.get_tensor(sdK_layout, dtype=self.dk_dtype)
        elif const_expr(not self.dKV_postprocess):
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
        thr_mma_S = tiled_mma_S.get_slice(mma_tile_coord_v)
        Sacc_shape = thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2])  # (M, N)
        tStS = thr_mma_S.make_fragment_C(Sacc_shape)
        # (MMA, MMA_M, MMA_N)
        tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS.layout)
        # dP
        thr_mma_dP = tiled_mma_dP.get_slice(mma_tile_coord_v)
        dPacc_shape = thr_mma_dP.partition_shape_C(self.mma_tiler_vdo[:2])
        tdPtdP = thr_mma_dP.make_fragment_C(dPacc_shape)
        tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPtdP.layout)
        # dV
        thr_mma_dV = tiled_mma_dV.get_slice(mma_tile_coord_v)
        dvacc_shape = thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2])
        tdVtdV = thr_mma_dV.make_fragment_C(dvacc_shape)
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)
        tP = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_P_offset, dtype=self.do_dtype), tP_layout.outer)
        # dK
        thr_mma_dK = tiled_mma_dK.get_slice(mma_tile_coord_v)
        dkacc_shape = thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2])
        tdKtdK = thr_mma_dK.make_fragment_C(dkacc_shape)
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)
        tdS = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_dS_offset, dtype=self.ds_dtype), tdS_layout.outer)
        # dQ
        thr_mma_dQ = tiled_mma_dQ.get_slice(mma_tile_coord_v)
        dQacc_shape = thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2])
        tdQtdQ = thr_mma_dQ.make_fragment_C(dQacc_shape)
        tdQtdQ = cute.make_tensor(tmem_ptr + self.tmem_dQ_offset, tdQtdQ.layout)

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n * self.cluster_shape_mnk[0],  # careful, this case is not very well-tested
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            tile_m=self.tile_m,
            tile_n=self.tile_n * self.cluster_shape_mnk[0],
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        #  EMPTY
        # (15)
        if warp_idx == self.empty_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)

        #  RELAY
        # (14)
        if warp_idx == self.relay_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma if self.use_2cta_instrs else self.num_regs_empty)
            if const_expr(self.use_2cta_instrs):
                self.relay(
                    dS_cluster_full_mbar_ptr,
                    dS_cluster_leader_mbar_ptr,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    blocksparse_tensors,
                )

        #  LOAD
        # (13)
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                thr_mma_dQ,
                mQ,
                mK,
                mKt,
                mV,
                mdO,
                mQt,
                mdOt,
                mLSE,
                mdPsum,
                sQ,
                sK,
                sKt,
                sV,
                sdO,
                sQt,
                sdOt,
                sLSE,
                sdPsum,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_Kt,
                tma_atom_V,
                tma_atom_dO,
                tma_atom_Qt,
                tma_atom_dOt,
                pipeline_Q,
                pipeline_Qt,
                pipeline_Kt,
                pipeline_dO,
                pipeline_LSE,
                pipeline_dPsum,
                cluster_layout_vmnk,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                should_load_Q=True,
                should_load_dO=True,
            )

        #  MMA
        # (12)
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)

            # Alloc tmem buffer
            tmem.allocate(self.tmem_alloc_cols)
            if const_expr(use_unaligned_tmem_barrier):
                barrier.sync_unaligned(
                    tmem_alloc_barrier.barrier_id,
                    tmem_alloc_barrier.num_threads,
                )
            else:
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
                dS_cluster_leader_mbar_ptr,
                pipeline_Q,
                pipeline_Qt,
                pipeline_Kt,
                pipeline_dO,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                pipeline_dQ,
                SeqlenInfoCls,
                TileSchedulerCls,
                is_leader_cta,
                blocksparse_tensors,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            if const_expr(use_unaligned_tmem_barrier):
                barrier.sync_unaligned(
                    tmem_alloc_barrier.barrier_id,
                    tmem_alloc_barrier.num_threads,
                )
            else:
                tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # Compute
        # (4, 5, 6, 7, 8, 9, 10, 11) --> 8 warps
        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_compute)  # 8 warps
            if const_expr(use_unaligned_tmem_barrier):
                barrier.sync_unaligned(
                    tmem_alloc_barrier.barrier_id,
                    tmem_alloc_barrier.num_threads,
                )
            else:
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
                sdS_xchg,
                pipeline_LSE,
                pipeline_dPsum,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                dS_cluster_full_mbar_ptr,
                dQaccum_empty_mbar_ptr,
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
                mdK_semaphore,
                mdV_semaphore,
                blocksparse_tensors,
            )
            if const_expr(use_unaligned_tmem_barrier):
                tmem_alloc_barrier.arrive_unaligned()
            else:
                tmem_alloc_barrier.arrive()

        # Reduce
        # (0, 1, 2, 3) - dQ
        if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_reduce)
            if const_expr(use_unaligned_tmem_barrier):
                barrier.sync_unaligned(
                    tmem_alloc_barrier.barrier_id,
                    tmem_alloc_barrier.num_threads,
                )
            else:
                tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.dQacc_reduce(
                mdQaccum,
                sdQaccum,
                thr_mma_dQ,
                tdQtdQ,
                pipeline_dQ,
                dQaccum_empty_mbar_ptr,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                mdQ_semaphore,
                blocksparse_tensors,
            )
            if const_expr(use_unaligned_tmem_barrier):
                tmem_alloc_barrier.arrive_unaligned()
            else:
                tmem_alloc_barrier.arrive()

        return

    @cute.jit
    def relay(
        self,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dS_cluster_leader_mbar_ptr: cute.Pointer,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        dS_cluster_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_sparse = n_block // self.cta_group_size
            n_blocks_per_sample = cute.ceil_div(seqlen.seqlen_k, self.tile_n * self.cta_group_size)

            num_iters = get_total_q_block_count_bwd(
                blocksparse_tensors,
                batch_idx,
                head_idx,
                n_block_sparse,
                subtile_factor=self.subtile_factor,
                n_blocks_per_sample=n_blocks_per_sample,
            )
            process_tile = num_iters > Int32(0)

            if process_tile:
                for _ in cutlass.range(num_iters, unroll=1):
                    # Wait for dS_xchg from peer CTA
                    cute.arch.mbarrier_wait(dS_cluster_full_mbar_ptr, phase=dS_cluster_phase)

                    # Arrive on MMA leader warp
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(dS_cluster_leader_mbar_ptr, Int32(0))

                    dS_cluster_phase ^= 1

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def load(
        self,
        thr_mma_S: cute.ThrMma,
        thr_mma_dP: cute.ThrMma,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        thr_mma_dQ: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mKt: Optional[cute.Tensor],
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mQt: Optional[cute.Tensor],
        mdOt: Optional[cute.Tensor],
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sKt: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sQt: cute.Tensor,
        sdOt: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_Kt: Optional[cute.CopyAtom],
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_Qt: Optional[cute.CopyAtom],
        tma_atom_dOt: Optional[cute.CopyAtom],  # 2-CTA only
        pipeline_Q: PipelineAsync,
        pipeline_Qt: PipelineAsync,
        pipeline_Kt: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        cluster_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        should_load_Q: bool = True,
        should_load_dO: bool = True,
    ):
        producer_state_Q_LSE = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.Q_stage)
        producer_state_Qt = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.Q_stage)
        producer_state_Kt = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.single_stage)
        producer_state_dO_dPsum = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.dO_stage)
        producer_state_Q_Qt = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.Q_stage)
        producer_state_O_Ot = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.dO_stage)
        producer_state_LSE = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.Q_stage)
        producer_state_dPsum = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.dO_stage)

        # Compute multicast mask for Q & dO buffer full
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        q_do_mcast_mask = None
        if const_expr(self.is_q_do_mcast):
            q_do_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_max = block_info.get_m_block_max(seqlen)
            head_idx_kv = head_idx // self.qhead_per_kvhead
            n_block_cta_group = n_block // self.cta_group_size
            n_block_sparse = n_block_cta_group if const_expr(self.use_2cta_instrs) else n_block
            n_blocks_per_sample = cute.ceil_div(seqlen.seqlen_k, self.tile_n * self.cta_group_size)

            # GMEM tensors (varlen-aware)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
            mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
            if const_expr(not seqlen.has_cu_seqlens_q):
                mdO_cur = mdO[None, None, head_idx, batch_idx]
            else:
                mdO_cur = cute.domain_offset((0, seqlen.offset_q), mdO[None, None, head_idx])
            mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2, padded=True)[None, head_idx]
            mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2, padded=True)[None, head_idx]

            if const_expr(self.use_2cta_instrs):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mQt_cur = mQt[None, None, head_idx, batch_idx]
                    mdOt_cur = mdOt[None, None, head_idx, batch_idx]
                else:
                    mQt_cur = cute.domain_offset((0, seqlen.offset_q, 0), mQt)[None, None, head_idx]
                    mdOt_cur = cute.domain_offset((seqlen.offset_q, 0, 0), mdOt)[None, None, head_idx]
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mKt_cur = mKt[None, None, head_idx_kv, batch_idx]
                else:
                    mKt_cur = cute.domain_offset((0, seqlen.offset_k, 0), mKt)[None, None, head_idx_kv]

            # (1) S.T = K @ Q.T
            gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block_cta_group, 0))
            tSgK = thr_mma_S.partition_A(gK)

            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0))
            tSgQ = thr_mma_S.partition_B(gQ)
            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
            gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))
            gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
            tdPgdO = thr_mma_dV.partition_B(gdO)

            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            load_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                tSgK,
                sK,
                single_stage=True,
            )

            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tSgQ,
                dst_tensor=sQ,
                mcast_mask=q_do_mcast_mask,
            )
            load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)

            # (2) dP = V @ dO.T
            gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (n_block_cta_group, 0))
            tdPgV = thr_mma_dP.partition_A(gV)

            load_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V,
                0,
                cute.make_layout(1),
                tdPgV,
                sV,
                single_stage=True,
            )

            if const_expr(tma_atom_dOt is not None):
                gdOt = cute.local_tile(mdOt_cur, cute.select(self.mma_tiler_vdo, mode=[1, 2]), (None, 0))
                tdPgdO = thr_mma_dP.partition_B(gdOt)
                load_dOt, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dOt,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=b_cta_layout,
                    src_tensor=tdPgdO,
                    dst_tensor=sdOt,
                    mcast_mask=q_do_mcast_mask,
                )
                load_dOt = copy_utils.tma_producer_copy_fn(load_dOt, pipeline_dO)

            # (3) dV += P.T @ dO
            gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
            tdVgdO = thr_mma_dV.partition_B(gdO)
            load_dO, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tdVgdO,
                dst_tensor=sdO,
                mcast_mask=q_do_mcast_mask,
            )
            load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)

            # (4) dK += dS.T @ Q (2-CTA: needs separate Qt load)
            if const_expr(tma_atom_Qt is not None):
                gQt = cute.local_tile(mQt_cur, cute.select(self.mma_tiler_dsq, mode=[1, 2]), (0, None))
                tdKgQt = thr_mma_dK.partition_B(gQt)
                load_Qt, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Qt,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=b_cta_layout,
                    src_tensor=tdKgQt,
                    dst_tensor=sQt,
                    mcast_mask=q_do_mcast_mask,
                )
                load_Qt = copy_utils.tma_producer_copy_fn(load_Qt, pipeline_Qt)

            # (5) dQ = dS @ K
            if const_expr(self.use_2cta_instrs):
                gKt = cute.local_tile(mKt_cur, cute.select(self.mma_tiler_dsk, mode=[1, 2]), (0, n_block_cta_group))
                tdQgK = thr_mma_dQ.partition_B(gKt)

                load_Kt, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Kt,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    tdQgK,
                    sKt,
                    single_stage=True,
                )

            copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
            copy_stats = partial(bulk_copy, copy_atom_stats)

            # some tiles might be empty due to block sparsity
            total_m_block_cnt = get_total_q_block_count_bwd(
                blocksparse_tensors,
                batch_idx,
                head_idx,
                n_block_sparse,
                subtile_factor=self.subtile_factor,
                n_blocks_per_sample=n_blocks_per_sample,
            )
            process_tile = total_m_block_cnt > Int32(0)

            if process_tile:
                if const_expr(self.use_2cta_instrs and self.tile_hdim == 192):
                    assert should_load_Q and should_load_dO
                    (
                        curr_q_cnt,
                        curr_q_idx,
                        curr_full_cnt,
                        curr_full_idx,
                        _,
                        _,
                        _,
                    ) = get_block_sparse_iteration_info_bwd(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block_sparse,
                        self.subtile_factor,
                        n_blocks_per_sample=n_blocks_per_sample,
                    )

                    for block_group in cutlass.range_constexpr(2):
                        group_loop_count = curr_full_cnt * self.subtile_factor
                        iter_offset = Int32(0)
                        if const_expr(block_group == 1):
                            group_loop_count = curr_q_cnt * self.subtile_factor
                            iter_offset = curr_full_cnt * self.subtile_factor

                        for group_iter_idx in cutlass.range(group_loop_count, unroll=1):
                            iter_idx = group_iter_idx + iter_offset
                            sparse_iter_idx = group_iter_idx // self.subtile_factor
                            subtile_offset = group_iter_idx % self.subtile_factor
                            if const_expr(block_group == 0):
                                m_block = curr_full_idx[sparse_iter_idx] * self.subtile_factor + subtile_offset
                            else:
                                m_block = curr_q_idx[sparse_iter_idx] * self.subtile_factor + subtile_offset
                            m_block_safe = m_block
                            if m_block_max > 0:
                                m_block_safe = cutlass.min(m_block, m_block_max - 1)

                            if iter_idx == 0:
                                # K & Q (for S)
                                pipeline_Q.producer_acquire(
                                    producer_state_Q_Qt,
                                    extra_tx_count=self.tma_copy_bytes["K"],
                                )
                                load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_Qt))
                                load_Q(m_block_safe, producer_state=producer_state_Q_Qt)
                                pipeline_Q.producer_commit(producer_state_Q_Qt)
                                producer_state_Q_Qt.advance()

                                # LSE
                                pipeline_LSE.producer_acquire(producer_state_LSE)
                                copy_stats(
                                    gLSE[None, m_block_safe],
                                    sLSE[None, producer_state_LSE.index],
                                    mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_LSE),
                                )
                                producer_state_LSE.advance()

                                # dOt + V, for dP.T = V @ dO.T
                                pipeline_dO.producer_acquire(
                                    producer_state_O_Ot,
                                    extra_tx_count=self.tma_copy_bytes["V"],
                                )
                                load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_O_Ot))
                                load_dOt(m_block_safe, producer_state=producer_state_O_Ot)
                                pipeline_dO.producer_commit(producer_state_O_Ot)
                                producer_state_O_Ot.advance()

                                # dPsum
                                pipeline_dPsum.producer_acquire(producer_state_dPsum)
                                copy_stats(
                                    gdPsum[None, m_block_safe],
                                    sdPsum[None, producer_state_dPsum.index],
                                    mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dPsum),
                                )
                                producer_state_dPsum.advance()

                                # Qt + Kt, for dK = dS.T @ Q and dQ = dS @ K
                                pipeline_Qt.producer_acquire(
                                    producer_state_Q_Qt,
                                    extra_tx_count=self.tma_copy_bytes["K"],
                                )
                                load_Qt(m_block_safe, producer_state=producer_state_Q_Qt)
                                load_Kt(tma_bar_ptr=pipeline_Qt.producer_get_barrier(producer_state_Q_Qt))
                                pipeline_Qt.producer_commit(producer_state_Q_Qt)
                                producer_state_Q_Qt.advance()

                                # dO, for dV = P.T @ dO
                                pipeline_dO.producer_acquire(producer_state_O_Ot)
                                load_dO(m_block_safe, producer_state=producer_state_O_Ot)
                                pipeline_dO.producer_commit(producer_state_O_Ot)
                                producer_state_O_Ot.advance()

                            else:
                                # LSE
                                pipeline_LSE.producer_acquire(producer_state_LSE)
                                copy_stats(
                                    gLSE[None, m_block_safe],
                                    sLSE[None, producer_state_LSE.index],
                                    mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_LSE),
                                )
                                producer_state_LSE.advance()

                                # Q
                                pipeline_Q.producer_acquire(producer_state_Q_Qt)
                                load_Q(m_block_safe, producer_state=producer_state_Q_Qt)
                                pipeline_Q.producer_commit(producer_state_Q_Qt)
                                producer_state_Q_Qt.advance()

                                # dPsum
                                pipeline_dPsum.producer_acquire(producer_state_dPsum)
                                copy_stats(
                                    gdPsum[None, m_block_safe],
                                    sdPsum[None, producer_state_dPsum.index],
                                    mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dPsum),
                                )
                                producer_state_dPsum.advance()

                                # dOt, for dP.T = V @ dO.T
                                pipeline_dO.producer_acquire(producer_state_O_Ot)
                                load_dOt(m_block_safe, producer_state=producer_state_O_Ot)
                                pipeline_dO.producer_commit(producer_state_O_Ot)
                                producer_state_O_Ot.advance()

                                # Qt, for dK = dS.T @ Q
                                pipeline_Qt.producer_acquire(producer_state_Q_Qt)
                                load_Qt(m_block_safe, producer_state=producer_state_Q_Qt)
                                pipeline_Qt.producer_commit(producer_state_Q_Qt)
                                producer_state_Q_Qt.advance()

                                # dO, for dV = P.T @ dO
                                pipeline_dO.producer_acquire(producer_state_O_Ot)
                                load_dO(m_block_safe, producer_state=producer_state_O_Ot)
                                pipeline_dO.producer_commit(producer_state_O_Ot)
                                producer_state_O_Ot.advance()
                elif const_expr(self.use_2cta_instrs and self.tile_hdim == 128):
                    assert should_load_Q and should_load_dO
                    (
                        curr_q_cnt,
                        curr_q_idx,
                        curr_full_cnt,
                        curr_full_idx,
                        _,
                        _,
                        loop_count,
                    ) = get_block_sparse_iteration_info_bwd(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block_sparse,
                        self.subtile_factor,
                        n_blocks_per_sample=n_blocks_per_sample,
                    )

                    prev_m_block_safe = Int32(0)
                    for iter_idx in cutlass.range(loop_count, unroll=1):
                        m_block, _ = get_m_block_from_iter_bwd(
                            iter_idx,
                            curr_q_cnt,
                            curr_q_idx,
                            curr_full_cnt,
                            curr_full_idx,
                            self.subtile_factor,
                            full_first=True,
                        )
                        m_block_safe = m_block
                        if m_block_max > 0:
                            m_block_safe = cutlass.min(m_block, m_block_max - 1)

                        if iter_idx == 0:
                            # K & Q, for S = K @ Q
                            pipeline_Q.producer_acquire(
                                producer_state_Q_LSE,
                                extra_tx_count=self.tma_copy_bytes["K"],
                            )
                            load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE))
                            load_Q(m_block_safe, producer_state=producer_state_Q_LSE)
                            pipeline_Q.producer_commit(producer_state_Q_LSE)

                            # LSE
                            pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                            copy_stats(
                                gLSE[None, m_block_safe],
                                sLSE[None, producer_state_Q_LSE.index],
                                mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                            )
                            producer_state_Q_LSE.advance()

                            # dO/dOt + V, for dP and dV
                            pipeline_dO.producer_acquire(
                                producer_state_dO_dPsum,
                                extra_tx_count=(
                                    self.tma_copy_bytes["V"] + self.tma_copy_bytes["dO"] if const_expr(tma_atom_dOt is not None) else self.tma_copy_bytes["V"]
                                ),
                            )
                            load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_dPsum))
                            load_dO(m_block_safe, producer_state=producer_state_dO_dPsum)
                            if const_expr(tma_atom_dOt is not None):
                                load_dOt(m_block_safe, producer_state=producer_state_dO_dPsum)
                            pipeline_dO.producer_commit(producer_state_dO_dPsum)

                            # dPsum
                            pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                            copy_stats(
                                gdPsum[None, m_block_safe],
                                sdPsum[None, producer_state_dO_dPsum.index],
                                mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                            )
                            producer_state_dO_dPsum.advance()

                            # Kt, for dQ = dS @ K
                            pipeline_Kt.producer_acquire(producer_state_Kt)
                            load_Kt(tma_bar_ptr=pipeline_Kt.producer_get_barrier(producer_state_Kt))
                            pipeline_Kt.producer_commit(producer_state_Kt)
                            producer_state_Kt.advance()
                        else:
                            # Qt is consumed one sparse iteration behind Q.
                            if const_expr(tma_atom_Qt is not None):
                                pipeline_Qt.producer_acquire(producer_state_Qt)
                                load_Qt(prev_m_block_safe, producer_state=producer_state_Qt)
                                pipeline_Qt.producer_commit(producer_state_Qt)
                                producer_state_Qt.advance()

                            # Q
                            pipeline_Q.producer_acquire(producer_state_Q_LSE)
                            load_Q(m_block_safe, producer_state=producer_state_Q_LSE)
                            pipeline_Q.producer_commit(producer_state_Q_LSE)

                            # LSE
                            pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                            copy_stats(
                                gLSE[None, m_block_safe],
                                sLSE[None, producer_state_Q_LSE.index],
                                mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                            )
                            producer_state_Q_LSE.advance()

                            # dO/dOt
                            pipeline_dO.producer_acquire(
                                producer_state_dO_dPsum,
                                extra_tx_count=self.tma_copy_bytes["dO"] if const_expr(tma_atom_dOt is not None) else 0,
                            )
                            load_dO(m_block_safe, producer_state=producer_state_dO_dPsum)
                            if const_expr(tma_atom_dOt is not None):
                                load_dOt(m_block_safe, producer_state=producer_state_dO_dPsum)
                            pipeline_dO.producer_commit(producer_state_dO_dPsum)

                            # dPsum
                            pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                            copy_stats(
                                gdPsum[None, m_block_safe],
                                sdPsum[None, producer_state_dO_dPsum.index],
                                mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                            )
                            producer_state_dO_dPsum.advance()

                        prev_m_block_safe = m_block_safe

                    # Tail Qt for the final sparse iteration.
                    if const_expr(tma_atom_Qt is not None):
                        pipeline_Qt.producer_acquire(producer_state_Qt)
                        load_Qt(prev_m_block_safe, producer_state=producer_state_Qt)
                        pipeline_Qt.producer_commit(producer_state_Qt)
                        producer_state_Qt.advance()
                else:
                    producer_state_Q_LSE, producer_state_dO_dPsum = produce_block_sparse_q_loads_bwd_sm100(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block_sparse,
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
                        should_load_Q=should_load_Q,
                        should_load_dO=should_load_dO,
                        subtile_factor=self.subtile_factor,
                        m_block_max=m_block_max,
                        n_blocks_per_sample=n_blocks_per_sample,
                    )

                if const_expr(self.use_2cta_instrs and self.tile_hdim == 192):
                    pipeline_Q.producer_tail(producer_state_Q_Qt)
                    pipeline_LSE.producer_tail(producer_state_LSE)
                    pipeline_dO.producer_tail(producer_state_O_Ot)
                    pipeline_dPsum.producer_tail(producer_state_dPsum)
                else:
                    if const_expr(should_load_Q):
                        pipeline_Q.producer_tail(producer_state_Q_LSE.clone())
                        pipeline_LSE.producer_tail(producer_state_Q_LSE)
                        if const_expr(tma_atom_Qt is not None):
                            pipeline_Qt.producer_tail(producer_state_Qt)
                    if const_expr(should_load_dO):
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
        dS_cluster_leader_mbar_ptr: cute.Pointer,
        pipeline_Q: PipelineAsync,
        pipeline_Qt: PipelineAsync,
        pipeline_Kt: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        pipeline_dQ: PipelineAsync,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        is_leader_cta: cutlass.Boolean,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        # [2025-10-21] For reasons I don't understand, putting these partitioning in the main
        # kernel (before warp specialization) is a lot slower tha putting them here.
        # Partition smem / tmem tensors
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
            cta_group=self.cta_group_size,
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
            cta_group=self.cta_group_size,
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
            cta_group=self.cta_group_size,
        )
        num_unroll_groups = 2 if const_expr(self.use_2cta_instrs) else 1
        mma_dsk_fn = partial(
            gemm_w_idx,
            tiled_mma_dQ,
            tdQtdQ,
            tdQrdS,
            tdQrK,
            zero_init=True,
            num_unroll_groups=num_unroll_groups,
        )
        # Need to explicitly pass in tA_addr for correctness.
        mma_dsq_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dK,
            tdKtdK,
            tdKrdS,
            tdKrQ,
            sA=None,
            sB=sQt,
            tA_addr=self.tmem_dS_offset,
            cta_group=self.cta_group_size,
        )

        pipeline_Q_consumer = pipeline_Q.make_consumer()

        consumer_state_Qt = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage)
        consumer_state_Q = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage)
        consumer_state_Kt = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.single_stage)
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage)
        producer_phase_acc = Int32(1)  # For S & P, dP, dQ
        producer_phase_dQ = Int32(1)  # 2-CTA: separate phase for dQ pipeline
        consumer_state_dS = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
        producer_phase_dKV = Int32(1)
        cta_group = pipeline_S_P.cta_group

        dS_cluster_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)  # must be seqlen_k
            n_block_sparse = n_block // self.cta_group_size if const_expr(self.use_2cta_instrs) else n_block
            n_blocks_per_sample = cute.ceil_div(seqlen.seqlen_k, self.tile_n * self.cta_group_size)

            block_iter_count = get_total_q_block_count_bwd(
                blocksparse_tensors,
                batch_idx,
                head_idx,
                n_block_sparse,
                subtile_factor=self.subtile_factor,
                n_blocks_per_sample=n_blocks_per_sample,
            )
            process_tile = block_iter_count > Int32(0)

            if const_expr(self.use_2cta_instrs and self.tile_hdim == 192):
                if is_leader_cta and process_tile:
                    accumulate_dK = False
                    accumulate_dV = False

                    # -----------------------------------------------------------
                    ###### MAIN LOOP
                    # -----------------------------------------------------------
                    # 1. S.T  = K    @ Q.T
                    # 2. dP.T = V    @ dO.T
                    # 3. dK   = dS.T @ Q
                    # 4. dV   = P.T  @ dO
                    # 5. dQ   = dS   @ K

                    main_loop_iters = block_iter_count

                    # empty waits

                    for _ in cutlass.range(main_loop_iters, unroll=1):
                        # 1) S.T = K @ Q.T
                        pipeline_Q.consumer_wait(consumer_state_Q)
                        pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)  # dQ tmem overlaps with S
                        mma_qk_fn(B_idx=consumer_state_Q.index)
                        pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()

                        producer_phase_acc ^= 1

                        # 2) dP.T = V @ dO.T
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)  # dP tmem overlaps with S
                        mma_dov_fn(B_idx=consumer_state_dO.index)
                        pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()

                        # 3) dK = dS.T @ Q
                        pipeline_Q.consumer_wait(consumer_state_Q)
                        pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)  # dP -> dS
                        mma_dsq_fn(B_idx=consumer_state_Q.index, zero_init=not accumulate_dK)
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()
                        accumulate_dK = True

                        # 4) dV = P.T @ dO
                        # Note: if dS is written to tmem, P must be written to tmem
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=not accumulate_dV)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()
                        accumulate_dV = True

                        # 5) dQ = dS @ K
                        pipeline_dS.consumer_wait(consumer_state_dS)
                        cute.arch.mbarrier_wait(dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase)
                        mma_dsk_fn()
                        pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()
                        dS_cluster_phase ^= 1

                    # signal to the epilogue that dV is ready
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                    pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                    # signal to the epilogue that dK is ready
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)
                    pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                    producer_phase_dKV ^= 1
            elif const_expr(self.use_2cta_instrs):
                if is_leader_cta and process_tile:
                    accumulate_dK = False
                    # -----------------------------------------------------------
                    ###### Prologue
                    # -----------------------------------------------------------
                    # 1. S  = Q0 @ K.T
                    # 2. dP = V @ dOt.T
                    # 3. dV = P @ dO

                    # 1) S = K @ Q
                    pipeline_Q.consumer_wait(consumer_state_Q)
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    mma_qk_fn(B_idx=consumer_state_Q.index)
                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                    pipeline_Q.consumer_release(consumer_state_Q)
                    consumer_state_Q.advance()

                    # 2) dP = V @ dOt.T
                    pipeline_dO.consumer_wait(consumer_state_dO)
                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                    mma_dov_fn(B_idx=consumer_state_dO.index)
                    pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                    # 3) dV = P.T @ dO
                    producer_phase_acc ^= 1
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()

                    pipeline_Kt.consumer_wait(consumer_state_Kt)
                    # -----------------------------------------------------------
                    ###### MAIN LOOP
                    # -----------------------------------------------------------
                    # 1. S.T  = K    @ Q.T
                    # 2. dK   = dS.T @ Q
                    # 3. dP.T = V    @ dO.T
                    # 4. dQ   = dS   @ K
                    # 5. dV   = P.T  @ dO

                    main_loop_iters = block_iter_count - 1

                    for _ in cutlass.range(main_loop_iters, unroll=1):
                        # (1) S.T = K @ Q.T (next)
                        pipeline_Q.consumer_wait(consumer_state_Q)
                        pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)
                        mma_qk_fn(B_idx=consumer_state_Q.index)
                        pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()

                        # (2) dK += dS.T @ Q (cur)
                        pipeline_Qt.consumer_wait(consumer_state_Qt)
                        pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)  # dP -> dS
                        mma_dsq_fn(B_idx=consumer_state_Qt.index, zero_init=not accumulate_dK)
                        accumulate_dK = True
                        pipeline_Qt.consumer_release(consumer_state_Qt)
                        consumer_state_Qt.advance()

                        # (3) dP.T = V @ dO.T (next)
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        mma_dov_fn(B_idx=consumer_state_dO.index)
                        pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                        # (5) dQ = dS @ K (cur)
                        pipeline_dS.consumer_wait(consumer_state_dS)
                        cute.arch.mbarrier_wait(dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase)
                        mma_dsk_fn()
                        pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()
                        dS_cluster_phase ^= 1
                        producer_phase_dQ ^= 1

                        # (4) dV += P.T @ dO (next)
                        producer_phase_acc ^= 1
                        pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)  # S -> P
                        mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()

                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                    # signal to the epilogue that dV is ready
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                    pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)

                    # -----------------------------------------------------------
                    # Tail: Remaining dK and dQ
                    # -----------------------------------------------------------
                    # dK += dS.T @ Q
                    pipeline_Qt.consumer_wait(consumer_state_Qt)
                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)  # dP -> dS
                    mma_dsq_fn(B_idx=consumer_state_Qt.index, zero_init=not accumulate_dK)
                    pipeline_Qt.consumer_release(consumer_state_Qt)
                    consumer_state_Qt.advance()
                    # signal to the epilogue that dK is ready
                    pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                    producer_phase_dKV ^= 1

                    # dQ = dS @ K
                    pipeline_dS.consumer_wait(consumer_state_dS)
                    cute.arch.mbarrier_wait(dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase)
                    pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)
                    mma_dsk_fn()
                    pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                    pipeline_dS.consumer_release(consumer_state_dS)
                    pipeline_Kt.consumer_release(consumer_state_Kt)
                    consumer_state_dS.advance()
                    consumer_state_Kt.advance()
                    dS_cluster_phase ^= 1
                    producer_phase_dQ ^= 1

                    producer_phase_acc ^= 1
            else:
                if is_leader_cta and process_tile:
                    accumulate_dK = False
                    # -----------------------------------------------------------
                    ###### Prologue
                    # -----------------------------------------------------------
                    # 1. S  = Q0 @ K.T
                    # 2. dP = V @ dOt.T
                    # 3. dV = P @ dO

                    # 1) S = K @ Q
                    handle_Q = pipeline_Q_consumer.wait_and_advance()
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    mma_qk_fn(B_idx=handle_Q.index)
                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                    # 2) dP = V @ dOt.T
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

                    # -----------------------------------------------------------
                    ###### MAIN LOOP
                    # -----------------------------------------------------------
                    # 1. S  = K    @ Q.T
                    # 2. dQ = dS   @ K
                    # 3. dK = dS.T @ Q
                    # 4. dP = V    @ dOt.T
                    # 5. dV = P.T  @ dO

                    # For block sparsity, we use block_iter_count; for dense, use m_block range
                    # MMA doesn't need actual m_block indices, just the iteration count
                    main_loop_iters = block_iter_count - 1

                    handle_Q_next = handle_Q
                    for _ in cutlass.range(main_loop_iters, unroll=1):
                        # (1) S.T = K @ Q.T
                        handle_Q_next = pipeline_Q_consumer.wait_and_advance()
                        mma_qk_fn(B_idx=handle_Q_next.index)
                        pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                        # (2) dK += dS.T @ Q
                        pipeline_dS.consumer_wait(consumer_state_dS)
                        mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                        accumulate_dK = True
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

                    # signal to the epilogue that dV is ready
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                    pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)

                    # -----------------------------------------------------------
                    # Tail: Remaining dK and dQ
                    # -----------------------------------------------------------
                    # 1) dK += dS.T @ Q
                    pipeline_dS.consumer_wait(consumer_state_dS)
                    mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                    # signal to the epilogue that dK is ready
                    pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                    producer_phase_dKV ^= 1

                    # 2) dQ = dS @ K
                    mma_dsk_fn()
                    pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                    handle_Q.release()
                    pipeline_dS.consumer_release(consumer_state_dS)
                    consumer_state_dS.advance()

                    producer_phase_acc ^= 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # Currently it hangs if we have this S_P.producer_tail, will need to understand why

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
            if const_expr(rank == 3 and reduced_shape[2] >= num_wg):
                t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1], reduced_shape[2] // num_wg))
                coord = (
                    None,
                    None,
                    (None, wg_idx),
                ) + (
                    None,
                ) * (rank - 3)
            elif const_expr(rank >= 4 and reduced_shape[3] >= num_wg):
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
            else:
                assert reduced_shape[0] % num_wg == 0, "The leading mode must divide the compute warp groups"
                t = cute.logical_divide(
                    t,
                    (reduced_shape[0] // num_wg,) + tuple(reduced_shape[1:]),
                )
                coord = ((None, wg_idx),) + (None,) * (rank - 1)
        return t[coord]

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
        sdS_xchg: cute.Tensor,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dQaccum_empty_mbar_ptr: cute.Pointer,
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
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        blocksparse_tensors: BlockSparseTensors = None,
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

        # 2-CTA assumes: repetiton should always be 32 & 16
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

        if const_expr(self.use_2cta_instrs):
            sdS_xchg_epi = cute.make_tensor(cute.recast_ptr(sdS_xchg.iterator, sdS_epi_layout.inner), sdS_layout)
            tRS_sdS_xchg = thr_copy_r2s.partition_D(sdS_xchg_epi)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        # 2-CTA: CTA 0 exchanges stage 1 (bottom half), CTA 1 exchanges stage 0 (top half)
        exchange_stage = cta_rank_in_cluster ^ 1 if const_expr(self.use_2cta_instrs) else Int32(0)

        consumer_state_S_P_dP = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)  # Our impl has shortcut for stage==1
        producer_state_dS = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, 1)  # Our impl has shortcut for stage==1
        consumer_state_dKV = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 2)
        consumer_state_LSE = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage)
        consumer_state_dPsum = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_max = block_info.get_m_block_max(seqlen)
            n_block_for_cluster = n_block // self.cta_group_size
            n_blocks_per_sample = cute.ceil_div(seqlen.seqlen_k, self.tile_n * self.cta_group_size)
            curr_q_cnt = Int32(0)
            curr_full_cnt = Int32(0)
            partial_payload_base = Int32(0)
            (
                curr_q_cnt,
                _,
                curr_full_cnt,
                _,
                partial_payload_base,
                _,
                loop_count,
            ) = get_block_sparse_iteration_info_bwd(
                blocksparse_tensors,
                batch_idx,
                head_idx,
                n_block_for_cluster,
                subtile_factor=self.subtile_factor,
                n_blocks_per_sample=n_blocks_per_sample,
            )
            process_tile = loop_count > Int32(0)

            # Mainloop
            # Block sparsity uses constexpr-split groups for partial and full blocks so the
            # full path can call the seqlen-only mask function instead of carrying the full mask.
            assert blocksparse_tensors.full_block_idx is not None
            full_first_sparse = const_expr(self.use_2cta_instrs)
            for block_group in cutlass.range_constexpr(2):
                is_full_group = const_expr((full_first_sparse and block_group == 0) or (not full_first_sparse and block_group == 1))
                iter_offset = Int32(0)
                if const_expr(is_full_group):
                    group_loop_count = curr_full_cnt * self.subtile_factor
                    if const_expr(not full_first_sparse):
                        iter_offset = curr_q_cnt * self.subtile_factor
                else:
                    group_loop_count = curr_q_cnt * self.subtile_factor
                    if const_expr(full_first_sparse):
                        iter_offset = curr_full_cnt * self.subtile_factor

                for group_iter_idx in cutlass.range(group_loop_count, unroll=1):
                    iter_idx = group_iter_idx + iter_offset
                    # Prefetch 1 stage of LSE
                    pipeline_LSE.consumer_wait(consumer_state_LSE)
                    tSrLSE_s2r = cute.make_rmem_tensor(tScS_t2r[None, 0, 0, 0].shape, Float32)
                    pipeline_S_P.consumer_wait(consumer_state_S_P_dP)
                    #### TMEM->RMEM (Load S from TMEM)
                    tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, Float32)
                    cute.copy(thr_copy_t2r, tStS_t2r, tSrS_t2r)

                    if const_expr(self.tile_hdim == 192):
                        # Signal S tmem load completion using pipeline_S_P when hdim 192
                        # dP is overlapped with S
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                    elif const_expr(self.use_2cta_instrs and self.tile_hdim <= 128):
                        # Signal S tmem load completion using pipeline_dS when 2cta hdim 128
                        # dQ is overlapped with S
                        if iter_idx > 0:
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                pipeline_dS.producer_commit(producer_state_dS)
                            producer_state_dS.advance()

                    if const_expr(not is_full_group):
                        assert blocksparse_tensors.mask_block_masks is not None
                        payload_sparse_idx = group_iter_idx // self.subtile_factor
                        payload_subtile_idx = cta_rank_in_cluster if const_expr(self.use_2cta_instrs) else group_iter_idx % self.subtile_factor
                        r_bitmask = load_mask_payload(
                            blocksparse_tensors.mask_block_masks,
                            partial_payload_base + payload_sparse_idx,
                            tidx,
                            subtile_idx=payload_subtile_idx,
                            payload_words=SM100_BWD_MASK_PAYLOAD_WORDS,
                        )
                        apply_loaded_arbitrary_mask(
                            tSrS_t2r,
                            r_bitmask,
                            SM100_BWD_MASK_PAYLOAD_WORDS,
                        )
                    num_stages = cute.size(tScS_t2r, mode=[1])
                    # ---------------------------------------------
                    #### P = exp(S - LSE)
                    # ---------------------------------------------
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
                    if const_expr(not self.tile_hdim == 192):
                        # Signal tmem store P completion with pipeline_S_P
                        with cute.arch.elect_one():
                            pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                    # Normally we'd need syncwarp here since only 1 thread will signal in
                    # consumer_release, but we already have the self.compute_sync_barrier before this
                    pipeline_LSE.consumer_release(consumer_state_LSE)
                    consumer_state_LSE.advance()
                    # ---------------------------------------------
                    # dS.T = P.T * (dP.T - D)
                    # ---------------------------------------------
                    pipeline_dPsum.consumer_wait(consumer_state_dPsum)
                    pipeline_dP.consumer_wait(consumer_state_S_P_dP)
                    # Keep the state unchanged until dS staging completes.

                    ##### dS.T = P.T * (dP.T - Psum)
                    for stage in cutlass.range_constexpr(num_stages):
                        tdPrdP_t2r = cute.make_rmem_tensor(tScS_t2r[None, 0, None, None].shape, Float32)
                        cute.copy(thr_copy_t2r, tdPtdP_t2r[None, stage, None, None], tdPrdP_t2r)
                        cute.arch.fence_view_async_tmem_load()
                        self.compute_sync_barrier.arrive_and_wait()
                        tdPrdP_cur = tdPrdP_t2r[None, 0, 0]
                        tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                        tSsdPsum_cur = tSsdPsum[None, stage, 0, 0, consumer_state_dPsum.index]
                        tSrdPsum = cute.make_fragment_like(tSsdPsum_cur, Float32)
                        cute.autovec_copy(tSsdPsum_cur, tSrdPsum)
                        for v in cutlass.range_constexpr(cute.size(tdPrdP_t2r, mode=[0]) // 2):
                            dPsum_pair = (tSrdPsum[2 * v], tSrdPsum[2 * v + 1])
                            tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = utils.sub_packed_f32x2((tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), dPsum_pair)
                            tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = cute.arch.mul_packed_f32x2(
                                (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                                (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                            )

                        tdPrdS_cvt = cute.make_fragment_like(tdPrdP_cur, self.ds_dtype)
                        utils.cvt_f16(tdPrdP_cur, tdPrdS_cvt)
                        if const_expr(stage == 0):
                            pipeline_dS.producer_acquire(producer_state_dS)
                            if const_expr(self.use_2cta_instrs):
                                tdPrdS_xchg = cute.make_fragment_like(tdPrdS_cvt, self.ds_dtype)

                        tdPrdS_r2t_f32 = cute.recast_tensor(tdPrdS_cvt, Float32)
                        cute.copy(thr_copy_r2t, tdPrdS_r2t_f32, tdPtdS_r2t[None, stage, 0, 0])

                        # RMEM->SMEM: For 2-CTA, keep exchange stage in registers, write non-exchange to sdS
                        if const_expr(self.use_2cta_instrs):
                            if exchange_stage == stage:
                                cute.autovec_copy(tdPrdS_cvt, tdPrdS_xchg)
                            else:
                                cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])
                        else:
                            cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])

                    cute.arch.fence_view_async_tmem_store()

                    if const_expr(self.use_2cta_instrs):
                        # use pipeline_dP to signal tmem store of dS
                        with cute.arch.elect_one():
                            pipeline_dP.consumer_release(consumer_state_S_P_dP)
                    consumer_state_S_P_dP.advance()

                    # After the loop: copy exchange registers to sdS_xchg buffer
                    if const_expr(self.use_2cta_instrs):
                        # when hdim 192, sdQaccum overlapped with sdS_xchg
                        if const_expr(self.tile_hdim == 192):
                            cute.arch.mbarrier_wait(dQaccum_empty_mbar_ptr, phase=producer_state_dS.phase)
                        cute.autovec_copy(tdPrdS_xchg, tRS_sdS_xchg[None, 0])

                    cute.arch.fence_view_async_shared()
                    self.compute_sync_barrier.arrive_and_wait()
                    # Normally we'd need syncwarp here since only 1 thread will signal in
                    # consumer_release, but we already have the self.compute_sync_barrier before this
                    pipeline_dPsum.consumer_release(consumer_state_dPsum)
                    consumer_state_dPsum.advance()
                    # when 2cta hdim 128, pipeline_dS also signals S tmem load completion so is deferred
                    if const_expr(not (self.use_2cta_instrs and self.tile_hdim == 128)):
                        with cute.arch.elect_one():
                            pipeline_dS.producer_commit(producer_state_dS)
                        producer_state_dS.advance()
                    # 2-CTA: DSMEM copy from sdS_xchg to peer's sdS buffer
                    if const_expr(self.use_2cta_instrs):
                        stage_copy_bytes = const_expr(self.tma_copy_bytes["dS"] // 2)
                        stage_copy_elems = const_expr(stage_copy_bytes // (self.ds_dtype.width // 8))
                        if tidx == 0:
                            peer_cta_rank_in_cluster = cta_rank_in_cluster ^ 1
                            smem_src_ptr = sdS_xchg.iterator
                            # Destination is peer's sdS at our CTA's offset (exchange_stage position)
                            smem_dst_ptr = sdS.iterator + cta_rank_in_cluster * stage_copy_elems
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                dS_cluster_full_mbar_ptr,
                                stage_copy_bytes,
                                peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                            )
                            copy_utils.cpasync_bulk_s2cluster(
                                smem_src_ptr,
                                smem_dst_ptr,
                                dS_cluster_full_mbar_ptr,
                                stage_copy_bytes,
                                peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                            )

            # Final signal for dS smem store completion
            if const_expr(self.use_2cta_instrs and self.tile_hdim == 128):
                if process_tile:
                    with cute.arch.elect_one():
                        pipeline_dS.producer_commit(producer_state_dS)
                    producer_state_dS.advance()

            # Epilogue
            # Run epilogue if we processed any m_blocks for this n_block
            if process_tile:
                if const_expr(not self.use_tma_store):
                    consumer_state_dKV = self.epilogue_dKV(
                        dp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        seqlen,
                        thr_mma_dV,
                        thr_mma_dK,
                        tdVtdV,
                        tdKtdK,
                        mdV,
                        mdK,
                        pipeline_dKV,
                        consumer_state_dKV,
                        softmax_scale,
                    )
                else:
                    thr_copy_r2s_dKV = tiled_copy_r2s_dKV.get_slice(dp_idx)
                    #### STORE dV
                    consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                        dp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        seqlen,
                        thr_mma_dV,
                        tdVtdV,
                        mdV_tma_tensor,
                        sdV,
                        tma_atom_dV,
                        thr_copy_r2s_dKV,
                        pipeline_dKV,
                        consumer_state_dKV,
                        None,  # Don't scale
                        int(NamedBarrierBwdSm100.EpilogueWG1),  # barrier_id
                        mdV_semaphore,
                        "V",
                    )
                    #### STORE dK
                    consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                        dp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        seqlen,
                        thr_mma_dK,
                        tdKtdK,
                        mdK_tma_tensor,
                        sdK,
                        tma_atom_dK,
                        thr_copy_r2s_dKV,
                        pipeline_dKV,
                        consumer_state_dKV,
                        softmax_scale if const_expr(not self.dKV_postprocess) else None,
                        int(NamedBarrierBwdSm100.EpilogueWG1),  # barrier_id
                        mdK_semaphore,
                        "K",
                    )
            if const_expr(self.deterministic and self.dKV_postprocess and self.qhead_per_kvhead > 1):
                if not process_tile:
                    # dK/dV accumulation is ordered by Q-head rank within each
                    # KV head.  An empty sparse row contributes zero, but it
                    # must still pass both semaphore tokens to the next head or
                    # a later non-empty head will wait forever.
                    assert mdK_semaphore is not None
                    assert mdV_semaphore is not None
                    head_idx_kv = head_idx // self.qhead_per_kvhead
                    head_rank = head_idx % self.qhead_per_kvhead
                    wg_idx = tidx // 128
                    mdV_semaphore_cur = mdV_semaphore[n_block, None, head_idx_kv, batch_idx]
                    mdK_semaphore_cur = mdK_semaphore[n_block, None, head_idx_kv, batch_idx]
                    barrier.wait_eq(
                        mdV_semaphore_cur.iterator,
                        dp_idx,
                        wg_idx,
                        head_rank,
                    )
                    barrier.arrive_inc(mdV_semaphore_cur.iterator, dp_idx, wg_idx)
                    barrier.wait_eq(
                        mdK_semaphore_cur.iterator,
                        dp_idx,
                        wg_idx,
                        head_rank,
                    )
                    barrier.arrive_inc(mdK_semaphore_cur.iterator, dp_idx, wg_idx)
            # Zero dK/dV for empty tiles (local attention or block sparsity)
            # When total_m_block_cnt == 0 for block sparsity, no Q tiles contribute to this KV tile
            if const_expr(not self.dKV_postprocess):
                should_zero_dKV = False
                if const_expr(self.is_varlen_q):
                    should_zero_dKV = m_block_max <= Int32(0)
                # For block sparsity, zero when no m_blocks contribute to this n_block
                if not process_tile:
                    should_zero_dKV = True

                # The zeroing tile is cluster-wide in 2-CTA mode.  Both CTAs
                # share the same empty K2Q row, so only the leader may write it;
                # the peer still follows the identical scheduler/barrier path.
                if const_expr(self.use_2cta_instrs):
                    should_zero_dKV = should_zero_dKV and cta_rank_in_cluster == Int32(0)

                if should_zero_dKV:
                    # For 2-CTA: use cluster-wide tile size (cta_group_size * tile_n)
                    cluster_tile_n = self.tile_n * self.cta_group_size
                    n_block_for_tile = n_block // self.cta_group_size
                    gmem_tiled_copy_zero_dK = copy_utils.tiled_copy_2d(
                        self.dk_dtype,
                        math.gcd(64, self.tile_hdim),
                        128,  # num_threads
                    )
                    gmem_tiled_copy_zero_dV = copy_utils.tiled_copy_2d(
                        self.dv_dtype,
                        math.gcd(64, self.tile_hdimv),
                        128,  # num_threads
                    )
                    gmem_thr_copy_zero_dK = gmem_tiled_copy_zero_dK.get_slice(dp_idx)
                    gmem_thr_copy_zero_dV = gmem_tiled_copy_zero_dV.get_slice(dp_idx)
                    mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3)[None, None, head_idx]
                    mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3)[None, None, head_idx]
                    gdK = cute.local_tile(mdK_cur, (cluster_tile_n, self.tile_hdim), (n_block_for_tile, 0))
                    gdV = cute.local_tile(mdV_cur, (cluster_tile_n, self.tile_hdimv), (n_block_for_tile, 0))
                    tdKgdK = gmem_thr_copy_zero_dK.partition_D(gdK)
                    tdVgdV = gmem_thr_copy_zero_dV.partition_D(gdV)
                    cdK = cute.make_identity_tensor((cluster_tile_n, self.tile_hdim))
                    cdV = cute.make_identity_tensor((cluster_tile_n, self.tile_hdimv))
                    tdKcdK = gmem_thr_copy_zero_dK.partition_D(cdK)
                    tdVcdV = gmem_thr_copy_zero_dV.partition_D(cdV)
                    assert cute.size(tdKgdK[None, 0, 0]) == cute.size(tdVgdV[None, 0, 0])
                    zero = cute.make_fragment_like(tdKgdK[None, 0, 0])
                    zero.fill(0.0)
                    if tidx < 128:
                        for i in cutlass.range_constexpr(tdKgdK.shape[1]):
                            row_idx = tdKcdK[0, i, 0][0]
                            if row_idx < seqlen.seqlen_k - cluster_tile_n * n_block_for_tile:
                                for j in cutlass.range_constexpr(tdKgdK.shape[2]):
                                    if not const_expr(self.check_hdim_oob) or tdKcdK[0, i, j][1] < self.head_dim:
                                        cute.copy(
                                            gmem_tiled_copy_zero_dK,
                                            zero,
                                            tdKgdK[None, i, j],
                                        )
                    else:
                        for i in cutlass.range_constexpr(tdVgdV.shape[1]):
                            row_idx = tdVcdV[0, i, 0][0]
                            if row_idx < seqlen.seqlen_k - cluster_tile_n * n_block_for_tile:
                                for j in cutlass.range_constexpr(tdVgdV.shape[2]):
                                    if not const_expr(self.check_hdim_v_oob) or tdVcdV[0, i, j][1] < self.head_dim_v:
                                        cute.copy(
                                            gmem_tiled_copy_zero_dV,
                                            zero,
                                            tdVgdV[None, i, j],
                                        )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def _dq_semaphore_lock_value(
        self,
        iter_idx: Int32,
        curr_q_cnt: Int32,
        curr_full_cnt: Int32,
        curr_dq_write_order: cute.Tensor,
        curr_dq_write_order_full: cute.Tensor,
        full_first: cutlass.Constexpr[bool] = False,
    ) -> Int32:
        sparse_iter = iter_idx // self.subtile_factor
        lock_value = Int32(0)
        if const_expr(full_first):
            if sparse_iter < curr_full_cnt:
                lock_value = curr_dq_write_order_full[sparse_iter]
            else:
                lock_value = curr_dq_write_order[sparse_iter - curr_full_cnt]
        else:
            if sparse_iter < curr_q_cnt:
                lock_value = curr_dq_write_order[sparse_iter]
            else:
                lock_value = curr_dq_write_order_full[sparse_iter - curr_q_cnt]
        return lock_value

    @cute.jit
    def dQacc_reduce(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        thr_mma_dQ: cute.ThrMma,
        tdQtdQ: cute.Tensor,
        pipeline_dQ: PipelineAsync,
        dQaccum_empty_mbar_ptr: Optional[cute.Pointer],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        mdQ_semaphore: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        num_reduce_threads = cute.arch.WARP_SIZE * len(self.reduce_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_reduce_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx() % len(self.reduce_warp_ids))
        is_tma_warp = warp_idx == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        # TMEM -> RMEM
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol_t2r)), Float32)
        thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ).get_slice(tidx)
        tdQtdQ_t2r = thr_copy_t2r.partition_S(tdQtdQ)
        tdQcdQ = thr_mma_dQ.partition_C(cute.make_identity_tensor(self.mma_tiler_dsk[:2]))
        tdQrdQ_t2r_shape = thr_copy_t2r.partition_D(tdQcdQ).shape
        # For 2-CTA: reduce_stage = dQaccum_reduce_stage_t2r / cta_group_size
        expected_reduce_stages_t2r = self.dQaccum_reduce_stage_t2r // self.cta_group_size
        assert cute.size(tdQrdQ_t2r_shape, mode=[1]) == expected_reduce_stages_t2r, "dQaccum t2r reduce stage mismatch"
        expected_reduce_stages = self.dQaccum_reduce_stage // self.cta_group_size
        # 2-CTA: CTA 0 -> (M/2, D) (stage 0, 1) & CTA 1 -> (M/2, D) (stage 2, 3)
        stage_offset = expected_reduce_stages * cta_rank_in_cluster if const_expr(self.use_2cta_instrs) else 0

        thr_copy_dQaccum_r2s = copy_utils.tiled_copy_1d(self.dqaccum_dtype, num_reduce_threads, num_copy_elems=128 // self.dqaccum_dtype.width).get_slice(tidx)
        tdQsdQ = thr_copy_dQaccum_r2s.partition_D(sdQaccum)

        read_flag = const_expr(not self.deterministic)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        dQ_consumer_state = pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, 1)
        dQ_tma_store_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.sdQaccum_stage)
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            n_block_cta_group = n_block // self.cta_group_size  # for 2cta
            n_block_sparse = n_block_cta_group if const_expr(self.use_2cta_instrs) else n_block
            seqlen = SeqlenInfoCls(batch_idx)
            n_blocks_per_sample = cute.ceil_div(seqlen.seqlen_k, self.tile_n * self.cta_group_size)
            m_block_max = block_info.get_m_block_max(seqlen)
            if const_expr(not seqlen.has_cu_seqlens_q):
                mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
            else:
                mdQaccum_cur = cute.domain_offset((seqlen.padded_offset_q * self.tile_hdim,), mdQaccum[None, head_idx])
            gdQaccum_ = cute.local_tile(mdQaccum_cur, (self.tile_m * self.tile_hdim,), (None,))
            # (M * K / STAGE, STAGE, _)
            gdQaccum = cute.flat_divide(gdQaccum_, (self.tile_m * self.tile_hdim // self.dQaccum_reduce_stage,))

            if const_expr(self.deterministic):
                assert mdQ_semaphore is not None
                mdQ_semaphore_cur = mdQ_semaphore[None, None, head_idx, batch_idx]

            curr_dq_write_order = None
            curr_dq_write_order_full = None
            assert blocksparse_tensors is not None
            (
                curr_q_cnt,
                curr_q_idx,
                curr_full_cnt,
                curr_full_idx,
                _,
                _,
                loop_count,
            ) = get_block_sparse_iteration_info_bwd(
                blocksparse_tensors,
                batch_idx,
                head_idx,
                n_block_sparse,
                subtile_factor=self.subtile_factor,
                n_blocks_per_sample=n_blocks_per_sample,
            )
            process_tile = loop_count > Int32(0)
            if const_expr(self.deterministic):
                assert blocksparse_tensors is not None
                curr_dq_write_order, curr_dq_write_order_full = get_curr_dq_write_order_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block_sparse,
                    n_blocks_per_sample=n_blocks_per_sample,
                )

            # dQacc_reduce mainloop
            # Match compute-loop sparse order (q-first normally, full-first for 2CTA).
            full_first_sparse = const_expr(self.use_2cta_instrs)
            for block_group in cutlass.range_constexpr(2):
                is_full_group = const_expr((full_first_sparse and block_group == 0) or (not full_first_sparse and block_group == 1))
                iter_offset = Int32(0)
                if const_expr(is_full_group):
                    group_loop_count = curr_full_cnt * self.subtile_factor
                    if const_expr(not full_first_sparse):
                        iter_offset = curr_q_cnt * self.subtile_factor
                else:
                    group_loop_count = curr_q_cnt * self.subtile_factor
                    if const_expr(full_first_sparse):
                        iter_offset = curr_full_cnt * self.subtile_factor

                for group_iter_idx in cutlass.range(group_loop_count, unroll=1):
                    iter_idx = group_iter_idx + iter_offset
                    sparse_iter_idx = group_iter_idx // self.subtile_factor
                    subtile_offset = group_iter_idx % self.subtile_factor
                    if const_expr(is_full_group):
                        m_block = curr_full_idx[sparse_iter_idx] * self.subtile_factor + subtile_offset
                    else:
                        m_block = curr_q_idx[sparse_iter_idx] * self.subtile_factor + subtile_offset
                    m_block_oob_upper = m_block >= m_block_max
                    pipeline_dQ.consumer_wait(dQ_consumer_state)
                    # TMEM -> RMEM
                    tdQrdQ_t2r = cute.make_rmem_tensor(tdQrdQ_t2r_shape, Float32)
                    cute.copy(thr_copy_t2r, tdQtdQ_t2r, tdQrdQ_t2r)
                    cute.arch.fence_view_async_tmem_load()
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        pipeline_dQ.consumer_release(dQ_consumer_state)
                    dQ_consumer_state.advance()

                    if m_block_max > 0:
                        m_block = cutlass.min(m_block, m_block_max - 1)
                    gdQaccum_cur = gdQaccum[None, None, m_block]

                    tdQrdQ_shape = (
                        self.dQ_reduce_ncol,
                        self.tile_hdim // self.cta_group_size // self.dQ_reduce_ncol,
                    )
                    tdQrdQ = cute.make_tensor(tdQrdQ_t2r.iterator, tdQrdQ_shape)

                    for stage in cutlass.range_constexpr(cute.size(tdQrdQ, mode=[1])):
                        smem_idx = dQ_tma_store_producer_state.index
                        tdQsdQ_r2s = tdQsdQ[None, None, smem_idx]
                        tdQrdQ_r2s = cute.make_tensor(tdQrdQ[None, stage].iterator, tdQsdQ_r2s.shape)
                        cute.copy(thr_copy_dQaccum_r2s, tdQrdQ_r2s, tdQsdQ_r2s)
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        cute.arch.fence_view_async_shared()
                        # semaphore acquire
                        if const_expr(self.deterministic and stage == 0):
                            if not m_block_oob_upper:
                                lock_value = self._dq_semaphore_lock_value(
                                    iter_idx,
                                    curr_q_cnt,
                                    curr_full_cnt,
                                    curr_dq_write_order,
                                    curr_dq_write_order_full,
                                    full_first=full_first_sparse,
                                )
                                barrier.wait_eq(
                                    mdQ_semaphore_cur[(m_block, None)].iterator,
                                    tidx,
                                    cta_rank_in_cluster,
                                    lock_value,
                                )
                        self.reduce_sync_barrier.arrive_and_wait()
                        # Copy from shared memory to global memory
                        if is_tma_warp and not m_block_oob_upper:
                            with cute.arch.elect_one():
                                copy_utils.cpasync_reduce_bulk_add_f32(
                                    sdQaccum[None, smem_idx].iterator,
                                    gdQaccum_cur[None, stage + stage_offset].iterator,
                                    self.tma_copy_bytes["dQ"] // 1,
                                )
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(self.sdQaccum_stage - 1, read=read_flag)
                        elif is_tma_warp:
                            # Drain pending TMA stores so SMEM buffers are safe to reuse
                            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                        self.reduce_sync_barrier.arrive_and_wait()
                        dQ_tma_store_producer_state.advance()

                    if const_expr(self.tile_hdim == 192):
                        if const_expr(self.sdQaccum_stage > 1):
                            if is_tma_warp:
                                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                            self.reduce_sync_barrier.arrive_and_wait()
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(dQaccum_empty_mbar_ptr)

                    # semaphore release
                    # NOTE: arrive_inc calls red_release which issues membar
                    if const_expr(self.deterministic):
                        if const_expr(self.sdQaccum_stage > 1 and not self.tile_hdim == 192):
                            if is_tma_warp and not m_block_oob_upper:
                                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                            self.reduce_sync_barrier.arrive_and_wait()
                        if not m_block_oob_upper:
                            barrier.arrive_inc(
                                mdQ_semaphore_cur[m_block, None].iterator,
                                tidx,
                                cta_rank_in_cluster,
                            )

            if process_tile:
                if is_tma_warp:
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                self.reduce_sync_barrier.arrive_and_wait()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if const_expr(not self.deterministic):
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def epilogue_dKV(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        seqlen,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        softmax_scale: Float32,
    ):
        wg_idx = (cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))) // 128
        num_wg = cute.arch.WARP_SIZE * len(self.compute_warp_ids) // 128

        assert self.qhead_per_kvhead == 1, "This epilogue path is only for MHA"
        mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3)[None, None, head_idx]
        mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3)[None, None, head_idx]

        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), Float32)
        # dV
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tiled_tmem_ld_dV = tcgen05.make_tmem_copy(tmem_load_atom, tdVtdV)
        thr_tmem_ld_dV = tiled_tmem_ld_dV.get_slice(tidx)

        tdVtdV_t2r_p = thr_tmem_ld_dV.partition_S(tdVtdV)
        tdVtdV_t2r = self.split_wg(tdVtdV_t2r_p, wg_idx, num_wg)

        cdV = cute.make_identity_tensor((self.mma_tiler_pdo[0], self.mma_tiler_pdo[1]))
        tdVcdV = thr_mma_dV.partition_C(cdV)
        tdVcdV_tensor = cute.make_tensor(tdVcdV.iterator, tdVcdV.layout)

        tdVcdV_t2r_p = thr_tmem_ld_dV.partition_D(tdVcdV_tensor)
        tdVcdV_t2r = self.split_wg(tdVcdV_t2r_p, wg_idx, num_wg)
        tdVrdV_t2r = cute.make_rmem_tensor(tdVcdV_t2r.shape, Float32)

        cute.copy(thr_tmem_ld_dV, tdVtdV_t2r, tdVrdV_t2r)
        cute.arch.fence_view_async_tmem_load()

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dv_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tiled_gmem_store_dV = cute.make_tiled_copy(
            atom_universal_copy,
            layout_tv=tiled_tmem_ld_dV.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_ld_dV.tiler_mn,
        )

        tdVrdV_r2s = cute.make_rmem_tensor(tdVrdV_t2r.shape, self.dv_dtype)
        for i in cutlass.range_constexpr(cute.size(tdVrdV_t2r, mode=[1])):
            dV_vec = tdVrdV_t2r[(None, i, 0, 0)].load()
            tdVrdV_r2s[(None, i, 0, 0)].store(dV_vec.to(self.dv_dtype))

        gdV = cute.local_tile(mdV_cur, (self.mma_tiler_pdo[0], self.tile_hdimv), (None, 0))
        gdV_tile = gdV[None, None, n_block // self.cta_group_size]

        tdVgdV = thr_mma_dV.partition_C(gdV_tile)
        tdVgdV_r2g_p = thr_tmem_ld_dV.partition_D(tdVgdV)
        tdVgdV_r2g = self.split_wg(tdVgdV_r2g_p, wg_idx, num_wg)
        if tidx < seqlen.seqlen_k - self.tile_n * n_block:
            cute.copy(
                tiled_gmem_store_dV,
                tdVrdV_r2s,
                tdVgdV_r2g,
            )

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()

        # dK
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tiled_tmem_ld_dK = tcgen05.make_tmem_copy(tmem_load_atom, tdKtdK)
        thr_tmem_ld_dK = tiled_tmem_ld_dK.get_slice(tidx)

        tdKtdK_t2r_p = thr_tmem_ld_dK.partition_S(tdKtdK)
        tdKtdK_t2r = self.split_wg(tdKtdK_t2r_p, wg_idx, num_wg)

        cdK = cute.make_identity_tensor((self.mma_tiler_dsq[0], self.mma_tiler_dsq[1]))
        tdKcdK = thr_mma_dK.partition_C(cdK)
        tdKcdK_tensor = cute.make_tensor(tdKcdK.iterator, tdKcdK.layout)

        tdKcdK_t2r_p = thr_tmem_ld_dK.partition_D(tdKcdK_tensor)
        tdKcdK_t2r = self.split_wg(tdKcdK_t2r_p, wg_idx, num_wg)
        tdKrdK_t2r = cute.make_rmem_tensor(tdKcdK_t2r.shape, Float32)

        cute.copy(tiled_tmem_ld_dK, tdKtdK_t2r, tdKrdK_t2r)
        cute.arch.fence_view_async_tmem_load()

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dk_dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        tiled_gmem_store_dK = cute.make_tiled_copy(
            atom_universal_copy,
            layout_tv=tiled_tmem_ld_dK.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_ld_dK.tiler_mn,
        )

        tdKrdK_r2s = cute.make_rmem_tensor(tdKrdK_t2r.shape, self.dk_dtype)

        for i in cutlass.range_constexpr(cute.size(tdKrdK_t2r, mode=[1])):
            dK_vec = tdKrdK_t2r[(None, i, 0, 0)].load() * softmax_scale
            tdKrdK_r2s[(None, i, 0, 0)].store(dK_vec.to(self.dk_dtype))

        gdK = cute.local_tile(mdK_cur, (self.mma_tiler_dsq[0], self.tile_hdim), (None, 0))
        gdK_tile = gdK[None, None, n_block // self.cta_group_size]

        tdKgdK = thr_mma_dK.partition_C(gdK_tile)
        tdKgdK_r2g_p = thr_tmem_ld_dK.partition_D(tdKgdK)
        tdKgdK_r2g = self.split_wg(tdKgdK_r2g_p, wg_idx, num_wg)
        if tidx < seqlen.seqlen_k - self.tile_n * n_block:
            cute.copy(
                tiled_gmem_store_dK,
                tdKrdK_r2s,
                tdKgdK_r2g,
            )

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        return consumer_state_dKV

    @cute.jit
    def epilogue_dK_or_dV_tma(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        seqlen,
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
        mdKV_semaphore: Optional[cute.Tensor],
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

        cta_group_tile_n = const_expr(self.tile_n * self.cta_group_size)

        if const_expr(not self.dKV_postprocess):
            sdKV = sdKV[None, None, wg_idx]  # (tile_n, 64) for bf16
        else:
            sdKV = sdKV[None, wg_idx]  # (tile_n * 32) for fp32

        # (8, tile_n / 128, 64 / 8) = (8, 1, 8) or (4, tile_n * 32 / (128 * 4)) = (4, 8)
        tdKVsdKV_r2s = thr_copy_r2s_dKV.partition_D(sdKV)

        head_idx_kv = head_idx // self.qhead_per_kvhead
        if const_expr(not self.dKV_postprocess):
            assert not seqlen.has_cu_seqlens_k, "varlen uses non tma store path"
            mdKV_cur = mdKV[None, None, head_idx_kv, batch_idx]  # (seqlen, hdim)
            gdKV_p = cute.local_tile(mdKV_cur, (self.tile_n, tile_hdim), (n_block, 0))  # (tile_n, hdim) - per CTA
            gdKV = self.split_wg(gdKV_p, wg_idx, num_wg)  # (tile_n, hdim / 2)
            gdKV_epi = cute.local_tile(gdKV, epi_tile, (0, None))  # (tile_n, 64, epi_stage = (hdim / 2) / 64)
        else:
            if const_expr(not seqlen.has_cu_seqlens_k):
                mdKV_cur = mdKV[None, head_idx_kv, batch_idx]  # (seqlen * hdim)
            else:
                mdKV_cur = cute.domain_offset((seqlen.padded_offset_k * tile_hdim,), mdKV[None, head_idx_kv])
            gdKV_p = cute.local_tile(mdKV_cur, (self.tile_n * tile_hdim,), (n_block,))  # (tile_n * hdim)
            gdKV = cute.logical_divide(gdKV_p, (self.tile_n * tile_hdim // num_wg,))[((None, wg_idx),)]  # (tile_n * hdim / 2)
            gdKV_epi = cute.flat_divide(gdKV, (flat_epi_tile,))  # (tile_n * hdim / 2 / epi_stage, epi_stage)

        deterministic_KV = self.deterministic and self.qhead_per_kvhead > 1
        if const_expr(deterministic_KV):
            assert mdKV_semaphore is not None
            mdKV_semaphore_cur = mdKV_semaphore[n_block, None, head_idx_kv, batch_idx]

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

        reduce_ncol = self.dK_reduce_ncol if const_expr(K_or_V == "K") else self.dV_reduce_ncol
        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(reduce_ncol)), Float32)

        read_flag = const_expr(not deterministic_KV)

        pipeline_dKV.consumer_wait(consumer_state_dKV)

        # semaphore acquire
        if const_expr(deterministic_KV):
            barrier.wait_eq(mdKV_semaphore_cur.iterator, tidx, wg_idx, head_idx % self.qhead_per_kvhead)
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

        for epi_stage in cutlass.range_constexpr(num_epi_stages):
            # TMEM -> RMEM -- setup
            thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(tidx)
            tdKVtdKV_t2r_p = thr_copy_t2r.partition_S(tdKVtdKV)
            tdKVtdKV_t2r = self.split_wg(tdKVtdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVtdKV_t2r = tdKVtdKV_t2r[None, epi_stage]

            cdKV = cute.make_identity_tensor((cta_group_tile_n, tile_hdim))
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
                        reduce_copy_bytes = self.tma_copy_bytes["dKacc"] if const_expr(K_or_V == "K") else self.tma_copy_bytes["dVacc"]
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdKV.iterator,
                            gdKV_epi[None, epi_stage].iterator,
                            reduce_copy_bytes,
                        )
                if const_expr(epi_stage < num_epi_stages - 1):
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                cute.arch.barrier_arrive(barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE)

            # Barrier since all warps need to wait for SMEM to be freed
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE)

        # semaphore release
        # NOTE: arrive_inc calls red_release which issues membar
        if const_expr(deterministic_KV):
            if leader_warp:
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)
            barrier.arrive_inc(mdKV_semaphore_cur.iterator, tidx, wg_idx)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV
