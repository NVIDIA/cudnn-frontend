# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 blk64 block-sparse backward kernels."""

import enum
import math
import operator
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple, Type, TypeAlias

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass.cute.nvgpu import LoadCacheMode, OperandMajorMode, cpasync, warp, warpgroup
from cutlass.cutlass_dsl import Arch, BaseDSL
from cutlass.utils import (
    LayoutEnum,
)

from cudnn.block_sparse_attention.csrc.utils import copy_utils, kernel_utils as utils, layout_utils, pipeline, sm90_utils
from cudnn.block_sparse_attention.csrc.utils.cute_dsl_utils import ParamsBase, assume_tensor_aligned
from cudnn.block_sparse_attention.csrc.utils.sm90_utils import gemm_w_idx, gemm_zero_init
from cudnn.block_sparse_attention.csrc.utils.tile_scheduler import SingleTileScheduler, TileSchedulerArguments, WorkTileInfo

SM90_BWD_SPARSE_BLOCK_SIZE = 64
SM90_BWD_HEAD_DIM = 128
SM90_BWD_BUCKETED_K2Q_SIZE_BLOCKS = 384
SM90_BWD_KV_IN_REGS = True
SM90_BWD_SDP_SWAP_AB = True
SM90_BWD_DQACCUM_STAGE = 1
SM90_BWD_PDS_STAGE = 1
SM90_BWD_QDO_STAGE = 3


def sm90_bwd_default_bucketed_k2q_size_blocks(num_q_blocks: int) -> int:
    return SM90_BWD_BUCKETED_K2Q_SIZE_BLOCKS


def sm90_bwd_auto_bucketed_k2q_size_blocks(num_q_blocks: int) -> int:
    return sm90_bwd_default_bucketed_k2q_size_blocks(num_q_blocks)


# =============================================================================
# Sequence metadata
# =============================================================================
# Keep these lightweight types before the public class because several kernel
# methods use them in evaluated type annotations.


@dataclass(frozen=True)
class SeqlenInfo:
    seqlen: Int32

    @staticmethod
    def create(
        batch_idx: Int32,
        seqlen_static: Int32,
        tile: cutlass.Constexpr[int] = 128,
    ):
        return SeqlenInfo(seqlen_static)

    def offset_batch(
        self,
        mT: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
        multiple: int = 1,
    ) -> cute.Tensor:
        idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mT) - 1 - dim)
        return mT[idx]


@dataclass(frozen=True)
class SeqlenInfoQK:
    seqlen_q: Int32
    seqlen_k: Int32

    @staticmethod
    def create(
        batch_idx: Int32,
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
        tile_m: cutlass.Constexpr[Int32] = 128,
        tile_n: cutlass.Constexpr[Int32] = 128,
    ):
        return SeqlenInfoQK(seqlen_q_static, seqlen_k_static)

    def offset_batch_Q(
        self,
        mQ: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
        ragged: cutlass.Constexpr[bool] = False,
    ) -> cute.Tensor:
        idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mQ) - 1 - dim)
        return mQ[idx]

    def offset_batch_K(
        self,
        mK: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
        ragged: cutlass.Constexpr[bool] = False,
        multiple: int = 1,
    ) -> cute.Tensor:
        idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mK) - 1 - dim)
        return mK[idx]


# =============================================================================
# Public kernel class
# =============================================================================
class BlockSparseAttnBackwardSm90Blk64:
    arch = 90
    tile_m = 64
    tile_n = 64

    # ---- Setup and capability checks ----
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
    ):
        assert head_dim == 128, "SM90 blk64 BSA bwd is currently specialized for D=128"
        assert head_dim_v in [None, 128], "SM90 blk64 BSA bwd is currently specialized for Dv=128"
        self.dtype = dtype
        self.use_pdl = BaseDSL._get_dsl().get_arch_enum() >= Arch.sm_90a
        self.preprocess_num_threads = 256
        self.postprocess_num_threads = 256
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        hdim_preprocess_multiple_of = 32
        self.head_dim_padded = int(math.ceil(head_dim / hdim_preprocess_multiple_of) * hdim_preprocess_multiple_of)
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_preprocess_multiple_of) * hdim_preprocess_multiple_of)
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.tile_m = 64
        self.tile_n = 64
        self.num_threads = 384
        self.Q_stage = SM90_BWD_QDO_STAGE
        self.dO_stage = SM90_BWD_QDO_STAGE
        self.PdS_stage = SM90_BWD_PDS_STAGE
        assert self.dO_stage in [1, self.Q_stage]
        assert self.PdS_stage == 1 or self.PdS_stage == self.Q_stage
        self.SdP_swapAB = SM90_BWD_SDP_SWAP_AB
        self.AtomLayoutNdKV = 2
        self.AtomLayoutMdQ = 1
        self.num_wg_mma = (self.num_threads // 128) - 1
        self.V_in_regs = SM90_BWD_KV_IN_REGS
        # These are tuned for speed
        # Do we keep the LSE and dPsum in each thread, or split them across 8 threads that share
        # them and then shuffle to get the value whenever we need? This can reduce register
        # pressure when SdP_swapAB, where each thread needs to keep statistics for (kBlockM / 4)
        # rows. If !SdP_swapAB, each thread only needs to keep statistics for 2 rows.
        self.shuffle_LSE = self.SdP_swapAB and self.tile_hdim <= 64
        self.shuffle_dPsum = self.SdP_swapAB and self.tile_hdim <= 64

        self.buffer_align_bytes = 1024
        self.num_wg_dQ = 1
        self.dQaccum_stage = SM90_BWD_DQACCUM_STAGE
        self.num_dQ_store_warps = 1
        assert self.num_wg_mma == 2, "WG-specialized pipeline assumes two MMA WGs"
        assert self.num_wg_dQ == 1, "WG-specialized pipeline has one dQ producer WG"
        assert self.SdP_swapAB, "Split dKV-RS requires SdP_swapAB"
        assert self.dQaccum_stage in [1, 2, 3], "WG-specialized dQaccum stage must be 1, 2, or 3"

    def _check_type(
        self,
        mQ_type: Type[cutlass.Numeric],
        mK_type: Type[cutlass.Numeric],
        mV_type: Type[cutlass.Numeric],
        mdO_type: Type[cutlass.Numeric],
        mLSE_type: Type[cutlass.Numeric],
        mdPsum_type: Type[cutlass.Numeric],
        mdQaccum_type: Type[cutlass.Numeric],
        mdK_type: Type[cutlass.Numeric],
        mdV_type: Type[cutlass.Numeric],
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(not (mQ_type == mK_type == mV_type == mdO_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mLSE_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mdPsum_type not in [Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if const_expr(mdQaccum_type not in [Float32]):
            raise TypeError("dQaccum tensor must be Float32")
        if const_expr(not (mdK_type == mdV_type == Float32)):
            raise TypeError("mdKaccum and mdVaccum tensors must have the data type Float32")
        assert mQ_type == self.dtype

    def _setup_attributes(self):
        # We need to accommodate both Q and Q^T (and dO and dO^T) in shared memory.
        # Q & dO are used in the SdP Mma and Q^T and dO^T are used in the dKV Mma.
        # The M dimension (tile_m) doesn't matter for the layout, only the K dimension
        wg_d_dKV = self.num_wg_mma // self.AtomLayoutNdKV
        self.sQ_layout, self.sdO_layout = [
            # Need to set major_mode_size (mms) to accommodate Q and Q.T
            sm90_utils.make_smem_layout(self.dtype, LayoutEnum.ROW_MAJOR, shape, stage, mms)
            for shape, stage, mms in [
                ((self.tile_m, self.tile_hdim), self.Q_stage, self.tile_hdim // wg_d_dKV),
                ((self.tile_m, self.tile_hdimv), self.dO_stage, self.tile_hdim // wg_d_dKV),
            ]
        ]
        wg_d_dQ = self.num_wg_dQ // self.AtomLayoutMdQ
        # Accomodate both K and K.T
        self.sK_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.tile_hdim),
            stage=None,
            major_mode_size=self.tile_hdim // wg_d_dQ,
        )
        # There's only V, no V.T, so layout is normal
        self.sV_layout = sm90_utils.make_smem_layout(self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdimv), None)
        # Accomodate both S and S.T
        wg_n_SdP = 1
        wg_n_dKV = 1
        self.sPdS_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_n),
            stage=self.PdS_stage,
            major_mode_size=math.gcd(self.tile_n // wg_n_SdP, self.tile_n // wg_n_dKV),
        )
        self.sdQaccum_layout = (
            cute.make_layout((self.tile_m * self.tile_hdim // self.num_wg_dQ, self.num_wg_dQ))
            if const_expr(self.dQaccum_stage == 1)
            else cute.make_layout(
                (
                    self.tile_m * self.tile_hdim // self.num_wg_dQ,
                    self.num_wg_dQ,
                    self.dQaccum_stage,
                )
            )
        )
        # dQaccum R->S
        self.r2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
            # thr_layout
            cute.make_layout((self.num_threads_per_warp_group, self.num_wg_dQ)),
            cute.make_layout(128 // Float32.width),  # val_layout
        )

    def _get_tiled_mma(self):
        tiled_mma_SdP = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(self.tile_m, self.tile_n),
            a_source=warpgroup.OperandSource.RMEM if self.V_in_regs else warpgroup.OperandSource.SMEM,
        )
        tiled_mma_dK, tiled_mma_dV = [
            sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype,
                self.dtype,
                OperandMajorMode.K,
                OperandMajorMode.MN,
                Float32,
                atom_layout_mnk=(1, 1, 1),
                tiler_mn=(self.tile_n, hdim),
                a_source=warpgroup.OperandSource.RMEM,
            )
            for hdim in (self.tile_hdim, self.tile_hdimv)
        ]
        tiled_mma_dQ = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(self.tile_m, self.tile_hdim),
        )
        return tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _get_shared_storage_cls(self):
        cosize_sK = cute.cosize(self.sK_layout)
        cosize_sV = cute.cosize(self.sV_layout)
        cosize_sK = max(cosize_sK, self.tile_n * self.tile_hdim * Float32.width // self.dtype.width)
        cosize_sV = max(cosize_sV, self.tile_n * self.tile_hdimv * Float32.width // self.dtype.width)
        sQ_struct, sK_struct, sV_struct, sdO_struct, sdQaccum_struct = [
            cute.struct.Align[cute.struct.MemRange[t, cosize], self.buffer_align_bytes]
            for (layout, t, cosize) in [
                (self.sQ_layout, self.dtype, cute.cosize(self.sQ_layout)),
                (self.sK_layout, self.dtype, cosize_sK),
                (self.sV_layout, self.dtype, cosize_sV),
                (self.sdO_layout, self.dtype, cute.cosize(self.sdO_layout)),
                (self.sdQaccum_layout, Float32, cute.cosize(self.sdQaccum_layout)),
            ]
        ]

        cosize_sdS = cute.cosize(self.sPdS_layout)
        cosize_sP = cute.cosize(self.sPdS_layout)
        sLSE_struct = cute.struct.Align[cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64) * self.Q_stage], 128]
        sdPsum_struct = cute.struct.Align[cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64) * self.dO_stage], 128]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, self.Q_stage * 2]
            mbar_ptr_dO: cute.struct.MemRange[cutlass.Int64, self.dO_stage * 2]
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sQ: sQ_struct
            sV: sV_struct
            sK: sK_struct
            sdO: sdO_struct
            sP: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
            sdS: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sdS], 1024]
            sdQaccum: sdQaccum_struct

        return SharedStorageQKV

    # ---- Workspace helpers ----
    def get_workspace_tensor(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        workspace: cute.Tensor,
    ) -> Tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]:
        q, k, d, hb = problem_shape
        h, b = cute.size(hb[0]), cute.size(hb[1])
        d = cute.round_up(d, 32)
        q = cute.round_up(q, self.tile_m)
        k = cute.round_up(k, self.tile_n)
        q_i64 = cutlass.Int64(q)
        k_i64 = cutlass.Int64(k)
        d_i64 = cutlass.Int64(d)
        h_i64 = cutlass.Int64(h)
        b_i64 = cutlass.Int64(b)

        dPsum_elems = cute.assume(b_i64 * h_i64 * q_i64, divby=4)
        lse_log2_elems = cute.assume(b_i64 * h_i64 * q_i64, divby=4)
        dQaccum_elems = cute.assume(b_i64 * h_i64 * q_i64 * d_i64, divby=4)
        dKaccum_elems = cute.assume(b_i64 * h_i64 * k_i64 * d_i64, divby=4)

        dPsum_iter = workspace.iterator
        lse_log2_iter = dPsum_iter + dPsum_elems
        dQaccum_iter = lse_log2_iter + lse_log2_elems
        dKaccum_iter = dQaccum_iter + dQaccum_elems
        dVaccum_iter = dKaccum_iter + dKaccum_elems

        dPsum = cute.make_tensor(
            dPsum_iter,
            cute.make_layout((b, h, q), stride=(h_i64 * q_i64, q_i64, 1)),
        )
        lse_log2 = cute.make_tensor(
            lse_log2_iter,
            cute.make_layout((b, h, q), stride=(h_i64 * q_i64, q_i64, 1)),
        )
        dQaccum = cute.make_tensor(
            dQaccum_iter,
            cute.make_layout(
                (b, h, q * d),
                stride=(h_i64 * q_i64 * d_i64, q_i64 * d_i64, 1),
            ),
        )
        dKaccum = cute.make_tensor(
            dKaccum_iter,
            cute.make_layout(
                (b, h, k * d),
                stride=(h_i64 * k_i64 * d_i64, k_i64 * d_i64, 1),
            ),
        )
        dVaccum = cute.make_tensor(
            dVaccum_iter,
            cute.make_layout(
                (b, h, k * d),
                stride=(h_i64 * k_i64 * d_i64, k_i64 * d_i64, 1),
            ),
        )
        return dPsum, lse_log2, dQaccum, dKaccum, dVaccum

    # ---- Preprocess / postprocess kernels ----
    def _setup_preprocess_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        # We want kBlockKGmem to be a power of 2 so that when we do the summing,
        # it's just between threads in the same warp
        gmem_k_block_size = (
            128 if self.head_dim_v_padded % 128 == 0 else (64 if self.head_dim_v_padded % 64 == 0 else (32 if self.head_dim_v_padded % 32 == 0 else 16))
        )
        num_copy_elems = 128 // self.dtype.width
        threads_per_row = gmem_k_block_size // num_copy_elems
        self.gmem_tiled_copy_O = copy_utils.tiled_copy_2d(self.dtype, threads_per_row, self.preprocess_num_threads, num_copy_elems)
        universal_copy_bits = 128
        num_copy_elems_dQaccum = universal_copy_bits // Float32.width
        assert (self.tile_m * self.head_dim_padded // num_copy_elems_dQaccum) % self.preprocess_num_threads == 0
        self.gmem_tiled_copy_dQaccum = copy_utils.tiled_copy_1d(Float32, self.preprocess_num_threads, num_copy_elems_dQaccum)

    @cute.jit
    def _preprocess_call(
        self,
        mO: cute.Tensor,  # (batch, seqlen, nheads, head_dim_v) or (total_q, nheads, head_dim_v)
        mdO: cute.Tensor,  # same shape as mO
        mPdPsum: cute.Tensor,  # (batch, nheads, seqlen_padded) or (nheads, total_q_padded)
        mLSE: Optional[cute.Tensor],  # (batch, nheads, seqlen) or (nheads, total_q)
        mLSElog2: Optional[cute.Tensor],  # same shape as mPdPsum
        # (batch, nheads, seqlen_padded * head_dim_v) or (nheads, total_q_padded * head_dim_v)
        mdQaccum: Optional[cute.Tensor],
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(not (mO.element_type == mdO.element_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mO.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mPdPsum.element_type not in [Float32]):
            raise TypeError("PdPsum tensor must be Float32")
        if const_expr(mdQaccum is not None):
            if const_expr(mdQaccum.element_type not in [Float32]):
                raise TypeError("dQaccum tensor must be Float32")
        if const_expr(mLSE is not None):
            assert mLSElog2 is not None, "If mLSE is provided, mLSElog2 must also be provided"
            if const_expr(mLSE.element_type not in [Float32]):
                raise TypeError("LSE tensor must be Float32")
            if const_expr(mLSElog2.element_type not in [Float32]):
                raise TypeError("LSElog2 tensor must be Float32")
        self._setup_preprocess_attributes()

        # (batch, nheads, seqlen) -> (seqlen, nheads, batch)
        transpose = [2, 1, 0]
        mPdPsum = layout_utils.select(mPdPsum, transpose)
        if const_expr(mLSE is not None):
            mLSE = layout_utils.select(mLSE, transpose)
            mLSElog2 = layout_utils.select(mLSElog2, transpose)
        if const_expr(mdQaccum is not None):
            mdQaccum = layout_utils.select(mdQaccum, transpose)

        TileScheduler = SingleTileScheduler
        num_head = mO.shape[2]
        num_batch = mO.shape[0]

        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(mO.shape[1], self.tile_m),
            num_head=num_head,
            num_batch=num_batch,
            num_splits=1,
            seqlen_k=0,
            headdim=0,
            headdim_v=mO.shape[2],
            total_q=mO.shape[0],
            tile_shape_mn=(self.tile_m, 1),
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        self.sum_OdO(
            mO,
            mdO,
            mPdPsum,
            mLSE,
            mLSElog2,
            mdQaccum,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_dQaccum,
            tile_sched_params,
            TileScheduler,
        ).launch(
            grid=grid_dim,
            block=[self.preprocess_num_threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def sum_OdO(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mPdPsum: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSElog2: Optional[cute.Tensor],
        mdQaccum: Optional[cute.Tensor],
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx

        # This kernel is launched with use_pdl=True, so the GPU may start executing it in
        # "prologue" mode while the previous stream kernel is still running. We must wait
        # before touching any upstream GMEM outputs (mO, mdO, mLSE); otherwise we risk
        # reading a partially-written dout, which silently corrupts dpsum = sum(O * dO) and
        # propagates to dQ/dK via dS = P * (dP - dpsum).
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        if work_tile.is_valid_tile:
            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # ///////////////////////////////////////////////////////////////////////////////
            seqlen = SeqlenInfo.create(batch_idx, mO.shape[1], tile=self.tile_m)
            mO_cur = seqlen.offset_batch(mO, batch_idx, dim=0)[None, head_idx, None]
            mdO_cur = seqlen.offset_batch(mdO, batch_idx, dim=0)[None, head_idx, None]
            mPdPsum_cur = seqlen.offset_batch(mPdPsum, batch_idx, dim=2, padded=True)[None, head_idx]
            seqlen_q = seqlen.seqlen
            seqlen_q_rounded = cute.round_up(seqlen_q, self.tile_m)
            seqlen_limit = seqlen_q - m_block * self.tile_m

            lse = None
            if const_expr(mLSE is not None):
                mLSE_cur = seqlen.offset_batch(mLSE, batch_idx, dim=2)[None, head_idx]
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                lse = Float32.inf
                if tidx < seqlen_limit:
                    lse = gLSE[tidx]

            blk_shape = (self.tile_m, self.head_dim_v_padded)
            gO = cute.local_tile(mO_cur, blk_shape, (m_block, 0))
            gdO = cute.local_tile(mdO_cur, blk_shape, (m_block, 0))
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            # (CPY_Atom, CPY_M, CPY_K)
            tOgO = gmem_thr_copy_O.partition_S(gO)
            tOgdO = gmem_thr_copy_O.partition_S(gdO)
            cO = cute.make_identity_tensor(blk_shape)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_thr_copy_O.get_slice(0).partition_S(cO)
            tOpO = None
            # Each copy will use the same predicate
            copy = partial(copy_utils.copy, pred=tOpO)

            tOrO = cute.make_rmem_tensor_like(tOgO)
            tOrdO = cute.make_rmem_tensor_like(tOgdO)
            assert tOgO.shape == tOgdO.shape
            for m in cutlass.range(cute.size(tOrO.shape[1]), unroll_full=True):
                # Instead of using tOcO, we using t0OcO and subtract the offset from the limit.
                # This is bc the entries of t0OcO are known at compile time.
                if t0OcO[0, m, 0][0] < seqlen_limit - tOcO[0][0]:
                    copy(tOgO[None, m, None], tOrO[None, m, None])
                    copy(tOgdO[None, m, None], tOrdO[None, m, None])
            # O and dO loads are done; signal that the next kernel can start.
            # Correctness is ensured by griddepcontrol_wait() in bwd_sm90 before it reads our outputs.
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()
            # Sum across the "k" dimension
            pdpsum = (tOrO.load().to(Float32) * tOrdO.load().to(Float32)).reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1))
            threads_per_row = gmem_tiled_copy_O.layout_src_tv_tiled[0].shape[0]
            assert cute.arch.WARP_SIZE % threads_per_row == 0
            pdpsum = utils.warp_reduce(pdpsum, operator.add, width=threads_per_row)
            PdP_sum = cute.make_rmem_tensor(cute.size(tOrO, mode=[1]), Float32)
            PdP_sum.store(pdpsum)

            # Write PdPsum from rmem -> gmem
            gPdPsum = cute.local_tile(mPdPsum_cur, (self.tile_m,), (m_block,))
            # Only the thread corresponding to column 0 writes out the PdPsum to gmem
            if tOcO[0, 0, 0][1] == 0:
                for m in cutlass.range(cute.size(PdP_sum), unroll_full=True):
                    row = tOcO[0, m, 0][0]
                    PdPsum_val = 0.0
                    if row < seqlen_limit:
                        PdPsum_val = PdP_sum[m]
                    gPdPsum[row] = PdPsum_val

            # Clear dQaccum
            if const_expr(mdQaccum is not None):
                mdQaccum_cur = seqlen.offset_batch(mdQaccum, batch_idx, dim=2, padded=True, multiple=self.head_dim_padded)[None, head_idx]
                blkdQaccum_shape = (self.tile_m * self.head_dim_padded,)
                gdQaccum = cute.local_tile(mdQaccum_cur, blkdQaccum_shape, (m_block,))
                gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx)
                tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)
                zero = cute.make_rmem_tensor_like(tdQgdQaccum)
                zero.fill(0.0)
                cute.copy(gmem_tiled_copy_dQaccum, zero, tdQgdQaccum)

            if const_expr(mLSE is not None):
                mLSElog2_cur = seqlen.offset_batch(mLSElog2, batch_idx, dim=2, padded=True)[None, head_idx]
                gLSElog2 = cute.local_tile(mLSElog2_cur, (self.tile_m,), (m_block,))
                LOG2_E = math.log2(math.e)
                if tidx < seqlen_q_rounded - m_block * self.tile_m:
                    gLSElog2[tidx] = lse * LOG2_E if lse != -Float32.inf else 0.0

    def _get_postprocess_tiled_mma(self):
        num_wg_mma = self.postprocess_num_threads // 128
        atom_layout_dQ = (self.AtomLayoutMdQ, num_wg_mma // self.AtomLayoutMdQ)
        tiler_mn_dQ = (self.tile_m // atom_layout_dQ[0], self.tile_hdim // atom_layout_dQ[1])
        tiled_mma = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            atom_layout_mnk=atom_layout_dQ + (1,),
            tiler_mn=tiler_mn_dQ,
        )
        assert self.postprocess_num_threads == tiled_mma.size
        return tiled_mma

    def _setup_postprocess_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        universal_copy_bits = 128
        async_copy_elems_accum = universal_copy_bits // Float32.width
        atom_async_copy_accum = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=LoadCacheMode.GLOBAL),
            Float32,
            num_bits_per_copy=universal_copy_bits,
        )
        # We don't do bound checking for the gmem -> smem load so we just assert here.
        assert (self.tile_m * self.tile_hdim // async_copy_elems_accum) % self.postprocess_num_threads == 0
        self.g2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            atom_async_copy_accum,
            cute.make_layout(self.postprocess_num_threads),
            cute.make_layout(async_copy_elems_accum),
        )
        num_threads_per_warp_group = 128
        num_wg_mma = self.postprocess_num_threads // 128
        self.s2r_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
            cute.make_layout((num_threads_per_warp_group, num_wg_mma)),  # thr_layout
            cute.make_layout(128 // Float32.width),  # val_layout
        )
        self.sdQaccum_layout = cute.make_layout((self.tile_m * self.tile_hdim // num_wg_mma, num_wg_mma))

        num_copy_elems = 128 // self.dtype.width
        threads_per_row = math.gcd(128, self.tile_hdim) // num_copy_elems
        self.gmem_tiled_copy_dQ = copy_utils.tiled_copy_2d(self.dtype, threads_per_row, self.postprocess_num_threads, num_copy_elems)
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: dQ
        # ///////////////////////////////////////////////////////////////////////////////
        # We can't just use kHeadDim here. E.g. if MMA shape is 64 x 96 but split across 2 WGs,
        # then setting kBlockKSmem to 32 will cause "Static shape_div failure".
        # We want to treat it as 64 x 48, so kBlockKSmem should be 16.
        wg_d_dQ = num_wg_mma // self.AtomLayoutMdQ
        self.sdQ_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_hdim),
            major_mode_size=self.tile_hdim // wg_d_dQ,
        )

    @cute.jit
    def _postprocess_call(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        scale: cutlass.Float32,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(mdQ.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mdQaccum is not None):
            if const_expr(mdQaccum.element_type not in [cutlass.Float32]):
                raise TypeError("dQaccum tensor must be Float32")

        mdQaccum, mdQ = [assume_tensor_aligned(t) for t in (mdQaccum, mdQ)]

        self.tiled_mma = self._get_postprocess_tiled_mma()
        self._setup_postprocess_attributes()

        smem_size = max(
            cute.size_in_bytes(cutlass.Float32, self.sdQaccum_layout),
            cute.size_in_bytes(self.dtype, self.sdQ_layout),
        )

        TileScheduler = SingleTileScheduler
        num_head = mdQ.shape[2]
        num_batch = mdQ.shape[0]
        num_block = cute.ceil_div(mdQ.shape[1], self.tile_m)

        tile_sched_args = TileSchedulerArguments(
            num_block=num_block,
            num_head=num_head,
            num_batch=num_batch,
            num_splits=1,
            seqlen_k=0,
            headdim=mdQ.shape[2],
            headdim_v=0,
            total_q=mdQ.shape[0],
            tile_shape_mn=(self.tile_m, 1),
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # grid_dim: (m_block, num_head, batch_size)
        self.convert(
            mdQaccum,
            mdQ,
            scale,
            self.tiled_mma,
            self.sdQaccum_layout,
            self.sdQ_layout,
            self.g2s_tiled_copy_dQaccum,
            self.s2r_tiled_copy_dQaccum,
            self.gmem_tiled_copy_dQ,
            tile_sched_params,
            TileScheduler,
        ).launch(
            grid=grid_dim,
            block=[self.postprocess_num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
        )

    @cute.kernel
    def convert(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        scale: cutlass.Float32,
        tiled_mma: cute.TiledMma,
        sdQaccum_layout: cute.Layout,
        sdQ_layout: cute.ComposedLayout,
        g2s_tiled_copy_dQaccum: cute.TiledCopy,
        s2r_tiled_copy_dQaccum: cute.TiledCopy,
        gmem_tiled_copy_dQ: cute.TiledCopy,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        sdQaccum = smem.allocate_tensor(cutlass.Float32, sdQaccum_layout, byte_alignment=1024)
        sdQaccum_flat = cute.make_tensor(sdQaccum.iterator, cute.make_layout(cute.size(sdQaccum)))
        sdQ = cute.make_tensor(cute.recast_ptr(sdQaccum.iterator, dtype=self.dtype), sdQ_layout)

        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()

        m_block, head_idx, batch_idx, _ = work_tile.tile_idx

        if work_tile.is_valid_tile:
            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # ///////////////////////////////////////////////////////////////////////////////

            seqlen = SeqlenInfoQK.create(
                batch_idx,
                mdQ.shape[1],
                0,
                tile_m=self.tile_m,
            )
            mdQ_cur = mdQ[batch_idx, None, head_idx, None]
            mdQaccum_cur = mdQaccum[batch_idx, head_idx, None]
            head_dim = mdQ.shape[3]

            gdQaccum = cute.local_tile(mdQaccum_cur, (self.tile_m * self.tile_hdim,), (m_block,))
            gdQ = cute.local_tile(mdQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))

            seqlen_q = seqlen.seqlen_q
            seqlen_q_rounded = cute.round_up(seqlen_q, self.tile_m)

            g2s_thr_copy_dQaccum = g2s_tiled_copy_dQaccum.get_slice(tidx)
            tdQgdQaccum = g2s_thr_copy_dQaccum.partition_S(gdQaccum)
            tdQsdQaccumg2s = g2s_thr_copy_dQaccum.partition_D(sdQaccum_flat)
            cute.copy(g2s_tiled_copy_dQaccum, tdQgdQaccum, tdQsdQaccumg2s)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            # Step 2: load dQ from smem to rmem
            s2r_thr_copy_dQaccum = s2r_tiled_copy_dQaccum.get_slice(tidx)
            tdQsdQaccum = s2r_thr_copy_dQaccum.partition_S(sdQaccum)
            tile_shape = (self.tile_m, self.tile_hdim)
            acc = None
            tiled_copy_t2r = None
            acc_shape = tiled_mma.partition_shape_C(tile_shape)
            acc = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
            assert cute.size(acc) == cute.size(tdQsdQaccum)
            tdQrdQaccum = cute.make_tensor(acc.iterator, cute.make_layout(tdQsdQaccum.shape))
            cute.autovec_copy(tdQsdQaccum, tdQrdQaccum)
            # Convert tdQrdQaccum from fp32 to fp16/bf16
            rdQ = cute.make_rmem_tensor_like(acc, self.dtype)
            rdQ.store((acc.load() * scale).to(self.dtype))

            # Step 3: Copy dQ from register to smem
            cute.arch.barrier()  # make sure all threads have finished loading dQaccum
            copy_atom_r2s_dQ = utils.get_smem_store_atom(self.arch, self.dtype, transpose=False)
            tiled_copy_r2s_dQ = cute.make_tiled_copy_C(copy_atom_r2s_dQ, tiled_mma)
            thr_copy_r2s_dQ = tiled_copy_r2s_dQ.get_slice(tidx)
            cdQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
            taccdQrdQ = thr_copy_r2s_dQ.retile(rdQ)
            taccdQsdQ = thr_copy_r2s_dQ.partition_D(sdQ)
            cute.copy(thr_copy_r2s_dQ, taccdQrdQ, taccdQsdQ)

            # Step 4: Copy dQ from smem to register to prepare for coalesced write to gmem
            cute.arch.barrier()  # make sure all smem stores are done
            gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_slice(tidx)
            tdQgdQ = gmem_thr_copy_dQ.partition_S(gdQ)
            tdQsdQ = gmem_thr_copy_dQ.partition_D(sdQ)
            tdQrdQ = cute.make_rmem_tensor_like(tdQsdQ, self.dtype)
            # The subsequent gmem predicate guards tail rows; sdQ is a full tile.
            cute.autovec_copy(tdQsdQ, tdQrdQ)

            # Step 5: Copy dQ from register to gmem
            tdQcdQ = gmem_thr_copy_dQ.partition_S(cdQ)
            tdQpdQ = utils.predicate_k(tdQcdQ, limit=head_dim)
            for rest_m in cutlass.range(cute.size(tdQrdQ.shape[1]), unroll_full=True):
                if tdQcdQ[0, rest_m, 0][0] < seqlen_q - m_block * self.tile_m:
                    cute.copy(
                        gmem_tiled_copy_dQ,
                        tdQrdQ[None, rest_m, None],
                        tdQgdQ[None, rest_m, None],
                        pred=tdQpdQ[None, rest_m, None],
                    )

    # ---- Launch orchestration ----
    @cute.jit
    def __call__(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        mdO: cute.Tensor,
        mO: cute.Tensor,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdQ: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        bucketed_k2q_offsets: cute.Tensor,
        bucketed_k2q_indices: cute.Tensor,
        mBlockSizes: Optional[cute.Tensor],
        workspace: cute.Tensor,
        softmax_scale: Float32,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # Public BSA tensors are BHSD. The localized SM90 kernels use the internal
        # BSHD contract, so we only create stride views here instead of materializing
        # temporary transposes in the Python interface.
        def _bhsd_to_bshd(t):
            return assume_tensor_aligned(cute.make_tensor(t.iterator, cute.select(t.layout, mode=[0, 2, 1, 3])))

        mdO_bshd = _bhsd_to_bshd(mdO)
        mO_bshd = _bhsd_to_bshd(mO)
        mQ_bshd = _bhsd_to_bshd(mQ)
        mK_bshd = _bhsd_to_bshd(mK)
        mV_bshd = _bhsd_to_bshd(mV)
        mdQ_bshd = _bhsd_to_bshd(mdQ)
        mdK_bshd = _bhsd_to_bshd(mdK)
        mdV_bshd = _bhsd_to_bshd(mdV)

        mdPsum, mLSElog2, mdQaccum, mdKaccum, mdVaccum = self.get_workspace_tensor(problem_shape, workspace)
        mdPsum, mLSElog2, mdQaccum, mdKaccum, mdVaccum = [assume_tensor_aligned(t) for t in (mdPsum, mLSElog2, mdQaccum, mdKaccum, mdVaccum)]

        self._preprocess_call(
            mO_bshd,
            mdO_bshd,
            mdPsum,
            mLSE,
            mLSElog2,
            mdQaccum,
            stream,
        )

        self._bwd_call(
            mQ_bshd,
            mK_bshd,
            mV_bshd,
            mdO_bshd,
            mLSElog2,
            mdPsum,
            mdQaccum,
            mdKaccum,
            mdVaccum,
            softmax_scale,
            mBlockSizes,
            bucketed_k2q_offsets,
            bucketed_k2q_indices,
            stream,
        )

        self._postprocess_call(mdQaccum, mdQ_bshd, softmax_scale, stream)
        self._postprocess_call(mdKaccum, mdK_bshd, softmax_scale, stream)
        self._postprocess_call(mdVaccum, mdV_bshd, Float32(1.0), stream)

    @cute.jit
    def _bwd_call(
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
        mBlockSizes: Optional[cute.Tensor] = None,
        bucketed_k2q_offsets: Optional[cute.Tensor] = None,
        bucketed_k2q_indices: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        self._check_type(*(t.element_type if t is not None else None for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)))

        mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)]

        # Non-varlen inputs are (b, s, n, h), varlen inputs are (s, n, h).
        # We convert both to a seqlen-major view with head-dim second.
        # Each tensor may have different rank when Q is padded (seqused_q) but K/V are unpadded (cu_seqlens_k).
        def _qkv_transpose(t):
            return layout_utils.select(t, [1, 3, 2, 0] if cute.rank(t.shape) == 4 else [0, 2, 1])

        mQ, mK, mV, mdO = [_qkv_transpose(t) for t in (mQ, mK, mV, mdO)]
        # Accum tensors are (b, n, s*h) for non-varlen and (n, s*h) for varlen.
        accum_transpose = [2, 1, 0] if cute.rank(mdK.shape) == 3 else [1, 0]
        mdK, mdV = [layout_utils.select(t, accum_transpose) for t in (mdK, mdV)]
        # Non-varlen stats are (b, n, s), varlen stats are (n, s).
        LSE_dPsum_dQaccum_transpose = [2, 1, 0] if cute.rank(mLSE.shape) == 3 else [1, 0]
        mLSE, mdPsum, mdQaccum = [layout_utils.select(t, LSE_dPsum_dQaccum_transpose) for t in (mLSE, mdPsum, mdQaccum)]
        assert bucketed_k2q_offsets is not None
        assert bucketed_k2q_indices is not None
        bucketed_k2q_offsets = cute.make_tensor(
            bucketed_k2q_offsets.iterator,
            cute.group_modes(
                cute.select(bucketed_k2q_offsets.layout, mode=[3, 2, 1, 0]),
                2,
                4,
            ),
        )
        bucketed_k2q_indices = cute.make_tensor(
            bucketed_k2q_indices.iterator,
            cute.group_modes(
                cute.select(bucketed_k2q_indices.layout, mode=[2, 1, 0]),
                1,
                3,
            ),
        )

        tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ = self._get_tiled_mma()

        self.num_mma_threads = self.num_threads - 128
        assert self.num_mma_threads + 128 == self.num_threads

        self.num_threads_per_warp_group = 128

        self.num_mma_regs_wg0 = 240
        self.num_mma_regs_wg1 = 240
        self.num_mma_regs = self.num_mma_regs_wg0  # for backward compat
        self.num_producer_regs = 24
        assert self.num_mma_regs_wg0 + self.num_mma_regs_wg1 + self.num_producer_regs <= 504

        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = self.tile_m * self.tile_hdim * Float32.width // 8 // self.num_wg_dQ
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.tile_hdim * Float32.width // 8
        self.tma_copy_bytes["dVacc"] = self.tile_n * self.tile_hdimv * Float32.width // 8

        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdim),
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
        )
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdimv),
        )
        tma_atom_dK = tma_atom_dV = None

        TileScheduler = SingleTileScheduler
        num_splits = cute.size(bucketed_k2q_offsets.shape[1])
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.tile_n),
            cute.size(mQ.shape[2]),
            cute.size(mK.shape[3]),
            num_splits,
            cute.size(mQ.shape[0]),  # pass seqlen_q or total_q for seqlen_k
            mQ.shape[1],  # headdim
            mV.shape[1],  # headdim_v
            total_q=cute.size(mK.shape[0]) * cute.size(mK.shape[3]),
            tile_shape_mn=(self.tile_n, self.tile_m),  # Swapping the role of Q & K
            element_size=self.dtype.width // 8,
            is_persistent=False,
            is_split_kv=True,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        self.bwd(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_dO,
            mdK,
            mdV,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dK,
            tma_atom_dV,
            mLSE,
            mdPsum,
            mdQaccum,
            mBlockSizes,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sPdS_layout,
            self.sdO_layout,
            self.sdQaccum_layout,
            self.r2s_tiled_copy_dQaccum,
            tiled_mma_SdP,
            tiled_mma_dK,
            tiled_mma_dV,
            tiled_mma_dQ,
            softmax_scale_log2,
            softmax_scale,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            bucketed_k2q_offsets,
            bucketed_k2q_indices,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    # ---- Mainloop kernel ----
    @cute.kernel
    def bwd(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mBlockSizes: Optional[cute.Tensor],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        softmax_scale_log2,
        softmax_scale,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        bucketed_k2q_offsets: Optional[cute.Tensor] = None,
        bucketed_k2q_indices: Optional[cute.Tensor] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # prefetch TMA descriptors
        if warp_idx == 0:
            for atom in [tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_dO, tma_atom_dK, tma_atom_dV]:
                if const_expr(atom is not None):
                    cpasync.prefetch_descriptor(atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE)
        pipeline_Q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"] + self.tma_copy_bytes["LSE"],
            defer_sync=True,
        )
        pipeline_dO = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_dO.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"] + self.tma_copy_bytes["dPsum"],
            defer_sync=False,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sLSE = storage.sLSE.get_tensor(
            cute.make_layout(
                (self.tile_m, self.Q_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdPsum = storage.sdPsum.get_tensor(
            cute.make_layout(
                (self.tile_m, self.dO_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            swap_AB=self.SdP_swapAB,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            if warp_idx == 0:
                self.load(
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
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    bucketed_k2q_offsets,
                    bucketed_k2q_indices,
                )
            if warp_idx == 1:
                self.dQaccum_store(
                    mdQaccum,
                    sdQaccum,
                    TileSchedulerCls,
                    SeqlenInfoCls,
                    bucketed_k2q_offsets,
                    bucketed_k2q_indices,
                )
        else:
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            mma_args = (
                tiled_mma_SdP,
                tiled_mma_dK,
                tiled_mma_dV,
                tiled_mma_dQ,
                mdK,
                mdV,
                mdQaccum,
                sQ,
                sK,
                sV,
                sdO,
                sP,
                sdS,
                sLSE,
                sdPsum,
                sdQaccum,
                pipeline_Q,
                pipeline_dO,
                tidx,
                tma_atom_dK,
                tma_atom_dV,
                r2s_tiled_copy_dQaccum,
                softmax_scale_log2,
                softmax_scale,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                bucketed_k2q_offsets,
                bucketed_k2q_indices,
                mBlockSizes,
            )
            warp_idx_in_mma = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - 4
            if warp_idx_in_mma < 4:
                cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)
                self.mma_wg1_qk_dv(*mma_args)
            else:
                cute.arch.setmaxregister_increase(self.num_mma_regs_wg1)
                self.mma_wg2_dov_dk_dq(*mma_args)

    @cute.jit
    def load(
        self,
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
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        bucketed_k2q_offsets: Optional[cute.Tensor] = None,
        bucketed_k2q_indices: Optional[cute.Tensor] = None,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        if warp_idx_in_wg == 0:
            producer_state_Q = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.Q_stage)
            producer_state_dO = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.dO_stage)
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                n_block, head_idx, batch_idx, q_group = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                head_idx_kv = head_idx
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
                mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0))

                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2, padded=True)[None, head_idx]
                mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None, head_idx]
                mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2, padded=True)[None, head_idx]
                gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (None, 0))
                gdO = cute.local_tile(mdO_cur, (self.tile_m, self.tile_hdimv), (None, 0))
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
                gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))

                load_K, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K, 0, cute.make_layout(1), gK, sK, single_stage=True)
                load_V, _, _ = copy_utils.tma_get_copy_fn(tma_atom_V, 0, cute.make_layout(1), gV, sV, single_stage=True)
                load_Q, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Q, 0, cute.make_layout(1), gQ, sQ)
                load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)
                load_dO, _, _ = copy_utils.tma_get_copy_fn(tma_atom_dO, 0, cute.make_layout(1), gdO, sdO)
                load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)
                load_LSE = copy_utils.cpasync_bulk_get_copy_fn(gLSE, sLSE)
                load_LSE = copy_utils.tma_producer_copy_fn(load_LSE, pipeline_Q)
                load_dPsum = copy_utils.cpasync_bulk_get_copy_fn(gdPsum, sdPsum)
                load_dPsum = copy_utils.tma_producer_copy_fn(load_dPsum, pipeline_dO)

                k2q_begin = bucketed_k2q_offsets[n_block, q_group, (head_idx, batch_idx)]
                k2q_end = bucketed_k2q_offsets[n_block + 1, q_group, (head_idx, batch_idx)]
                total_m_block_cnt = k2q_end - k2q_begin
                process_tile = total_m_block_cnt > Int32(0)

                if process_tile:
                    first_m_block = bucketed_k2q_indices[k2q_begin, (head_idx, batch_idx)]
                    pipeline_Q.producer_acquire(producer_state_Q, extra_tx_count=self.tma_copy_bytes["K"])
                    load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q))
                    load_Q(first_m_block, producer_state=producer_state_Q)
                    # Wait for bwd preprocess to finish writing LSE and dPsum
                    cute.arch.griddepcontrol_wait()
                    load_LSE(first_m_block, producer_state=producer_state_Q)
                    producer_state_dO_cur = producer_state_dO if const_expr(self.Q_stage != self.dO_stage) else producer_state_Q
                    pipeline_dO.producer_acquire(producer_state_dO_cur, extra_tx_count=self.tma_copy_bytes["V"])
                    load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_cur))
                    load_dO(first_m_block, producer_state=producer_state_dO_cur)
                    load_dPsum(first_m_block, producer_state=producer_state_dO_cur)
                    producer_state_Q.advance()
                    producer_state_dO.advance()

                    for iter_idx in cutlass.range(1, total_m_block_cnt, unroll=1):
                        m_block = bucketed_k2q_indices[k2q_begin + iter_idx, (head_idx, batch_idx)]
                        pipeline_Q.producer_acquire(producer_state_Q)
                        load_Q(m_block, producer_state=producer_state_Q)
                        load_LSE(m_block, producer_state=producer_state_Q)
                        producer_state_dO_cur = producer_state_dO if const_expr(self.Q_stage != self.dO_stage) else producer_state_Q
                        pipeline_dO.producer_acquire(producer_state_dO_cur)
                        load_dO(m_block, producer_state=producer_state_dO_cur)
                        load_dPsum(m_block, producer_state=producer_state_dO_cur)
                        producer_state_Q.advance()
                        producer_state_dO.advance()

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

    @staticmethod
    @cute.jit
    def _get_stat(tSrS: cute.Tensor, row: Int32, lane: Int32, shuffle: bool) -> Float32:
        if const_expr(not shuffle):
            return tSrS[row]
        vecsize = cute.size(tSrS, mode=[0, 0])
        idx0, off, idx1 = cute.idx2crd(row, (vecsize, 8, cute.shape(tSrS, mode=[0, 1])))
        return utils.shuffle_sync(tSrS[idx0 + idx1 * vecsize], offset=off * 4 + (lane % 4))

    @cute.jit
    def mma_wg1_qk_dv(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdQaccum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sP: Optional[cute.Tensor],
        sdS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdQaccum: cute.Tensor,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tidx: Int32,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        bucketed_k2q_offsets: Optional[cute.Tensor] = None,
        bucketed_k2q_indices: Optional[cute.Tensor] = None,
        mBlockSizes: Optional[cute.Tensor] = None,
    ):
        wg_tidx = tidx % self.num_threads_per_warp_group
        thr_mma_SdP = tiled_mma_SdP.get_slice(wg_tidx)
        wg_mma_SdP = tiled_mma_SdP.get_slice(wg_tidx)
        wg_mma_dV = tiled_mma_dV.get_slice(wg_tidx)

        shape_mnk_S = (self.tile_m, self.tile_n, self.tile_hdim)
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(wg_mma_SdP, shape_mnk_S, sQ, sK, swap_AB=self.SdP_swapAB)
        mma_qk_fn = partial(gemm_zero_init, tiled_mma_SdP, shape_mnk_S[:2], tSrQ, tSrK, swap_AB=self.SdP_swapAB)
        smem_thr_copy_K = None
        tSsK = None
        tSrK_copy_view = None
        if const_expr(self.V_in_regs):
            smem_copy_atom_K = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype)
            smem_thr_copy_K = utils.make_tiled_copy_B(smem_copy_atom_K, tiled_mma_SdP, swapAB=self.SdP_swapAB).get_slice(wg_tidx)
            tSsK = smem_thr_copy_K.partition_S(sK)
            tSrK_copy_view = smem_thr_copy_K.retile(tSrK)

        sdOt = layout_utils.transpose_view(sdO)
        shape_mnk_dV = (self.tile_n, self.tile_hdimv, self.tile_m)
        acc_dV, _, tdVrdOt = sm90_utils.partition_fragment_ABC(wg_mma_dV, shape_mnk_dV, None, sdOt, swap_AB=False)

        sP_cpy = sP if const_expr(not self.SdP_swapAB) else layout_utils.transpose_view(sP)
        copy_P_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_SdP,
            sP_cpy,
            wg_tidx,
            transpose=self.SdP_swapAB,
            position_independent=True,
            major_mode_size=self.tile_n,
        )
        tLSEsLSE = layout_utils.mma_partition_C_vec(sLSE, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB)

        consumer_state_Q = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage)
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, q_group = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mask = AttentionMaskCls(seqlen)
            k2q_begin = bucketed_k2q_offsets[n_block, q_group, (head_idx, batch_idx)]
            k2q_end = bucketed_k2q_offsets[n_block + 1, q_group, (head_idx, batch_idx)]
            loop_count = k2q_end - k2q_begin
            process_tile = loop_count > Int32(0)
            block_size_k = None
            if const_expr(mBlockSizes is not None):
                block_size_k = mBlockSizes[batch_idx, n_block]
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                n_block=n_block,
                thr_mma=thr_mma_SdP,
                mask_seqlen=True,
                block_size_k=block_size_k,
            )
            pds_iter = Int32(0)
            dV_accumulate = False
            if process_tile:
                for iter_idx in cutlass.range(loop_count, unroll=1):
                    m_block = bucketed_k2q_indices[k2q_begin + iter_idx, (head_idx, batch_idx)]
                    consumer_state_dO_cur = consumer_state_Q if const_expr(self.Q_stage == self.dO_stage) else consumer_state_dO
                    smem_idx_Q = consumer_state_Q.index
                    smem_idx_dO = consumer_state_dO_cur.index if const_expr(self.dO_stage > 1) else 0
                    smem_idx_PdS = smem_idx_Q if const_expr(self.PdS_stage > 1) else 0

                    if pds_iter >= self.PdS_stage:
                        cute.arch.barrier(
                            barrier_id=int(NamedBarrierBwd.PdSConsumed) + smem_idx_PdS,
                            number_of_threads=self.num_mma_threads,
                        )

                    pipeline_Q.consumer_wait(consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q))
                    if const_expr(self.V_in_regs):
                        if pds_iter == 0:
                            cute.copy(smem_thr_copy_K, tSsK, tSrK_copy_view)
                    acc_S = mma_qk_fn(A_idx=smem_idx_Q, wg_wait=-1)
                    tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, smem_idx_Q])
                    warpgroup.wait_group(0)
                    pipeline_Q.consumer_release(consumer_state_Q)

                    if const_expr(mBlockSizes is not None):
                        if block_size_k < self.tile_n:
                            mask_fn(acc_S, m_block=m_block)
                    else:
                        mask_fn(acc_S, m_block=m_block)
                    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.SdP_swapAB)
                    lane_idx = cute.arch.lane_idx()
                    for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
                        lse_val = self._get_stat(tLSErLSE, r, lane_idx, shuffle=self.shuffle_LSE)
                        for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                            acc_S_mn[r, c] = cute.math.exp2(acc_S_mn[r, c] * softmax_scale_log2 - lse_val, fastmath=True)
                    tdVrP = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), self.dtype)
                    copy_P_r2s(tdVrP, dst_idx=smem_idx_PdS)
                    cute.arch.fence_view_async_shared()
                    cute.arch.barrier_arrive(
                        barrier_id=int(NamedBarrierBwd.PReady) + smem_idx_PdS,
                        number_of_threads=self.num_mma_threads,
                    )

                    pipeline_dO.consumer_wait(
                        consumer_state_dO_cur,
                        pipeline_dO.consumer_try_wait(consumer_state_dO_cur),
                    )
                    gemm_w_idx(
                        tiled_mma_dV,
                        acc_dV,
                        tdVrP,
                        tdVrdOt,
                        zero_init=not dV_accumulate,
                        B_idx=smem_idx_dO,
                        wg_wait=-1,
                    )
                    warpgroup.wait_group(0)
                    pipeline_dO.consumer_release(consumer_state_dO_cur)
                    consumer_state_Q.advance()
                    consumer_state_dO.advance()
                    pds_iter += 1
                    dV_accumulate = True
                self.epilogue_dV_accum_one_wg(
                    acc_dV,
                    mdV,
                    sV,
                    seqlen,
                    wg_tidx,
                    n_block,
                    head_idx,
                    batch_idx,
                )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma_wg2_dov_dk_dq(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdQaccum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sP: Optional[cute.Tensor],
        sdS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdQaccum: cute.Tensor,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tidx: Int32,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        bucketed_k2q_offsets: Optional[cute.Tensor] = None,
        bucketed_k2q_indices: Optional[cute.Tensor] = None,
        mBlockSizes: Optional[cute.Tensor] = None,
    ):
        wg_tidx = tidx % self.num_threads_per_warp_group
        thr_mma_SdP = tiled_mma_SdP.get_slice(wg_tidx)
        wg_mma_SdP = tiled_mma_SdP.get_slice(wg_tidx)
        wg_mma_dK = tiled_mma_dK.get_slice(wg_tidx)
        wg_mma_dQ = tiled_mma_dQ.get_slice(wg_tidx)

        shape_mnk_dP = (self.tile_m, self.tile_n, self.tile_hdimv)
        _, tdPrdO, tdPrV = sm90_utils.partition_fragment_ABC(wg_mma_SdP, shape_mnk_dP, sdO, sV, swap_AB=self.SdP_swapAB)
        mma_dov_fn = partial(gemm_zero_init, tiled_mma_SdP, shape_mnk_dP[:2], tdPrdO, tdPrV, swap_AB=self.SdP_swapAB)
        smem_thr_copy_V = None
        tdPsV = None
        tdPrV_copy_view = None
        if const_expr(self.V_in_regs):
            smem_copy_atom_V = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype)
            smem_thr_copy_V = utils.make_tiled_copy_B(smem_copy_atom_V, tiled_mma_SdP, swapAB=self.SdP_swapAB).get_slice(wg_tidx)
            tdPsV = smem_thr_copy_V.partition_S(sV)
            tdPrV_copy_view = smem_thr_copy_V.retile(tdPrV)

        sP_cpy = sP if const_expr(not self.SdP_swapAB) else layout_utils.transpose_view(sP)
        _, tiled_copy_P_s2r, tSRsP = copy_utils.get_smem_load_C(
            tiled_mma_SdP,
            sP_cpy,
            wg_tidx,
            transpose=self.SdP_swapAB,
            position_independent=True,
        )
        sdS_cpy = sdS if const_expr(not self.SdP_swapAB) else layout_utils.transpose_view(sdS)
        copy_dS_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_SdP,
            sdS_cpy,
            wg_tidx,
            transpose=self.SdP_swapAB,
            position_independent=True,
            major_mode_size=self.tile_n,
        )
        tLSEsdPsum = layout_utils.mma_partition_C_vec(sdPsum, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB)

        sKt = layout_utils.transpose_view(sK)
        shape_mnk_dQ = (self.tile_m, self.tile_hdim, self.tile_n)
        _, tdQrdS, tdQrKt = sm90_utils.partition_fragment_ABC(wg_mma_dQ, shape_mnk_dQ, sdS, sKt, swap_AB=False)
        mma_dsk_fn = partial(gemm_zero_init, tiled_mma_dQ, shape_mnk_dQ[:2], tdQrdS, tdQrKt, swap_AB=False)
        sQt = layout_utils.transpose_view(sQ)
        shape_mnk_dK = (self.tile_n, self.tile_hdim, self.tile_m)
        acc_dK, _, tdKrQt = sm90_utils.partition_fragment_ABC(wg_mma_dK, shape_mnk_dK, None, sQt, swap_AB=False)
        smem_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_slice(wg_tidx)

        consumer_state_Q = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage)
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, q_group = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            k2q_begin = bucketed_k2q_offsets[n_block, q_group, (head_idx, batch_idx)]
            k2q_end = bucketed_k2q_offsets[n_block + 1, q_group, (head_idx, batch_idx)]
            loop_count = k2q_end - k2q_begin
            process_tile = loop_count > Int32(0)
            pds_iter = Int32(0)
            dK_accumulate = False
            if process_tile:
                for iter_idx in cutlass.range(loop_count, unroll=1):
                    m_block = bucketed_k2q_indices[k2q_begin + iter_idx, (head_idx, batch_idx)]
                    consumer_state_dO_cur = consumer_state_Q if const_expr(self.Q_stage == self.dO_stage) else consumer_state_dO
                    smem_idx_Q = consumer_state_Q.index
                    smem_idx_dO = consumer_state_dO_cur.index if const_expr(self.dO_stage > 1) else 0
                    smem_idx_PdS = smem_idx_Q if const_expr(self.PdS_stage > 1) else 0

                    pipeline_Q.consumer_wait(consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q))
                    pipeline_dO.consumer_wait(
                        consumer_state_dO_cur,
                        pipeline_dO.consumer_try_wait(consumer_state_dO_cur),
                    )
                    if const_expr(self.V_in_regs):
                        if pds_iter == 0:
                            cute.copy(smem_thr_copy_V, tdPsV, tdPrV_copy_view)
                    acc_dP = mma_dov_fn(A_idx=smem_idx_dO, wg_wait=-1)
                    cute.arch.barrier(
                        barrier_id=int(NamedBarrierBwd.PReady) + smem_idx_PdS,
                        number_of_threads=self.num_mma_threads,
                    )
                    tdPrP = cute.make_rmem_tensor(acc_dP.shape, self.dtype)
                    copy_utils.load_s2r_retile(
                        tiled_copy_P_s2r,
                        tSRsP[None, None, None, smem_idx_PdS],
                        tdPrP,
                    )
                    cute.arch.barrier_arrive(
                        barrier_id=int(NamedBarrierBwd.PdSConsumed) + smem_idx_PdS,
                        number_of_threads=self.num_mma_threads,
                    )
                    warpgroup.wait_group(0)
                    tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, smem_idx_dO])
                    pipeline_dO.consumer_release(consumer_state_dO_cur)

                    tdPrP_mn = layout_utils.reshape_acc_to_mn(tdPrP, transpose=self.SdP_swapAB)
                    acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=self.SdP_swapAB)
                    lane_idx = cute.arch.lane_idx()
                    for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
                        dpsum_val = self._get_stat(tLSErdPsum, r, lane_idx, shuffle=self.shuffle_dPsum)
                        for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                            acc_dP_mn[r, c] = tdPrP_mn[r, c].to(Float32) * (acc_dP_mn[r, c] - dpsum_val)
                    tdKrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)
                    copy_dS_r2s(tdKrdS, dst_idx=smem_idx_PdS)
                    cute.arch.fence_view_async_shared()
                    acc_dQ = mma_dsk_fn(A_idx=smem_idx_PdS, wg_wait=-1)
                    gemm_w_idx(
                        tiled_mma_dK,
                        acc_dK,
                        tdKrdS,
                        tdKrQt,
                        zero_init=not dK_accumulate,
                        B_idx=smem_idx_Q,
                        wg_wait=1,
                    )
                    smem_idx_dQaccum = pds_iter % self.dQaccum_stage if const_expr(self.dQaccum_stage > 1) else 0
                    cute.arch.barrier(
                        barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + smem_idx_dQaccum,
                        number_of_threads=self.num_threads_per_warp_group + self.num_dQ_store_warps * cute.arch.WARP_SIZE,
                    )
                    sdQaccum_cur = sdQaccum[None, None, smem_idx_dQaccum] if const_expr(self.dQaccum_stage > 1) else sdQaccum
                    tdQsdQaccum = smem_thr_copy_dQaccum.partition_D(sdQaccum_cur)
                    tdQrdQaccum_flat = cute.make_tensor(acc_dQ.iterator, cute.make_layout(tdQsdQaccum.shape))
                    cute.autovec_copy(tdQrdQaccum_flat, tdQsdQaccum)
                    cute.arch.fence_view_async_shared()
                    cute.arch.barrier_arrive(
                        barrier_id=int(NamedBarrierBwd.dQFullWG0) + smem_idx_dQaccum,
                        number_of_threads=self.num_threads_per_warp_group + self.num_dQ_store_warps * cute.arch.WARP_SIZE,
                    )
                    warpgroup.wait_group(0)
                    pipeline_Q.consumer_release(consumer_state_Q)
                    consumer_state_Q.advance()
                    consumer_state_dO.advance()
                    pds_iter += 1
                    dK_accumulate = True
                self.epilogue_dK_accum_one_wg(
                    acc_dK,
                    mdK,
                    sK,
                    seqlen,
                    wg_tidx,
                    n_block,
                    head_idx,
                    batch_idx,
                )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    # ---- dK/dV epilogue and dQ reduce store ----
    @cute.jit
    def epilogue_dV_accum_one_wg(
        self,
        acc_dV: cute.Tensor,
        mdV: cute.Tensor,
        sV: cute.Tensor,
        seqlen: SeqlenInfoQK,
        wg_tidx: Int32,
        n_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        epi_barrier = cutlass.pipeline.NamedBarrier(barrier_id=int(NamedBarrierBwd.EpilogueV), num_threads=self.num_threads_per_warp_group)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        mdVaccum_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=2, padded=True, multiple=self.tile_hdimv)[None, head_idx]
        gdVaccum = cute.local_tile(mdVaccum_cur, (self.tile_n * self.tile_hdimv,), (n_block,))
        sdVaccum_layout = cute.make_layout(self.tile_n * self.tile_hdimv)
        sdVaccum = cute.make_tensor(cute.recast_ptr(sV.iterator, dtype=Float32), sdVaccum_layout)
        tiled_copy_dVaccum_r2s = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
            cute.make_layout(self.num_threads_per_warp_group),
            cute.make_layout(128 // Float32.width),
        )
        thr_copy_dVaccum_r2s = tiled_copy_dVaccum_r2s.get_slice(wg_tidx)
        tdVsdVaccum = thr_copy_dVaccum_r2s.partition_D(sdVaccum)

        cute.arch.cp_async_bulk_wait_group(0, read=True)
        epi_barrier.arrive_and_wait()
        tdVrdVaccum_flat = cute.make_tensor(acc_dV.iterator, cute.make_layout(tdVsdVaccum.shape))
        cute.autovec_copy(tdVrdVaccum_flat, tdVsdVaccum)
        cute.arch.fence_view_async_shared()
        epi_barrier.arrive_and_wait()
        if warp_idx % 4 == 0:
            with cute.arch.elect_one():
                copy_utils.cpasync_reduce_bulk_add_f32(
                    sdVaccum.iterator,
                    gdVaccum.iterator,
                    self.tma_copy_bytes["dVacc"],
                )
            cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def epilogue_dK_accum_one_wg(
        self,
        acc_dK: cute.Tensor,
        mdK: cute.Tensor,
        sK: cute.Tensor,
        seqlen: SeqlenInfoQK,
        wg_tidx: Int32,
        n_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        epi_barrier = cutlass.pipeline.NamedBarrier(barrier_id=int(NamedBarrierBwd.EpilogueK), num_threads=self.num_threads_per_warp_group)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        mdKaccum_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=2, padded=True, multiple=self.tile_hdim)[None, head_idx]
        gdKaccum = cute.local_tile(mdKaccum_cur, (self.tile_n * self.tile_hdim,), (n_block,))
        sdKaccum_layout = cute.make_layout(self.tile_n * self.tile_hdim)
        sdKaccum = cute.make_tensor(cute.recast_ptr(sK.iterator, dtype=Float32), sdKaccum_layout)
        tiled_copy_dKaccum_r2s = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
            cute.make_layout(self.num_threads_per_warp_group),
            cute.make_layout(128 // Float32.width),
        )
        thr_copy_dKaccum_r2s = tiled_copy_dKaccum_r2s.get_slice(wg_tidx)
        tdKsdKaccum = thr_copy_dKaccum_r2s.partition_D(sdKaccum)

        cute.arch.cp_async_bulk_wait_group(0, read=True)
        epi_barrier.arrive_and_wait()
        tdKrdKaccum_flat = cute.make_tensor(acc_dK.iterator, cute.make_layout(tdKsdKaccum.shape))
        cute.autovec_copy(tdKrdKaccum_flat, tdKsdKaccum)
        cute.arch.fence_view_async_shared()
        epi_barrier.arrive_and_wait()
        if warp_idx % 4 == 0:
            with cute.arch.elect_one():
                copy_utils.cpasync_reduce_bulk_add_f32(
                    sdKaccum.iterator,
                    gdKaccum.iterator,
                    self.tma_copy_bytes["dKacc"],
                )
            cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def dQaccum_store(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        SeqlenInfoCls: cutlass.Constexpr[Callable],
        bucketed_k2q_offsets: Optional[cute.Tensor] = None,
        bucketed_k2q_indices: Optional[cute.Tensor] = None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        read_flag = True

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, q_group = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
            # ((M * K / num_wg_dQ, num_wg_dQ), num_m_blocks)
            gdQaccum = cute.local_tile(
                mdQaccum_cur,
                (cute.make_layout((self.tile_m * self.tile_hdim // self.num_wg_dQ, self.num_wg_dQ)),),
                (None,),
            )

            k2q_begin = bucketed_k2q_offsets[n_block, q_group, (head_idx, batch_idx)]
            k2q_end = bucketed_k2q_offsets[n_block + 1, q_group, (head_idx, batch_idx)]
            loop_count = k2q_end - k2q_begin
            process_tile = loop_count > Int32(0)

            if process_tile:
                if const_expr(self.dQaccum_stage > 1):
                    for stage_idx in cutlass.range_constexpr(self.dQaccum_stage):
                        cute.arch.barrier_arrive(
                            barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + stage_idx,
                            number_of_threads=self.num_threads_per_warp_group + self.num_dQ_store_warps * cute.arch.WARP_SIZE,
                        )
                for iter_idx in cutlass.range(loop_count, unroll=1):
                    m_block = bucketed_k2q_indices[k2q_begin + iter_idx, (head_idx, batch_idx)]
                    m_block_safe = m_block
                    smem_idx_dQaccum = iter_idx % self.dQaccum_stage if const_expr(self.dQaccum_stage > 1) else 0

                    num_dQ_chunks = self.num_wg_dQ
                    for warp_group_idx in cutlass.range_constexpr(num_dQ_chunks):
                        if const_expr(self.dQaccum_stage == 1):
                            cute.arch.cp_async_bulk_wait_group(num_dQ_chunks - 1 - warp_group_idx, read=read_flag)
                            cute.arch.barrier_arrive(
                                barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                                number_of_threads=self.num_threads_per_warp_group + self.num_dQ_store_warps * cute.arch.WARP_SIZE,
                            )

                    if const_expr(self.dQaccum_stage > 1):
                        if iter_idx >= self.dQaccum_stage:
                            cute.arch.cp_async_bulk_wait_group(self.dQaccum_stage - 1, read=read_flag)
                            cute.arch.barrier_arrive(
                                barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + smem_idx_dQaccum,
                                number_of_threads=self.num_threads_per_warp_group + self.num_dQ_store_warps * cute.arch.WARP_SIZE,
                            )

                    for warp_group_idx in cutlass.range_constexpr(num_dQ_chunks):
                        cute.arch.barrier(
                            barrier_id=int(NamedBarrierBwd.dQFullWG0) + (smem_idx_dQaccum if const_expr(self.dQaccum_stage > 1) else warp_group_idx),
                            number_of_threads=self.num_threads_per_warp_group + self.num_dQ_store_warps * cute.arch.WARP_SIZE,
                        )
                        sdQaccum_cur = (
                            sdQaccum[None, warp_group_idx, smem_idx_dQaccum] if const_expr(self.dQaccum_stage > 1) else sdQaccum[None, warp_group_idx]
                        )
                        with cute.arch.elect_one():
                            copy_utils.cpasync_reduce_bulk_add_f32(
                                sdQaccum_cur.iterator,
                                gdQaccum[(None, warp_group_idx), m_block_safe].iterator,
                                self.tma_copy_bytes["dQ"],
                            )
                        cute.arch.cp_async_bulk_commit_group()

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        cute.arch.cp_async_bulk_wait_group(0, read=True)


# =============================================================================
# SM90 backward-specific barriers and masks
# =============================================================================
class NamedBarrierBwd(enum.IntEnum):
    # Hopper named barrier ids are 0..15, but id 0 is also used by driver
    # sync_threads paths.  P/dS and dQ reserve consecutive ids so dense
    # WG-specialized buffers can be staged without barrier-id collisions.
    Epilogue = 1
    PdS = 2
    PReady = 3
    dSReady = 5
    PdSConsumed = 7
    dQFullWG0 = 9
    dQFullWG1 = 10
    dQFullWG2 = 11
    dQEmptyWG0 = 12
    dQEmptyWG1 = 13
    dQEmptyWG2 = 14
    EpilogueV = 11
    EpilogueK = 14
    WarpSchedulerWG1 = 15
    WarpSchedulerWG2 = 15
    WarpSchedulerWG3 = 15


MaskGenFn: TypeAlias = Callable[[int], Uint32]
PREDICATE_MASK_CHUNK_SIZE: int = 32


@cute.jit
def predicate_bitmask_below(limit: Int32, s: int) -> Uint32:
    """32-bit register-to-predicate bitmask keeping positions < limit.

    Positions 0..limit-1 in chunk `s` get bit=1 (keep), the rest bit=0 (mask).
    Uses inline PTX to avoid shift-by-type-width UB.
    """
    m = max((s + 1) * PREDICATE_MASK_CHUNK_SIZE - limit, 0)
    return utils.shr_u32(Uint32(0xFFFFFFFF), Uint32(m))


@cute.jit
def apply_predicate_mask(
    X: cute.Tensor,
    mask_gen_fn: cutlass.Constexpr[MaskGenFn],
    rank1: bool = False,
) -> None:
    """Apply register-to-predicate masking with a custom bitmask generator.

    mask_gen_fn(chunk_idx: constexpr int) -> Uint32:
        Returns a 32-bit bitmask for the chunk. Bit i set means column
        chunk_idx * chunk_size + i is KEPT; bit i clear means masked to -inf.
    """
    ncol = const_expr(cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape))
    # 32-column chunks. The mask_gen_fn returns a Uint32 bitmask (1=keep).
    CHUNK_SIZE = PREDICATE_MASK_CHUNK_SIZE
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, CHUNK_SIZE)):
        mask = mask_gen_fn(s)
        # This must be range_constexpr so the compiler can generate the register-to-predicate instruction.
        for i in cutlass.range_constexpr(min(CHUNK_SIZE, ncol - s * CHUNK_SIZE)):
            in_bound = cutlass.Boolean(mask & (Uint32(1) << i))
            c = s * CHUNK_SIZE + i
            if const_expr(rank1):
                X[c] = X[c] if in_bound else -Float32.inf
            else:
                for r in cutlass.range_constexpr(cute.size(X.shape[0])):
                    X[r, c] = X[r, c] if in_bound else -Float32.inf


@cute.jit
def sm90_col_to_predicate_idx(col_limit: Int32) -> Int32:
    """Transform SM90 MMA column coordinate to register-to-predicate element index.

    SM90 MMA accumulator column indices are non-contiguous: 0, 1, 8, 9, 16, 17, ...
    Element indices are contiguous: 0, 1, 2, 3, 4, 5, ...
    This converts a column-space threshold to element-space for predicate bitmasks.
    """
    return col_limit // 8 * 2 + min(col_limit % 8, 2)


@dataclass(frozen=True)
class AttentionMask:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    seqlen_info: SeqlenInfoQK
    swap_AB: cutlass.Constexpr[bool] = False

    @property
    def seqlen_k(self) -> Int32:
        return self.seqlen_info.seqlen_k

    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool] = True,
        block_size_k: Optional[Int32] = None,
    ) -> None:
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.swap_AB)
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(acc_shape if not self.swap_AB else acc_shape[::-1])
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS), transpose=self.swap_AB)
        t0ScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cS), transpose=self.swap_AB)
        COL = 1 if const_expr(not self.swap_AB) else 0
        thr_col_offset = tScS_mn[0][COL]
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n
        if const_expr(block_size_k is not None):
            seqlenk_col_limit = cutlass.min(seqlenk_col_limit, block_size_k)
        seqlenk_col_limit = seqlenk_col_limit - thr_col_offset
        if const_expr(mask_seqlen):
            use_predicate_mask = const_expr(not self.swap_AB)
            if const_expr(not use_predicate_mask):
                for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                    oob = t0ScS_mn[0, c][COL] >= seqlenk_col_limit
                    for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                        acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]
            else:
                seqlenk_col_limit_predicate = sm90_col_to_predicate_idx(seqlenk_col_limit)
                apply_predicate_mask(
                    acc_S_mn,
                    lambda s: predicate_bitmask_below(seqlenk_col_limit_predicate, s),
                )
