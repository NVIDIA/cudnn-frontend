# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import AddressSpace
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils

from cutlass._mlir import ir
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, Int32 as _epi_Int32, Int64
from cutlass.cute.typing import Float32

from ......helpers.iket_compat import iket
from ......helpers.flag_batch import GpuReleaseFlagBatchTracker
from ......helpers.ptx_helpers import (
    red_add_relaxed_sys_f32 as _red_add_relaxed_sys_f32,
    red_add_relaxed_sys_v2_bf16x2 as _red_add_relaxed_sys_v2_bf16x2,
    stg_e8m0_from_f32,
    stg_e8m0x8_from_f32,
)
from ..helpers.utils import swiglu_act, dswiglu_act, quant_sfd_row, quant_sfd_col
from ......quant_def import CombineFormat
from .....schedulers import BlockPhase
from ..tmem_transpose import _TmemTranspose16x32Core
from ..fwd_glu.glu_mxfp8_fc12_epilogue import Fc2OutputDest

Fc1GateUpInterleave = 32
EpilogueTileN = 32
Fc1EpilogueOutputTileM = 128
Fc1EpilogueOutputTileN = 128
WarpThreadCount = 32
EpiWarpCount = 4


@cute.jit
def dprob_reduce_gmem(
    real_dprob: cute.Tensor,
    dprob_val: cutlass.Float32,
    is_valid: bool,
    expert_local_token_idx,
    system_scope: bool = False,
) -> None:
    """Atomically reduce per-tile dprob accumulator into GMEM."""
    if is_valid:
        if cutlass.const_expr(system_scope):
            _red_add_relaxed_sys_f32(
                real_dprob.iterator + expert_local_token_idx,
                cutlass.Float32(dprob_val),
            )
        else:
            cute.arch.atomic_add(
                real_dprob.iterator + expert_local_token_idx,
                cutlass.Float32(dprob_val),
                sem="relaxed",
                scope="gpu",
            )

# =============================================================================
# DgluMxfp8Epilogue
# =============================================================================

class DgluMxfp8Epilogue:

    _SubtileBarIdBase = 4

    def __init__(
        self,
        *,
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        use_2cta_instrs: bool,
        sf_vec_size: int,
        fc1_output_dtype: Type[cutlass.Numeric],
        fc1_output_layout: utils.LayoutEnum,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E8M0FNU,
        epilog_sync_bar_id: int = 1,
        epilogue_warp_ids: Tuple[int, ...] = (0, 1, 2, 3),
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        token_back_by_dispatch: bool = False,
        epi_flag_batch: Optional[Tuple[int, int]] = (1, 1),
        dfc2_recompute: bool = False,
        dfc2_col_output: bool = False,
        fc2_in_kernel_topk_reduce: bool = False,
        combine_format: Optional[CombineFormat] = None,
        combine_hidden: Optional[int] = None,
        act_func: str = "swiglu",
        gate_up_clamp: Optional[float] = None,
    ) -> None:
        self._act_func = act_func
        self._gate_up_clamp = (
            cutlass.Float32(gate_up_clamp) if gate_up_clamp is not None else None
        )
        self.fc1_output_dtype = fc1_output_dtype
        self.fc1_output_layout = fc1_output_layout
        self.acc_dtype = acc_dtype
        self.sf_dtype = sf_dtype
        self._sf_vec_size = sf_vec_size
        self._epilog_sync_bar_id = epilog_sync_bar_id
        self._epilogue_warp_ids = epilogue_warp_ids
        self._use_2cta_instrs = use_2cta_instrs

        self._atom_thr_size = 2 if use_2cta_instrs else 1
        self._cta_tile_m = mma_tiler_mnk[0] // self._atom_thr_size
        self._cta_tile_n = mma_tiler_mnk[1]
        self._mma_tiler_k = mma_tiler_mnk[2]
        self._mma_tiler = tuple(mma_tiler_mnk)  # for partition_C in the C-load
        self._cta_tile_n_sfb = ((mma_tiler_mnk[1] + 127) // 128) * 128
        self._static_expert_shape = static_expert_shape
        if (
            static_expert_shape is not None
            and static_expert_shape[2] % (self._cta_tile_m * cluster_shape_mn[0]) == 0
        ):
            self._fc2_stg_needs_predicate: bool = False
        else:
            self._fc2_stg_needs_predicate: bool = True

        self._epi_tile = (EpilogueTileN, Fc1EpilogueOutputTileM)
        self._subtile_cnt = self._cta_tile_n // 2 // EpilogueTileN

        self._num_acc_stage = 2
        self._num_acc_pipeline_stages = self._num_acc_stage

        k = self._mma_tiler_k
        self._num_sfa_tmem_cols = self._cta_tile_m * k // sf_vec_size * 4 // 4 // 128
        self._num_sfb_tmem_cols = (
            self._cta_tile_n_sfb * k // sf_vec_size * 4 // 4 // 128
        )
        self._num_sf_tmem_cols = 32 # self._num_sfa_tmem_cols + self._num_sfb_tmem_cols

        self._num_accumulator_tmem_cols = self._cta_tile_n * self._num_acc_stage

        self._token_back_by_dispatch = token_back_by_dispatch
        # In-kernel top-k reduce
        self._reduce_topk_in_epilogue = (
            fc2_in_kernel_topk_reduce and not token_back_by_dispatch
        )
        _fc1_batch, _fc2_batch = (1, 1) if epi_flag_batch is None else epi_flag_batch
        self._epi_fc1_batch = max(1, min(32, int(_fc1_batch)))
        self._epi_fc2_batch = max(1, min(32, int(_fc2_batch)))

        self._dfc2_recompute = dfc2_recompute
        self._dfc2_col_output = dfc2_col_output
        # One PipelineTmaStore stage holds every data plane produced by a dFC2 subtile
        self._d_output_slots = (
            2
            + (2 if dfc2_col_output else 0)
            + (1 if dfc2_recompute else 0)
        )

        # combine_format determines the dfc1 (final grad_x) combine encoding.
        if combine_format is None:
            combine_format = CombineFormat.parse("bf16")
        self._combine_format = combine_format
        self._combine_mxfp8 = combine_format.is_quantized
        # sf_block_pad for the dfc1 MXFP8 combine
        if self._combine_mxfp8 and combine_hidden is not None:
            _hidden_dfc1 = combine_hidden
            _sf_blocks_dfc1 = _hidden_dfc1 // EpilogueTileN
            self._dfc1_sf_block_pad = ((_sf_blocks_dfc1 + 15) // 16) * 16
            self._hidden_dfc1 = _hidden_dfc1
        else:
            self._dfc1_sf_block_pad = 0
            self._hidden_dfc1 = 0
        # batching stg.64 SF
        self._dfc1_sf_batch8 = (
            self._combine_mxfp8
            and self._hidden_dfc1 > 0
            and (self._hidden_dfc1 % self._cta_tile_n == 0)
            and (self._cta_tile_n // EpilogueTileN == 8)
        )

        pass
        
    # -- Codegen-time queries  --

    @property
    def epi_tile(self) -> Tuple[int, int]:
        return self._epi_tile

    @property
    def num_acc_pipeline_stages(self) -> int:
        return self._num_acc_pipeline_stages

    @property
    def num_acc_stage(self) -> int:
        return self._num_acc_stage

    @property
    def d_output_slots(self) -> int:
        return self._d_output_slots

    @property
    def subtile_cnt(self) -> int:
        return self._subtile_cnt

    @property
    def cta_tile_n(self) -> int:
        return self._cta_tile_n

    @property
    def num_sf_tmem_cols(self) -> int:
        return self._num_sf_tmem_cols

    @property
    def num_sfa_tmem_cols(self) -> int:
        return self._num_sfa_tmem_cols

    @property
    def num_sfb_tmem_cols(self) -> int:
        return self._num_sfb_tmem_cols

    @property
    def num_accumulator_tmem_cols(self) -> int:
        return self._num_accumulator_tmem_cols

    def staged_smem_layout(
        self,
        n_stages: int,
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        return sm100_utils.make_smem_layout_epi(
            self.fc1_output_dtype,
            self.fc1_output_layout,
            self._epi_tile,
            n_stages,
        )

    @property
    def smem_layout_one_stage(self) -> Union[cute.Layout, cute.ComposedLayout]:
        staged = self.staged_smem_layout(1)
        return cute.select(staged, mode=[0, 1])

    @property
    def bytes_per_stage(self) -> int:
        return cute.size_in_bytes(self.fc1_output_dtype, self.smem_layout_one_stage)

    # -- grad_y1 (dfc2 output) sD staging: reference-style shared (128×32) box --
    @property
    def d_epi_tile(self) -> Tuple[int, int]:
        return (self._cta_tile_m, EpilogueTileN)

    def d_staged_smem_layout(
        self,
        n_stages: int,
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        return sm100_utils.make_smem_layout_epi(
            self.fc1_output_dtype,
            self.fc1_output_layout,
            self.d_epi_tile,
            n_stages,
        )

    @property
    def d_smem_layout_one_stage(self) -> Union[cute.Layout, cute.ComposedLayout]:
        staged = self.d_staged_smem_layout(1)
        return cute.select(staged, mode=[0, 1])

    @property
    def d_bytes_per_stage(self) -> int:
        return cute.size_in_bytes(self.fc1_output_dtype, self.d_smem_layout_one_stage)

    # forward pre-activation (dswiglu C input) staging
    @property
    def preact_epi_tile(self) -> Tuple[int, int]:
        return (self._cta_tile_m, EpilogueTileN)

    def preact_staged_smem_layout(
        self, n_stages: int
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        return sm100_utils.make_smem_layout_epi(
            cutlass.BFloat16,
            self.fc1_output_layout,
            self.preact_epi_tile,
            n_stages,
        )

    @property
    def preact_smem_layout_one_stage(self) -> Union[cute.Layout, cute.ComposedLayout]:
        staged = self.preact_staged_smem_layout(1)
        return cute.select(staged, mode=[0, 1])

    @property
    def preact_bytes_per_stage(self) -> int:
        return cute.size_in_bytes(cutlass.BFloat16, self.preact_smem_layout_one_stage)

    @cute.jit
    def _store_aux_row_smem(self, r_data: cute.Tensor, s_data: cute.Tensor) -> None:
        """Store one 32-byte token row as two swizzle-safe 128-bit segments."""
        segment_elements = 16
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.fc1_output_dtype,
            num_bits_per_copy=128,
        )
        r_segments = cute.zipped_divide(r_data, (segment_elements,))
        s_segments = cute.zipped_divide(s_data, (segment_elements,))
        for segment in cutlass.range_constexpr(EpilogueTileN // segment_elements):
            cute.copy(
                store_atom,
                cute.coalesce(r_segments[None, segment]),
                cute.coalesce(s_segments[None, segment]),
            )

    @cute.jit
    def _run_dfc2_task_tile(
        self,
        work_tile_info,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        sched_ext,
        gmem_fc1_output: cute.Tensor,
        gmem_fc1_output_sf: cute.Tensor,
        tma_atom_fc1_recompute: cute.CopyAtom,
        gmem_fc1_recompute: cute.Tensor,
        gmem_fc1_recompute_sf: cute.Tensor,
        tma_atom_fc1_col_output: cute.CopyAtom,
        gmem_fc1_col_output: cute.Tensor,
        gmem_fc1_col_output_sf: cute.Tensor,
        c_pipeline,
        smem_preact_buffer: cute.Tensor,
        c_consumer_state,
        smem_d_buffer: cute.Tensor,
        tma_atom_grad_y1: cute.CopyAtom,
        warp_idx: int,
        tidx,
        norm_const,
        gmem_topk_scores: cute.Tensor,
        gmem_beta: cute.Tensor,
        gmem_dprob: cute.Tensor,
        d_pipeline,
        d_num_stage,
        token_comm_args=None,
    ):
        """dfc2 task-tile — c_pipeline CONSUMER. """
        real_fc1_output, _    = sched_ext.get_gmem_tensor("d",   gmem_fc1_output,    work_tile_info)
        real_fc1_output_sf, _ = sched_ext.get_gmem_tensor("sfd", gmem_fc1_output_sf, work_tile_info)
        if cutlass.const_expr(token_comm_args is None):
            real_dprob, _ = sched_ext.get_gmem_tensor("topk", gmem_dprob, work_tile_info)
        else:
            real_dprob = None
        if cutlass.const_expr(self._dfc2_recompute):
            real_fc1_recompute, _    = sched_ext.get_gmem_tensor("recompute",   gmem_fc1_recompute,    work_tile_info)
            real_fc1_recompute_sf, _ = sched_ext.get_gmem_tensor("sfrecompute", gmem_fc1_recompute_sf, work_tile_info)
        else:
            real_fc1_recompute = None
            real_fc1_recompute_sf = None
        if cutlass.const_expr(self._dfc2_col_output):
            real_fc1_col_output, _    = sched_ext.get_gmem_tensor("col_output",   gmem_fc1_col_output,    work_tile_info)
            real_fc1_col_output_sf, _ = sched_ext.get_gmem_tensor("sfcol_output", gmem_fc1_col_output_sf, work_tile_info)
        else:
            real_fc1_col_output = None
            real_fc1_col_output_sf = None

        acc_pipeline.consumer_wait(acc_consumer_state)
        iket.range_push("mxfp8_dfc2_epi_tile")

        subtile_cnt = self._cta_tile_n // EpilogueTileN  # 8 (256 / 32)
        start_subtile = 0
        tmem_t = self._subtile_dfc12_tmem_tensor(
            tmem_acc_tensor, cutlass.Int32(start_subtile), warp_idx,
        )
        tmem_forward_cols = EpilogueTileN

        rmem_sf = cute.make_rmem_tensor(
            cute.make_layout(2 * (self._cta_tile_n // EpilogueTileN)).shape, self.acc_dtype,
        )
        # fc1_recompute SF accumulator: ONE SF per subtile (recompute N = half of dfc2).
        rmem_sf_recompute = cute.make_rmem_tensor(
            cute.make_layout(self._cta_tile_n // EpilogueTileN).shape, self.acc_dtype,
        )
        # fc1_col_output SF accumulator
        rmem_sf_col_output = cute.make_rmem_tensor(
            cute.make_layout(2 * (self._cta_tile_n // EpilogueTileN)).shape, self.acc_dtype,
        )
        thread_in_warp = tidx % WarpThreadCount
        token_row_in_cta = cutlass.Int32(warp_idx * WarpThreadCount) + thread_in_warp
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile
        expert_local_token_idx = (
            work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m) + token_row_in_cta
        )

        # beta / prob / dprob setup
        beta_val = gmem_beta[work_tile_info.expert_idx]
        # mProb: load from topk_scores for valid tokens; default 1.0 (unused) for invalid.
        rmem_prob = cute.make_rmem_tensor(cute.make_layout(1).shape, self.acc_dtype)
        rmem_prob[0] = cutlass.Float32(1.0)
        if token_row_in_cta < valid_tokens:
            real_topk, _ = sched_ext.get_gmem_tensor("topk", gmem_topk_scores, work_tile_info)
            rmem_prob[0] = real_topk[expert_local_token_idx]

        # Per-tile dprob accumulator (single scalar).
        dprob = cutlass.Float32(0.0)

        # Output col-strips per N-tile.
        n_col_strips_per_tile = (self._cta_tile_n * 2) // EpilogueTileN
        base_token_tile = work_tile_info.tile_m_idx  # 128-row tile index

        _epilog_sync = pipeline.NamedBarrier(
            barrier_id=self._epilog_sync_bar_id,
            num_threads=WarpThreadCount * len(self._epilogue_warp_ids),
        )

        # Build tiled copies for SMEM↔REG (reference pattern).
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self._mma_tiler,
            self.fc1_output_layout,
            self.fc1_output_dtype,
            self.acc_dtype,
            self.d_epi_tile,
            self._use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(
            tmem_acc_tensor[((None, None), 0, 0)],
            self.d_epi_tile,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_rAcc_full = thr_copy_t2r.partition_D(tAcc_epi)
        copy_atom_s2r = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16)
        tiled_copy_s2r = cute.make_tiled_copy_D(copy_atom_s2r, tiled_copy_t2r)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        tRS_sPre = thr_copy_s2r.partition_D(smem_preact_buffer)

        r_layout = cute.make_layout(((1, EpilogueTileN,), 1, 1,), stride=((0, 1,), 0, 0,))
        r_gate_bf = cute.make_rmem_tensor(r_layout, cutlass.BFloat16)
        r_up_bf   = cute.make_rmem_tensor(r_layout, cutlass.BFloat16)
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.fc1_output_layout, self.fc1_output_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)

        for i in cutlass.range(0, subtile_cnt, 1):
            subtile_idx = cutlass.Int32(i)
            c_consumer_state, subtile_dprob = self._run_dfc2_subtile(
                subtile_idx=subtile_idx,
                subtile_i=i,
                t_subtile=tmem_t,
                smem_d=smem_d_buffer,
                tiled_copy_r2s=tiled_copy_r2s,
                tiled_copy_s2r=tiled_copy_s2r,
                tRS_sPre=tRS_sPre,
                c_pipeline=c_pipeline,
                c_consumer_state=c_consumer_state,
                acc_pipeline=acc_pipeline,
                acc_consumer_state=acc_consumer_state,
                r_gate_bf=r_gate_bf,
                r_up_bf=r_up_bf,
                work_tile_info=work_tile_info,
                warp_idx=warp_idx,
                tidx=tidx,
                norm_const=norm_const,
                rmem_sf=rmem_sf,
                rmem_sf_recompute=rmem_sf_recompute,
                real_fc1_recompute=real_fc1_recompute,
                rmem_sf_col_output=rmem_sf_col_output,
                real_fc1_col_output=real_fc1_col_output,
                beta=beta_val,
                prob=rmem_prob[0],
                epilog_sync=_epilog_sync,
                d_pipeline=d_pipeline,
                d_num_stage=d_num_stage,
            )
            dprob = dprob + subtile_dprob

            tmem_t = self._advance_fc2_tmem_tensor(tmem_t, tmem_forward_cols)

            # BARRIER: fence_proxy makes R2S (stmatrix) writes visible to TMA
            cute.arch.fence_proxy("async.shared", space="cta")
            _epilog_sync.arrive_and_wait()

            # Compute GMEM tile pointers for this subtile.
            gate_col_idx = (
                work_tile_info.tile_n_idx * cutlass.Int32(n_col_strips_per_tile)
                + subtile_idx * cutlass.Int32(2)
            )
            g_gate = cute.local_tile(
                real_fc1_output,
                (self._cta_tile_m, EpilogueTileN, 1),
                (base_token_tile, gate_col_idx, cutlass.Int32(0)),
            )[(None, None, 0)]
            g_up = cute.local_tile(
                real_fc1_output,
                (self._cta_tile_m, EpilogueTileN, 1),
                (base_token_tile, gate_col_idx + cutlass.Int32(1), cutlass.Int32(0)),
            )[(None, None, 0)]
            g_col_gate = None
            g_col_up = None
            if cutlass.const_expr(self._dfc2_col_output):
                g_col_gate = cute.local_tile(
                    real_fc1_col_output,
                    (self._cta_tile_m, EpilogueTileN, 1),
                    (base_token_tile, gate_col_idx, cutlass.Int32(0)),
                )[(None, None, 0)]
                g_col_up = cute.local_tile(
                    real_fc1_col_output,
                    (self._cta_tile_m, EpilogueTileN, 1),
                    (base_token_tile, gate_col_idx + cutlass.Int32(1), cutlass.Int32(0)),
                )[(None, None, 0)]
            g_recompute = None
            if cutlass.const_expr(self._dfc2_recompute):
                recompute_col_idx = (
                    work_tile_info.tile_n_idx
                    * cutlass.Int32(self._cta_tile_n // EpilogueTileN)
                    + subtile_idx
                )
                g_recompute = cute.local_tile(
                    real_fc1_recompute,
                    (self._cta_tile_m, EpilogueTileN, 1),
                    (base_token_tile, recompute_col_idx, cutlass.Int32(0)),
                )[(None, None, 0)]
            # TMA issue (warp 0 only).
            d_outputs_per_stage = cutlass.const_expr(self._d_output_slots)
            d_n_stages = cutlass.const_expr(d_num_stage // d_outputs_per_stage)
            d_slot = cutlass.Int32(d_outputs_per_stage) * (
                cutlass.Int32(i) % cutlass.Int32(d_n_stages)
            )
            if warp_idx == self._epilogue_warp_ids[0]:
                self.tma_store_dfc2_outputs(
                    smem_d_buffer,
                    tma_atom_grad_y1,
                    g_gate,
                    g_up,
                    tma_atom_fc1_col_output,
                    g_col_gate,
                    g_col_up,
                    tma_atom_fc1_recompute,
                    g_recompute,
                    valid_tokens,
                    d_pipeline,
                    d_slot,
                )

        self._acc_pipeline_consumer_release(acc_pipeline, acc_consumer_state, True)

        valid_inter = real_fc1_output.shape[1]
        self._stg_sf_dfc2(rmem_sf, real_fc1_output_sf, work_tile_info, tidx, valid_inter)
    
        # fc1_recompute SFs
        if cutlass.const_expr(self._dfc2_recompute):
            valid_inter_recompute = real_fc1_recompute.shape[1]
            self._stg_sf_recompute(
                rmem_sf_recompute, real_fc1_recompute_sf,
                work_tile_info, tidx, valid_inter_recompute, valid_tokens,
            )
        # fc1_col_output SFs
        if cutlass.const_expr(self._dfc2_col_output):
            valid_inter_col_output = real_fc1_col_output.shape[1]
            self._stg_sf_col_output(
                rmem_sf_col_output, real_fc1_col_output_sf,
                work_tile_info, tidx, valid_inter_col_output, valid_tokens,
            )
        # MegaMoE maps the receiver-pool row back to the source rank's combine slot.
        if cutlass.const_expr(token_comm_args is not None):
            if token_row_in_cta < valid_tokens:
                pool_token_global = (
                    work_tile_info.cumulative_data_physical_row
                    + work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m)
                    + token_row_in_cta
                )
                metadata_u32 = cute.recast_tensor(
                    token_comm_args.token_src_metadata, cutlass.Uint32,
                )
                dprob_output_dest = Fc2OutputDest(
                    tensor=token_comm_args.dprob_output,
                    metadata=metadata_u32,
                    peer_rank_ptr_mapper=token_comm_args.peer_rank_ptr_mapper,
                )
                dest_row = dprob_output_dest.resolve_token_row(pool_token_global)
                dprob_reduce_gmem(
                    dest_row,
                    dprob,
                    True,
                    cutlass.Int32(0),
                    system_scope=True,
                )
        else:
            dprob_reduce_gmem(
                real_dprob,
                dprob,
                token_row_in_cta < valid_tokens,
                expert_local_token_idx,
            )
        iket.range_pop()
        return c_consumer_state

    @cute.jit
    def _dfc2_load_c(
        self,
        c_pipeline,
        c_consumer_state,
        tiled_copy_s2r,
        tRS_sPre: cute.Tensor,
        r_gate_bf,
        r_up_bf,
    ):
        """Consume gate then up c_pipeline stages into register tiles."""
        c_pipeline.consumer_wait(c_consumer_state)
        c_slot = 2 * c_consumer_state.index
        cute.copy(
            tiled_copy_s2r,
            tRS_sPre[(None, None, None, c_slot)],
            r_gate_bf,
        )
        cute.copy(
            tiled_copy_s2r,
            tRS_sPre[(None, None, None, c_slot + 1)],
            r_up_bf,
        )
        cute.arch.fence_proxy("async.shared", space="cta")
        c_pipeline.consumer_release(c_consumer_state)
        c_consumer_state.advance()

        return c_consumer_state

    @cute.jit
    def _run_dfc2_subtile(
        self,
        subtile_idx,
        subtile_i,
        t_subtile: cute.Tensor,
        smem_d: cute.Tensor,
        tiled_copy_r2s,
        tiled_copy_s2r,
        tRS_sPre: cute.Tensor,
        c_pipeline,
        c_consumer_state,
        acc_pipeline,
        acc_consumer_state,
        r_gate_bf: cute.Tensor,
        r_up_bf: cute.Tensor,
        work_tile_info,
        warp_idx: int,
        tidx,
        norm_const,
        rmem_sf: cute.Tensor,
        rmem_sf_recompute: cute.Tensor,
        real_fc1_recompute: cute.Tensor,
        rmem_sf_col_output: cute.Tensor,
        real_fc1_col_output: cute.Tensor,
        beta: cutlass.Float32,
        prob: cutlass.Float32,
        epilog_sync,
        d_pipeline,
        d_num_stage,
    ):
        iket.range_push("mxfp8_dfc2_epilogue_subtile")
        EN = EpilogueTileN

        r_layout = cute.make_layout((((EN,), 1),), stride=(((1,), 0),))
        atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), self.acc_dtype,
        )
        r_acc = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
        cute.copy(atom_t2r, t_subtile, r_acc)

        thread_in_warp = tidx % WarpThreadCount
        token_row_in_cta = cutlass.Int32(warp_idx * WarpThreadCount) + thread_in_warp
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile
        subtile_dprob = cutlass.Float32(0.0)

        # load c from shared memory to registers
        c_consumer_state = self._dfc2_load_c(
            c_pipeline,
            c_consumer_state,
            tiled_copy_s2r,
            tRS_sPre,
            r_gate_bf,
            r_up_bf,
        )

        # c_gate / c_up declared outside the validity guard: stmatrix (tiled_copy_r2s)
        # is warp-cooperative and all threads must call it regardless of token validity.
        c_shape = cute.make_layout(((1, EN,), 1, 1), stride=((0, 1,), 0, 0)).shape
        c_gate = cute.make_rmem_tensor(c_shape, self.fc1_output_dtype)
        c_up   = cute.make_rmem_tensor(c_shape, self.fc1_output_dtype)
        # c_recompute: flat MXFP8 row for token-major output staging
        c_recompute = cute.make_rmem_tensor(cute.make_layout(EN).shape, self.fc1_output_dtype)

        is_valid_row = token_row_in_cta < valid_tokens

        r_gate = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
        r_up = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
        for j in cutlass.range_constexpr(EN):
            r_gate[j] = r_gate_bf[j].to(self.acc_dtype)
            r_up[j] = r_up_bf[j].to(self.acc_dtype)
        # Zero invalid rows' inputs (their r_acc/r_gate/r_up are padding garbage) so the
        # warp-wide column amax is not polluted and no NaN propagates into the reduction.
        if token_row_in_cta >= valid_tokens:
            for j in cutlass.range_constexpr(EN):
                r_acc[j] = self.acc_dtype(0.0)
                r_gate[j] = self.acc_dtype(0.0)
                r_up[j] = self.acc_dtype(0.0)

        # dswiglu backward: acc(grad_h) x (gate, up) -> (d_gate, d_up) ----
        d_gate = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
        d_up = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
        if cutlass.const_expr(self._act_func == "swiglu"):
            subtile_dprob = dswiglu_act(
                d_gate, d_up, r_acc, r_gate, r_up, beta, prob, self._gate_up_clamp
            )

        if cutlass.const_expr(self._dfc2_col_output):
            # Snapshot d_gate / d_up BEFORE quant_sfd_row mutates them in place; the col
            # path col-quants these copies (quant_sfd_col mutates its input).
            d_gate_col = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
            d_up_col   = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
            for j in cutlass.range_constexpr(EN):
                d_gate_col[j] = d_gate[j]
                d_up_col[j]   = d_up[j]
            c_gate_col = cute.make_rmem_tensor(cute.make_layout(EN).shape, self.fc1_output_dtype)
            c_up_col   = cute.make_rmem_tensor(cute.make_layout(EN).shape, self.fc1_output_dtype)
            qg_col = quant_sfd_col(
                d_gate_col, c_gate_col, norm_const,
                self._sf_vec_size, self.sf_dtype, self.fc1_output_dtype,
            )
            qu_col = quant_sfd_col(
                d_up_col, c_up_col, norm_const,
                self._sf_vec_size, self.sf_dtype, self.fc1_output_dtype,
            )
            for _k in cutlass.range_constexpr(self._cta_tile_n // EN):
                if subtile_idx == cutlass.Int32(_k):
                    rmem_sf_col_output[2 * _k]     = qg_col
                    rmem_sf_col_output[2 * _k + 1] = qu_col

        # quantize each half to MXFP8 + E8M0 row SF (per-thread, no warp reduction) ----
        qg = quant_sfd_row(
            d_gate, c_gate, norm_const, self._sf_vec_size, self.sf_dtype, self.fc1_output_dtype,
        )
        qu = quant_sfd_row(
            d_up, c_up, norm_const, self._sf_vec_size, self.sf_dtype, self.fc1_output_dtype,
        )

        # accumulate the 2 E8M0 row SFs into rmem
        for _k in cutlass.range_constexpr(self._cta_tile_n // EN):
            if subtile_idx == cutlass.Int32(_k):
                rmem_sf[2 * _k] = qg
                rmem_sf[2 * _k + 1] = qu

        # dfc2_recompute: forward swiglu + column quantization
        if cutlass.const_expr(self._dfc2_recompute):
            c_recompute_f32 = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
            if cutlass.const_expr(self._act_func == "swiglu"):
                swiglu_act(
                    c_recompute_f32,
                    r_up,
                    r_gate,
                    prob,
                    self._gate_up_clamp,
                )
            qc = quant_sfd_col(
                c_recompute_f32, c_recompute, norm_const,
                self._sf_vec_size, self.sf_dtype, self.fc1_output_dtype,
            )
            for _k in cutlass.range_constexpr(self._cta_tile_n // EN):
                if subtile_idx == cutlass.Int32(_k):
                    rmem_sf_recompute[_k] = qc

        # BARRIER: drain PREVIOUS subtile's TMA BEFORE R2S.
        if warp_idx == self._epilogue_warp_ids[0]:
            d_pipeline.producer_acquire()
        epilog_sync.arrive_and_wait()

        # Write d to smem.
        d_outputs_per_stage = cutlass.const_expr(self._d_output_slots)
        d_n_stages = cutlass.const_expr(d_num_stage // d_outputs_per_stage)
        d_slot = cutlass.Int32(d_outputs_per_stage) * (
            subtile_i % cutlass.Int32(d_n_stages)
        )
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        sd = thr_copy_r2s.partition_D(smem_d)
        cute.copy(tiled_copy_r2s, c_gate, sd[(None, None, None, d_slot)])
        cute.copy(tiled_copy_r2s, c_up,   sd[(None, None, None, d_slot + cutlass.Int32(1))])

        # Auxiliary data planes use the public token-major ABI.
        next_slot = d_slot + cutlass.Int32(2)
        if cutlass.const_expr(self._dfc2_col_output):
            s_col_gate = cute.slice_(smem_d, (token_row_in_cta, None, next_slot))
            s_col_up = cute.slice_(
                smem_d, (token_row_in_cta, None, next_slot + cutlass.Int32(1))
            )
            self._store_aux_row_smem(c_gate_col, s_col_gate)
            self._store_aux_row_smem(c_up_col, s_col_up)
            next_slot = next_slot + cutlass.Int32(2)
        if cutlass.const_expr(self._dfc2_recompute):
            s_recompute = cute.slice_(smem_d, (token_row_in_cta, None, next_slot))
            self._store_aux_row_smem(c_recompute, s_recompute)

        iket.range_pop()
        return c_consumer_state, subtile_dprob

    @cute.jit
    def _stg_sf_dfc2(
        self,
        rmem_sf_f32: cute.Tensor,
        real_fc1_output_sf: cute.Tensor,
        work_tile_info,
        tidx,
        valid_inter,
    ) -> None:
        """Store the dfc2 grad_y1 E8M0 row SFs, 4 blocks per 128-col region."""
        if tidx < work_tile_info.valid_tokens_in_cta_tile:
            token_idx = (
                work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m) + tidx
            )
            n_regions = (self._cta_tile_n * 2) // Fc1EpilogueOutputTileN
            region_col = (
                work_tile_info.tile_n_idx * cutlass.Int32(self._cta_tile_n * 2)
            )
            for r in cutlass.range_constexpr(n_regions):
                sf_base = cute.local_tile(
                    real_fc1_output_sf, (1, 1, 1),
                    (token_idx, region_col, cutlass.Int32(0)),
                )
                r_sf4_f32 = cute.make_rmem_tensor(cute.make_layout(4).shape, self.acc_dtype)
                for idx in cutlass.range_constexpr(4):
                    r_sf4_f32[idx] = rmem_sf_f32[r * 4 + idx]
                if region_col < valid_inter:
                    sf_ptr = cute.make_ptr(
                        self.sf_dtype,
                        sf_base.iterator.toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    )
                    gmem_sf4 = cute.make_tensor(sf_ptr, cute.make_layout(4))
                    r_sf4 = cute.make_rmem_tensor(cute.make_layout(4).shape, self.sf_dtype)
                    r_sf4.store(r_sf4_f32.load().to(self.sf_dtype))
                    cute.autovec_copy(r_sf4, gmem_sf4)
                region_col += cutlass.Int32(Fc1EpilogueOutputTileN)

    @cute.jit
    def _stg_col_sf_atom_value(
        self,
        real_sf: cute.Tensor,
        row_block,
        feature,
        _feature_atoms,
        sf_value,
    ) -> None:
        """Store one SF in a 128-feature × 4-token-block atom."""
        token_atom = row_block // cutlass.Int32(4)
        token_bank = row_block % cutlass.Int32(4)
        feature_atom = feature // cutlass.Int32(128)
        feature_bank = (feature // cutlass.Int32(32)) % cutlass.Int32(4)
        feature_lane = feature % cutlass.Int32(32)
        atom_byte = (
            feature_lane * cutlass.Int32(16)
            + feature_bank * cutlass.Int32(4)
            + token_bank
        )
        if feature_atom < real_sf.shape[0] and token_atom < real_sf.shape[1]:
            real_sf[feature_atom, token_atom, atom_byte] = sf_value.to(self.sf_dtype)

    @cute.jit
    def _stg_sf_recompute(
        self,
        rmem_sf_f32: cute.Tensor,
        real_fc1_recompute_sf: cute.Tensor,
        work_tile_info,
        tidx,
        valid_inter,
        valid_tokens,
    ) -> None:
        """Store fc1_recompute SFs in MN-major 128×4 atoms."""
        EN = EpilogueTileN  # 32
        sf_vec_size = self._sf_vec_size
        warp_lane_idx = tidx % cutlass.Int32(32)
        warp_idx_local = tidx // cutlass.Int32(32)
        hidden_atoms = (valid_inter + cutlass.Int32(127)) // cutlass.Int32(128)

        # Row-block within the M-tile: 4 warps × 1 row-block each (128 / 32).
        row_blocks_per_m_tile = self._cta_tile_m // sf_vec_size
        row_block = (
            work_tile_info.tile_m_idx * cutlass.Int32(row_blocks_per_m_tile)
            + warp_idx_local
        )
        col_base = (
            work_tile_info.tile_n_idx * cutlass.Int32(self._cta_tile_n)
        )
        for s in cutlass.range_constexpr(self._cta_tile_n // EN):
            col = col_base + cutlass.Int32(s * EN) + warp_lane_idx
            if col < valid_inter:
                self._stg_col_sf_atom_value(
                    real_fc1_recompute_sf,
                    row_block,
                    col,
                    hidden_atoms,
                    rmem_sf_f32[s],
                )

    @cute.jit
    def _stg_sf_col_output(
        self,
        rmem_sf_f32: cute.Tensor,
        real_fc1_col_output_sf: cute.Tensor,
        work_tile_info,
        tidx,
        valid_inter,
        valid_tokens,
    ) -> None:
        """Store fc1_col_output SFs in MN-major 128×4 atoms."""
        EN = EpilogueTileN  # 32
        sf_vec_size = self._sf_vec_size
        warp_lane_idx = tidx % cutlass.Int32(32)
        warp_idx_local = tidx // cutlass.Int32(32)
        hidden_atoms = (valid_inter + cutlass.Int32(127)) // cutlass.Int32(128)

        row_blocks_per_m_tile = self._cta_tile_m // sf_vec_size
        row_block = (
            work_tile_info.tile_m_idx * cutlass.Int32(row_blocks_per_m_tile)
            + warp_idx_local
        )
        # Doubled N: cta_tile_n * 2 cols per fc1 N-tile.
        col_base = (
            work_tile_info.tile_n_idx * cutlass.Int32(self._cta_tile_n * 2)
        )
        for s in cutlass.range_constexpr(self._cta_tile_n // EN):
            for gu in cutlass.range_constexpr(2):
                col = col_base + cutlass.Int32((2 * s + gu) * EN) + warp_lane_idx
                if col < valid_inter:
                    self._stg_col_sf_atom_value(
                        real_fc1_col_output_sf,
                        row_block,
                        col,
                        hidden_atoms,
                        rmem_sf_f32[2 * s + gu],
                    )

    @cute.jit
    def _tma_store_tile(
        self,
        smem_tile: cute.Tensor,
        tma_atom: cute.CopyAtom,
        gmem_tile: cute.Tensor,
    ) -> None:
        tma_smem_src, tma_gmem_dst = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem_tile, 0, 2),
            cute.group_modes(gmem_tile, 0, 2),
        )
        cute.copy(tma_atom, tma_smem_src, tma_gmem_dst)

    @cute.jit
    def tma_store_dfc2_outputs(
        self,
        smem_d_buffer: cute.Tensor,
        tma_atom_grad_y1: cute.CopyAtom,
        g_gate_2d: cute.Tensor,
        g_up_2d: cute.Tensor,
        tma_atom_col_output: cute.CopyAtom,
        g_col_gate_2d,
        g_col_up_2d,
        tma_atom_recompute: cute.CopyAtom,
        g_recompute_2d,
        valid_tokens,
        d_pipeline,
        d_slot,
    ) -> None:
        """Issue one TMA store group for every dFC2 data plane."""
        tile_is_valid = valid_tokens > cutlass.Int32(0)
        if tile_is_valid:
            self._tma_store_tile(
                cute.slice_(smem_d_buffer, (None, None, d_slot)),
                tma_atom_grad_y1,
                g_gate_2d,
            )
            self._tma_store_tile(
                cute.slice_(
                    smem_d_buffer, (None, None, d_slot + cutlass.Int32(1))
                ),
                tma_atom_grad_y1,
                g_up_2d,
            )
            next_slot = d_slot + cutlass.Int32(2)
            if cutlass.const_expr(self._dfc2_col_output):
                self._tma_store_tile(
                    cute.slice_(smem_d_buffer, (None, None, next_slot)),
                    tma_atom_col_output,
                    g_col_gate_2d,
                )
                self._tma_store_tile(
                    cute.slice_(
                        smem_d_buffer,
                        (None, None, next_slot + cutlass.Int32(1)),
                    ),
                    tma_atom_col_output,
                    g_col_up_2d,
                )
                next_slot = next_slot + cutlass.Int32(2)
            if cutlass.const_expr(self._dfc2_recompute):
                self._tma_store_tile(
                    cute.slice_(smem_d_buffer, (None, None, next_slot)),
                    tma_atom_recompute,
                    g_recompute_2d,
                )
        d_pipeline.producer_commit()


    @cute.jit
    def _subtile_dfc12_tmem_tensor(
        self,
        tmem_acc_tensor: cute.Tensor,
        subtile_idx,
        warp_idx,
    ) -> cute.Tensor:
        """
        Per-warp TMEM view for one fc2 subtile (EpilogueTileN=32 cols).
        """
        base = tmem_acc_tensor.iterator
        warp_lane_off = warp_idx * WarpThreadCount
        subtile_col_off = subtile_idx * EpilogueTileN
        total = (warp_lane_off << 16) + subtile_col_off
        subtile_ptr = base + cute.assume(total, divby=16)
        return cute.make_tensor(
            subtile_ptr,
            _TmemTranspose16x32Core._tmem_layout(32, EpilogueTileN),
        )

    @cute.jit
    def _advance_fc2_tmem_tensor(
        self,
        tmem_tensor: cute.Tensor,
        col_offset: int,
    ) -> cute.Tensor:
        new_ptr = tmem_tensor.iterator + cute.assume(col_offset, divby=16)
        return cute.make_tensor(
            new_ptr,
            _TmemTranspose16x32Core._tmem_layout(32, EpilogueTileN),
        )

    @cute.jit
    def _acc_pipeline_consumer_release(
        self,
        acc_pipeline,
        acc_consumer_state,
        is_release: bool,
    ) -> None:
        """Release the acc pipeline consumer."""
        if is_release:
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

    @cute.jit
    def _run_dfc1_subtile(
        self,
        subtile_idx,
        subtile_i,
        t_subtile: cute.Tensor,
        real_fc2_output: cute.Tensor,
        work_tile_info,
        valid_hidden,
        warp_idx: int,
        tidx,
        acc_pipeline,
        acc_consumer_state,
        token_comm_args=None,
        rmem_sf_dfc1=None,
        *,
        preload_acc=None,
    ) -> None:
        """fc2 subtile: LDTM + fp32->bf16 + STG."""
        iket.range_push("mxfp8_fc2_epilogue_subtile")
        dfc1_subtile_cnt = self._cta_tile_n // EpilogueTileN  # = 8
        r_acc_layout = cute.make_layout((((EpilogueTileN,), 1),), stride=(((1,), 0),))
        atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), self.acc_dtype,
        )
        r_acc = cute.make_rmem_tensor(r_acc_layout.shape, self.acc_dtype)
        cute.copy(atom_t2r, t_subtile, r_acc)

        hidden_group = (
            work_tile_info.tile_n_idx * cutlass.Int32(dfc1_subtile_cnt) + subtile_idx
        )
        hidden_col_start = (
            work_tile_info.tile_n_idx * cutlass.Int32(self._cta_tile_n)
            + subtile_idx * cutlass.Int32(EpilogueTileN)
        )
        r_bf16 = cute.make_rmem_tensor(r_acc_layout.shape, cutlass.BFloat16)
        r_bf16.store(r_acc.load().to(cutlass.BFloat16))
        thread_in_warp = tidx % WarpThreadCount
        token_row_in_cta = cutlass.Int32(warp_idx * WarpThreadCount) + thread_in_warp
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile
        is_valid = token_row_in_cta < valid_tokens and hidden_col_start < valid_hidden

        if cutlass.const_expr(
            token_comm_args is not None
            and not self._token_back_by_dispatch
            and self._combine_mxfp8
        ):
            fp8_dtype = self._combine_format.act_dtype
            r_fp8 = cute.make_rmem_tensor(r_acc_layout.shape, fp8_dtype)
            qpvscale = quant_sfd_row(
                r_acc, r_fp8, 1.0, EpilogueTileN,
                cutlass.Float8E8M0FNU, fp8_dtype,
            )
            pool_token_global = (
                work_tile_info.cumulative_data_physical_row
                + work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m)
                + token_row_in_cta
            )
            metadata_u32 = cute.recast_tensor(
                token_comm_args.token_src_metadata, cutlass.Uint32,
            )
            fc2_output_dest = Fc2OutputDest(
                tensor=token_comm_args.combine_output,
                metadata=metadata_u32,
                peer_rank_ptr_mapper=token_comm_args.peer_rank_ptr_mapper,
            )
            dest_row = fc2_output_dest.resolve_token_row(pool_token_global)
            r_fp8_flat = cute.make_tensor(r_fp8.iterator, cute.make_layout(32))
            stg_fp8_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), fp8_dtype, num_bits_per_copy=256,
            )
            dest_fp8_ptr = cute.make_ptr(
                fp8_dtype,
                dest_row.iterator.toint() + Int64(hidden_col_start),
                cute.AddressSpace.gmem,
                assumed_align=32,
            )
            if is_valid:
                cute.copy(
                    stg_fp8_atom, r_fp8_flat,
                    cute.make_tensor(dest_fp8_ptr, cute.make_layout(32)),
                )
                self._write_sf_dfc1_buffer(rmem_sf_dfc1, subtile_idx, qpvscale)
        elif cutlass.const_expr(
            self._token_back_by_dispatch and self._combine_mxfp8
        ):
            pool_token_global = (
                work_tile_info.cumulative_data_physical_row
                + work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m)
                + token_row_in_cta
            )
            fp8_dtype = self._combine_format.act_dtype
            r_fp8 = cute.make_rmem_tensor(r_acc_layout.shape, fp8_dtype)
            qpvscale = quant_sfd_row(
                r_acc, r_fp8, 1.0, EpilogueTileN,
                cutlass.Float8E8M0FNU, fp8_dtype,
            )
            fp8_byte_addr = (
                token_comm_args.fc2_output_workspace.iterator.toint()
                + Int64(pool_token_global) * Int64(self._hidden_dfc1)
                + Int64(hidden_col_start)
            )
            stg_fp8_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), fp8_dtype, num_bits_per_copy=256,
            )
            aligned_fp8_iter = cute.make_ptr(
                fp8_dtype,
                fp8_byte_addr,
                cute.AddressSpace.gmem,
                assumed_align=32,
            )
            r_fp8_flat = cute.make_tensor(r_fp8.iterator, cute.make_layout(EpilogueTileN))
            if is_valid:
                cute.copy(
                    stg_fp8_atom, r_fp8_flat,
                    cute.make_tensor(aligned_fp8_iter, cute.make_layout(EpilogueTileN)),
                )
                self._write_sf_dfc1_buffer(rmem_sf_dfc1, subtile_idx, qpvscale)
        else:
            # BF16 path (default): fp32->bf16, two 256-bit STGs.
            stg_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=256,
            )
            if cutlass.const_expr(
                token_comm_args is not None
                and not self._token_back_by_dispatch
            ):
                pool_token_global = (
                    work_tile_info.cumulative_data_physical_row
                    + work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m)
                    + token_row_in_cta
                )
                metadata_u32 = cute.recast_tensor(
                    token_comm_args.token_src_metadata, cutlass.Uint32,
                )
                fc2_output_dest = Fc2OutputDest(
                    tensor=token_comm_args.combine_output,
                    metadata=metadata_u32,
                    peer_rank_ptr_mapper=token_comm_args.peer_rank_ptr_mapper,
                    # Collapse topk -> src_topk=0 so every contribution of a
                    # source token resolves to the SAME combine row (red-added
                    # below).  No-op when reduce is off.
                    reduce_topk_in_kernel=self._reduce_topk_in_epilogue,
                )
                dest_row = fc2_output_dest.resolve_token_row(pool_token_global)
            for stg_half in cutlass.range(EpilogueTileN // 16, unroll_full=True):
                reg_view = cute.make_tensor(
                    r_bf16.iterator + stg_half * 16,
                    cute.make_layout(16),
                )
                if cutlass.const_expr(
                    token_comm_args is not None
                    and not self._token_back_by_dispatch
                ):
                    # epi_warps: peer-write grad_x directly to combine_output
                    hidden_off = hidden_col_start + cutlass.Int32(stg_half * 16)
                    if cutlass.const_expr(self._reduce_topk_in_epilogue):
                        if is_valid:
                            reg_u32 = cute.recast_tensor(reg_view, cutlass.Uint32)
                            for redg_i in cutlass.range_constexpr(16 // 4):
                                chunk_ptr = cute.make_ptr(
                                    cutlass.BFloat16,
                                    dest_row.iterator.toint()
                                    + (hidden_off + cutlass.Int32(redg_i * 4))
                                    * cutlass.Int64(2),
                                    cute.AddressSpace.gmem,
                                    assumed_align=8,
                                )
                                _red_add_relaxed_sys_v2_bf16x2(
                                    chunk_ptr,
                                    cutlass.Uint32(reg_u32[2 * redg_i]),
                                    cutlass.Uint32(reg_u32[2 * redg_i + 1]),
                                )
                    else:
                        dest_ptr = cute.make_ptr(
                            cutlass.BFloat16,
                            dest_row.iterator.toint() + hidden_off * cutlass.Int64(2),
                            cute.AddressSpace.gmem,
                            assumed_align=32,
                        )
                        if is_valid:
                            cute.copy(
                                stg_atom, reg_view,
                                cute.make_tensor(dest_ptr, cute.make_layout(16)),
                            )
                else:
                    # Lean path (token_comm_args is None) OR dispatch-push
                    # (token_back_by_dispatch)
                    g_fc2_output_tile = cute.local_tile(
                        real_fc2_output,
                        (self._cta_tile_m, EpilogueTileN, 1),
                        (work_tile_info.tile_m_idx, hidden_group, 0),
                    )
                    g_fc2_slice = cute.slice_(g_fc2_output_tile, (None, None, 0))
                    g_thread_row = cute.local_tile(
                        g_fc2_slice, (1, 16), (token_row_in_cta, stg_half),
                    )
                    g_flat = cute.coalesce(g_thread_row)
                    aligned_iter = cute.make_ptr(
                        cutlass.BFloat16,
                        g_flat.iterator.toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=32,
                    )
                    if is_valid:
                        cute.copy(stg_atom, reg_view, cute.make_tensor(aligned_iter, g_flat.layout))

        iket.range_pop()

    @cute.jit
    def _write_sf_dfc1_buffer(self, rmem_sf_dfc1, subtile_idx, qpvscale) -> None:
        """Scatter one subtile's E8M0 scale into the per-tile SF buffer."""
        for j in cutlass.range_constexpr(self._cta_tile_n // EpilogueTileN):
            if subtile_idx == cutlass.Int32(j):
                rmem_sf_dfc1[j] = qpvscale

    @cute.jit
    def _stg_sf_dfc1(
        self,
        rmem_sf_dfc1: cute.Tensor,
        token_comm_args,
        work_tile_info,
        valid_hidden,
        warp_idx: int,
        tidx,
    ) -> None:
        """Flush a task tile's dfc1 E8M0 scales to local fc2_output_sf."""
        dfc1_subtile_cnt = self._cta_tile_n // EpilogueTileN
        thread_in_warp = tidx % WarpThreadCount
        token_row_in_cta = cutlass.Int32(warp_idx * WarpThreadCount) + thread_in_warp
        if token_row_in_cta < work_tile_info.valid_tokens_in_cta_tile:
            pool_token_global = (
                work_tile_info.cumulative_data_physical_row
                + work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m)
                + token_row_in_cta
            )
            hidden_group_base = (
                work_tile_info.tile_n_idx * cutlass.Int32(dfc1_subtile_cnt)
            )
            sf_byte_addr = (
                token_comm_args.fc2_output_sf.iterator.toint()
                + Int64(pool_token_global) * Int64(self._dfc1_sf_block_pad)
                + Int64(hidden_group_base)
            )
            if cutlass.const_expr(self._dfc1_sf_batch8):
                stg_e8m0x8_from_f32(
                    sf_byte_addr,
                    rmem_sf_dfc1[0], rmem_sf_dfc1[1], rmem_sf_dfc1[2], rmem_sf_dfc1[3],
                    rmem_sf_dfc1[4], rmem_sf_dfc1[5], rmem_sf_dfc1[6], rmem_sf_dfc1[7],
                )
            else:
                for j in cutlass.range_constexpr(dfc1_subtile_cnt):
                    block_hidden_start = (
                        work_tile_info.tile_n_idx * cutlass.Int32(self._cta_tile_n)
                        + cutlass.Int32(j * EpilogueTileN)
                    )
                    if block_hidden_start < valid_hidden:
                        stg_e8m0_from_f32(sf_byte_addr + Int64(j), rmem_sf_dfc1[j])


    @cute.jit
    def _run_dfc1_task_tile(
        self,
        work_tile_info,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        sched_ext,
        gmem_fc2_output: cute.Tensor,
        valid_hidden,
        warp_idx: int,
        tidx,
        token_comm_args=None,
    ) -> None:
        """fc2 (Linear2) task-tile body following fc1 pattern exactly."""
        real_fc2_output, _ = sched_ext.get_gmem_tensor(
            "d", gmem_fc2_output, work_tile_info,
        )
        acc_pipeline.consumer_wait(acc_consumer_state)
        iket.range_push("mxfp8_dfc1_epi_tile")

        dfc1_subtile_cnt = self._cta_tile_n // EpilogueTileN  # = 8

        # Start subtile mirrors fc1: last for odd turn, first for even.
        start_subtile = 0
        tmem_t = self._subtile_dfc12_tmem_tensor(
            tmem_acc_tensor, cutlass.Int32(start_subtile), warp_idx,
        )
        tmem_forward_cols = EpilogueTileN

        # Quantized combine: buffer the per-subtile E8M0 scales and flush them
        # in one stg.64 after the loop (see _stg_sf_dfc1).
        if cutlass.const_expr(self._combine_mxfp8 and token_comm_args is not None):
            layout_sf_dfc1 = cute.make_layout(dfc1_subtile_cnt)
            rmem_sf_dfc1 = cute.make_rmem_tensor(layout_sf_dfc1.shape, self.acc_dtype)
        else:
            rmem_sf_dfc1 = None

        for i in cutlass.range(0, dfc1_subtile_cnt, 1, unroll=1):
            self._run_dfc1_subtile(
                subtile_idx=cutlass.Int32(i),
                subtile_i=i,
                t_subtile=tmem_t,
                real_fc2_output=real_fc2_output,
                work_tile_info=work_tile_info,
                valid_hidden=valid_hidden,
                warp_idx=warp_idx,
                tidx=tidx,
                acc_pipeline=acc_pipeline,
                acc_consumer_state=acc_consumer_state,
                token_comm_args=token_comm_args,
                rmem_sf_dfc1=rmem_sf_dfc1,
            )

            tmem_t = self._advance_fc2_tmem_tensor(tmem_t, tmem_forward_cols)

        # Release AFTER all subtile reads (never early-release for FC2).
        self._acc_pipeline_consumer_release(acc_pipeline, acc_consumer_state, True)

        # Flush the buffered E8M0 scales (one stg.64 per thread when aligned).
        if cutlass.const_expr(self._combine_mxfp8 and token_comm_args is not None):
            self._stg_sf_dfc1(
                rmem_sf_dfc1=rmem_sf_dfc1,
                token_comm_args=token_comm_args,
                work_tile_info=work_tile_info,
                valid_hidden=valid_hidden,
                warp_idx=warp_idx,
                tidx=tidx,
            )

        iket.range_pop()


    @cute.jit
    def run(
        self,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        sched_consumer,
        sched_ext,
        gmem_fc1_output: cute.Tensor,
        gmem_fc1_output_sf: cute.Tensor,
        tma_atom_fc1_recompute: cute.CopyAtom,
        gmem_fc1_recompute: Optional[cute.Tensor],
        gmem_fc1_recompute_sf: Optional[cute.Tensor],
        tma_atom_fc1_col_output: cute.CopyAtom,
        gmem_fc1_col_output: Optional[cute.Tensor],
        gmem_fc1_col_output_sf: Optional[cute.Tensor],
        smem_preact_buffer: cute.Tensor,
        c_pipeline,
        c_num_stage,
        smem_d_buffer: cute.Tensor,
        d_pipeline,
        d_num_stage,
        tma_atom_grad_y1: cute.CopyAtom,
        gmem_topk_scores: cute.Tensor,
        gmem_fc2_output: cute.Tensor,
        gmem_fc1_done_counter: cute.Tensor,
        warp_idx: int,
        tidx,
        alpha,
        norm_const,
        gmem_beta: cute.Tensor,
        gmem_dprob: cute.Tensor,
        token_comm_args=None,
    ) -> None:
        """
        Run the full MXFP8 dfc2+dfc1-fused (backward) epilogue task-tile loop.
        """
        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self._num_acc_pipeline_stages
        )
        task_tile_boundary_bar = pipeline.NamedBarrier(
            barrier_id=self._epilog_sync_bar_id,
            num_threads=32 * len(self._epilogue_warp_ids),
        )

        valid_hidden = cutlass.Int32(gmem_fc2_output.shape[1])

        c_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, c_num_stage
        )

        bidx, bidy, bidz = cute.arch.block_idx()
        work_tile_info = sched_consumer.consume_work()

        flag_tracker = GpuReleaseFlagBatchTracker(
            flag_address=Int64(0),
            accumulated_flags=cutlass.Int32(0),
            phase=cutlass.Int32(work_tile_info.phase),
            thread_idx=tidx % (len(self._epilogue_warp_ids) * WarpThreadCount),
        )

        while work_tile_info.is_valid_tile:
            acc_stage_index = acc_consumer_state.index
            tmem_acc_stage_tesnor = tmem_acc_tensor[(None, None, None, acc_stage_index)]

            if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                c_consumer_state = self._run_dfc2_task_tile(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_stage_tesnor,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    sched_ext=sched_ext,
                    gmem_fc1_output=gmem_fc1_output,
                    gmem_fc1_output_sf=gmem_fc1_output_sf,
                    tma_atom_fc1_recompute=tma_atom_fc1_recompute,
                    gmem_fc1_recompute=gmem_fc1_recompute,
                    gmem_fc1_recompute_sf=gmem_fc1_recompute_sf,
                    tma_atom_fc1_col_output=tma_atom_fc1_col_output,
                    gmem_fc1_col_output=gmem_fc1_col_output,
                    gmem_fc1_col_output_sf=gmem_fc1_col_output_sf,
                    c_pipeline=c_pipeline,
                    smem_preact_buffer=smem_preact_buffer,
                    c_consumer_state=c_consumer_state,
                    smem_d_buffer=smem_d_buffer,
                    tma_atom_grad_y1=tma_atom_grad_y1,
                    warp_idx=warp_idx,
                    tidx=tidx,
                    norm_const=norm_const,
                    gmem_topk_scores=gmem_topk_scores,
                    gmem_beta=gmem_beta,
                    gmem_dprob=gmem_dprob,
                    d_pipeline=d_pipeline,
                    d_num_stage=d_num_stage,
                    token_comm_args=token_comm_args,
                )
            else:
                self._run_dfc1_task_tile(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_stage_tesnor,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    sched_ext=sched_ext,
                    gmem_fc2_output=gmem_fc2_output,
                    valid_hidden=valid_hidden,
                    warp_idx=warp_idx,
                    tidx=tidx,
                    token_comm_args=token_comm_args,
                )

            acc_consumer_state.advance()

            cur_was_linear1 = work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
            cur_fc1_counter_slot = (
                work_tile_info.cumulative_token_block_count
                + work_tile_info.tile_m_idx // cutlass.Int32(self._atom_thr_size)
            )
            cur_fc2_expert_idx = work_tile_info.expert_idx

            work_tile_info = sched_consumer.consume_work()

            if cur_was_linear1:
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)
                cute.arch.fence_proxy("async")
                cute.arch.fence_acq_rel_gpu()

            task_tile_boundary_bar.arrive_and_wait()

            if cur_was_linear1:
                flag_tracker = flag_tracker.accumulate(
                    work_tile_info.phase,
                    self._epi_fc1_batch,
                    (gmem_fc1_done_counter.iterator + cur_fc1_counter_slot).toint(),
                )
            else:
                if cutlass.const_expr(
                    self._token_back_by_dispatch or self._combine_mxfp8
                ):
                    # Fence before (deferred) counter release: make the fc2
                    # pool-output STG writes device-visible.
                    cute.arch.fence_acq_rel_gpu()
                    fc2_flag_addr = (
                        token_comm_args.fc2_done_counter.iterator + cur_fc2_expert_idx
                    ).toint()
                else:
                    fc2_flag_addr = Int64(0)
                no_fire: cutlass.Constexpr = not (
                    self._token_back_by_dispatch or self._combine_mxfp8
                )
                flag_tracker = flag_tracker.accumulate(
                    work_tile_info.phase,
                    self._epi_fc2_batch,
                    fc2_flag_addr,
                    no_fire,
                )

        flag_tracker.fire()

        d_pipeline.producer_tail()
