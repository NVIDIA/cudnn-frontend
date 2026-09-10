# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MoE Block-Scaled Grouped GEMM Kernel — Weight Gradient (2Dx2D) with a
SECOND-LEVEL A scale (SFA2 only; wgrad has no SFB2).

Computes:  A(hidden, tokens_sum) x B(tokens_sum, intermediate)
        -> C(experts, hidden, intermediate)

where C is the weight gradient.  K (tokens) varies per expert;
M (hidden) and N (intermediate) are fixed across all experts.

Copy-and-extend of kernels/wgrad_baseline/kernel.py with the dgrad_quant
kernel's accumulator-splitting machinery grafted in: K is partitioned into
sgk-token scale blocks (k_tile_same_scale_factor = sgk / mma_tiler_k MMA
K-tiles each); the MMA warp produces one CLEAN TMEM partial accumulator per
block; a dedicated 4-warp accumulator-update warpgroup rescales each partial
by its per-M-row f32 SFA2 (sgm = 1: one scale per hidden row per sgk-token
group, the dgrad_dglu_rht_2 producer's rht_quant_sf2 grid) and sums in f32
registers; the running tile lands in an SMEM sFinalAcc buffer that the
epilogue warpgroup reads instead of TMEM. A/SFA/B/SFB TMA loads, the
per-expert tensormap helper kernel, and the CLC scheduler are byte-identical
to wgrad_baseline; SFA2 rides a cp.async (LDGSTS) pipeline on a dedicated
scale warp — 12 warps = 3 whole 128-thread warpgroups (setmaxregister-safe).

Ragged K x sgk: the wrapper enforces token_counts[e] % sgk == 0 (the RHT
producer's own gate), so every expert's k_tile_cnt divides exactly into
whole scale blocks and an expert's SFA2 columns are a domain_offset by
token_offset // sgk on the global (hidden, tokens_sum/sgk) grid.

Supports (unchanged from wgrad_baseline):
    - CLC-based dynamic persistent tile scheduling
    - Dense (contiguous 3-D C) / Discrete (per-expert pointer array C) output
    - accumulate_on_output (TMA reduce for atomic accumulation)
    - NVFP4 global_scale support (epilogue alpha)
    - k_tile_cnt == 0 handling (zero output for empty experts — the
      acc-update warpgroup zero-fills and still produces the epi stage)

Scheduler: moe_persistent_scheduler.py (CLC mode, scenario="2Dx2D")
Extension: moe_sched_extension.WgradScaledGemmSchedExtension (sgk-aware "sfa2" branch)
Source provenance: bs_ggemm_harness kernels/wgrad/kernel.py
"""

import re
from importlib.metadata import PackageNotFoundError, version
from typing import Literal, Type, Tuple, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from ..moe_persistent_scheduler import (
    MoEPersistentTileScheduler,
    MoESchedulerParams,
    MoEWorkTileInfo,
)
from ..moe_utils import (
    MoEWeightMode,
    WGradInputOrder,
    WgradSfTensormapConstructor,
)
from ..moe_sched_extension import (
    WgradScaledGemmSchedExtension,
)
from ..moe_kernel_helpers import (
    compute_stages_wgrad_2nd_level,
    epilog_gmem_copy_and_partition,
)

# Same launch-config space as wgrad_baseline; 2-CTA instructions pair with
# the 256-M tilers. N=256 tilers run with 1 TMEM acc stage and 1 AB stage
# (SMEM: full-tile f32 sFinalAcc) — correct but unpipelined, like
# dgrad_quant's (256, 256) config.
VALID_CTA_SHAPES = ((128, 128), (256, 128))
VALID_CLUSTER_SHAPES = (
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 4),
    (4, 1),
    (2, 4),
    (4, 2),
    (4, 4),
)
DEFAULT_CTA_SHAPE = (256, 128)
DEFAULT_CLUSTER_SHAPE = (2, 1)


def _using_internal_cutlass_dsl() -> bool:
    try:
        version("nvidia-cutlass-dsl-internal")
    except PackageNotFoundError:
        return False
    return True


def _cutlass_dsl_needs_fp4_layout_workaround() -> bool:
    # Public cutlass-dsl wheels before 4.8 interpret packed sub-byte
    # from_dlpack layouts in byte units, so the FP4 A/B layouts must be
    # recast to element units.
    if _using_internal_cutlass_dsl():
        return False
    match = re.match(r"(\d+)\.(\d+)", getattr(cutlass, "__version__", "") or "")
    if match is None:
        return False
    return (int(match.group(1)), int(match.group(2))) < (4, 8)


_NEEDS_FP4_LAYOUT_WORKAROUND = _cutlass_dsl_needs_fp4_layout_workaround()


class BlockScaledSubChannelMoEGroupedGemmWgradKernel:
    """Block-scaled grouped GEMM kernel for MoE weight gradient (2Dx2D) with
    a K-grouped second-level A scale (SFA2 only).

    :param sf_vec_size: First-level scale-factor vector size (16 or 32).
    :param sgk: Second-level scale group size along K (tokens); one f32 SFA2
        per (1 M row x sgk tokens). Must be a multiple of the MMA K tile
        (256 for NVFP4) — sgk in {256, 512}.
    :param acc_dtype: Accumulator data type (Float32).
    :param use_2cta_instrs: Use 2-CTA MMA instructions.
    :param mma_tiler_mn: MMA tile shape (M, N).
    :param cluster_shape_mn: Cluster shape (M, N).
    :param accumulate_on_output: Use TMA reduce for atomic accumulation.
    :param expert_cnt: Number of experts.
    :param weight_mode: ``MoEWeightMode.DENSE`` or ``MoEWeightMode.DISCRETE`` for output.
    :param input_order: ``WGradInputOrder.Tensor2D`` (default) or
        ``WGradInputOrder.TensorRagged`` — see wgrad_baseline.
    """

    FIX_PAD_SIZE = 256  # every expert's token count (K_e) must be 256-aligned
    # sgm is fixed: SFA2 is per-M-row (the rht_quant_sf2 grid)
    SGM = 1

    def __init__(
        self,
        sf_vec_size: int,
        sgk: int,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        use_2cta_instrs: bool = False,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        accumulate_on_output: bool = False,
        expert_cnt: int = 1,
        weight_mode: MoEWeightMode = MoEWeightMode.DENSE,
        input_order: WGradInputOrder = WGradInputOrder.Tensor2D,
        sf_fp8_dtype_override: Optional[Literal["e5m3"]] = None,
    ):
        self.sf_vec_size = sf_vec_size
        self.sf_dtype_override: Optional[Type[cutlass.Numeric]] = cutlass.FloatNV8E5M3FNU if sf_fp8_dtype_override == "e5m3" else None
        self.sgm = self.SGM
        self.sgk = sgk
        self.expert_cnt = expert_cnt
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.accumulate_on_output = accumulate_on_output
        self.weight_mode = weight_mode
        self.input_order = input_order

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

        # Warp specialization (the dgrad_quant layout): 12 warps = 3 whole
        # 128-thread warpgroups — a partial warpgroup is illegal for
        # setmaxregister, so every role must pack into whole groups.
        self.occupancy = 1
        self.accumulator_update_warp_id = (0, 1, 2, 3)
        self.epilog_warp_id = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.sched_warp_id = 10
        self.scale_warp_id = 11
        self.threads_per_warp = 32
        all_warps = [
            *self.accumulator_update_warp_id,
            *self.epilog_warp_id,
            self.mma_warp_id,
            self.tma_warp_id,
            self.sched_warp_id,
            self.scale_warp_id,
        ]
        # Per-role register budgets applied via setmaxregister (uniform per
        # warpgroup: warps 0-3 acc-update, 4-7 epilogue, 8-11 uniform)
        self.num_regs_uniform_warps = 24
        self.num_regs_epilogue_warps = 168
        self.num_regs_acc_update_warps = 176
        self.threads_per_cta = self.threads_per_warp * len(all_warps)

        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3

        self.architecture = "sm_100"
        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.architecture)
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.architecture)

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------

    def get_workspace_bytes(self) -> int:
        return WgradSfTensormapConstructor.get_workspace_size(self.input_order, self.weight_mode, self.expert_cnt)

    # ------------------------------------------------------------------
    # _setup_attributes
    # ------------------------------------------------------------------

    def _setup_attributes(self) -> None:
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = self._create_tiled_mma()
        tiled_mma_sfb = self._create_tiled_mma_sfb()

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        mma_tiler_k = mma_inst_shape_k * mma_inst_tile_k
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_tiler_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_tiler_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        # Every expert's k_tile_cnt must divide into whole scale blocks
        # (wrapper gate token_counts % sgk == 0 makes this per-expert exact).
        assert self.sgk % self.mma_tiler[2] == 0, f"sgk ({self.sgk}) must be a multiple of the MMA K tile " f"({self.mma_tiler[2]})"

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])
        # The acc-update warpgroup's SFA2-as-C partition fixes the epi M
        # iterator at 0 (the dgrad_quant scheme) — needs one epi tile per M.
        assert cute.size(self.epi_tile[0]) == self.cta_tile_shape_mnk[0], (
            f"epi tile M ({cute.size(self.epi_tile[0])}) must equal the CTA " f"tile M ({self.cta_tile_shape_mnk[0]})"
        )

        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_epi_stage,
        ) = compute_stages_wgrad_2nd_level(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
            self.acc_dtype,
            self.sf2_dtype,
            self.sgk,
        )
        self.num_sched_stages = 2
        # Second-level scale pipeline runs in lockstep with the AB pipeline
        self.num_scale_stage = self.num_ab_stage

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        # Per-CTA-tile scale block sizes, clamped to the tile (dgrad_quant's
        # scheme; here sgm = 1 so the M axis is per-row and only K clamps).
        size_m = self.sgm if self.sgm < self.cta_tile_shape_mnk[0] else self.cta_tile_shape_mnk[0]
        size_k = self.sgk if self.sgk < self.cta_tile_shape_mnk[2] else self.cta_tile_shape_mnk[2]
        self.scale_size_m = size_m
        self.scale_size_k = size_k
        self.scale_m_per_tile = self.cta_tile_shape_mnk[0] // size_m
        self.scale_k_per_tile = self.cta_tile_shape_mnk[2] // size_k

        # Staged SMEM layout for the second-level (f32) A scale; extent-0
        # inner modes broadcast one stored f32 across its whole block.
        self.sfa2_smem_layout_staged = cute.make_layout(
            (
                (size_m, self.scale_m_per_tile),
                (size_k, self.scale_k_per_tile),
                self.num_scale_stage,
            ),
            stride=(
                (0, self.scale_k_per_tile),
                (0, 1),
                self.scale_k_per_tile * self.scale_m_per_tile,
            ),
        )

        # Full-CTA-tile SMEM buffer for the final accumulator; the "stage"
        # mode of the epi layout doubles as the subtile-slot index
        # (final_acc_subtile_cnt slots per epi stage).
        self.final_acc_subtile_cnt = (self.cta_tile_shape_mnk[0] // cute.size(self.epi_tile[0])) * (self.cta_tile_shape_mnk[1] // cute.size(self.epi_tile[1]))
        self.final_acc_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.acc_dtype,
            self.c_layout,
            self.epi_tile,
            self.final_acc_subtile_cnt * self.num_epi_stage,
        )

        # TMEM column budget: partial accumulators followed by SFA/SFB scale
        # columns (no overlapping-accum trick: at N=256 num_acc_stage is 1, so
        # 256 acc + 48 SF <= 512; at N=128, 384 + 32 <= 512).
        sf_atom_mn = 32
        mma_inst_tile_k = 4
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = self.cta_tile_shape_mnk[1] * self.num_acc_stage
        assert self.num_accumulator_tmem_cols + self.num_sf_tmem_cols <= self.num_tmem_alloc_cols, (
            f"TMEM overflow: {self.num_accumulator_tmem_cols} acc + " f"{self.num_sf_tmem_cols} SF columns > {self.num_tmem_alloc_cols}"
        )

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size) * atom_thr_size

    # ------------------------------------------------------------------
    # MMA helpers
    # ------------------------------------------------------------------

    def _create_tiled_mma(self):
        return sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

    def _create_tiled_mma_sfb(self):
        return sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

    def mainloop_s2t_copy_and_partition(self, sSF, tSF):
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    # ------------------------------------------------------------------
    # Acc-update / epilogue partition helpers (ported from dgrad_quant)
    # ------------------------------------------------------------------

    def acc_update_tmem_copy_and_partition(
        self,
        tidx,
        tCtAcc_base,
        gC_mnl,
        sSFA2,
        sFinalAcc,
        epi_tile,
    ):
        """T2R (partial accumulator TMEM->reg), R2S (final acc reg->SMEM) and
        the SFA2-as-C SMEM partition for the accumulator-update warpgroup.
        One subtile's partial (tTR_rAcc) is live at a time; tTR_rAcc_final
        spans the full CTA tile grouped by epi subtile."""
        if cutlass.const_expr(self.mma_tiler[0] == 64):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
                self.acc_dtype,
            )
        else:
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
                self.acc_dtype,
            )

        tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

        # R2S store of the final accumulator: reuse the T2R copy's TV layout
        # with a universal copy atom (swizzle-safe for f32)
        copy_atom_r2s_acc = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype)
        tiled_copy_r2s_acc = cute.make_tiled_copy_D(copy_atom_r2s_acc, tiled_copy_t2r)
        thr_copy_r2s_acc = tiled_copy_r2s_acc.get_slice(tidx)
        # (R2S, R2S_M, R2S_N, SUBTILE * STAGE)
        tRS_sFinalAcc = thr_copy_r2s_acc.partition_D(sFinalAcc)

        tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_mnl_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        sSFA2_epi = cute.flat_divide(sSFA2, epi_tile)

        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_sSFA2 = thr_copy_t2r.partition_D(sSFA2_epi)

        # One k_tile's partial accumulator: (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        # Final accumulated result across all scale blocks:
        # (T2R, T2R_M, T2R_N, (EPI_M, EPI_N))
        tTR_rAcc_final_ = cute.make_rmem_tensor(tTR_gC[(None, None, None, None, None, 0, 0, 0)].shape, self.acc_dtype)
        tTR_rAcc_final = cute.group_modes(tTR_rAcc_final_, 3, cute.rank(tTR_rAcc_final_))

        return (
            tiled_copy_t2r,
            tiled_copy_r2s_acc,
            tTR_tAcc_base,
            tTR_rAcc,
            tTR_rAcc_final,
            tRS_sFinalAcc,
            tTR_sSFA2,
        )

    def epilog_tmem_copy_and_partition(self, tidx, tAcc, gC_mnl, epi_tile, use_2cta_instrs):
        # Shape/TV-layout carrier only — the epilogue never reads TMEM data;
        # this defines the tiled T2R copy whose TV layout the r2s/s2r reuse.
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(self, tiled_copy_t2r, tTR_rC, tidx, sC):
        copy_atom_r2s = sm100_utils.get_smem_store_op(self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_smem_load_copy_and_partition(self, tiled_copy_t2r, tTR_rAcc, tidx, sFinalAcc):
        # S2R load of the final accumulator: reuse the T2R copy's TV layout
        # with a universal copy atom; tSR_rAcc is a register view of tTR_rAcc
        copy_atom_s2r = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype)
        tiled_copy_s2r = cute.make_tiled_copy_D(copy_atom_s2r, tiled_copy_t2r)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        # (S2R, S2R_M, S2R_N, SUBTILE * STAGE)
        tSR_sFinalAcc = thr_copy_s2r.partition_D(sFinalAcc)
        tSR_rAcc = tiled_copy_s2r.retile(tTR_rAcc)
        return tiled_copy_s2r, tSR_rAcc, tSR_sFinalAcc

    @cute.jit
    def __call__(
        self,
        mat_a: cute.Tensor,  # (hidden, tokens_sum) — activation-grad^T
        mat_b: cute.Tensor,  # (tokens_sum, intermediate) — activation
        scale_a: cute.Tensor,  # SFA (assembled block-scaled layout)
        scale_b: cute.Tensor,  # SFB (assembled block-scaled layout)
        scale_a2: cute.Tensor,  # SFA2 (hidden, tokens_sum/sgk[, 1]) f32, hidden contiguous
        out,  # Dense: cute.Tensor (experts, hidden, intermediate)
        # Discrete: cute.Pointer to int64[]
        offs,  # Union[cute.Tensor, cute.Pointer] (experts,) cumsum end offsets, int32
        workspace: cute.Tensor,  # expert-wise TMA desc (discrete only)
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        global_scale_a: Optional[cute.Tensor] = None,
        global_scale_b: Optional[cute.Tensor] = None,
        # Discrete-only: template tensor for a single expert's output (M, N) or (M, N, 1)
        out_single_expert: Optional[cute.Tensor] = None,
    ) -> None:

        # Public CUTLASS DSL < 4.8 needs the packed-FP4 from_dlpack layout
        # workaround. Rubin, the internal DSL wheel, and public wheels >= 4.8
        # consume the native 4-bit layout directly.
        needs_fp4_layout_workaround = self.architecture != "sm_107" and _NEEDS_FP4_LAYOUT_WORKAROUND
        if cutlass.const_expr(needs_fp4_layout_workaround and mat_a.iterator.dtype.width < 8):
            mat_a = cute.make_tensor(
                mat_a.iterator,
                cute.recast_layout(mat_a.iterator.dtype.width, 8, mat_a.layout),
            )
        if cutlass.const_expr(needs_fp4_layout_workaround and mat_b.iterator.dtype.width < 8):
            mat_b = cute.make_tensor(
                mat_b.iterator,
                cute.recast_layout(mat_b.iterator.dtype.width, 8, mat_b.layout),
            )

        # =================================================================
        # Step 1: Transform to GEMM domain (2Dx2D)
        # =================================================================
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        # mat_a: (hidden, tokens_sum) -> A: (M=hidden, K=tokens_sum, L=1)
        hidden, tokens_sum = mat_a.shape
        a_gemm = cute.make_tensor(
            mat_a.iterator,
            cute.make_layout(
                (hidden, tokens_sum, c1),
                stride=(mat_a.stride[0], mat_a.stride[1], c0),
            ),
        )
        # mat_b: (tokens_sum, intermediate) -> B: (N=intermediate, K=tokens_sum, L=1)
        tokens_sum_b, intermediate = mat_b.shape
        b_gemm = cute.make_tensor(
            mat_b.iterator,
            cute.make_layout(
                (intermediate, tokens_sum_b, c1),
                stride=(mat_b.stride[1], mat_b.stride[0], c0),
            ),
        )

        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
            # out: (experts, hidden, intermediate) -> C: (M=hidden, N=intermediate, L=experts)
            experts, hidden_c, intermediate_c = out.shape
            c_gemm = cute.make_tensor(
                out.iterator,
                cute.make_layout(
                    (hidden_c, intermediate_c, experts),
                    stride=(out.stride[1], out.stride[2], out.stride[0]),
                ),
            )
            expert_cnt = experts
        else:
            # Discrete: out is a Pointer to int64[] of per-expert base addresses
            expert_cnt = self.expert_cnt
            # Normalize out_single_expert to rank-3 (M, N, 1) if rank-2
            if cutlass.const_expr(cute.rank(out_single_expert.layout) == 2):
                out_single_expert = cute.make_tensor(
                    out_single_expert.iterator,
                    cute.make_layout(
                        (*out_single_expert.shape, c1),
                        stride=(*out_single_expert.stride, c0),
                    ),
                )
            c_gemm = out_single_expert

        intermediate_dim = intermediate
        hidden_dim = hidden

        # SFA: (hidden_padded, tokens_sum_padded_sf)
        hidden_padded = scale_a.shape[0]
        tokens_sum_padded = scale_a.shape[1] * self.sf_vec_size
        sfa_gemm = cute.make_tensor(
            scale_a.iterator,
            blockscaled_utils.tile_atom_to_shape_SF((hidden_padded, tokens_sum_padded, c1), self.sf_vec_size),
        )
        # SFB: (intermediate_padded, tokens_sum_padded_sf)
        intermediate_padded = scale_b.shape[0]
        sfb_gemm = cute.make_tensor(
            scale_b.iterator,
            blockscaled_utils.tile_atom_to_shape_SF((intermediate_padded, tokens_sum_padded, c1), self.sf_vec_size),
        )

        # Accept the producer's rank-2 (hidden, tokens_sum/sgk) SFA2 view directly.
        if cutlass.const_expr(cute.rank(scale_a2.layout) == 2):
            scale_a2 = cute.make_tensor(
                scale_a2.iterator,
                cute.make_layout((*scale_a2.shape, c1), stride=(*scale_a2.stride, c0)),
            )

        # SFA2: (hidden, tokens_sum/sgk, 1) f32 → hierarchical broadcast view
        # ((sgm, hidden), (sgk, scale_cols), 1) with stride-0 inner modes so
        # one f32 covers its whole (1 row x sgk tokens) block (the dgrad_quant
        # sfa2 template with the group axis on K).
        sfa2_gemm = cute.make_tensor(
            scale_a2.iterator,
            cute.make_layout(
                (
                    (self.sgm, scale_a2.shape[0]),
                    (self.sgk, scale_a2.shape[1]),
                    scale_a2.shape[2],
                ),
                stride=(
                    (0, scale_a2.layout.stride[0]),
                    (0, scale_a2.layout.stride[1]),
                    scale_a2.layout.stride[2],
                ),
            ),
        )

        # =================================================================
        # Step 2: Infer dtypes and major modes
        # =================================================================
        self.a_dtype = a_gemm.element_type
        self.b_dtype = b_gemm.element_type
        self.c_dtype = c_gemm.element_type
        # Scale factors may arrive under a stand-in element type: FloatNV8E5M3FNU has
        # no torch dtype and TVM-FFI cannot marshal it, so e5m3 scales are passed as
        # Float8E4M3FN storage of the same width and reinterpreted here. This must
        # happen before _setup_attributes(), which picks the MMA atom off sf_dtype.
        if cutlass.const_expr(self.sf_dtype_override is not None):
            self.sf_dtype = self.sf_dtype_override
        else:
            self.sf_dtype = sfa_gemm.element_type
        self.sf2_dtype = scale_a2.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_gemm).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_gemm).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_gemm)

        # =================================================================
        # Step 3: Setup kernel attributes
        # =================================================================
        self._setup_attributes()
        tiled_mma = self._create_tiled_mma()
        tiled_mma_sfb = self._create_tiled_mma_sfb()

        # =================================================================
        # Step 4: Create TMA ops (atoms built after helper kernel)
        # =================================================================
        # TMA load A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))

        # TMA load B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))

        # TMA ops for SFA/SFB
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)

        # TMA store/reduce C
        if cutlass.const_expr(self.accumulate_on_output):
            c_tma_op = cpasync.CopyReduceBulkTensorTileS2GOp()
        else:
            c_tma_op = cpasync.CopyBulkTensorTileS2GOp()

        # =================================================================
        # Step 5: Scheduler params and grid
        # =================================================================
        sched_params = MoESchedulerParams(
            scenario="2Dx2D",
            expert_shape=(expert_cnt, intermediate_dim, hidden_dim),
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
        )
        grid = MoESchedulerParams.get_grid_shape(sched_params, max_active_clusters)

        # =================================================================
        # Step 6: Launch helper kernel (both Dense and Discrete)
        # =================================================================
        # Identical to wgrad_baseline — SFA2 has no tensormap slot (LDGSTS).
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        epi_smem_layout_helper = cute.select(self.c_smem_layout_staged, mode=[0, 1]) if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else None
        if cutlass.const_expr(self.input_order == WGradInputOrder.TensorRagged):
            a_smem_layout_helper = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
            b_smem_layout_helper = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
            a_gemm_helper = a_gemm
            b_gemm_helper = b_gemm
            a_op_helper = a_op
            b_op_helper = b_op
        else:
            a_smem_layout_helper = None
            b_smem_layout_helper = None
            a_gemm_helper = None
            b_gemm_helper = None
            a_op_helper = None
            b_op_helper = None

        self.helper_kernel(
            sfa_gemm,
            sfb_gemm,
            offs,
            workspace.iterator,
            sfa_op,
            sfa_smem_layout,
            sfb_op,
            sfb_smem_layout,
            tiled_mma,
            tiled_mma_sfb,
            self.cluster_layout_vmnk.shape,
            self.cluster_layout_sfb_vmnk.shape,
            out if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else None,
            c_gemm if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else None,
            c_tma_op if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else None,
            epi_smem_layout_helper,
            self.epi_tile if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else None,
            a_gemm_helper,
            b_gemm_helper,
            a_op_helper,
            b_op_helper,
            a_smem_layout_helper,
            b_smem_layout_helper,
        ).launch(
            grid=(expert_cnt, 1, 1),
            block=(1, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

        # Build A, B, SFA, SFB, C TMA atoms AFTER the helper kernel launch
        # (see wgrad_baseline for the host-region contamination rationale).
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_gemm,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_gemm,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_gemm,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Uint64,
        )
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_gemm,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Uint64,
        )
        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            c_tma_op,
            c_gemm,
            epi_smem_layout,
            self.epi_tile,
        )

        # =================================================================
        # Step 7: Launch main kernel
        # =================================================================
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            sfa2_gemm,
            tma_atom_c,
            tma_tensor_c,
            a_gemm,
            b_gemm,
            c_gemm,
            sfa_gemm,
            sfb_gemm,
            offs,
            sched_params,
            workspace.iterator,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.sfa2_smem_layout_staged,
            self.c_smem_layout_staged,
            self.final_acc_smem_layout_staged,
            self.epi_tile,
            global_scale_a,
            global_scale_b,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=self.occupancy,
        )

    # ------------------------------------------------------------------
    # helper_kernel (expert-wise TMA desc init via construct_and_write)
    # ------------------------------------------------------------------

    @cute.kernel
    def helper_kernel(
        self,
        sfa_gemm: cute.Tensor,
        sfb_gemm: cute.Tensor,
        offs: cute.Tensor,
        workspace_ptr,
        sfa_tma_op: cutlass.Constexpr,
        sfa_smem_layout,
        sfb_tma_op: cutlass.Constexpr,
        sfb_smem_layout,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        cluster_layout_vmnk_shape: cutlass.Constexpr,
        cluster_layout_sfb_vmnk_shape: cutlass.Constexpr,
        c_ptrs=None,
        c_single_expert=None,
        c_tma_op: cutlass.Constexpr = None,
        epi_smem_layout=None,
        epi_tile=None,
        a_tensor=None,
        b_tensor=None,
        a_tma_op: cutlass.Constexpr = None,
        b_tma_op: cutlass.Constexpr = None,
        a_smem_layout=None,
        b_smem_layout=None,
    ):
        """Build per-expert TMA descriptors (identical to wgrad_baseline)."""
        from ..moe_utils import (
            WgradSfTensormapConstructor,
        )

        ctor = WgradSfTensormapConstructor(
            sf_vec_size=self.sf_vec_size,
            weight_mode=self.weight_mode,
            sfa_tma_op=sfa_tma_op,
            sfb_tma_op=sfb_tma_op,
            sfa_smem_layout=sfa_smem_layout,
            sfb_smem_layout=sfb_smem_layout,
            tiled_mma=tiled_mma,
            tiled_mma_sfb=tiled_mma_sfb,
            mma_tiler=self.mma_tiler,
            mma_tiler_sfb=self.mma_tiler_sfb,
            cluster_layout_vmnk_shape=cluster_layout_vmnk_shape,
            cluster_layout_sfb_vmnk_shape=cluster_layout_sfb_vmnk_shape,
            sfa_tensor=sfa_gemm,
            sfb_tensor=sfb_gemm,
            offs=offs,
            workspace_ptr=workspace_ptr,
            c_tma_op=c_tma_op,
            epi_smem_layout=epi_smem_layout,
            epi_tile=epi_tile,
            c_ptrs=c_ptrs,
            c_single_expert=c_single_expert,
            expert_cnt=self.expert_cnt,
            input_order=self.input_order,
            a_tma_op=a_tma_op,
            b_tma_op=b_tma_op,
            a_smem_layout=a_smem_layout,
            b_smem_layout=b_smem_layout,
            a_major_mode=self.a_major_mode,
            b_major_mode=self.b_major_mode,
            a_tensor=a_tensor,
            b_tensor=b_tensor,
        )
        expert_idx = cute.arch.block_idx()[0]
        ctor.construct_and_write(expert_idx)

    # ------------------------------------------------------------------
    # kernel (GPU device kernel)
    # ------------------------------------------------------------------

    helper_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        sfa2_tensor: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        a_gemm: cute.Tensor,
        b_gemm: cute.Tensor,
        c_gemm: cute.Tensor,
        sfa_gemm: cute.Tensor,
        sfb_gemm: cute.Tensor,
        offs: cute.Tensor,
        sched_params: MoESchedulerParams,
        workspace_ptr,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        sfa2_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: cute.ComposedLayout,
        final_acc_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        global_scale_a: Optional[cute.Tensor],
        global_scale_b: Optional[cute.Tensor],
    ):
        """GPU device kernel for MoE wgrad with second-level A scaling."""

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        # =================================================================
        # SharedStorage
        # =================================================================
        SchedulerStorage = MoEPersistentTileScheduler.make_storage_struct(self.num_sched_stages, use_dynamic_sched=True)

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            scale_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_scale_stage * 2]
            epi_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_epi_stage * 2]
            scheduler: SchedulerStorage
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sched_storage = storage.scheduler

        # =================================================================
        # Pipelines
        # =================================================================

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        # Partial-accumulator pipeline: MMA warp (producer, one stage per
        # scale block) -> acc-update warpgroup (consumer, elect_one release
        # per warp: 4 arrivals, x2 across the 2-CTA pair).
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.accumulator_update_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # SFA2 pipeline: scale warp (cp.async producer) -> acc-update warps
        scale_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        scale_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.accumulator_update_warp_id),
        )
        scale_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.scale_full_mbar_ptr.data_ptr(),
            num_stages=self.num_scale_stage,
            producer_group=scale_pipeline_producer_group,
            consumer_group=scale_pipeline_consumer_group,
            defer_sync=True,
        )

        # Final-accumulator pipeline: acc-update warps -> epilogue warps
        epi_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.accumulator_update_warp_id),
        )
        epi_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.epilog_warp_id),
        )
        epi_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.epi_full_mbar_ptr.data_ptr(),
            num_stages=self.num_epi_stage,
            producer_group=epi_pipeline_producer_group,
            consumer_group=epi_pipeline_consumer_group,
            defer_sync=True,
        )

        # Scheduler pipeline (sched warp -> every other warp)
        sched_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        num_sched_consumer_threads = 32 * len(
            (
                self.tma_warp_id,
                self.mma_warp_id,
                self.scale_warp_id,
                *self.accumulator_update_warp_id,
                *self.epilog_warp_id,
            )
        )
        sched_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_sched_consumer_threads)
        sched_pipeline = pipeline.PipelineAsync.create(
            num_stages=self.num_sched_stages,
            producer_group=sched_producer_group,
            consumer_group=sched_consumer_group,
            barrier_storage=sched_storage.tile_info_mbar.data_ptr(),
            defer_sync=True,
        )

        # TMEM allocator (allocated by epilogue warp 4; MMA + acc-update +
        # epilogue warps all retrieve → the barrier spans those 9 warps)
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32
            * len(
                (
                    self.mma_warp_id,
                    *self.accumulator_update_warp_id,
                    *self.epilog_warp_id,
                )
            ),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
            arch=self.architecture,
        )

        # Scheduler (CLC-based for 2Dx2D)
        scheduler = MoEPersistentTileScheduler.create(
            sched_params,
            offs,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            counter_ptr=None,
            sched_storage=sched_storage,
        )
        scheduler.internal_init()

        # Cluster barrier sync after init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # =================================================================
        # SMEM tensors
        # =================================================================
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        sSFB = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfb_smem_layout_staged,
            byte_alignment=128,
        )
        sSFA2 = smem.allocate_tensor(
            element_type=self.sf2_dtype,
            layout=sfa2_smem_layout_staged,
            byte_alignment=128,
        )
        sFinalAcc = smem.allocate_tensor(
            element_type=self.acc_dtype,
            layout=final_acc_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=final_acc_smem_layout_staged.inner,
        )

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # Scheduler buf tensor for sched_pipeline broadcast
        sched_buf_ptr = sched_storage.sInfo.data_ptr()
        sched_copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128)
        sched_buf_tensor = cute.make_tensor(
            sched_buf_ptr,
            cute.make_layout((4, self.num_sched_stages), stride=(1, 4)),
        )

        # LDGSTS (cp.async) setup for the second-level A scale
        atom_scale2_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            self.sf2_dtype,
            num_bits_per_copy=self.sf2_dtype.width,
        )
        tiled_copy_sfa2 = cute.make_tiled_copy_tv(atom_scale2_copy, cute.make_layout((32,)), cute.make_layout((1,)))
        thr_copy_sfa2 = tiled_copy_sfa2.get_slice(lane_idx)
        tAsSFA2 = thr_copy_sfa2.partition_D(sSFA2)

        # SFA2 viewed as a C-tile tensor: M is the CTA-tile-clamped scale
        # block grid (sgm = 1 → per-row), N fully broadcast (no SFB2).
        sSFA2_view_as_C_layout = cute.make_layout(
            (
                (self.scale_size_m, self.scale_m_per_tile),
                self.cta_tile_shape_mnk[1],
                self.num_scale_stage,
            ),
            stride=((0, 1), 0, self.scale_m_per_tile),
        )
        sSFA2_view_as_C = cute.make_tensor(sSFA2.iterator, sSFA2_view_as_C_layout)

        # Number of K MMA tiles that share a single second-level scale block
        k_tile_same_scale_factor = self.sgk // self.mma_tiler[2]

        # Build extension
        from ..moe_utils import (
            TensormapWorkspace,
            WgradSfTensormapConstructor,
        )

        slot_names = WgradSfTensormapConstructor.slot_names(self.input_order, self.weight_mode)
        desc_workspace = TensormapWorkspace(workspace_ptr, slot_names)
        ext = WgradScaledGemmSchedExtension(
            tensormap_ctor=desc_workspace,
            sf_vec_size=self.sf_vec_size,
            sgk=self.sgk,
            weight_mode=self.weight_mode,
            input_order=self.input_order,
        )

        # Cluster wait
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # =================================================================
        # Scheduler warp (warp 10)
        # =================================================================
        if warp_idx == self.sched_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)
            sched_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_sched_stages)

            work_tile_info = scheduler.initial_work_tile_info()

            sched_pipeline.producer_acquire(sched_producer_state)
            rmem = work_tile_info.to_rmem_tensor()
            cute.copy(
                sched_copy_atom,
                rmem,
                sched_buf_tensor[(None, sched_producer_state.index)],
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            sched_pipeline.producer_commit(sched_producer_state)
            sched_producer_state.advance()

            work_tile_info = scheduler.advance_to_next_work()
            while work_tile_info.is_valid_tile:
                sched_pipeline.producer_acquire(sched_producer_state)
                rmem = work_tile_info.to_rmem_tensor()
                cute.copy(
                    sched_copy_atom,
                    rmem,
                    sched_buf_tensor[(None, sched_producer_state.index)],
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                sched_pipeline.producer_commit(sched_producer_state)
                sched_producer_state.advance()

                work_tile_info = scheduler.advance_to_next_work()

            sched_pipeline.producer_acquire(sched_producer_state)
            sentinel = MoEWorkTileInfo(
                cutlass.Int32(-1),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
            )
            rmem = sentinel.to_rmem_tensor()
            cute.copy(
                sched_copy_atom,
                rmem,
                sched_buf_tensor[(None, sched_producer_state.index)],
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            sched_pipeline.producer_commit(sched_producer_state)
            sched_pipeline.producer_tail(sched_producer_state)

        # =================================================================
        # TMA load warp (warp 9) — identical to wgrad_baseline
        # =================================================================
        if warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)
            a_full_mcast_mask = None
            b_full_mcast_mask = None
            sfa_full_mcast_mask = None
            sfb_full_mcast_mask = None
            if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
                a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
                b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
                sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
                sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_sfb_vmnk,
                    block_in_cluster_coord_sfb_vmnk,
                    mcast_mode=1,
                )

            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_sched_stages)

            sched_pipeline.consumer_wait(sched_consumer_state)
            rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
            cute.copy(
                sched_copy_atom,
                sched_buf_tensor[(None, sched_consumer_state.index)],
                rmem,
            )
            work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
            cute.arch.fence_acq_rel_cta()
            sched_pipeline.consumer_release(sched_consumer_state)
            sched_consumer_state.advance()

            while work_tile_info.is_valid_tile:
                k_tile_cnt = work_tile_info.k_tile_cnt
                ext.update_expert_info(offs, work_tile_info.expert_idx)

                real_a, desc_ptr_a = ext.get_gmem_tensor(
                    "a",
                    mA_mkl,
                    offs,
                    work_tile_info,
                )
                real_b, desc_ptr_b = ext.get_gmem_tensor(
                    "b",
                    mB_nkl,
                    offs,
                    work_tile_info,
                )
                real_sfa, desc_ptr_sfa = ext.get_gmem_tensor(
                    "sfa",
                    mSFA_mkl,
                    offs,
                    work_tile_info,
                )
                real_sfb, desc_ptr_sfb = ext.get_gmem_tensor(
                    "sfb",
                    mSFB_nkl,
                    offs,
                    work_tile_info,
                )

                gA_mkl = cute.local_tile(
                    real_a,
                    cute.slice_(self.mma_tiler, (None, 0, None)),
                    (None, None, None),
                )
                gB_nkl = cute.local_tile(
                    real_b,
                    cute.slice_(self.mma_tiler, (0, None, None)),
                    (None, None, None),
                )
                gSFA_mkl = cute.local_tile(
                    real_sfa,
                    cute.slice_(self.mma_tiler, (None, 0, None)),
                    (None, None, None),
                )
                gSFB_nkl = cute.local_tile(
                    real_sfb,
                    cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                    (None, None, None),
                )

                thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
                thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
                tCgA = thr_mma.partition_A(gA_mkl)
                tCgB = thr_mma.partition_B(gB_nkl)
                tCgSFA = thr_mma.partition_A(gSFA_mkl)
                tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

                a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
                tAsA, tAgA = cpasync.tma_partition(
                    tma_atom_a,
                    block_in_cluster_coord_vmnk[2],
                    a_cta_layout,
                    cute.group_modes(sA, 0, 3),
                    cute.group_modes(tCgA, 0, 3),
                )
                b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    cute.group_modes(sB, 0, 3),
                    cute.group_modes(tCgB, 0, 3),
                )
                sfa_cta_layout = a_cta_layout
                tAsSFA, tAgSFA = cpasync.tma_partition(
                    tma_atom_sfa,
                    block_in_cluster_coord_vmnk[2],
                    sfa_cta_layout,
                    cute.group_modes(sSFA, 0, 3),
                    cute.group_modes(tCgSFA, 0, 3),
                )
                tAsSFA = cute.filter_zeros(tAsSFA)
                tAgSFA = cute.filter_zeros(tAgSFA)
                sfb_cta_layout = cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
                tBsSFB, tBgSFB = cpasync.tma_partition(
                    tma_atom_sfb,
                    block_in_cluster_coord_sfb_vmnk[1],
                    sfb_cta_layout,
                    cute.group_modes(sSFB, 0, 3),
                    cute.group_modes(tCgSFB, 0, 3),
                )
                tBsSFB = cute.filter_zeros(tBsSFB)
                tBgSFB = cute.filter_zeros(tBgSFB)

                mma_tile_m = work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)
                tAgA_slice = tAgA[(None, mma_tile_m, None, 0)]
                tBgB_slice = tBgB[(None, work_tile_info.tile_n_idx, None, 0)]
                tAgSFA_slice = tAgSFA[(None, mma_tile_m, None, 0)]
                slice_n = work_tile_info.tile_n_idx
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = work_tile_info.tile_n_idx // 2
                tBgSFB_slice = tBgSFB[(None, slice_n, None, 0)]

                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        tma_desc_ptr=desc_ptr_a,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        tma_desc_ptr=desc_ptr_b,
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, handle.count)],
                        tAsSFA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        tma_desc_ptr=desc_ptr_sfa,
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, handle.count)],
                        tBsSFB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        tma_desc_ptr=desc_ptr_sfb,
                        mcast_mask=sfb_full_mcast_mask,
                    )

                sched_pipeline.consumer_wait(sched_consumer_state)
                rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
                cute.copy(
                    sched_copy_atom,
                    sched_buf_tensor[(None, sched_consumer_state.index)],
                    rmem,
                )
                work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
                cute.arch.fence_acq_rel_cta()
                sched_pipeline.consumer_release(sched_consumer_state)
                sched_consumer_state.advance()
            ab_producer.tail()

        # =================================================================
        # Second-level scale load warp (warp 11)
        # =================================================================
        if warp_idx == self.scale_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)

            scale_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_scale_stage)
            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_sched_stages)

            sched_pipeline.consumer_wait(sched_consumer_state)
            rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
            cute.copy(
                sched_copy_atom,
                sched_buf_tensor[(None, sched_consumer_state.index)],
                rmem,
            )
            work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
            cute.arch.fence_acq_rel_cta()
            sched_pipeline.consumer_release(sched_consumer_state)
            sched_consumer_state.advance()

            while work_tile_info.is_valid_tile:
                k_tile_cnt = work_tile_info.k_tile_cnt
                ext.update_expert_info(offs, work_tile_info.expert_idx)

                mSFA2_mkl_current, _ = ext.get_gmem_tensor(
                    "sfa2",
                    sfa2_tensor,
                    offs,
                    work_tile_info,
                )

                gSFA2_mkl = cute.local_tile(
                    mSFA2_mkl_current,
                    cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
                    (None, None, None),
                )
                cSFA2_mkl = cute.make_identity_tensor(cute.shape(mSFA2_mkl_current))
                cSFA2 = cute.local_tile(
                    cSFA2_mkl,
                    cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
                    (None, None, None),
                )

                tAgSFA2_mkl = thr_copy_sfa2.partition_S(gSFA2_mkl)
                tAcSFA2 = thr_copy_sfa2.partition_S(cSFA2)

                # SFA2 tiling is per-CTA: raw (CTA-granular) tile_m_idx
                tile_m_cta = work_tile_info.tile_m_idx

                # OOB predication mask over the M rows this lane loads
                tApSFA2 = cute.make_rmem_tensor(
                    cute.make_layout(cute.filter_zeros(cute.slice_(tAsSFA2, (None, None, None, 0))).shape),
                    cutlass.Boolean,
                )

                scale_producer_state.reset_count()
                peek_scale_empty_status = cutlass.Boolean(1)
                if scale_producer_state.count < k_tile_cnt:
                    peek_scale_empty_status = scale_pipeline.producer_try_acquire(scale_producer_state)

                for k_tile in cutlass.range(0, k_tile_cnt // k_tile_same_scale_factor, 1, unroll=1):
                    tAsSFA2_pipe = cute.filter_zeros(tAsSFA2[(None, None, None, scale_producer_state.index)])
                    tAgSFA2_k = cute.filter_zeros(
                        tAgSFA2_mkl[
                            (
                                None,
                                None,
                                None,
                                tile_m_cta,
                                scale_producer_state.count * k_tile_same_scale_factor,
                                0,
                            )
                        ]
                    )
                    tAcSFA2_compact = cute.filter_zeros(
                        cute.slice_(
                            tAcSFA2,
                            (
                                None,
                                None,
                                None,
                                tile_m_cta,
                                scale_producer_state.count * k_tile_same_scale_factor,
                                0,
                            ),
                        )
                    )

                    for i in cutlass.range_constexpr(cute.size(tApSFA2, mode=[1])):
                        tApSFA2[((0, 0), i, (0, 0))] = cute.elem_less(tAcSFA2_compact[(i)][0], mSFA2_mkl_current.shape[0])

                    scale_pipeline.producer_acquire(scale_producer_state, peek_scale_empty_status)
                    cute.copy(tiled_copy_sfa2, tAgSFA2_k, tAsSFA2_pipe, pred=tApSFA2)
                    scale_pipeline.producer_commit(scale_producer_state)

                    scale_producer_state.advance()
                    peek_scale_empty_status = cutlass.Boolean(1)
                    if scale_producer_state.count < k_tile_cnt:
                        peek_scale_empty_status = scale_pipeline.producer_try_acquire(scale_producer_state)

                sched_pipeline.consumer_wait(sched_consumer_state)
                rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
                cute.copy(
                    sched_copy_atom,
                    sched_buf_tensor[(None, sched_consumer_state.index)],
                    rmem,
                )
                work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
                cute.arch.fence_acq_rel_cta()
                sched_pipeline.consumer_release(sched_consumer_state)
                sched_consumer_state.advance()

        # =================================================================
        # MMA warp (warp 8) — one clean partial accumulator per scale block
        # =================================================================
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)

            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)
            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_sched_stages)

            sched_pipeline.consumer_wait(sched_consumer_state)
            rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
            cute.copy(
                sched_copy_atom,
                sched_buf_tensor[(None, sched_consumer_state.index)],
                rmem,
            )
            work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
            cute.arch.fence_acq_rel_cta()
            sched_pipeline.consumer_release(sched_consumer_state)
            sched_consumer_state.advance()

            while work_tile_info.is_valid_tile:
                k_tile_cnt = work_tile_info.k_tile_cnt

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    offset = cutlass.Int32((work_tile_info.tile_n_idx % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                acc_producer_state.reset_count()
                peek_acc_empty_status = cutlass.Boolean(1)
                if is_leader_cta and k_tile_cnt > 0:
                    peek_acc_empty_status = acc_pipeline.producer_try_acquire(acc_producer_state)

                if is_leader_cta:
                    ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta and k_tile_cnt > 0:
                    peek_ab_full_status = ab_consumer.try_wait()

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Acquire a fresh accumulator stage at each scale block start
                    if k_tile % k_tile_same_scale_factor == 0:
                        if is_leader_cta:
                            acc_pipeline.producer_acquire(acc_producer_state, peek_acc_empty_status)

                    if is_leader_cta:
                        tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                        s2t_stage_coord = (None, None, None, None, handle.index)
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t[s2t_stage_coord],
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t[s2t_stage_coord],
                            tCtSFB_compact_s2t,
                        )

                        # Clean accumulator on the first tile of each block
                        tiled_mma.set(
                            tcgen05.Field.ACCUMULATE,
                            k_tile % k_tile_same_scale_factor > 0,
                        )
                        tile_crd = (None, None, None, handle.index)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            [tCrA[tile_crd], tCtSFA],
                            [tCrB[tile_crd], tCtSFB_mma],
                            tCtAcc,
                        )
                        handle.release()

                    # Commit the block's partial accumulator to the consumers
                    if k_tile % k_tile_same_scale_factor == k_tile_same_scale_factor - 1:
                        if is_leader_cta:
                            acc_pipeline.producer_commit(acc_producer_state)
                        acc_producer_state.advance()
                        peek_acc_empty_status = cutlass.Boolean(1)
                        if acc_producer_state.count < k_tile_cnt:
                            if is_leader_cta:
                                peek_acc_empty_status = acc_pipeline.producer_try_acquire(acc_producer_state)

                sched_pipeline.consumer_wait(sched_consumer_state)
                rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
                cute.copy(
                    sched_copy_atom,
                    sched_buf_tensor[(None, sched_consumer_state.index)],
                    rmem,
                )
                work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
                cute.arch.fence_acq_rel_cta()
                sched_pipeline.consumer_release(sched_consumer_state)
                sched_consumer_state.advance()

            acc_pipeline.producer_tail(acc_producer_state)

        # =================================================================
        # SMEM tensor C (allocated after MMA section)
        # =================================================================
        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )

        # =================================================================
        # Accumulator-update warps (warps 0-3): per scale block,
        # final += partial * sfa2 in f32 registers; result -> sFinalAcc.
        # NO empty-work early exit: k_tile_cnt == 0 tiles still zero-fill
        # and produce the epi stage so empty experts write zeros.
        # =================================================================
        if warp_idx in self.accumulator_update_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_acc_update_warps)

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # Shape-only partition on the output tensor (invariant t2r setup;
            # never dereferenced — expert-uniform C shape)
            gC_mnl_shape = cute.local_tile(
                mC_mnl,
                cute.slice_(self.mma_tiler, (None, None, 0)),
                (None, None, None),
            )
            thr_mma_acc = tiled_mma.get_slice(mma_tile_coord_v)
            tCgC_shape = thr_mma_acc.partition_C(gC_mnl_shape)

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)
            scale_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_scale_stage)
            epi_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_epi_stage)

            (
                tiled_copy_t2r,
                tiled_copy_r2s_acc,
                tTR_tAcc_base,
                tTR_rAcc,
                tTR_rAcc_final,
                tRS_sFinalAcc,
                tTR_sSFA2,
            ) = self.acc_update_tmem_copy_and_partition(
                tidx,
                tCtAcc_base,
                tCgC_shape,
                sSFA2_view_as_C,
                sFinalAcc,
                epi_tile,
            )

            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_sched_stages)

            sched_pipeline.consumer_wait(sched_consumer_state)
            rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
            cute.copy(
                sched_copy_atom,
                sched_buf_tensor[(None, sched_consumer_state.index)],
                rmem,
            )
            work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
            cute.arch.fence_acq_rel_cta()
            sched_pipeline.consumer_release(sched_consumer_state)
            sched_consumer_state.advance()

            while work_tile_info.is_valid_tile:
                k_tile_cnt = work_tile_info.k_tile_cnt

                tTR_rAcc_final.fill(0.0)

                tTR_rSFA2 = cute.make_rmem_tensor(
                    cute.slice_(tTR_sSFA2, (None, None, None, 0, None, 0)).shape,
                    self.acc_dtype,
                )

                acc_consumer_state.reset_count()
                peek_acc_full_status = cutlass.Boolean(1)
                if acc_consumer_state.count < k_tile_cnt:
                    peek_acc_full_status = acc_pipeline.consumer_try_wait(acc_consumer_state)

                scale_consumer_state.reset_count()
                peek_scale_full_status = cutlass.Boolean(1)
                if scale_consumer_state.count < k_tile_cnt:
                    peek_scale_full_status = scale_pipeline.consumer_try_wait(scale_consumer_state)

                for k_tile in cutlass.range(0, k_tile_cnt // k_tile_same_scale_factor, 1, unroll=1):
                    # This scale block's per-row SFA2 from SMEM into registers
                    scale_pipeline.consumer_wait(scale_consumer_state, peek_scale_full_status)
                    tTR_sSFA2_slice = cute.slice_(
                        tTR_sSFA2,
                        (None, None, None, 0, None, scale_consumer_state.index),
                    )
                    scale_atom_copy = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.acc_dtype,
                        num_bits_per_copy=self.acc_dtype.width,
                    )
                    cute.copy(scale_atom_copy, tTR_sSFA2_slice, tTR_rSFA2)

                    scale_pipeline.consumer_release(scale_consumer_state)
                    scale_consumer_state.advance()

                    # Wait for the block's partial accumulator from the MMA
                    acc_pipeline.consumer_wait(acc_consumer_state, peek_acc_full_status)

                    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]
                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                    assert cute.size(tTR_rAcc) % 2 == 0
                    for subtile_idx in cutlass.range(subtile_cnt):
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                        # final += partial * sfa2, as SEPARATELY-ROUNDED f32
                        # mul then add: mul.rn is non-contractible (PTX), so
                        # ptxas cannot fuse this into an FFMA — keeping the
                        # kernel byte-exact vs the torch mul+add reference
                        # for ARBITRARY f32 scales (the RHT producer's
                        # amax/(6*448) descales are not powers of two).
                        tTR_rAcc_subtile = tTR_rAcc_final[(None, None, None, subtile_idx)]
                        tTR_rSFA2_sub = tTR_rSFA2[(None, None, None, subtile_idx)]
                        for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc), 2):
                            s0, s1 = cute.arch.mul_packed_f32x2(
                                (tTR_rAcc[i], tTR_rAcc[i + 1]),
                                (tTR_rSFA2_sub[i], tTR_rSFA2_sub[i + 1]),
                                rnd="rn",
                                ftz=False,
                            )
                            tTR_rAcc_subtile[i] = tTR_rAcc_subtile[i] + s0
                            tTR_rAcc_subtile[i + 1] = tTR_rAcc_subtile[i + 1] + s1

                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                    peek_acc_full_status = cutlass.Boolean(1)
                    if acc_consumer_state.count < k_tile_cnt:
                        peek_acc_full_status = acc_pipeline.consumer_try_wait(acc_consumer_state)
                    peek_scale_full_status = cutlass.Boolean(1)
                    if scale_consumer_state.count < k_tile_cnt:
                        peek_scale_full_status = scale_pipeline.consumer_try_wait(scale_consumer_state)

                # All scale blocks accumulated (or none: zeros) — publish the
                # final tile to the epilogue through sFinalAcc.
                epi_pipeline.producer_acquire(epi_producer_state)

                final_subtile_cnt = cute.size(tTR_rAcc_final.shape, mode=[3])
                slot_base = epi_producer_state.index * final_subtile_cnt
                for subtile_idx in cutlass.range(final_subtile_cnt, unroll_full=True):
                    tRS_rFinal = tiled_copy_r2s_acc.retile(tTR_rAcc_final[(None, None, None, subtile_idx)])
                    cute.copy(
                        tiled_copy_r2s_acc,
                        tRS_rFinal,
                        tRS_sFinalAcc[(None, None, None, slot_base + subtile_idx)],
                    )

                epi_pipeline.producer_commit(epi_producer_state)
                epi_producer_state.advance()

                sched_pipeline.consumer_wait(sched_consumer_state)
                rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
                cute.copy(
                    sched_copy_atom,
                    sched_buf_tensor[(None, sched_consumer_state.index)],
                    rmem,
                )
                work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
                cute.arch.fence_acq_rel_cta()
                sched_pipeline.consumer_release(sched_consumer_state)
                sched_consumer_state.advance()

        # =================================================================
        # Epilogue warps (warps 4-7): sFinalAcc -> alpha -> bf16 -> TMA C
        # =================================================================
        if warp_idx in self.epilog_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_epilogue_warps)
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            # Shape/TV-layout carrier only — the final accumulator lives in
            # SMEM (sFinalAcc); TMEM holds just MMA partials + SF.
            tCtAcc_base_ = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            gC_mnl_shape = cute.local_tile(
                mC_mnl,
                cute.slice_(self.mma_tiler, (None, None, 0)),
                (None, None, None),
            )
            thr_mma_epi = tiled_mma.get_slice(mma_tile_coord_v)
            tCgC_shape = thr_mma_epi.partition_C(gC_mnl_shape)

            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx,
                tCtAcc_base_,
                tCgC_shape,
                epi_tile,
                use_2cta_instrs,
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r,
                tTR_rC,
                epi_tidx,
                sC,
            )
            tiled_copy_s2r, tSR_rAcc, tSR_sFinalAcc = self.epilog_smem_load_copy_and_partition(
                tiled_copy_t2r,
                tTR_rAcc,
                epi_tidx,
                sFinalAcc,
            )

            epi_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_epi_stage)
            sched_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_sched_stages)
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_c_stage, producer_group=c_producer_group)

            epilog_sync_barrier = pipeline.NamedBarrier(
                barrier_id=self.epilog_sync_bar_id,
                num_threads=32 * len(self.epilog_warp_id),
            )

            num_prev_subtiles = cutlass.Int32(0)

            sched_pipeline.consumer_wait(sched_consumer_state)
            rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
            cute.copy(
                sched_copy_atom,
                sched_buf_tensor[(None, sched_consumer_state.index)],
                rmem,
            )
            work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
            cute.arch.fence_acq_rel_cta()
            sched_pipeline.consumer_release(sched_consumer_state)
            sched_consumer_state.advance()

            while work_tile_info.is_valid_tile:
                ext.update_expert_info(offs, work_tile_info.expert_idx)

                real_c, desc_ptr_c = ext.get_gmem_tensor(
                    "c",
                    mC_mnl,
                    offs,
                    work_tile_info,
                )

                gC_mnl_loop = cute.local_tile(
                    real_c,
                    cute.slice_(self.mma_tiler, (None, None, 0)),
                    (None, None, None),
                )
                tCgC_loop = thr_mma_epi.partition_C(gC_mnl_loop)
                _, bSG_sC, bSG_gC_partitioned = epilog_gmem_copy_and_partition(
                    epi_tidx,
                    tma_atom_c,
                    tCgC_loop,
                    epi_tile,
                    sC,
                )

                mma_tile_coord_mnl = (
                    work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape),
                    work_tile_info.tile_n_idx,
                    cutlass.Int32(0),
                )
                bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                if cutlass.const_expr(global_scale_a is not None):
                    expert_idx = work_tile_info.expert_idx
                    current_scale_a_iter = global_scale_a.iterator + expert_idx
                    current_scale_b_iter = global_scale_b.iterator + expert_idx
                    alpha = cute.arch.load(current_scale_a_iter.llvm_ptr, cutlass.Float32) * cute.arch.load(current_scale_b_iter.llvm_ptr, cutlass.Float32)
                else:
                    alpha = None

                epi_stage_index = epi_consumer_state.index

                # Wait for the final accumulator from the acc-update warps
                epi_pipeline.consumer_wait(epi_consumer_state)

                subtile_cnt = self.final_acc_subtile_cnt
                for subtile_idx in cutlass.range(0, subtile_cnt, 1, unroll=1):
                    final_slot = epi_stage_index * subtile_cnt + subtile_idx
                    cute.copy(
                        tiled_copy_s2r,
                        tSR_sFinalAcc[(None, None, None, final_slot)],
                        tSR_rAcc,
                    )

                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    if cutlass.const_expr(global_scale_a is not None):
                        acc_vec = acc_vec * alpha
                    acc_vec = acc_vec.to(self.c_dtype)
                    tRS_rC.store(acc_vec)

                    c_buffer = num_prev_subtiles % self.num_c_stage
                    num_prev_subtiles = num_prev_subtiles + 1
                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    epilog_sync_barrier.arrive_and_wait()

                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, subtile_idx)],
                            tma_desc_ptr=desc_ptr_c,
                        )
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    epilog_sync_barrier.arrive_and_wait()

                epi_pipeline.consumer_release(epi_consumer_state)
                epi_consumer_state.advance()

                sched_pipeline.consumer_wait(sched_consumer_state)
                rmem = cute.make_rmem_tensor((4,), cutlass.Int32)
                cute.copy(
                    sched_copy_atom,
                    sched_buf_tensor[(None, sched_consumer_state.index)],
                    rmem,
                )
                work_tile_info = MoEWorkTileInfo.from_rmem_tensor(rmem)
                cute.arch.fence_acq_rel_cta()
                sched_pipeline.consumer_release(sched_consumer_state)
                sched_consumer_state.advance()

            c_pipeline.producer_tail()

            tmem.relinquish_alloc_permit()
            epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
