# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Second-Level-Scaled (Subchannel-Scaled) MoE Block-Scaled Grouped GEMM Kernel.

Supports:
    - Static / Dynamic persistent tile scheduling (MoEPersistentTileScheduler)
    - Dense (contiguous 3-D B) / Discrete (per-expert pointer array B) weight layout
    - Optional bias and routing-probability (prob) fusion

This module contains only the kernel class.
MoE scheduler components live in moe_persistent_scheduler.py / moe_sched_extension.py / moe_utils.py.
"""

from typing import Type, Tuple, Union, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu import OperandMajorMode
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.typing import Float32, Int32, AddressSpace

from cutlass._mlir.dialects.nvvm import (
    SetMaxRegisterAction,
)

from ..moe_persistent_scheduler import (
    MoEPersistentTileScheduler,
    MoESchedulerParams,
    MoEWorkTileInfo,
)
from ..moe_utils import (
    compute_expert_token_range,
    MoEWeightMode,
    TensormapWorkspace,
    store_tma_desc,
)
from ..moe_sched_extension import (
    DiscreteWeightScaledGemmSchedExtension,
    ContiguousAndConsistentGroupedGemmSchedExtension,
)
from ..moe_kernel_helpers import (
    compute_stages,
    compute_grid,
    can_implement,
    epilog_gmem_copy_and_partition,
)

# Valid launch-config space for THIS kernel (can_implement still prunes per
# problem). cta shape == mma_tiler_mn. Default matches dgrad_dglu's pinned
# config for iso-tile benchmark baselines.
VALID_CTA_SHAPES = ((256, 128), (256, 256))
VALID_CLUSTER_SHAPES = ((1, 1), (2, 1), (2, 2), (4, 1), (4, 2))
DEFAULT_CTA_SHAPE = (256, 128)
DEFAULT_CLUSTER_SHAPE = (2, 1)


class BlockScaledSubChannelMoEGroupedGemmKernel:
    """Sub-channel scaled block-scaled grouped GEMM kernel with MoE tile scheduling.

    Supports both dense and discrete weight layouts and static and dynamic
    scheduling. Inputs carry two scale levels (per-(1,16) SFA/SFB plus f32
    SFA2/SFB2 subchannel scales); the output D is plain BF16.

    :param sf_vec_size: Scale-factor vector size (16 or 32).
    :param acc_dtype: Accumulator data type (Float32).
    :param use_2cta_instrs: Use 2-CTA MMA instructions.
    :param mma_tiler_mn: MMA tile shape (M, N).
    :param cluster_shape_mn: Cluster shape (M, N).
    :param vectorized_f32: Use packed FP32 arithmetic.
    :param enable_bias: Fuse bias addition.
    :param expert_cnt: Number of experts.
    :param weight_mode: ``MoEWeightMode.DENSE`` or ``MoEWeightMode.DISCRETE``.
    :param use_dynamic_sched: Enable dynamic tile scheduling.
    """

    FIX_PAD_SIZE = 256

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        acc_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        cd_major: str,
        m_aligned: int,
        weight_mode: MoEWeightMode = MoEWeightMode.DENSE,
        sgn: int = 256,
        sgk: int = 256,
    ) -> bool:
        result = can_implement(
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            acc_dtype,
            d_dtype,
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            m,
            n,
            k,
            l,
            a_major,
            b_major,
            cd_major,
            m_aligned,
            fix_pad_size=BlockScaledSubChannelMoEGroupedGemmKernel.FIX_PAD_SIZE,
            allowed_mma_tiler_n=(256, 128),
        )
        # Discrete SFB2 per-expert base pointers are declared 16B-aligned in
        # the sched extension (assumed_align=16). With back-to-back per-expert
        # f32 (ceil(n/sgn), ceil(k/sgk)) blocks, every base past the first is
        # misaligned unless the block byte size is a multiple of 16.
        if weight_mode == MoEWeightMode.DISCRETE and l > 1:
            sfb2_block_bytes = ((n + sgn - 1) // sgn) * ((k + sgk - 1) // sgk) * 4
            if sfb2_block_bytes % 16 != 0:
                result = False
        return result

    def __init__(
        self,
        sf_vec_size: int,
        sgm: int,
        sgn: int,
        sgk: int,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        vectorized_f32: bool,
        enable_bias: bool,
        expert_cnt: int,
        weight_mode: MoEWeightMode = MoEWeightMode.DENSE,
        use_dynamic_sched: bool = False,
    ):
        # Validate constructor arguments before storing any configuration
        mma_tile_m = mma_tiler_mn[0]
        if self.FIX_PAD_SIZE % mma_tile_m != 0:
            raise ValueError(
                f"FIX_PAD_SIZE ({self.FIX_PAD_SIZE}) must be divisible by " f"mma_tiler_mn[0] ({mma_tile_m}). " f"Supported mma_tiler_mn[0] values: 128, 256."
            )
        if expert_cnt > 1024:
            raise ValueError("Expert count > 1024 is not supported.")
        if not isinstance(weight_mode, MoEWeightMode):
            raise TypeError(f"weight_mode must be a MoEWeightMode, got {type(weight_mode)}")

        # Store core GEMM / block-scale configuration
        self.sf_vec_size = sf_vec_size
        self.sgm = sgm
        self.sgn = sgn
        self.sgk = sgk
        self.expert_cnt = expert_cnt
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)

        # Select 1-CTA or 2-CTA MMA instruction group
        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

        # Warp specialization: assign a fixed warp id to each role
        # The bias load rides on the scale warp: a 13th warp would leave a
        # partial warpgroup, which is illegal for setmaxregister (the CTA must
        # be a whole number of 128-thread warpgroups).
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
        # Per-role register budgets applied via setmaxregister (uniform vs compute warps)
        self.num_regs_uniform_warps = 24
        self.num_regs_sched_warps = 24
        self.num_regs_epilogue_warps = 168
        self.num_regs_acc_update_warps = 176
        warps_wo_sched = [*self.accumulator_update_warp_id, *self.epilog_warp_id, self.mma_warp_id, self.tma_warp_id, self.scale_warp_id]
        # Total threads per CTA (with and without the scheduler warp)
        self.threads_per_cta = self.threads_per_warp * len(all_warps)
        self.threads_wo_sched = self.threads_per_warp * len(warps_wo_sched)

        # Named barriers for cross-warp synchronization
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * len((self.mma_warp_id, *self.accumulator_update_warp_id, *self.epilog_warp_id)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )
        # Query SMEM/TMEM hardware capacities used later for stage sizing
        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        # Store remaining feature flags and modes
        self.vectorized_f32 = vectorized_f32
        self.enable_bias = enable_bias

        self.weight_mode = weight_mode
        self.use_dynamic_sched = use_dynamic_sched

        self.epilogue_use_functor = False

        # Cache warp-group sizes for the compute warps
        self.num_accumulator_update_warps = len(self.accumulator_update_warp_id)
        self.num_epilog_warps = len(self.epilog_warp_id)

    # ------------------------------------------------------------------
    # _setup_attributes
    # ------------------------------------------------------------------

    def _setup_attributes(self):
        """Configure MMA / tile / stage / SMEM layouts from GEMM inputs."""

        # MMA instruction MN shape, plus a separate SFB variant (1-CTA, N rounded to 128)
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        # Build the tiled MMA (and an SFB-only tiled MMA) for the block-scaled GEMM
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        # Extend the MMA tiler with the K extent (instruction K times tiles-per-loop)
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        # Per-CTA tile shapes (MMA tiler divided by the number of CTAs in the MMA group)
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

        # Output (D) MMA tiler and its per-CTA tile shape
        self.mma_tiler_d = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.cta_tile_shape_mnk_d = (
            self.mma_tiler_d[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_d[1],
            self.mma_tiler_d[2],
        )

        # Cluster layouts (V, M, N, K) for A/B and for SFB multicast partitioning
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Number of CTAs each A/B tile is multicast to across the cluster
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Epilogue subtile shape used to stream the accumulator out to D
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.d_layout,
            self.d_dtype,
        )

        # Derive pipeline stage counts from the available SMEM budget
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_d_stage,
            self.num_tile_stage,
            self.num_bias_stage,
            self.num_epi_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.d_dtype,
            self.d_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.num_smem_capacity,
            self.occupancy,
            self.bias_dtype if self.enable_bias else None,
            acc_dtype=self.acc_dtype,
            sf2_dtype=self.sf2_dtype,
            sgm=self.sgm,
            sgn=self.sgn,
            sgk=self.sgk,
        )

        # Second-level scale pipeline runs in lockstep with the AB pipeline
        self.num_scale_stage = self.num_ab_stage

        # Staged SMEM layouts for A/B operands and their first-level scale factors
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

        # Store the per-tile scale block sizes (each clamped to the CTA tile) so
        # the device kernel builds C-shaped scale views that match the CTA tile
        # exactly, even when sg{m,n,k} > the tile extent (e.g. sgn=512 with a
        # 256-wide N tile spans one scale block, broadcast across the tile).
        if self.sgm < self.cta_tile_shape_mnk[0]:
            size_m = self.sgm
        else:
            size_m = self.cta_tile_shape_mnk[0]

        if self.sgn < self.cta_tile_shape_mnk[1]:
            size_n = self.sgn
        else:
            size_n = self.cta_tile_shape_mnk[1]

        if self.sgk < self.cta_tile_shape_mnk[2]:
            size_k = self.sgk
        else:
            size_k = self.cta_tile_shape_mnk[2]

        self.scale_size_m = size_m
        self.scale_size_n = size_n
        self.scale_size_k = size_k

        # How many scale blocks fit along each CTA-tile dimension
        self.scale_m_per_tile = self.cta_tile_shape_mnk[0] // size_m
        self.scale_n_per_tile = self.cta_tile_shape_mnk[1] // size_n
        self.scale_k_per_tile = self.cta_tile_shape_mnk[2] // size_k

        # Staged SMEM layouts for the second-level (f32) A/B scales; extent-0 modes broadcast
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
        self.sfb2_smem_layout_staged = cute.make_layout(
            (
                (size_n, self.scale_n_per_tile),
                (size_k, self.scale_k_per_tile),
                self.num_scale_stage,
            ),
            stride=(
                (0, self.scale_k_per_tile),
                (0, 1),
                self.scale_k_per_tile * self.scale_n_per_tile,
            ),
        )

        # Staged SMEM layout for the epilogue output (D) buffer
        self.d_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.d_dtype,
            self.d_layout,
            self.epi_tile,
            self.num_d_stage,
        )

        # Full-CTA-tile SMEM buffer for the final accumulator; the "stage" mode of the
        # epi layout doubles as the subtile-slot index (subtile_cnt slots per epi stage)
        self.final_acc_subtile_cnt = (self.cta_tile_shape_mnk[0] // cute.size(self.epi_tile[0])) * (self.cta_tile_shape_mnk[1] // cute.size(self.epi_tile[1]))
        self.final_acc_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.acc_dtype,
            self.d_layout,
            self.epi_tile,
            self.final_acc_subtile_cnt * self.num_epi_stage,
        )

        # Staged SMEM layout for bias (only allocated when bias fusion is enabled)
        if self.enable_bias:
            self.bias_smem_layout_staged = cute.make_layout(
                (self.mma_tiler[1], self.num_bias_stage),
                stride=(1, self.mma_tiler[1]),
            )
        else:
            self.bias_smem_layout_staged = cute.make_layout((1, 1))

        self.overlapping_accum = self.num_acc_stage == 1 and self.mma_tiler[1] == 256
        self.epilogue_prefetch_more = False

        # TMEM column budget: accumulator columns followed by SFA/SFB scale columns
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = self.cta_tile_shape_mnk[1] * self.num_acc_stage

        # Epilogue N tile width and how early the accumulator TMEM can be released
        self.epi_tile_n_required = cute.size(self.epi_tile[1])
        self.iter_acc_early_release_in_epilogue = (self.num_sf_tmem_cols + self.epi_tile_n_required - 1) // self.epi_tile_n_required - 1

    # ------------------------------------------------------------------
    # _compute_stages (with bias support)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stages(
        tiled_mma,
        mma_tiler_mnk,
        a_dtype,
        b_dtype,
        epi_tile,
        d_dtype,
        d_layout,
        sf_dtype,
        sf_vec_size,
        num_smem_capacity,
        occupancy,
        bias_dtype,
        acc_dtype=None,
        sf2_dtype=None,
        sgm=1,
        sgn=256,
        sgk=256,
    ):
        # Fixed stage counts; the AB stage count is derived from leftover SMEM below
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 3
        num_d_stage = 1
        num_tile_stage = 2

        # Single-stage layouts, used only to measure per-stage byte sizes
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        d_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(d_dtype, d_layout, epi_tile, 1)

        # Compute second level scale factor layout
        # Store the per-tile scale block sizes (each clamped to the CTA tile) so
        # the device kernel builds C-shaped scale views that match the CTA tile
        # exactly, even when sg{m,n,k} > the tile extent (e.g. sgn=512 with a
        # 256-wide N tile spans one scale block, broadcast across the tile).
        if sgm < mma_tiler_mnk[0]:
            size_m = sgm
        else:
            size_m = mma_tiler_mnk[0]

        if sgn < mma_tiler_mnk[1]:
            size_n = sgn
        else:
            size_n = mma_tiler_mnk[1]

        if sgk < mma_tiler_mnk[2]:
            size_k = sgk
        else:
            size_k = mma_tiler_mnk[2]

        scale_m_per_tile = mma_tiler_mnk[0] // size_m
        scale_n_per_tile = mma_tiler_mnk[1] // size_n
        scale_k_per_tile = mma_tiler_mnk[2] // size_k

        # Second-level scale factor layouts
        sfa2_smem_layout_staged_one = cute.make_layout(
            (
                (size_m, scale_m_per_tile),
                (size_k, scale_k_per_tile),
                1,
            ),
            stride=(
                (0, scale_k_per_tile),
                (0, 1),
                scale_k_per_tile * scale_m_per_tile,
            ),
        )
        sfb2_smem_layout_staged_one = cute.make_layout(
            (
                (size_n, scale_n_per_tile),
                (size_k, scale_k_per_tile),
                1,
            ),
            stride=(
                (0, scale_k_per_tile),
                (0, 1),
                scale_k_per_tile * scale_n_per_tile,
            ),
        )

        # Bytes consumed by one AB stage (operands plus both scale-factor levels)
        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
            + cute.size_in_bytes(sf2_dtype, sfa2_smem_layout_staged_one)
            + cute.size_in_bytes(sf2_dtype, sfb2_smem_layout_staged_one)
        )
        # Fixed overheads: mbarriers, scheduler tile-info, and staged D output
        mbar_helpers_bytes = 1024
        sinfo_bytes = 4 * 4 * num_tile_stage
        d_bytes_per_stage = cute.size_in_bytes(d_dtype, d_smem_layout_staged_one)
        d_bytes = d_bytes_per_stage * num_d_stage

        # Bias staging bytes (zero when bias is disabled)
        if bias_dtype is not None:
            num_bias_stage = 2
            bias_epi_tile_n = mma_tiler_mnk[1]
            bias_bytes = bias_epi_tile_n * num_bias_stage * (bias_dtype.width // 8)
        else:
            num_bias_stage = 0
            bias_bytes = 0

        num_epi_stage = 1  # Pipeline stages for epi_pipeline (accumulator update -> epilogue)

        # Full-CTA-tile final accumulator buffer in SMEM (replaces the TMEM final region)
        cta_m = mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id.shape)
        cta_n = mma_tiler_mnk[1]
        subtile_cnt = (cta_m // cute.size(epi_tile[0])) * (cta_n // cute.size(epi_tile[1]))
        final_acc_smem_layout_one = sm100_utils.make_smem_layout_epi(
            acc_dtype,
            d_layout,
            epi_tile,
            subtile_cnt * num_epi_stage,
        )
        final_acc_bytes = cute.size_in_bytes(acc_dtype, final_acc_smem_layout_one)

        # SMEM left after fixed + epilogue usage is divided into as many AB stages as fit
        epi_bytes = d_bytes + bias_bytes + final_acc_bytes
        num_ab_stage = (num_smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes + sinfo_bytes)) // ab_bytes_per_stage
        assert num_ab_stage >= 1, (
            f"SMEM overflow: full-tile sFinalAcc ({final_acc_bytes} B) leaves no room for " f"AB stages (mma_tiler={mma_tiler_mnk}); reduce mma_tiler_mn"
        )

        return num_acc_stage, num_ab_stage, num_d_stage, num_tile_stage, num_bias_stage, num_epi_stage

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

    def get_desc_workspace_bytes(self) -> int:
        # Bytes for per-expert TMA descriptors (discrete weight mode only)
        if self.weight_mode == MoEWeightMode.DISCRETE:
            from ..moe_utils import DiscreteWeightTensormapConstructor

            return DiscreteWeightTensormapConstructor.get_workspace_size(self.expert_cnt)
        return 0

    def get_workspace_bytes(self) -> int:
        # Total workspace = TMA descriptor region + optional 4B dynamic-sched counter
        desc_workspace_bytes = self.get_desc_workspace_bytes()
        dynamic_sched_bytes = 4 if self.use_dynamic_sched else 0
        return desc_workspace_bytes + dynamic_sched_bytes

    @cute.jit
    def _get_sched_counter_ptr(self, workspace_ptr):
        # Atomic tile counter lives just past the descriptor region in the workspace
        counter_addr = workspace_ptr.toint() + self.get_desc_workspace_bytes()
        return cute.make_ptr(
            cutlass.Int32,
            counter_addr,
            AddressSpace.gmem,
            assumed_align=4,
        )

    # ------------------------------------------------------------------
    # helper_kernel: pre-main-kernel initialization
    #   - discrete weight: build per-expert B/SFB TMA descriptors
    #   - dynamic sched: reset the atomic tile counter
    # ------------------------------------------------------------------
    @cute.kernel
    def helper_kernel(
        self,
        # Discrete-only params (unused in dense mode, but must be present for signature)
        ptrs_b: cute.Pointer,
        ptrs_sfb: cute.Pointer,
        n: Int32,
        k: Int32,
        b_stride_size: cutlass.Int64,
        b_major_mode: cutlass.Constexpr,
        workspace_ptr,
        tiled_mma_arg: cute.TiledMma,
        tiled_mma_sfb_arg: cute.TiledMma,
        b_smem_layout_arg,
        sfb_smem_layout_arg,
        cluster_layout_vmnk_shape_arg: cutlass.Constexpr,
        cluster_layout_sfb_vmnk_shape_arg: cutlass.Constexpr,
    ):
        """Pre-main-kernel initialization.

        Launched with grid=(expert_cnt, 1, 1) for discrete mode, or
        grid=(1, 1, 1) for dense+dynamic mode.

        Discrete weight: each block builds B/SFB TMA descriptors for one expert.
        Dynamic sched: block 0 resets the atomic tile counter to 0.
        """
        # One block per expert (discrete) or a single block (dense + dynamic sched)
        expert_idx = cute.arch.block_idx()[0]

        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE):
            # TMA copy atoms for B / SFB given the cluster shape
            b_tma_op_arg = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma_arg.thr_id)
            sfb_tma_op_arg = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma_arg.thr_id)

            # Wrap the device pointer arrays as tensors indexed by expert
            b_ptr_tensor = cute.make_tensor(
                cute.make_ptr(cutlass.Int64, ptrs_b.toint(), AddressSpace.gmem, assumed_align=8), cute.make_layout((self.expert_cnt,))
            )
            sfb_ptr_tensor = cute.make_tensor(
                cute.make_ptr(cutlass.Int64, ptrs_sfb.toint(), AddressSpace.gmem, assumed_align=8), cute.make_layout((self.expert_cnt,))
            )

            # B strides depend on whether B is K-major or N-major
            c1 = cutlass.Int32(1)
            c0 = cutlass.Int64(0)
            c1_64 = 1
            if cutlass.const_expr(b_major_mode == OperandMajorMode.K):
                stride_n = b_stride_size
                stride_k = c1_64
            else:
                stride_n = c1_64
                stride_k = b_stride_size

            # Build this expert's B tensor and its TMA descriptor
            b_ptr_val = b_ptr_tensor[expert_idx]
            b_ptr = cute.make_ptr(self.b_dtype, b_ptr_val, AddressSpace.gmem)
            b_tensor_i = cute.make_tensor(
                b_ptr,
                cute.make_layout((n, k, c1), stride=(stride_n, stride_k, c0)),
            )
            tma_atom_b, _ = cute.nvgpu.make_tiled_tma_atom_B(
                b_tma_op_arg,
                b_tensor_i,
                b_smem_layout_arg,
                self.mma_tiler,
                tiled_mma_arg,
                cluster_layout_vmnk_shape_arg,
            )

            # Store the B descriptor into this expert's workspace slot
            workspace = TensormapWorkspace(workspace_ptr, ["b", "sfb"])
            store_tma_desc(tma_atom_b, workspace.get_ptr("b", expert_idx))

            # Build this expert's SFB tensor and its TMA descriptor, then store it
            sfb_ptr_val = sfb_ptr_tensor[expert_idx]
            sfb_ptr = cute.make_ptr(self.sf_dtype, sfb_ptr_val, AddressSpace.gmem)
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF((n, k, c1), self.sf_vec_size)
            sfb_tensor_i = cute.make_tensor(sfb_ptr, sfb_layout)
            tma_atom_sfb, _ = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_tma_op_arg,
                sfb_tensor_i,
                sfb_smem_layout_arg,
                self.mma_tiler_sfb,
                tiled_mma_sfb_arg,
                cluster_layout_sfb_vmnk_shape_arg,
                internal_type=cutlass.Uint64,
            )
            store_tma_desc(tma_atom_sfb, workspace.get_ptr("sfb", expert_idx))

        # Dynamic sched: block 0 zeroes the atomic tile counter
        if cutlass.const_expr(self.use_dynamic_sched):
            if expert_idx == cutlass.Int32(0):
                sched_counter = cute.make_tensor(
                    self._get_sched_counter_ptr(workspace_ptr),
                    cute.make_layout(1),
                )
                sched_counter[0] = cutlass.Int32(0)

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b,  # Dense: cute.Tensor (N,K,L) | Discrete: cute.Pointer to int64[]
        sfb,  # Dense: cute.Tensor         | Discrete: cute.Pointer to int64[]
        sfb2,  # Dense: cute.Tensor         | Discrete: cute.Pointer to int64[]
        n: Int32,  # Ignored for dense mode
        k: Int32,  # Ignored for dense mode
        b_stride_size: cutlass.Int64,  # Ignored for dense mode
        b_major_mode: cutlass.Constexpr,  # Ignored for dense mode
        workspace_ptr,
        d: cute.Tensor,
        sfa: cute.Tensor,
        sfa2: cute.Tensor,
        padded_offsets: cute.Tensor,
        alpha: cute.Tensor,
        bias: Optional[cute.Tensor],
        prob: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM.

        Dense mode: ``b`` and ``sfb`` are 3-D cute.Tensor (N, K, L).
        Discrete mode: ``b`` and ``sfb`` are cute.Pointer to device int64[]
        arrays of per-expert base addresses; ``n``, ``k``, ``b_stride_size``,
        ``b_major_mode`` describe the uniform per-expert layout.
        """
        # Resolve operand/output dtypes and A/D major modes from the input tensors
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = a.element_type
        self.d_dtype: Type[cutlass.Numeric] = d.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.sf2_dtype: Type[cutlass.Numeric] = sfa2.element_type
        self.bias_dtype = bias.element_type if cutlass.const_expr(self.enable_bias) else cutlass.BFloat16
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.d_layout = utils.LayoutEnum.from_tensor(d)

        # Dense B carries its own layout; discrete mode passes b_major_mode explicitly
        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
            self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        else:
            self.b_major_mode = b_major_mode

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"A/B dtype must match: {self.a_dtype} != {self.b_dtype}")

        # Compute derived tile / stage / SMEM-layout attributes
        self._setup_attributes()

        # ---- SFA layout ----
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        sfa = cute.make_tensor(sfa.iterator, sfa_layout)

        # ---- SFA2 layout ----
        sfa2_tensor = cute.make_tensor(
            sfa2.iterator,
            cute.make_layout(
                (
                    (self.sgm, sfa2.shape[0]),
                    (self.sgk, sfa2.shape[1]),
                    sfa2.shape[2],
                ),
                stride=(
                    (0, sfa2.layout.stride[0]),
                    (0, sfa2.layout.stride[1]),
                    sfa2.layout.stride[2],
                ),
            ),
        )

        c1 = cutlass.Int32(1)
        c0 = cutlass.Int64(0)
        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
            sfb2_ptr_typed = sfb2.iterator
            sfb2_shape_0 = sfb2.shape[0]
            sfb2_shape_1 = sfb2.shape[1]
            sfb2_mode_2 = sfb2.shape[2]
            sfb2_stride_0 = sfb2.layout.stride[0]
            sfb2_stride_1 = sfb2.layout.stride[1]
            sfb2_stride_2 = sfb2.layout.stride[2]
        else:  # DISCRETE mode
            # In discrete mode, sfb2 is a pointer to array of per-expert pointers (Int64[])
            # Create a template tensor with correct element type (sf2_dtype) for get_gmem_tensor
            # The pointer address is the pointer array address; get_gmem_tensor will load actual per-expert pointer
            sfb2_ptr_typed = cute.make_ptr(
                self.sf2_dtype,  # Correct data type so element_type is right
                sfb2.toint(),  # Pointer array address (will be reinterpreted in get_gmem_tensor)
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            # Use computed scale values for template shape/stride
            scale_n = cute.ceil_div(n, self.sgn)
            scale_k = cute.ceil_div(k, self.sgk)
            sfb2_shape_0 = scale_n
            sfb2_shape_1 = scale_k
            sfb2_mode_2 = c1
            sfb2_stride_0 = c1
            sfb2_stride_1 = scale_n
            sfb2_stride_2 = c0

        # ---- SFB2 layout ----
        # Dense: use actual tensor shape/stride
        # Discrete: use template shape/stride (get_gmem_tensor will load per-expert pointer)
        sfb2_tensor = cute.make_tensor(
            sfb2_ptr_typed,
            cute.make_layout(
                (
                    (self.sgn, sfb2_shape_0),
                    (self.sgk, sfb2_shape_1),
                    sfb2_mode_2,
                ),
                stride=(
                    (0, sfb2_stride_0),
                    (0, sfb2_stride_1),
                    sfb2_stride_2,
                ),
            ),
        )

        # ---- B / SFB setup (mode-dependent) ----
        # Save the call-arg b/sfb before the discrete branch overwrites them
        # with template tensors.  helper_kernel needs the original Pointers.
        b_from_call_arg = b
        sfb_from_call_arg = sfb
        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size)
            sfb = cute.make_tensor(sfb.iterator, sfb_layout)
        else:
            c1 = cutlass.Int32(1)
            c0 = cutlass.Int64(0)
            c1_64 = 1
            if cutlass.const_expr(b_major_mode == OperandMajorMode.K):
                b_template_stride = (b_stride_size, c1_64, c0)
            else:
                b_template_stride = (c1_64, b_stride_size, c0)

            # Creating a template layout and B tensor for a single expert
            b_template_layout = cute.make_layout((n, k, c1), stride=b_template_stride)
            b_ptr_typed = cute.make_ptr(self.b_dtype, b.toint(), AddressSpace.gmem, assumed_align=16)
            b = cute.make_tensor(b_ptr_typed, b_template_layout)

            # Creating a template layout and SFB tensor for a single expert
            sfb_ptr_typed = cute.make_ptr(self.sf_dtype, sfb.toint(), AddressSpace.gmem, assumed_align=16)
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF((n, k, c1), self.sf_vec_size)
            sfb = cute.make_tensor(sfb_ptr_typed, sfb_layout)

        # ---- TMA atoms ----
        # Rebuild the tiled MMAs on the now-resolved dtypes / major modes
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # A operand TMA atom
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # B operand TMA atom
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # SFA (first-level A scale) TMA atom
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # SFB (first-level B scale) TMA atom
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Uint64,
        )

        # Total bytes moved per TMA load (used to size the AB pipeline barrier)
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size) * atom_thr_size

        # D output TMA atom (SMEM -> GMEM bulk tensor store)
        d_smem_layout = cute.slice_(self.d_smem_layout_staged, (None, None, 0))
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            d,
            d_smem_layout,
            self.epi_tile,
        )

        # ---- Helper kernel: TMA desc init (discrete) + sched counter reset (dynamic) ----
        _need_helper = cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE or self.use_dynamic_sched)
        if cutlass.const_expr(_need_helper):
            _helper_grid_x = self.expert_cnt if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else 1
            _helper_args = (
                b_from_call_arg if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cute.make_ptr(cutlass.Int64, 0, AddressSpace.gmem),
                sfb_from_call_arg if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cute.make_ptr(cutlass.Int64, 0, AddressSpace.gmem),
                n if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cutlass.Int32(0),
                k if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cutlass.Int32(0),
                b_stride_size if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else cutlass.Int64(0),
                b_major_mode if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE) else self.b_major_mode,
                workspace_ptr,
                tiled_mma,
                tiled_mma_sfb,
                b_smem_layout,
                sfb_smem_layout,
                self.cluster_layout_vmnk.shape,
                self.cluster_layout_sfb_vmnk.shape,
            )
            self.helper_kernel(*_helper_args).launch(
                grid=(_helper_grid_x, 1, 1),
                block=(1, 1, 1),
                stream=stream,
                min_blocks_per_mp=1,
            )

        # ---- Grid computation via MoE scheduler ----
        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
            b_n, b_k, b_l = cute.shape(b)  # B is (N, K, L)
            sched_expert_shape = (self.expert_cnt, b_n, b_k)
        else:
            sched_expert_shape = (self.expert_cnt, n, k)

        sched_params = MoESchedulerParams(
            scenario="2Dx3D",
            expert_shape=sched_expert_shape,
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            use_dynamic_sched=self.use_dynamic_sched,
        )
        self.sched_params, grid = compute_grid(sched_params, max_active_clusters, self.use_2cta_instrs)

        self.buffer_align_bytes = 1024

        # ---- Shared storage ----
        SchedulerStorage = MoEPersistentTileScheduler.make_storage_struct(self.num_tile_stage, self.use_dynamic_sched)

        # Per-CTA SMEM layout: pipeline mbarriers, scheduler state, and all staged buffers
        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            scale_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_scale_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            epi_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_epi_stage * 2]
            if cutlass.const_expr(self.enable_bias):
                bias_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_bias_stage * 2]
            scheduler: SchedulerStorage
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, cute.cosize(self.d_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sFinalAcc: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(self.final_acc_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFA2: cute.struct.Align[
                cute.struct.MemRange[self.sf2_dtype, cute.cosize(self.sfa2_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB2: cute.struct.Align[
                cute.struct.MemRange[self.sf2_dtype, cute.cosize(self.sfb2_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            if cutlass.const_expr(self.enable_bias):
                sBias: cute.struct.Align[
                    cute.struct.MemRange[self.bias_dtype, cute.cosize(self.bias_smem_layout_staged)],
                    16,
                ]

        self.shared_storage = SharedStorage

        # ---- Launch ----
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
            sfa2_tensor,
            sfb2_tensor,
            tma_atom_d,
            tma_tensor_d,
            padded_offsets,
            alpha,
            bias,
            prob,
            workspace_ptr,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.sfa2_smem_layout_staged,
            self.sfb2_smem_layout_staged,
            self.d_smem_layout_staged,
            self.final_acc_smem_layout_staged,
            self.bias_smem_layout_staged,
            self.epi_tile,
            self.sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            max_number_threads=[self.threads_per_cta, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def mainloop_s2t_copy_and_partition(self, sSF, tSF):
        # Build the SMEM->TMEM copy for scale factors and partition src/dst (zeros filtered out)
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # Convert the SMEM source into an S2T descriptor tensor expected by the copy
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def acc_update_tmem_copy_and_partition(self, tidx, tCtAcc_base, gD_mnl, sSFA, sSFB, sFinalAcc, epi_tile):
        """
        Create tiled copy operations and partition tensors for accumulator update warp.

        This function sets up T2R (TMEM-to-Register) and R2S (Register-to-SMEM) copies
        for the accumulator update warp to:
        1. Load partial accumulators from TMEM (tCtAcc_base) to registers
        2. Store final accumulated result from registers to SMEM (sFinalAcc)

        :param tidx: Thread index within accumulator update warp group
        :param tCtAcc_base: TMEM tensor for reading partial accumulators (from MMA)
        :param sFinalAcc: SMEM tensor for writing final accumulator (for epilogue)
        :param epi_tile: Epilogue tile size
        :return: Tuple of (tiled_copy_t2r, tiled_copy_r2s_acc, tTR_tAcc_base, tTR_rAcc,
                          tTR_rAcc_final, tRS_sFinalAcc, tTR_sSFA, tTR_sSFB)
        """
        # Create TMEM load atom based on MMA tile size
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

        # Partition accumulator tensors by epilogue tile
        tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)

        # Create tiled copy for T2R (TMEM to Register)
        tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tAcc_epi[(None, None, 0, 0, 0)])
        # Get thread-specific slices
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)

        # R2S (Register to SMEM) store of the final accumulator: reuse the T2R copy's
        # thread-value layout with a universal copy atom (swizzle-safe for f32)
        copy_atom_r2s_acc = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype)
        tiled_copy_r2s_acc = cute.make_tiled_copy_D(copy_atom_r2s_acc, tiled_copy_t2r)
        thr_copy_r2s_acc = tiled_copy_r2s_acc.get_slice(tidx)

        # (R2S, R2S_M, R2S_N, SUBTILE * STAGE)
        tRS_sFinalAcc = thr_copy_r2s_acc.partition_D(sFinalAcc)

        # Partition source for T2R: tCtAcc_base (read partial accumulators)
        tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gD_mnl_epi = cute.flat_divide(gD_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        sSFA_epi = cute.flat_divide(sSFA, epi_tile)
        sSFB_epi = cute.flat_divide(sSFB, epi_tile)

        tTR_gC = thr_copy_t2r.partition_D(gD_mnl_epi)
        tTR_sSFA = thr_copy_t2r.partition_D(sSFA_epi)
        tTR_sSFB = thr_copy_t2r.partition_D(sSFB_epi)

        # Create register tensor for T2R destination (holds one k_tile's partial accumulator)
        # Shape: (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)

        # Create register tensor for final accumulated result across all k_tiles
        # Shape: (T2R, T2R_M, T2R_N, EPI_M, EPI_N) -> grouped to (T2R, T2R_M, T2R_N, (EPI_M, EPI_N))
        tTR_rAcc_final_ = cute.make_rmem_tensor(tTR_gC[(None, None, None, None, None, 0, 0, 0)].shape, self.acc_dtype)
        tTR_rAcc_final = cute.group_modes(tTR_rAcc_final_, 3, cute.rank(tTR_rAcc_final_))

        return (
            tiled_copy_t2r,
            tiled_copy_r2s_acc,
            tTR_tAcc_base,
            tTR_rAcc,
            tTR_rAcc_final,
            tRS_sFinalAcc,
            tTR_sSFA,
            tTR_sSFB,
        )

    def epilog_tmem_copy_and_partition(self, tidx, tAcc, gD_mnl, epi_tile, use_2cta_instrs):
        # Build the TMEM->register load for the epilogue and partition acc/output by epi tile
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.d_layout,
            self.d_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gD_mnl_epi = cute.flat_divide(gD_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gD_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(self, tiled_copy_t2r, tTR_rD, tidx, sD):
        # Build the register->SMEM store for the D output and partition it
        copy_atom_r2s = sm100_utils.get_smem_store_op(self.d_layout, self.d_dtype, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD)
        tRS_rD = tiled_copy_r2s.retile(tTR_rD)
        return tiled_copy_r2s, tRS_rD, tRS_sD

    def epilog_smem_load_copy_and_partition(self, tiled_copy_t2r, tTR_rAcc, tidx, sFinalAcc):
        # S2R load of the final accumulator: reuse the T2R copy's thread-value layout
        # with a universal copy atom; tSR_rAcc is a register view of tTR_rAcc
        copy_atom_s2r = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype)
        tiled_copy_s2r = cute.make_tiled_copy_D(copy_atom_s2r, tiled_copy_t2r)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        # (S2R, S2R_M, S2R_N, SUBTILE * STAGE)
        tSR_sFinalAcc = thr_copy_s2r.partition_D(sFinalAcc)
        tSR_rAcc = tiled_copy_s2r.retile(tTR_rAcc)
        return tiled_copy_s2r, tSR_rAcc, tSR_sFinalAcc

    # ------------------------------------------------------------------
    # GPU device kernel
    # ------------------------------------------------------------------

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
        sfb2_tensor,  # can be cute.Tensor or cute.Pointer
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        padded_offsets: cute.Tensor,
        alpha: cute.Tensor,
        mBias_nl: Optional[cute.Tensor],
        prob: cute.Tensor,
        workspace_ptr,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        sfa2_smem_layout_staged: cute.Layout,
        sfb2_smem_layout_staged: cute.Layout,
        d_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        final_acc_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        bias_smem_layout_staged: Optional[cute.Layout],
        epi_tile: cute.Tile,
        sched_params: MoESchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """GPU device kernel for the persistent second-level-scaled MoE grouped GEMM."""
        # Per-thread warp / lane identity used for warp specialization
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()

        # TMA warp prefetches all TMA descriptors up front (B/SFB only for dense)
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            if cutlass.const_expr(self.weight_mode == MoEWeightMode.DENSE):
                cpasync.prefetch_descriptor(tma_atom_b)
                cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_d)

        # total_token is the last padded offset (0 => nothing to do for this launch)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        total_token = padded_offsets[self.expert_cnt - 1]

        # Block/cluster coordinates and this CTA's MMA V (leader/follower) coordinate
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        # Allocate this CTA's shared storage struct
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sched_storage = storage.scheduler

        # Initialize mainloop ab_pipeline and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Initialize scale_pipeline and states
        scale_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        scale_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * len(self.epilog_warp_id),
        )
        scale_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.scale_mbar_ptr.data_ptr(),
            num_stages=self.num_scale_stage,
            producer_group=scale_pipeline_producer_group,
            consumer_group=scale_pipeline_consumer_group,
            defer_sync=True,
        )

        # Initialize acc_pipeline and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.accumulator_update_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Initialize epi_pipeline and states
        epi_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp * len(self.accumulator_update_warp_id))
        epi_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp * len(self.epilog_warp_id))
        epi_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.epi_mbar_ptr.data_ptr(),
            num_stages=self.num_epi_stage,
            producer_group=epi_pipeline_producer_group,
            consumer_group=epi_pipeline_consumer_group,
        )

        # Initialize tile_info_pipeline and states
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp * 1)
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_wo_sched)
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=sched_storage.tile_info_mbar.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        if cutlass.const_expr(self.enable_bias):
            # Initialize bias_pipeline and states
            bias_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp)
            bias_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.epilog_warp_id),
            )
            bias_pipeline = pipeline.PipelineCpAsync.create(
                barrier_storage=storage.bias_mbar_ptr.data_ptr(),
                num_stages=self.num_bias_stage,
                producer_group=bias_pipeline_producer_group,
                consumer_group=bias_pipeline_consumer_group,
            )
            sBias = storage.sBias.get_tensor(bias_smem_layout_staged)

        # Create and initialize the persistent MoE tile scheduler
        scheduler = MoEPersistentTileScheduler.create(
            sched_params,
            padded_offsets,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            counter_ptr=self._get_sched_counter_ptr(workspace_ptr),
            sched_storage=sched_storage,
        )
        scheduler.internal_init()

        # TMEM allocator holding the accumulator plus SFA/SFB scale columns
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )

        # Relaxed cluster arrive; the matching cluster_wait happens before specialization
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        # Materialize SMEM tensor views over the staged buffers
        sD = storage.sD.get_tensor(d_smem_layout_staged.outer, swizzle=d_smem_layout_staged.inner)
        sFinalAcc = storage.sFinalAcc.get_tensor(final_acc_smem_layout_staged.outer, swizzle=final_acc_smem_layout_staged.inner)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        sSFA2 = storage.sSFA2.get_tensor(sfa2_smem_layout_staged)
        sSFB2 = storage.sSFB2.get_tensor(sfb2_smem_layout_staged)
        info_layout = cute.make_layout((4, self.num_tile_stage), stride=(1, 4))
        sInfo = sched_storage.sInfo.get_tensor(info_layout)

        # Multicast masks — must create ALL when any mcast or 2CTA is active
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1)

        # MMA partition (for tCtAcc_fake shape computation only)
        thr_mma_common = tiled_mma.get_slice(0)
        tCsA_common = thr_mma_common.partition_A(sA)
        tCsB_common = thr_mma_common.partition_B(sB)
        tCsA_common = cute.filter_zeros(tCsA_common)
        tCsB_common = cute.filter_zeros(tCsB_common)

        # Preparing LDGSTS partition for second level scale factor tensor
        atom_scale2_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            self.sf2_dtype,
            num_bits_per_copy=self.sf2_dtype.width,
        )
        tiled_copy_sfa2 = cute.make_tiled_copy_tv(atom_scale2_copy, cute.make_layout((32,)), cute.make_layout((1,)))
        tiled_copy_sfb2 = cute.make_tiled_copy_tv(atom_scale2_copy, cute.make_layout((32,)), cute.make_layout((1,)))

        # Per-lane G2S partition of the SFA2/SFB2 SMEM destinations
        thr_copy_sfa2 = tiled_copy_sfa2.get_slice(lane_idx)
        thr_copy_sfb2 = tiled_copy_sfb2.get_slice(lane_idx)
        tAsSFA2 = thr_copy_sfa2.partition_D(sSFA2)
        tBsSFB2 = thr_copy_sfb2.partition_D(sSFB2)

        # Scale viewed as C tensor
        # N/M extents use the CTA-tile-clamped scale block size (scale_size_*),
        # so the view spans exactly one CTA tile (size * per_tile == cta extent)
        # regardless of whether sg{m,n} exceed the tile (e.g. sgn=512 > 256).
        sSFA2_view_as_C_layout = cute.make_layout(
            (
                (self.scale_size_m, self.scale_m_per_tile),
                self.cta_tile_shape_mnk[1],
                self.num_scale_stage,
            ),
            stride=((0, 1), 0, self.scale_m_per_tile),
        )
        sSFB2_view_as_C_layout = cute.make_layout(
            (
                self.cta_tile_shape_mnk[0],
                (self.scale_size_n, self.scale_n_per_tile),
                self.num_scale_stage,
            ),
            stride=(0, (0, 1), self.scale_n_per_tile),
        )
        sSFA2_view_as_C = cute.make_tensor(sSFA2.iterator, sSFA2_view_as_C_layout)
        sSFB2_view_as_C = cute.make_tensor(sSFB2.iterator, sSFB2_view_as_C_layout)

        # Number of K MMA tiles that share a single second-level scale block
        k_tile_same_scale_factor = self.sgk // self.mma_tiler[2]
        # SMEM fragments for MMA (used by MMA warp)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        # TMEM accumulator shape
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # Cluster sync before warp specialization
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

        # No tokens routed to any expert this launch => every warp exits early
        if total_token <= 0:
            cute.arch.nvvm.exit()

        # ==============================================================
        # Scheduler warp (MoE Persistent Tile Scheduler)
        # ==============================================================
        if warp_idx == self.sched_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_sched_warps)
            work_tile_info = scheduler.initial_work_tile_info()
            tile_info_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_tile_stage)
            # Publish each valid work tile's info into the tile-info pipeline
            while work_tile_info.is_valid_tile:
                tile_info_pipeline.producer_acquire(tile_info_producer_state)
                with cute.arch.elect_one():
                    sInfo[(0, tile_info_producer_state.index)] = work_tile_info.expert_idx
                    sInfo[(1, tile_info_producer_state.index)] = work_tile_info.tile_m_idx
                    sInfo[(2, tile_info_producer_state.index)] = work_tile_info.tile_n_idx
                    sInfo[(3, tile_info_producer_state.index)] = work_tile_info.k_tile_cnt
                cute.arch.fence_proxy("async.shared", space="cta")
                self.sched_sync_barrier.arrive_and_wait()
                tile_info_pipeline.producer_commit(tile_info_producer_state)
                tile_info_producer_state.advance()
                work_tile_info = scheduler.advance_to_next_work()

            # Push a sentinel tile (expert_idx = -1) to signal end of work
            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = cutlass.Int32(-1)
                sInfo[(1, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(2, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        # ==============================================================
        # DMA / TMA load warp
        # ==============================================================
        if warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)

            ext = self._make_extension(workspace_ptr)
            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)
            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)

            # Prime the loop with the first work tile from the scheduler
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                work_tile_info = MoEWorkTileInfo(
                    expert_idx=tile_info[0],
                    tile_m_idx=tile_info[1],
                    tile_n_idx=tile_info[2],
                    k_tile_cnt=tile_info[3],
                )
                k_tile_cnt = work_tile_info.k_tile_cnt
                ext.update_expert_info(padded_offsets, work_tile_info.expert_idx)

                # Resolve this expert's A/B and first-level SFA/SFB gmem tensors, then tile them
                real_a, _ = ext.get_gmem_tensor("a", mA_mkl, padded_offsets, work_tile_info)
                real_b, desc_ptr_b = ext.get_gmem_tensor("b", mB_nkl, padded_offsets, work_tile_info)
                real_sfa, _ = ext.get_gmem_tensor("sfa", mSFA_mkl, padded_offsets, work_tile_info)
                real_sfb, desc_ptr_sfb = ext.get_gmem_tensor("sfb", mSFB_nkl, padded_offsets, work_tile_info)

                gA_mkl = cute.local_tile(real_a, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gB_nkl = cute.local_tile(real_b, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
                gSFA_mkl = cute.local_tile(real_sfa, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                gSFB_nkl = cute.local_tile(real_sfb, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))
                # MMA partition on gmem tensors
                thr_mma_dma = tiled_mma.get_slice(mma_tile_coord_v)
                thr_mma_sfb_dma = tiled_mma_sfb.get_slice(mma_tile_coord_v)
                tCgA = thr_mma_dma.partition_A(gA_mkl)
                tCgB = thr_mma_dma.partition_B(gB_nkl)
                tCgSFA = thr_mma_dma.partition_A(gSFA_mkl)
                tCgSFB = thr_mma_sfb_dma.partition_B(gSFB_nkl)

                # TMA partition A
                a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
                tAsA, tAgA = cpasync.tma_partition(
                    tma_atom_a,
                    block_in_cluster_coord_vmnk[2],
                    a_cta_layout,
                    cute.group_modes(sA, 0, 3),
                    cute.group_modes(tCgA, 0, 3),
                )
                # TMA partition B
                b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    cute.group_modes(sB, 0, 3),
                    cute.group_modes(tCgB, 0, 3),
                )
                # TMA partition SFA
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
                # TMA partition SFB
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

                # Fix this tile's M/N coordinate and slice each operand down to it
                mma_tile_coord_m = work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)
                mma_tile_coord_n = work_tile_info.tile_n_idx
                tAgA_slice = tAgA[(None, mma_tile_coord_m, None, 0)]
                tBgB_slice = tBgB[(None, mma_tile_coord_n, None, 0)]
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_m, None, 0)]
                slice_n = mma_tile_coord_n
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_n // 2
                tBgSFB_slice = tBgSFB[(None, slice_n, None, 0)]

                # Peek whether the first AB buffer stage is free before the k loop
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                # k_tile loop
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    tAgA_k = tAgA_slice[(None, ab_producer_state.count)]
                    tBgB_k = tBgB_slice[(None, ab_producer_state.count)]
                    tAgSFA_k = tAgSFA_slice[(None, ab_producer_state.count)]
                    tBgSFB_k = tBgSFB_slice[(None, ab_producer_state.count)]
                    tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tBsB_pipe = tBsB[(None, ab_producer_state.index)]
                    tAsSFA_pipe = tAsSFA[(None, ab_producer_state.index)]
                    tBsSFB_pipe = tBsSFB[(None, ab_producer_state.index)]

                    # Acquire the stage, then TMA-load A/B/SFA/SFB for this k tile on one barrier
                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)

                    cute.copy(tma_atom_a, tAgA_k, tAsA_pipe, tma_bar_ptr=tma_bar, mcast_mask=a_full_mcast_mask)
                    cute.copy(tma_atom_b, tBgB_k, tBsB_pipe, tma_bar_ptr=tma_bar, mcast_mask=b_full_mcast_mask, tma_desc_ptr=desc_ptr_b)
                    cute.copy(tma_atom_sfa, tAgSFA_k, tAsSFA_pipe, tma_bar_ptr=tma_bar, mcast_mask=sfa_full_mcast_mask)
                    cute.copy(tma_atom_sfb, tBgSFB_k, tBsSFB_pipe, tma_bar_ptr=tma_bar, mcast_mask=sfb_full_mcast_mask, tma_desc_ptr=desc_ptr_sfb)

                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                # Get next tile from scheduler
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            ab_pipeline.producer_tail(ab_producer_state)

        # ==============================================================
        # Second level scale load warp
        # ==============================================================
        if warp_idx == self.scale_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)

            ext = self._make_extension(workspace_ptr)
            scale_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_scale_stage)

            # Bias load shares this warp (a dedicated 13th warp would break the
            # setmaxregister warpgroup alignment). One 2-stage cp.async produce
            # per tile, consumed by the epilogue warpgroup.
            if cutlass.const_expr(self.enable_bias):
                bias_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_bias_stage)
                # Per-lane G2S copy for streaming a bias column into SMEM
                bias_g2s_atom = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
                    self.bias_dtype,
                    num_bits_per_copy=128,
                )
                bias_g2s_tiled = cute.make_tiled_copy_tv(
                    bias_g2s_atom,
                    cute.make_layout((32,)),
                    cute.make_layout((8,)),
                )
                thr_bias_g2s = bias_g2s_tiled.get_slice(cute.arch.lane_idx())
                tBs_sBias = thr_bias_g2s.partition_D(sBias)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)
            # Prime the loop with the first work tile from the scheduler
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            scale_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_scale_stage)

            while is_valid_tile:

                work_tile_info = MoEWorkTileInfo(
                    expert_idx=tile_info[0],
                    tile_m_idx=tile_info[1],
                    tile_n_idx=tile_info[2],
                    k_tile_cnt=tile_info[3],
                )
                k_tile_cnt = work_tile_info.k_tile_cnt
                ext.update_expert_info(padded_offsets, work_tile_info.expert_idx)

                # Get current tensors
                mSFA2_mkl_current, _ = ext.get_gmem_tensor("sfa2", sfa2_tensor, padded_offsets, work_tile_info)
                mSFB2_nkl_current, _ = ext.get_gmem_tensor("sfb2", sfb2_tensor, padded_offsets, work_tile_info)

                # Create global memory tiled tensors
                gSFA2_mkl = cute.local_tile(mSFA2_mkl_current, cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)), (None, None, None))
                gSFB2_nkl = cute.local_tile(mSFB2_nkl_current, cute.slice_(self.cta_tile_shape_mnk, (0, None, None)), (None, None, None))

                # Create coordinate tensors
                cSFA2_mkl = cute.make_identity_tensor(cute.shape(mSFA2_mkl_current))
                cSFB2_nkl = cute.make_identity_tensor(cute.shape(mSFB2_nkl_current))
                cSFA2 = cute.local_tile(cSFA2_mkl, cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)), (None, None, None))
                cSFB2 = cute.local_tile(cSFB2_nkl, cute.slice_(self.cta_tile_shape_mnk, (0, None, None)), (None, None, None))

                # Partition tensors
                tAgSFA2_mkl = thr_copy_sfa2.partition_S(gSFA2_mkl)
                tBgSFB2_nkl = thr_copy_sfb2.partition_S(gSFB2_nkl)
                tAcSFA2 = thr_copy_sfa2.partition_S(cSFA2)
                tBcSFB2 = thr_copy_sfb2.partition_S(cSFB2)

                mma_tile_coord_mnl = (work_tile_info.tile_m_idx, work_tile_info.tile_n_idx, 0)

                # Prepare the mask for scaleA/scaleB
                tApSFA2 = cute.make_rmem_tensor(
                    cute.make_layout(cute.filter_zeros(cute.slice_(tAsSFA2, (None, None, None, 0))).shape),
                    cutlass.Boolean,
                )
                tBpSFB2 = cute.make_rmem_tensor(
                    cute.make_layout(cute.filter_zeros(cute.slice_(tBsSFB2, (None, None, None, 0))).shape),
                    cutlass.Boolean,
                )

                # Peek (try_wait) SCALE buffer empty
                scale_producer_state.reset_count()
                peek_scale_empty_status = cutlass.Boolean(1)
                if scale_producer_state.count < k_tile_cnt:
                    peek_scale_empty_status = scale_pipeline.producer_try_acquire(scale_producer_state)

                # k_tile loop
                for k_tile in cutlass.range(0, k_tile_cnt // k_tile_same_scale_factor, 1, unroll=1):

                    # Slice to per mma tile index
                    tAsSFA2_pipe = cute.filter_zeros(tAsSFA2[(None, None, None, scale_producer_state.index)])
                    tBsSFB2_pipe = cute.filter_zeros(tBsSFB2[(None, None, None, scale_producer_state.index)])

                    tAgSFA2_k = cute.filter_zeros(
                        tAgSFA2_mkl[
                            (
                                None,
                                None,
                                None,
                                mma_tile_coord_mnl[0],
                                scale_producer_state.count * k_tile_same_scale_factor,
                                mma_tile_coord_mnl[2],
                            )
                        ]
                    )
                    tBgSFB2_k = cute.filter_zeros(
                        tBgSFB2_nkl[
                            (
                                None,
                                None,
                                None,
                                mma_tile_coord_mnl[1],
                                scale_producer_state.count * k_tile_same_scale_factor,
                                mma_tile_coord_mnl[2],
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
                                mma_tile_coord_mnl[0],
                                scale_producer_state.count * k_tile_same_scale_factor,
                                mma_tile_coord_mnl[2],
                            ),
                        )
                    )
                    tBcSFB2_compact = cute.filter_zeros(
                        cute.slice_(
                            tBcSFB2,
                            (
                                None,
                                None,
                                None,
                                mma_tile_coord_mnl[1],
                                scale_producer_state.count * k_tile_same_scale_factor,
                                mma_tile_coord_mnl[2],
                            ),
                        )
                    )

                    # Skip more unnecessary load
                    for i in cutlass.range_constexpr(cute.size(tApSFA2, mode=[1])):
                        tApSFA2[((0, 0), i, (0, 0))] = cute.elem_less(tAcSFA2_compact[(i)][0], mSFA2_mkl_current.shape[0])
                    for i in cutlass.range_constexpr(cute.size(tBpSFB2, mode=[1])):
                        tBpSFB2[((0, 0), i, (0, 0))] = cute.elem_less(tBcSFB2_compact[(i)][0], mSFB2_nkl_current.shape[0])

                    # Conditionally wait for Scale buffer empty
                    scale_pipeline.producer_acquire(scale_producer_state, peek_scale_empty_status)

                    # load scaleA/scaleB
                    cute.copy(tiled_copy_sfa2, tAgSFA2_k, tAsSFA2_pipe, pred=tApSFA2)
                    cute.copy(tiled_copy_sfb2, tBgSFB2_k, tBsSFB2_pipe, pred=tBpSFB2)

                    scale_pipeline.producer_commit(scale_producer_state)

                    # Peek (try_wait) Scale buffer empty
                    scale_producer_state.advance()
                    peek_scale_empty_status = cutlass.Boolean(1)
                    if scale_producer_state.count < k_tile_cnt:
                        peek_scale_empty_status = scale_pipeline.producer_try_acquire(scale_producer_state)

                # Bias produce for this tile — after the k-tile loop so a
                # blocking bias acquire never throttles scale prefetch.
                if cutlass.const_expr(self.enable_bias):
                    bias_producer_state.reset_count()
                    # Slice this expert's bias down to the current N tile
                    real_bias, _ = ext.get_gmem_tensor("bias", mBias_nl, padded_offsets, work_tile_info)
                    gBias_expert = cute.local_tile(real_bias, cute.slice_(self.mma_tiler[:2], (0, None)), (None, None))
                    bias_tile = gBias_expert[(None, work_tile_info.tile_n_idx, 0)]
                    bias_identity_tensor = cute.make_identity_tensor(bias_tile.shape)
                    bias_partitioned_by_g2s = thr_bias_g2s.partition_S(bias_tile)
                    bias_coord_partitioned_by_g2s = thr_bias_g2s.partition_S(bias_identity_tensor)

                    # Predicate out lanes past the valid N extent (ragged final tile)
                    residue_n = sched_params.intermediate - work_tile_info.tile_n_idx * self.cta_tile_shape_mnk[1]
                    bias_pred_tensor = cute.make_rmem_tensor(bias_coord_partitioned_by_g2s[(None, 0)].shape, cutlass.Boolean)
                    for vi in cutlass.range_constexpr(cute.size(bias_pred_tensor)):
                        bias_pred_tensor[vi] = cute.elem_less(bias_coord_partitioned_by_g2s[(vi, 0)], (residue_n,))
                    bias_pred_tensor = bias_pred_tensor[((0, None),)]

                    # Issue the predicated G2S bias load into the bias pipeline
                    bias_pipeline.producer_acquire(bias_producer_state)
                    cute.copy(bias_g2s_tiled, bias_partitioned_by_g2s[(None, 0)], tBs_sBias[(None, 0, bias_producer_state.index)], pred=bias_pred_tensor)
                    bias_pipeline.producer_commit(bias_producer_state)
                    bias_producer_state.advance()

                # Get next tile from scheduler
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            if cutlass.const_expr(self.enable_bias):
                bias_pipeline.producer_tail(bias_producer_state)

        # ==============================================================
        # MMA warp
        # ==============================================================
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_uniform_warps)
            # Wait for TMEM allocation and view the accumulator region
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # SFA TMEM tensor
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

            # SFB TMEM tensor
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

            # S2T copy partition for SFA/SFB
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

            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)
            # Prime the loop with the first work tile from the scheduler
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                k_tile_cnt = tile_info[3]

                # Peek AB buffer full
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                # Peek Acc buffer empty
                acc_producer_state.reset_count()
                peek_acc_empty_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_acc_empty_status = acc_pipeline.producer_try_acquire(acc_producer_state)

                # This tile's (M, N, L=expert) MMA coordinate
                mma_tile_coord_mnl = (
                    tile_info[1] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[2],
                    tile_info[0],
                )

                acc_stage_index = acc_producer_state.index

                # k_tile loop
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):

                    tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                    # Acquire a fresh accumulator stage at the start of each scale block
                    if k_tile % k_tile_same_scale_factor == 0:
                        if is_leader_cta:
                            acc_pipeline.producer_acquire(acc_producer_state, peek_acc_empty_status)

                    # Reset the ACCUMULATE field for the first tile within each scale block
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile % k_tile_same_scale_factor > 0)

                    if is_leader_cta:
                        # Wait for this k tile's operands, then stage SFA/SFB from SMEM into TMEM
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        s2t_stage_coord = (None, None, None, None, ab_consumer_state.index)
                        cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t[s2t_stage_coord], tCtSFA_compact_s2t)
                        cute.copy(tiled_copy_s2t_sfb, tCsSFB_compact_s2t[s2t_stage_coord], tCtSFB_compact_s2t)

                        # Peek the next stage's fullness so the acquire can be non-blocking
                        num_kblocks = cute.size(tCrA, mode=[2])
                        ab_consumer_state_next = ab_consumer_state.clone()
                        ab_consumer_state_next.advance()
                        if ab_consumer_state_next.count < k_tile_cnt:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state_next)

                        # Issue one block-scaled MMA per K block, binding SFA/SFB each time
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
                            tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator)
                            cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_coord], tCrB[kblock_coord], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        ab_pipeline.consumer_release(ab_consumer_state)
                        ab_consumer_state = ab_consumer_state_next

                    # At the end of each scale block, commit the accumulator stage to consumers
                    if k_tile % k_tile_same_scale_factor == k_tile_same_scale_factor - 1:
                        if is_leader_cta:
                            acc_pipeline.producer_commit(acc_producer_state)

                        acc_producer_state.advance()
                        if acc_producer_state.count < k_tile_cnt:
                            if is_leader_cta:
                                peek_acc_empty_status = acc_pipeline.producer_try_acquire(acc_producer_state)

                # Get next tile from scheduler
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            # Wait for accumulator buffer empty
            acc_pipeline.producer_tail(acc_producer_state)

        # ==============================================================
        # Accumulator Update warps
        # ==============================================================
        if warp_idx in self.accumulator_update_warp_id and total_token > 0:
            cute.arch.setmaxregister_increase(self.num_regs_acc_update_warps)

            # Wait for TMEM allocation (done by epilogue warp)
            tmem.wait_for_alloc()

            # Retrieve TMEM pointer and create accumulator tensors
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            # tCtAcc_base: Read partial accumulators from MMA (via acc_pipeline stages)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # Shape-only partition on global tensor (invariant setup for t2r copy atom)
            gD_mnl_shape = cute.local_tile(mD_mnl, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
            thr_mma_epi = tiled_mma.get_slice(mma_tile_coord_v)
            tCgD_shape = thr_mma_epi.partition_C(gD_mnl_shape)

            # Consumer of acc_pipeline (receives partial accumulators from MMA)
            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

            # Consumer of scale_pipeline (receives partial accumulators from Scale Load Warp)
            scale_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_scale_stage)

            # Producer for epi_pipeline (sends final accumulator to epilogue)
            epi_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_epi_stage)

            # Call partitioning function to setup copy operations
            acc_tidx = tidx
            (
                tiled_copy_t2r,
                tiled_copy_r2s_acc,
                tTR_tAcc_base,
                tTR_rAcc,
                tTR_rAcc_final,
                tRS_sFinalAcc,
                tTR_sSFA,
                tTR_sSFB,
            ) = self.acc_update_tmem_copy_and_partition(
                acc_tidx,
                tCtAcc_base,
                tCgD_shape,
                sSFA2_view_as_C,
                sSFB2_view_as_C,
                sFinalAcc,
                epi_tile,
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)
            # Prime the loop with the first work tile from the scheduler
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            # Main tile processing loop
            while is_valid_tile:

                # Extract k_tile_cnt from scheduler
                k_tile_cnt = tile_info[3]

                # Initialize final accumulator to zero
                tTR_rAcc_final.fill(0.0)

                tTR_rSFA = cute.make_rmem_tensor(
                    cute.slice_(tTR_sSFA, (None, None, None, 0, None, 0)).shape,
                    self.acc_dtype,
                )
                tTR_rSFB = cute.make_rmem_tensor(
                    cute.slice_(tTR_sSFB, (None, None, None, 0, None, 0)).shape,
                    self.acc_dtype,
                )

                # Reset and peek acc_pipeline (MMA produces partial accumulators)
                acc_consumer_state.reset_count()
                peek_acc_full_status = cutlass.Boolean(1)
                if acc_consumer_state.count < k_tile_cnt:
                    peek_acc_full_status = acc_pipeline.consumer_try_wait(acc_consumer_state)

                # Peek (try_wait) Scale buffer full for k_tile = 0
                scale_consumer_state.reset_count()
                peek_scale_full_status = cutlass.Boolean(1)
                if scale_consumer_state.count < k_tile_cnt:
                    peek_scale_full_status = scale_pipeline.consumer_try_wait(scale_consumer_state)

                # k_tile loop
                for k_tile in cutlass.range(0, k_tile_cnt // k_tile_same_scale_factor, 1, unroll=1):

                    # Wait for scale buffer full
                    scale_pipeline.consumer_wait(scale_consumer_state, peek_scale_full_status)

                    tTR_sSFA_slice = cute.slice_(
                        tTR_sSFA,
                        (None, None, None, 0, None, scale_consumer_state.index),
                    )
                    tTR_sSFB_slice = cute.slice_(
                        tTR_sSFB,
                        (None, None, None, 0, None, scale_consumer_state.index),
                    )
                    scale_atom_copy = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.acc_dtype,
                        num_bits_per_copy=self.acc_dtype.width,
                    )

                    cute.copy(scale_atom_copy, tTR_sSFA_slice, tTR_rSFA)
                    cute.copy(scale_atom_copy, tTR_sSFB_slice, tTR_rSFB)

                    # Async arrive scale buffer empty
                    scale_pipeline.consumer_release(scale_consumer_state)
                    scale_consumer_state.advance()

                    # Wait for MMA to produce partial accumulator for this k_tile
                    acc_pipeline.consumer_wait(acc_consumer_state, peek_acc_full_status)

                    # Index into TMEM buffer for current pipeline stage
                    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

                    # Group modes for subtile iteration
                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                    # Process each subtile
                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                    for subtile_idx in cutlass.range(subtile_cnt):

                        # Load partial accumulator from TMEM to register
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                        # Accumulate: final += partial (no scaling)
                        tTR_rAcc_subtile = tTR_rAcc_final[(None, None, None, subtile_idx)]
                        acc_vec = tTR_rAcc.load()
                        final_vec = tTR_rAcc_subtile.load()
                        scale_a = tTR_rSFA[(None, None, None, subtile_idx)].load()
                        scale_b = tTR_rSFB[(None, None, None, subtile_idx)].load()
                        scale = scale_a * scale_b
                        final_vec = acc_vec * scale + final_vec  # Simple accumulation
                        tTR_rAcc_subtile.store(final_vec.to(self.acc_dtype))

                    # Release acc_pipeline buffer (MMA can reuse it)
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                    # Peek next k_tile
                    peek_acc_full_status = cutlass.Boolean(1)
                    if acc_consumer_state.count < k_tile_cnt:
                        peek_acc_full_status = acc_pipeline.consumer_try_wait(acc_consumer_state)

                    peek_scale_full_status = cutlass.Boolean(1)
                    if scale_consumer_state.count < k_tile_cnt:
                        peek_scale_full_status = scale_pipeline.consumer_try_wait(scale_consumer_state)

                # All k_tiles accumulated, now store final result to SMEM for epilogue
                # Acquire epi_pipeline (wait for epilogue to be done with the buffer)
                epi_pipeline.producer_acquire(epi_producer_state)

                # Copy final accumulator from registers to SMEM, one epi subtile per slot.
                # No proxy fence needed: producer and consumer both use the generic proxy,
                # and producer_commit's mbarrier arrive has release semantics.
                final_subtile_cnt = cute.size(tTR_rAcc_final.shape, mode=[3])
                slot_base = epi_producer_state.index * final_subtile_cnt
                for subtile_idx in cutlass.range(final_subtile_cnt, unroll_full=True):
                    tRS_rFinal = tiled_copy_r2s_acc.retile(tTR_rAcc_final[(None, None, None, subtile_idx)])
                    cute.copy(
                        tiled_copy_r2s_acc,
                        tRS_rFinal,
                        tRS_sFinalAcc[(None, None, None, slot_base + subtile_idx)],
                    )

                # Commit to epi_pipeline (signal epilogue that data is ready)
                epi_pipeline.producer_commit(epi_producer_state)
                epi_producer_state.advance()

                # Get next tile from scheduler
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

        # ==============================================================
        # Epilogue warps
        # ==============================================================
        if warp_idx in self.epilog_warp_id and total_token > 0:
            cute.arch.setmaxregister_increase(self.num_regs_epilogue_warps)
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            # Final accumulator now lives in SMEM (sFinalAcc); TMEM here is only used
            # by MMA partials + SF. tCtAcc_base_ is a shape/TV-layout carrier for the
            # t2r template below — it is never dereferenced in the epilogue.
            tCtAcc_base_ = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # Shape-only partition on global tensor (invariant setup for t2r copy atom)
            gD_mnl_shape = cute.local_tile(mD_mnl, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
            thr_mma_epi = tiled_mma.get_slice(mma_tile_coord_v)
            tCgD_shape = thr_mma_epi.partition_C(gD_mnl_shape)

            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx,
                tCtAcc_base_,
                tCgD_shape,
                epi_tile,
                use_2cta_instrs,
            )

            tTR_rD = cute.make_rmem_tensor(tTR_rAcc.shape, self.d_dtype)
            tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r,
                tTR_rD,
                epi_tidx,
                sD,
            )
            tiled_copy_s2r, tSR_rAcc, tSR_sFinalAcc = self.epilog_smem_load_copy_and_partition(
                tiled_copy_t2r,
                tTR_rAcc,
                epi_tidx,
                sFinalAcc,
            )

            epi_ext = self._make_extension(workspace_ptr)

            # Consume from epi_pipeline (from accumulator update warp)
            epi_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_epi_stage)
            d_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * len(self.epilog_warp_id))
            d_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_d_stage, producer_group=d_producer_group)

            tile_info_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_tile_stage)
            # Prime the loop with the first work tile from the scheduler
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[0] >= cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            if cutlass.const_expr(self.enable_bias):
                bias_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_bias_stage)
                bias_s2r_tom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.bias_dtype, num_bits_per_copy=128)
                # epi_tile_n_required: compute_epilogue_tile_shape returns Layout
                # modes on current DSLs; make_layout needs the plain host-side extent.
                tTR_rBias = cute.make_rmem_tensor(cute.make_layout(self.epi_tile_n_required), self.bias_dtype)

            num_prev_subtiles = cutlass.Int32(0)
            while is_valid_tile:
                epi_work_tile_info = MoEWorkTileInfo(
                    expert_idx=tile_info[0],
                    tile_m_idx=tile_info[1],
                    tile_n_idx=tile_info[2],
                    k_tile_cnt=tile_info[3],
                )
                expert_idx = epi_work_tile_info.expert_idx
                epi_ext.update_expert_info(padded_offsets, expert_idx)

                alpha_val = alpha[expert_idx]

                if cutlass.const_expr(self.enable_bias):
                    bias_consumer_state.reset_count()
                    bias_pipeline.consumer_wait(bias_consumer_state)
                    sBias_stage = sBias[(None, bias_consumer_state.index)]
                    sBias_subtiles = cute.flat_divide(sBias_stage, cute.make_layout(self.epi_tile_n_required))

                real_d, _ = epi_ext.get_gmem_tensor("d", mD_mnl, padded_offsets, epi_work_tile_info)

                thr_mma_epi_loop = tiled_mma.get_slice(mma_tile_coord_v)

                gD_mnl_loop = cute.local_tile(real_d, cute.slice_(self.mma_tiler_d, (None, None, 0)), (None, None, None))
                tCgD_loop = thr_mma_epi_loop.partition_C(gD_mnl_loop)
                _, bSG_sD, bSG_gD_partitioned = epilog_gmem_copy_and_partition(
                    epi_tidx,
                    tma_atom_d,
                    tCgD_loop,
                    epi_tile,
                    sD,
                )

                epi_mma_tile_coord = (
                    epi_work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape),
                    epi_work_tile_info.tile_n_idx,
                    0,
                )

                bSG_gD = bSG_gD_partitioned[(None, None, None, *epi_mma_tile_coord)]
                bSG_gD = cute.group_modes(bSG_gD, 1, cute.rank(bSG_gD))

                mPosition = epi_work_tile_info.tile_m_idx * self.cta_tile_shape_mnk[0] + tidx % 128
                real_prob, _ = epi_ext.get_gmem_tensor("prob", prob, padded_offsets, epi_work_tile_info)
                mProb = real_prob[mPosition, 0, 0]

                epi_stage_index = 0  # epi_pipeline has num_epi_stage=1

                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, epi_stage_index)]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                # Wait on epi_pipeline for final accumulator from accumulator update warp
                epi_pipeline.consumer_wait(epi_consumer_state)

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

                for subtile_idx in cutlass.range(0, subtile_cnt, 1, unroll=1):
                    # No reverse subtile logic needed (epi_pipeline has 1 stage)
                    real_subtile_idx = subtile_idx

                    # Load final accumulator subtile from SMEM (written by acc-update warp)
                    final_slot = epi_stage_index * subtile_cnt + real_subtile_idx
                    cute.copy(
                        tiled_copy_s2r,
                        tSR_sFinalAcc[(None, None, None, final_slot)],
                        tSR_rAcc,
                    )

                    if cutlass.const_expr(self.enable_bias):
                        # m7 fix: use real_subtile_idx directly (matches contiguous)
                        sBias_sub = sBias_subtiles[(None, real_subtile_idx)]
                        cute.copy(bias_s2r_tom, sBias_sub, tTR_rBias)
                        bias_vec = tTR_rBias.load()
                        if cutlass.const_expr(self.vectorized_f32):
                            for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc), 2):
                                bias_f32_0 = bias_vec[i].to(cutlass.Float32)
                                bias_f32_1 = bias_vec[i + 1].to(cutlass.Float32)
                                bias_f32_0, bias_f32_1 = cute.arch.mul_packed_f32x2(
                                    (mProb, mProb),
                                    (bias_f32_0, bias_f32_1),
                                    rnd="rn",
                                    ftz=False,
                                )
                                tTR_rAcc[i], tTR_rAcc[i + 1] = cute.arch.fma_packed_f32x2(
                                    (tTR_rAcc[i], tTR_rAcc[i + 1]),
                                    (cutlass.Float32(alpha_val), cutlass.Float32(alpha_val)),
                                    (bias_f32_0, bias_f32_1),
                                    rnd="rn",
                                    ftz=False,
                                )
                        else:
                            for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                                tTR_rAcc[i] = tTR_rAcc[i] * cutlass.Float32(alpha_val) + bias_vec[i].to(cutlass.Float32) * mProb
                    else:
                        if cutlass.const_expr(self.vectorized_f32):
                            for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc), 2):
                                tTR_rAcc[i], tTR_rAcc[i + 1] = cute.arch.mul_packed_f32x2(
                                    (tTR_rAcc[i], tTR_rAcc[i + 1]),
                                    (cutlass.Float32(alpha_val), cutlass.Float32(alpha_val)),
                                    rnd="rn",
                                    ftz=False,
                                )
                        else:
                            for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                                tTR_rAcc[i] = tTR_rAcc[i] * cutlass.Float32(alpha_val)

                    acc_vec = tTR_rAcc.load()
                    if cutlass.const_expr(not self.enable_bias):
                        tCompute = cute.make_rmem_tensor(acc_vec.shape, self.acc_dtype)
                        if cutlass.const_expr(self.vectorized_f32):
                            for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc), 2):
                                tCompute[i], tCompute[i + 1] = cute.arch.mul_packed_f32x2(
                                    (acc_vec[i], acc_vec[i + 1]),
                                    (mProb, mProb),
                                    rnd="rn",
                                    ftz=False,
                                )
                        else:
                            for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                                tCompute[i] = acc_vec[i] * mProb
                    else:
                        tCompute = tTR_rAcc

                    acc_vec = tiled_copy_r2s.retile(tCompute).load()
                    tRS_rD.store(acc_vec.to(self.d_dtype))

                    d_buffer = num_prev_subtiles % self.num_d_stage
                    num_prev_subtiles = num_prev_subtiles + 1
                    cute.copy(tiled_copy_r2s, tRS_rD, tRS_sD[(None, None, None, d_buffer)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(tma_atom_d, bSG_sD[(None, d_buffer)], bSG_gD[(None, real_subtile_idx)])
                        d_pipeline.producer_commit()
                        d_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                epi_pipeline.consumer_release(epi_consumer_state)
                epi_consumer_state.advance()

                if cutlass.const_expr(self.enable_bias):
                    bias_pipeline.consumer_release(bias_consumer_state)
                    bias_consumer_state.advance()

                # Get next tile from scheduler
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[0] >= cutlass.Int32(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            d_pipeline.producer_tail()

    # ------------------------------------------------------------------
    # Internal: create extension based on weight_mode
    # ------------------------------------------------------------------

    @cute.jit
    def _make_extension(self, workspace_ptr):
        # Discrete mode resolves per-expert B/SFB via prebuilt TMA descriptors;
        # dense mode uses the contiguous grouped-GEMM scheduler extension
        if cutlass.const_expr(self.weight_mode == MoEWeightMode.DISCRETE):
            desc_workspace = TensormapWorkspace(workspace_ptr, ["b", "sfb"])
            return DiscreteWeightScaledGemmSchedExtension(
                tensormap_ctor=desc_workspace,
                sf_vec_size=self.sf_vec_size,
            )
        else:
            return ContiguousAndConsistentGroupedGemmSchedExtension(
                sf_vec_size=self.sf_vec_size,
            )
