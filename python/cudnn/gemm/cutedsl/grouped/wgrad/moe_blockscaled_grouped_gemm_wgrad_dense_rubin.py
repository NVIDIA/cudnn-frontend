# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Rubin (SM107) MXFP8 wgrad from global descriptors: ``BlockScaledMoEGroupedGemmWgradDenseKernel`` (global A / B / SF
descriptors, no helper kernel, per-warp dW stores, no A reuse) on the Rubin kernel's K64 MMA, 256-token K tile and
disjoint TMEM plan, so dW is the Rubin kernel's bit for bit. The wgrad API picks it on SM107 under the same conditions
as the global-descriptor kernel on SM100/SM103.
"""

from .moe_blockscaled_grouped_gemm_wgrad_dense import BlockScaledMoEGroupedGemmWgradDenseKernel
from .moe_blockscaled_grouped_gemm_wgrad_rubin import BlockScaledMoEGroupedGemmWgradRubinKernel, _make_tmem_plan


class BlockScaledMoEGroupedGemmWgradDenseRubinKernel(BlockScaledMoEGroupedGemmWgradDenseKernel, BlockScaledMoEGroupedGemmWgradRubinKernel):
    """``BlockScaledMoEGroupedGemmWgradDenseKernel`` with the Rubin kernel's architecture, MMA atoms and SF copies."""

    K_TILE = 256  # 4 x K64: the Rubin kernel's MXFP8 K tile
    # A/B stages: the Rubin kernel's depth. A stage is 67 KB per CTA (A, B and their scale factors), and the 4 per-warp
    # dW buffers take 16 KB, so a fifth stage does not fit.
    AB_STAGES = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_ab_stage = self.AB_STAGES

    def _setup_attributes(self) -> None:
        if self.a_dtype.width != 8:
            raise ValueError("The Rubin global-descriptor wgrad supports MXFP8 only.")
        # Rubin's MXFP8 MMA atoms, read by _create_tiled_mma(_sfb) in the dense kernel's setup.
        self.instruction_k = 64
        self.mma_inst_shape_mnk = (self.mma_tiler[0], self.mma_tiler[1], self.instruction_k)
        self.mma_inst_shape_mnk_sfb = (self.mma_tiler[0] // 2, self.mma_tiler[1], self.instruction_k)
        super()._setup_attributes()
        # Rubin's TMEM: two disjoint accumulator stages next to the SF columns, one pipeline stage each.
        tmem_plan = _make_tmem_plan(
            tile_n=self.mma_tiler[1],
            tile_k=self.mma_tiler[2],
            sf_vec_size=self.sf_vec_size,
            architecture=self.architecture,
        )
        assert tmem_plan.accumulator_stage_count == 2, "the dense kernel holds two accumulator stages"
        self.num_tmem_alloc_cols = tmem_plan.allocation_columns
        self.num_accumulator_tmem_cols = tmem_plan.accumulator_columns
        self.acc_stage_stride = self.mma_tiler[1]
        self.num_sfa_tmem_cols = tmem_plan.sfa_columns
        self.num_sfb_tmem_cols = tmem_plan.sfb_columns
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.overlapping_accum = False
        self.num_acc_pipeline_stages = tmem_plan.accumulator_pipeline_stages


__all__ = ["BlockScaledMoEGroupedGemmWgradDenseRubinKernel"]
