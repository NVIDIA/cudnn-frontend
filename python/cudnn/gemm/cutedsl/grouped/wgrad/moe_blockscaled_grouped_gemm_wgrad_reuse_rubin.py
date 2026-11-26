# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Rubin (SM107) MXFP8 wgrad for short experts: ``BlockScaledMoEGroupedGemmWgradReuseKernel`` (A kept in shared memory
across the N tiles of a work unit, global A / B / SF descriptors, per-warp dW stores) on the Rubin kernel's K64 MMA,
256-token K tile and disjoint TMEM plan, so dW is meant to be the Rubin kernel's bit for bit.

Not compiled or run on Rubin hardware yet, and not picked by the wgrad API.
"""

from .moe_blockscaled_grouped_gemm_wgrad_reuse import BlockScaledMoEGroupedGemmWgradReuseKernel
from .moe_blockscaled_grouped_gemm_wgrad_rubin import BlockScaledMoEGroupedGemmWgradRubinKernel, _make_tmem_plan


class BlockScaledMoEGroupedGemmWgradReuseRubinKernel(BlockScaledMoEGroupedGemmWgradReuseKernel, BlockScaledMoEGroupedGemmWgradRubinKernel):
    """``BlockScaledMoEGroupedGemmWgradReuseKernel`` with the Rubin kernel's architecture, MMA atoms and SF copies."""

    K_TILE = 256  # 4 x K64: the Rubin kernel's MXFP8 K tile
    # The A ring at most: the shared memory left after 4 B stages, the Rubin kernel's A/B depth.
    A_STAGES = 5

    @classmethod
    def a_stages_for(cls, tokens: int, experts: int) -> int:
        """A stages for ``tokens`` over ``experts``: an average expert's K tiles (B stages take the rest), 2..A_STAGES."""
        k_tiles = -(-tokens // (max(experts, 1) * cls.K_TILE))
        return max(2, min(cls.A_STAGES, k_tiles))

    def __init__(self, *args, a_stages: int = A_STAGES, **kwargs):
        super().__init__(*args, a_stages=a_stages, **kwargs)

    def _setup_attributes(self) -> None:
        if self.a_dtype.width != 8:
            raise ValueError("The Rubin A-reuse wgrad supports MXFP8 only.")
        # Rubin's MXFP8 MMA atoms, read by _create_tiled_mma(_sfb) in the reuse kernel's setup.
        self.instruction_k = 64
        self.mma_inst_shape_mnk = (self.mma_tiler[0], self.mma_tiler[1], self.instruction_k)
        self.mma_inst_shape_mnk_sfb = (self.mma_tiler[0] // 2, self.mma_tiler[1], self.instruction_k)
        super()._setup_attributes()
        # Rubin's TMEM: disjoint accumulator stages next to the SF columns.
        tmem_plan = _make_tmem_plan(
            tile_n=self.mma_tiler[1],
            tile_k=self.sf_window_k,
            sf_vec_size=self.sf_vec_size,
            architecture=self.architecture,
        )
        self.num_tmem_alloc_cols = tmem_plan.allocation_columns
        self.num_accumulator_tmem_cols = tmem_plan.accumulator_columns
        self.num_acc_stage = tmem_plan.accumulator_stage_count
        self.num_acc_pipeline_stages = tmem_plan.accumulator_pipeline_stages
        self.num_sfa_tmem_cols = tmem_plan.sfa_columns
        self.num_sfb_tmem_cols = tmem_plan.sfb_columns
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.overlapping_accum = False
        self.epi_release_chunks = self.epi_chunks_per_warp


__all__ = ["BlockScaledMoEGroupedGemmWgradReuseRubinKernel"]
