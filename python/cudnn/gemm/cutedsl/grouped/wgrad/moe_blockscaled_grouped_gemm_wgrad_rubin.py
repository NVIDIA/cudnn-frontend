# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Rubin (SM107) block-scaled MoE grouped GEMM weight-gradient kernel."""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05

from ..moe_kernel_helpers import compute_stages_wgrad
from .moe_blockscaled_grouped_gemm_wgrad import (
    BlockScaledMoEGroupedGemmWgradKernel,
)


@dataclass(frozen=True)
class _TmemPlan:
    allocation_columns: int
    accumulator_columns: int
    accumulator_stage_count: int
    accumulator_pipeline_stages: int
    sfa_columns: int
    sfb_columns: int


def _round_tmem_allocation_columns(used_columns: int, capacity_columns: int) -> int:
    if used_columns <= 512:
        allocation_columns = max(32, 1 << (used_columns - 1).bit_length())
    else:
        allocation_columns = (used_columns + 31) // 32 * 32
    if allocation_columns > capacity_columns:
        raise ValueError(
            f"Rubin wgrad needs {allocation_columns} TMEM columns, "
            f"exceeding {capacity_columns}."
        )
    return allocation_columns


def _make_tmem_plan(
    *,
    tile_n: int,
    tile_k: int,
    sf_vec_size: int,
    architecture: str,
) -> _TmemPlan:
    """Plan disjoint accumulator and scale-factor TMEM regions for Rubin."""
    if not isinstance(sf_vec_size, int) or isinstance(sf_vec_size, bool):
        raise TypeError("sf_vec_size must be a Python integer.")
    if sf_vec_size <= 0:
        raise ValueError("sf_vec_size must be positive.")
    if tile_k % sf_vec_size != 0:
        raise ValueError("SF window K must be divisible by sf_vec_size.")
    if tile_k % (sf_vec_size * 4) != 0:
        raise ValueError(
            "SF window K must contain complete block-scaled basic chunks."
        )
    capacity_columns = cute.arch.get_max_tmem_alloc_cols(architecture)
    sfa_columns = tile_k // sf_vec_size
    sfb_columns = max(tile_n // 128, 1) * tile_k // sf_vec_size
    scale_factor_columns = sfa_columns + sfb_columns
    accumulator_stage_count = (
        capacity_columns - scale_factor_columns
    ) // tile_n
    if accumulator_stage_count < 1:
        raise ValueError("Scale-factor TMEM leaves no accumulator stage.")

    accumulator_columns = accumulator_stage_count * tile_n
    allocation_columns = _round_tmem_allocation_columns(
        accumulator_columns + scale_factor_columns,
        capacity_columns,
    )
    return _TmemPlan(
        allocation_columns=allocation_columns,
        accumulator_columns=accumulator_columns,
        accumulator_stage_count=accumulator_stage_count,
        accumulator_pipeline_stages=accumulator_stage_count,
        sfa_columns=sfa_columns,
        sfb_columns=sfb_columns,
    )


class BlockScaledMoEGroupedGemmWgradRubinKernel(
    BlockScaledMoEGroupedGemmWgradKernel
):
    """SM107 specialization retaining the cuDNN FE wgrad scheduler topology."""

    FIX_PAD_SIZE = 256

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.architecture = "sm_107"
        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.architecture)
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(
            self.architecture
        )

    def _setup_attributes(self) -> None:
        mma_cta_count = 2 if self.use_2cta_instrs else 1
        self.instruction_k = 128 if self.a_dtype.width == 4 else 64
        mma_inst_tile_k = 4
        mma_tiler_k = self.instruction_k * mma_inst_tile_k

        valid_quantization = (
            self.a_dtype is cutlass.Float4E2M1FN
            and self.b_dtype is cutlass.Float4E2M1FN
            and (
                (
                    self.sf_dtype is cutlass.Float8E4M3FN
                    and self.sf_vec_size == 16
                )
                or (
                    self.sf_dtype is cutlass.Float8E8M0FNU
                    and self.sf_vec_size == 32
                )
            )
        ) or (
            self.a_dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2)
            and self.b_dtype is self.a_dtype
            and self.sf_dtype is cutlass.Float8E8M0FNU
            and self.sf_vec_size == 32
        )
        if not valid_quantization:
            raise ValueError(
                "Rubin wgrad supports NVFP4, MXFP4, MXFP8-E4M3, "
                "or MXFP8-E5M2 block scaling."
            )
        if self.acc_dtype is not cutlass.Float32:
            raise ValueError("Rubin wgrad requires Float32 accumulators.")
        if self.a_dtype.width == 4 and (
            self.a_major_mode != OperandMajorMode.K
            or self.b_major_mode != OperandMajorMode.K
        ):
            raise ValueError(
                "Four-bit Rubin TCGen05 operands require K-major A and B."
            )

        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mnk = (*self.mma_inst_shape_mn, self.instruction_k)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // mma_cta_count,
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )
        self.mma_inst_shape_mnk_sfb = (
            *self.mma_inst_shape_mn_sfb,
            self.instruction_k,
        )
        self.mma_tiler = (*self.mma_inst_shape_mn, mma_tiler_k)
        self.mma_tiler_sfb = (*self.mma_inst_shape_mn_sfb, mma_tiler_k)

        use_sf_window = (
            self.sf_vec_size == 16
            and self.sf_dtype is cutlass.Float8E4M3FN
            and self.mma_tiler[1] == 256
            and self.mma_tiler[2] == 512
        )
        self.sf_window_k = (
            self.instruction_k * 2 if use_sf_window else self.mma_tiler[2]
        )
        self.num_mma_instructions_per_sf_window = (
            self.sf_window_k // self.instruction_k
        )
        self.num_sf_windows_per_ab_stage = (
            self.mma_tiler[2] // self.sf_window_k
        )

        tiled_mma = self._create_tiled_mma()
        tiled_mma_sfb = self._create_tiled_mma_sfb()
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

        if self.cta_tile_shape_mnk[0] != 128:
            raise ValueError("Rubin wgrad requires a per-CTA M tile of 128.")
        if self.mma_tiler[1] not in (128, 256):
            raise ValueError("Rubin wgrad supports MMA tile N of 128 or 256.")

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

        self.epi_tile = (128, 64)
        self.epi_tile_n = 64
        _, self.num_ab_stage, self.num_c_stage = compute_stages_wgrad(
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
        )
        self.num_sched_stages = 2

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

        tmem_plan = _make_tmem_plan(
            tile_n=self.mma_tiler[1],
            tile_k=self.sf_window_k,
            sf_vec_size=self.sf_vec_size,
            architecture=self.architecture,
        )
        self.num_tmem_alloc_cols = tmem_plan.allocation_columns
        self.num_accumulator_tmem_cols = tmem_plan.accumulator_columns
        self.num_acc_stage = tmem_plan.accumulator_stage_count
        self.num_acc_pipeline_stages = (
            tmem_plan.accumulator_pipeline_stages
        )
        self.num_sfa_tmem_cols = tmem_plan.sfa_columns
        self.num_sfb_tmem_cols = tmem_plan.sfb_columns
        self.num_sf_tmem_cols = (
            self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        )
        self.overlapping_accum = False
        self.iter_acc_early_release_in_epilogue = 0

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        stage = (None, None, None, 0)
        a_copy_size = cute.size_in_bytes(
            self.a_dtype, cute.slice_(self.a_smem_layout_staged, stage)
        )
        b_copy_size = cute.size_in_bytes(
            self.b_dtype, cute.slice_(self.b_smem_layout_staged, stage)
        )
        sfa_copy_size = cute.size_in_bytes(
            self.sf_dtype, cute.slice_(self.sfa_smem_layout_staged, stage)
        )
        sfb_copy_size = cute.size_in_bytes(
            self.sf_dtype, cute.slice_(self.sfb_smem_layout_staged, stage)
        )
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

    def _create_tiled_mma(self):
        return sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mnk,
        )

    def _create_tiled_mma_sfb(self):
        return sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mnk_sfb,
        )

    def mainloop_s2t_copy_and_partition(self, sSF, tSF):
        compact_smem = cute.filter_zeros(sSF)
        compact_tmem = cute.filter_zeros(tSF)
        copy_atom = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy = tcgen05.make_s2t_copy(copy_atom, compact_tmem)
        thread_copy = tiled_copy.get_slice(0)

        mn_mode = cute.get(compact_smem.layout, mode=[0, 0])
        mn_mode = cute.append(mn_mode, cute.make_layout((4,), stride=(0,)))
        broadcast_layout = cute.append(
            cute.group_modes(mn_mode, 0),
            cute.get(compact_smem.layout, mode=[0, 1]),
        )
        broadcast_layout = cute.append(
            cute.group_modes(broadcast_layout, 0),
            cute.get(compact_smem.layout, mode=[1]),
        )
        broadcast_layout = cute.append(
            broadcast_layout, cute.get(compact_smem.layout, mode=[2])
        )
        broadcast_layout = cute.append(
            broadcast_layout, cute.get(compact_smem.layout, mode=[3])
        )
        broadcast_smem = cute.make_tensor(
            compact_smem.iterator, broadcast_layout
        )

        partitioned_smem = thread_copy.partition_S(broadcast_smem)
        partitioned_smem = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy, partitioned_smem
        )
        partitioned_tmem = thread_copy.partition_D(compact_tmem)
        return tiled_copy, partitioned_smem, partitioned_tmem


__all__ = ["BlockScaledMoEGroupedGemmWgradRubinKernel"]
