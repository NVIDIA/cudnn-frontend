"""Mainloop component for Rubin block-scaled MoE weight gradients."""
# Needs the TMA descriptor overrides (cute.copy on the *Override* TMA atoms with a residue
# tensor); helpers/dsl_helpers.py builds them on the public cuTeDSL release wheel.

from typing import ClassVar, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cutlass_dsl import Boolean, Int32

from ......api import ImplDesc, KernelComponent, OptionalRequirement, ProblemDesc, StaticOrRuntimeIntegerType
from ......helpers.cute_py_helpers import Tcgen05MmaInstruction, Tcgen05TmemPlan, make_smem_layouts
from ......helpers.dsl_helpers import (
    guarded_override_atom,
    is_fully_oob_tile,
    make_override_tma_atom,
    override_load_op,
    tma_multicast_mask,
)
from ......helpers.smem_workspace import SmemWorkspace
from ......helpers.utils import round_up
from ......quant_def import QuantKind
from .....schedulers.base import SchedulerConsumer
from ....custom_mix_cga_helpers import (
    PipelineTmaUmmaMixedCga,
    TmaAtomOrPair,
    bind_executable_tma_load_fields,
    make_executable_tma_atom_or_pair,
)


def issue_wgrad_k_tile(
    k_tile_index, tma_atoms: tuple, global_slices: list, residue_slices: list, smem_stages: list,
    multicast_masks: tuple, is_mixed_cga: bool, tma_barrier, stage_index,
):
    """Issue the four TMA loads of one K tile.

    Plain Python (not ``@cute.jit``): it is inlined into the staged caller, so only Python-level
    loops / branches on static values are allowed here. ``residue_slices[i]`` is None for a plain
    TMA operand, else the TMA residue slice whose ``[(None, k_tile_index)]`` supplies the copy's
    remaining extents.
    """
    for operand_index in range(4):
        tma_atom = tma_atoms[operand_index]
        if residue_slices[operand_index] is None:
            source = global_slices[operand_index][(None, k_tile_index)]
        else:
            source = (
                global_slices[operand_index][(None, k_tile_index)],
                residue_slices[operand_index][(None, k_tile_index)],
            )
        if is_mixed_cga:
            # Mixed CGA atoms carry no executable TMA fields; bind them per issue.
            tma_atom = bind_executable_tma_load_fields(
                make_executable_tma_atom_or_pair(tma_atom), tma_bar_ptr=tma_barrier,
                tma_desc_ptr=None, mcast_mask=multicast_masks[operand_index],
            )
            cute.copy(tma_atom, source, smem_stages[operand_index][(None, stage_index)])
        else:
            cute.copy(
                tma_atom, source, smem_stages[operand_index][(None, stage_index)],
                tma_bar_ptr=tma_barrier, mcast_mask=multicast_masks[operand_index],
            )

class BlockScaledMoeWgradMainloop(KernelComponent):
    """Own the TMA, MMA, pipeline, SMEM, and TMEM weight-gradient mainloop."""

    ab_pipeline_mbarriers_region: ClassVar[str] = "rubin.training.block_scaled_moe_wgrad.mainloop.ab_pipeline_mbarriers"
    a_smem_tensor_region: ClassVar[str] = "rubin.training.block_scaled_moe_wgrad.mainloop.a_smem_tensor"
    b_smem_tensor_region: ClassVar[str] = "rubin.training.block_scaled_moe_wgrad.mainloop.b_smem_tensor"
    sfa_smem_tensor_region: ClassVar[str] = "rubin.training.block_scaled_moe_wgrad.mainloop.sfa_smem_tensor"
    sfb_smem_tensor_region: ClassVar[str] = "rubin.training.block_scaled_moe_wgrad.mainloop.sfb_smem_tensor"
    acc_pipeline_mbarriers_region: ClassVar[str] = (
        "rubin.training.block_scaled_moe_wgrad.mainloop.acc_pipeline_mbarriers"
    )
    tmem_holding_buffer_region: ClassVar[str] = "rubin.training.block_scaled_moe_wgrad.mainloop.tmem_holding_buffer"
    tmem_deallocation_mbarrier_region: ClassVar[str] = (
        "rubin.training.block_scaled_moe_wgrad.mainloop.tmem_deallocation_mbarrier"
    )
    tmem_allocation_barrier_id: ClassVar[int] = 2

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "expert_count": int,
            "intermediate": StaticOrRuntimeIntegerType,
            "hidden": StaticOrRuntimeIntegerType,
            "a_dtype": type,
            "b_dtype": type,
            "sf_dtype": type,
            "acc_dtype": type,
            "ab_sf_vec_size": int,
            "ab_quant_kind": str,
            "a_major_mode": OperandMajorMode,
            "b_major_mode": OperandMajorMode,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            "mma_tiler_mnk": tuple,
            "cluster_shape_mn": tuple,
            "fallback_cluster_shape_mn": OptionalRequirement(Optional[tuple]),
            "use_2cta_instrs": bool,
            "mainloop_smem_budget_bytes": int,
            "num_accumulator_consumer_warps_per_cta": int,
            "accumulator_shape": tuple,
            "accumulator_stride": tuple,
            "sf_window_k": int,
            "tmem_plan": Tcgen05TmemPlan,
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.expert_count = problem_desc["expert_count"]
        self.intermediate = problem_desc["intermediate"]
        self.hidden = problem_desc["hidden"]
        self.a_dtype = problem_desc["a_dtype"]
        self.b_dtype = problem_desc["b_dtype"]
        self.sf_dtype = problem_desc["sf_dtype"]
        self.acc_dtype = problem_desc["acc_dtype"]
        self.ab_sf_vec_size = problem_desc["ab_sf_vec_size"]
        self.ab_quant_kind = QuantKind(problem_desc["ab_quant_kind"])
        self.a_major_mode = problem_desc["a_major_mode"]
        self.b_major_mode = problem_desc["b_major_mode"]
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.cluster_shape_mn = impl_desc["cluster_shape_mn"]
        self.fallback_cluster_shape_mn = impl_desc.get("fallback_cluster_shape_mn")
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        self.mainloop_smem_budget_bytes = impl_desc["mainloop_smem_budget_bytes"]
        self.num_accumulator_consumer_warps_per_cta = impl_desc["num_accumulator_consumer_warps_per_cta"]

        self.architecture = "sm_107"
        self.mma_cta_count = 2 if self.use_2cta_instrs else 1
        self.instruction_m = 128 * self.mma_cta_count
        self.instruction_k = 128 if self.a_dtype.width == 4 else 64
        self.sf_window_k = impl_desc["sf_window_k"]
        self.num_mma_instructions_per_sf_window = self.sf_window_k // self.instruction_k
        self.num_sf_windows_per_ab_stage = self.mma_tiler_mnk[2] // self.sf_window_k
        self._validate_configuration()

        # B-reuse: tile M repeats the MMA instruction twice over one shared B operand.
        self.m_repeat = self.mma_tiler_mnk[0] // self.instruction_m
        self.enable_breuse = self.m_repeat == 2
        self.cta_tile_m = self.mma_tiler_mnk[0] // self.mma_cta_count
        self.cta_tile_n = self.mma_tiler_mnk[1]
        self.mma_tile_k = self.mma_tiler_mnk[2]
        tmem_plan = impl_desc["tmem_plan"]
        self.tmem_plan = tmem_plan
        self.num_sfa_tmem_cols = tmem_plan.sfa_columns
        self.num_sfb_tmem_cols = tmem_plan.sfb_columns
        self.num_sf_tmem_cols = tmem_plan.sfa_columns + tmem_plan.sfb_columns
        self.num_tmem_alloc_cols = tmem_plan.allocation_columns
        required_tmem_cols = tmem_plan.accumulator_columns + self.num_sf_tmem_cols
        if required_tmem_cols > self.num_tmem_alloc_cols:
            raise ValueError(
                f"The accumulator and SF TMEM stage need {required_tmem_cols} columns, "
                f"but the allocation has {self.num_tmem_alloc_cols}."
            )
        self.num_accumulator_stages = tmem_plan.accumulator_stage_count
        self.num_accumulator_pipeline_stages = tmem_plan.accumulator_pipeline_stages
        if tmem_plan.accumulator_stage_stride_columns != tmem_plan.accumulator_stage_columns:
            raise ValueError("Rubin Wgrad does not support overlapping accumulator stages.")
        if tmem_plan.accumulator_pipeline_stages != tmem_plan.accumulator_stage_count:
            raise ValueError("Rubin Wgrad requires one pipeline stage per accumulator stage.")
        if tmem_plan.m_repeat != self.m_repeat:
            raise ValueError(
                f"TMEM plan M repeat {tmem_plan.m_repeat} does not match the mainloop tile ({self.m_repeat})."
            )
        self.num_accumulator_tmem_cols = tmem_plan.accumulator_columns
        self.accumulator_shape = impl_desc["accumulator_shape"]
        self.accumulator_stride = impl_desc["accumulator_stride"]
        self.mma_instruction = Tcgen05MmaInstruction(
            a_type=self.a_dtype,
            b_type=self.b_dtype,
            acc_type=self.acc_dtype,
            instruction_mnk=(self.instruction_m, self.mma_tiler_mnk[1], self.instruction_k),
            participates=self.mma_cta_count,
            sfa_type=self.sf_dtype,
            sfb_type=self.sf_dtype,
            sf_vec_size=self.ab_sf_vec_size,
        )

        # SFB is shared by both B-reuse halves, so its plan tracks the instruction, not the tile.
        self.sfb_instruction_shape_mnk = (
            self.instruction_m // self.mma_cta_count,
            round_up(self.cta_tile_n, 128),
            self.instruction_k,
        )
        self.sfb_mma_instruction = Tcgen05MmaInstruction(
            a_type=self.a_dtype,
            b_type=self.b_dtype,
            acc_type=self.acc_dtype,
            instruction_mnk=self.sfb_instruction_shape_mnk,
            participates=1,
            sfa_type=self.sf_dtype,
            sfb_type=self.sf_dtype,
            sf_vec_size=self.ab_sf_vec_size,
        )
        self.mma_tiler_sfb = (self.sfb_instruction_shape_mnk[0], self.sfb_instruction_shape_mnk[1], self.mma_tile_k)

        self.cluster_layout_shape_vmnk = (
            (self.mma_cta_count,),
            self.cluster_shape_mn[0] // self.mma_cta_count,
            self.cluster_shape_mn[1],
            1,
        )
        self.cluster_layout_sfb_shape_vmnk = ((1,), *self.cluster_shape_mn, 1)
        self.resolved_fallback_cluster_shape_mn = (
            self.cluster_shape_mn if self.fallback_cluster_shape_mn is None else self.fallback_cluster_shape_mn
        )
        self.is_mixed_cga = self.resolved_fallback_cluster_shape_mn != self.cluster_shape_mn
        self.num_mcast_ctas_a = self.cluster_layout_shape_vmnk[2]
        self.num_mcast_ctas_b = self.cluster_layout_shape_vmnk[1]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self._select_ab_stage_plan()
        self.num_tma_load_bytes = self.ab_stage_payload_bytes * self.mma_cta_count

    def _validate_configuration(self) -> None:
        expected_ab_sf_vec_size = self.ab_quant_kind.sf_vec_size
        if self.ab_sf_vec_size != expected_ab_sf_vec_size:
            raise ValueError(
                f"{self.ab_quant_kind} requires ab_sf_vec_size {expected_ab_sf_vec_size}, got {self.ab_sf_vec_size}."
            )
        expected_dtypes = {
            "nvfp4": (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN),
            "mxfp4": (cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU),
            "mxfp8_e4m3": (cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU),
            "mxfp8_e5m2": (cutlass.Float8E5M2, cutlass.Float8E8M0FNU),
        }
        expected_operand_dtype, expected_sf_dtype = expected_dtypes[self.ab_quant_kind]
        if self.a_dtype is not expected_operand_dtype or self.b_dtype is not expected_operand_dtype:
            raise ValueError(f"{self.ab_quant_kind} requires A and B dtype {expected_operand_dtype}.")
        if self.sf_dtype is not expected_sf_dtype:
            raise ValueError(f"{self.ab_quant_kind} requires scale-factor dtype {expected_sf_dtype}.")

        for dtype_name, dtype in (
            ("a_dtype", self.a_dtype),
            ("b_dtype", self.b_dtype),
            ("sf_dtype", self.sf_dtype),
            ("acc_dtype", self.acc_dtype),
        ):
            if not hasattr(dtype, "width"):
                raise TypeError(f"{dtype_name} must be a CUTLASS numeric type.")
        if self.a_dtype.width not in (4, 8) or self.b_dtype.width != self.a_dtype.width:
            raise ValueError(f"{self.ab_quant_kind} requires 4- or 8-bit A and B operands of one width.")
        if self.sf_dtype.width != 8:
            raise ValueError("Block-scaled MMA scale factors must use an 8-bit dtype.")
        if self.acc_dtype.width != 32:
            raise ValueError("Block-scaled weight-gradient accumulators must use a 32-bit dtype.")
        if self.a_dtype.width == 4 and (
            self.a_major_mode != OperandMajorMode.K or self.b_major_mode != OperandMajorMode.K
        ):
            raise ValueError("Four-bit TCGen05 operands require K-major A and B layouts.")

        if len(self.mma_tiler_mnk) != 3:
            raise ValueError("mma_tiler_mnk must contain three dimensions.")
        if not all(isinstance(dimension, int) and not isinstance(dimension, bool) for dimension in self.mma_tiler_mnk):
            raise TypeError("mma_tiler_mnk dimensions must be Python integers.")
        if any(dimension <= 0 for dimension in self.mma_tiler_mnk):
            raise ValueError("mma_tiler_mnk dimensions must be positive.")

        supported_mma_m = (self.instruction_m, 2 * self.instruction_m)
        if self.mma_tiler_mnk[0] not in supported_mma_m:
            raise ValueError(
                f"{self.mma_cta_count}-CTA MMA requires tile M in {supported_mma_m} "
                f"(the second is the B-reuse tile), got {self.mma_tiler_mnk[0]}."
            )
        if self.mma_tiler_mnk[1] not in (128, 256):
            raise ValueError("Rubin BlockScaledMoeWgradMainloop supports tile N in (128, 256).")
        supported_mma_tile_k = (self.instruction_k * 2, self.instruction_k * 4)
        if self.mma_tiler_mnk[2] not in supported_mma_tile_k:
            raise ValueError(
                f"Rubin block-scaled Wgrad requires tile K in {supported_mma_tile_k}, got {self.mma_tiler_mnk[2]}."
            )
        if self.mma_tiler_mnk[2] % self.ab_sf_vec_size != 0:
            raise ValueError("Tile K must be divisible by ab_sf_vec_size.")
        if self.sf_window_k <= 0 or self.sf_window_k % self.instruction_k != 0:
            raise ValueError("SF window K must be a positive multiple of instruction K.")
        if self.mma_tiler_mnk[2] % self.sf_window_k != 0:
            raise ValueError("SF window K must divide tile K.")

        if len(self.cluster_shape_mn) != 2:
            raise ValueError("cluster_shape_mn must contain two dimensions.")
        if not all(
            isinstance(dimension, int) and not isinstance(dimension, bool) for dimension in self.cluster_shape_mn
        ):
            raise TypeError("cluster_shape_mn dimensions must be Python integers.")
        if any(dimension <= 0 for dimension in self.cluster_shape_mn):
            raise ValueError("cluster_shape_mn dimensions must be positive.")
        if self.cluster_shape_mn[0] % self.mma_cta_count != 0:
            raise ValueError("Cluster M must be divisible by the MMA CTA participation count.")
        if self.cluster_shape_mn[0] * self.cluster_shape_mn[1] > 16:
            raise ValueError("A Rubin thread-block cluster may contain at most 16 CTAs.")
        if self.fallback_cluster_shape_mn is not None:
            if len(self.fallback_cluster_shape_mn) != 2:
                raise ValueError("fallback_cluster_shape_mn must contain two dimensions.")
            if not all(
                isinstance(dimension, int) and not isinstance(dimension, bool)
                for dimension in self.fallback_cluster_shape_mn
            ):
                raise TypeError("fallback_cluster_shape_mn dimensions must be Python integers.")
            if any(dimension <= 0 for dimension in self.fallback_cluster_shape_mn):
                raise ValueError("fallback_cluster_shape_mn dimensions must be positive.")
            if self.fallback_cluster_shape_mn[0] % self.mma_cta_count != 0:
                raise ValueError("Fallback cluster M must be divisible by the MMA CTA participation count.")
            if any(
                preferred % fallback != 0
                for preferred, fallback in zip(self.cluster_shape_mn, self.fallback_cluster_shape_mn)
            ):
                raise ValueError("Preferred cluster dimensions must be divisible by fallback dimensions.")

        if isinstance(self.expert_count, bool) or self.expert_count <= 0:
            raise ValueError("expert_count must be positive.")
        for dimension_name in ("intermediate", "hidden"):
            dimension = getattr(self, dimension_name)
            if isinstance(dimension, int) and (isinstance(dimension, bool) or dimension <= 0):
                raise ValueError(f"{dimension_name} must be positive.")
        if isinstance(self.mainloop_smem_budget_bytes, bool) or self.mainloop_smem_budget_bytes <= 0:
            raise ValueError("mainloop_smem_budget_bytes must be positive.")
        if (
            isinstance(self.num_accumulator_consumer_warps_per_cta, bool)
            or self.num_accumulator_consumer_warps_per_cta <= 0
        ):
            raise ValueError("num_accumulator_consumer_warps_per_cta must be positive.")

    def _select_ab_stage_plan(self) -> None:
        stage_one_regions = make_smem_layouts(
            self.mma_instruction, self.mma_tiler_mnk, 1, (self.a_major_mode, self.b_major_mode), self.architecture
        )
        independent_sfb_region = make_smem_layouts(
            self.sfb_mma_instruction, self.mma_tiler_sfb, 1, (self.a_major_mode, self.b_major_mode), self.architecture
        )[3]
        planned_sfb_region = stage_one_regions[3]
        if (
            independent_sfb_region.shape != planned_sfb_region.shape
            or independent_sfb_region.stride != planned_sfb_region.stride
            or independent_sfb_region.nbytes != planned_sfb_region.nbytes
        ):
            raise ValueError("The independent SFB MMA plan does not match the main MMA SFB operand.")
        self.ab_data_stage_payload_bytes = sum(region.nbytes for region in stage_one_regions[:2])
        self.sf_stage_payload_bytes = sum(region.nbytes for region in stage_one_regions[2:])
        self.ab_stage_payload_bytes = self.ab_data_stage_payload_bytes + self.sf_stage_payload_bytes

        mbarrier_bytes = cutlass.Int64.width // 8
        self.acc_pipeline_mbarrier_bytes = self.num_accumulator_pipeline_stages * 2 * mbarrier_bytes
        self.tmem_deallocation_mbarrier_bytes = mbarrier_bytes
        self.tmem_holding_buffer_bytes = 8
        self.mainloop_fixed_smem_bytes = (
            self.acc_pipeline_mbarrier_bytes + self.tmem_deallocation_mbarrier_bytes + self.tmem_holding_buffer_bytes
        )
        self.ab_pipeline_metadata_bytes_per_stage = 2 * mbarrier_bytes
        bytes_per_ab_stage = (
            self.ab_stage_payload_bytes
            + self.ab_pipeline_metadata_bytes_per_stage
        )
        remaining_bytes = self.mainloop_smem_budget_bytes - self.mainloop_fixed_smem_bytes
        self.num_ab_pipeline_stages = remaining_bytes // bytes_per_ab_stage
        if self.num_ab_pipeline_stages < 1:
            required_bytes = self.mainloop_fixed_smem_bytes + bytes_per_ab_stage
            raise ValueError(
                f"One AB stage and fixed mainloop state need {required_bytes} bytes, "
                f"exceeding the {self.mainloop_smem_budget_bytes}-byte budget."
            )

        (self.a_smem_region, self.b_smem_region, self.sfa_smem_region, self.sfb_smem_region) = make_smem_layouts(
            self.mma_instruction,
            self.mma_tiler_mnk,
            self.num_ab_pipeline_stages,
            (self.a_major_mode, self.b_major_mode),
            self.architecture,
        )
        self.selected_smem_bytes = (
            self.mainloop_fixed_smem_bytes
            + self.num_ab_pipeline_stages
            * self.ab_pipeline_metadata_bytes_per_stage
            + sum(
                region.nbytes
                for region in (self.a_smem_region, self.b_smem_region, self.sfa_smem_region, self.sfb_smem_region)
            )
        )

    def register_smem_regions(self, smem_workspace: SmemWorkspace) -> None:
        """Register every mainloop-owned barrier and tensor region."""
        smem_workspace.register_mbarrier(self.ab_pipeline_mbarriers_region, self.num_ab_pipeline_stages * 2)
        for region_name, region in (
            (self.a_smem_tensor_region, self.a_smem_region),
            (self.b_smem_tensor_region, self.b_smem_region),
            (self.sfa_smem_tensor_region, self.sfa_smem_region),
            (self.sfb_smem_tensor_region, self.sfb_smem_region),
        ):
            smem_workspace.register_tensor(
                region_name,
                region.dtype,
                region.shape,
                stride=region.stride,
                swizzle=region.swizzle,
                byte_alignment=region.byte_alignment,
            )
        smem_workspace.register_mbarrier(self.acc_pipeline_mbarriers_region, self.num_accumulator_pipeline_stages * 2)
        smem_workspace.register_mbarrier(self.tmem_deallocation_mbarrier_region, 1)
        smem_workspace.register_tensor(self.tmem_holding_buffer_region, cutlass.Int32, (1,), byte_alignment=8)

    def mma_permutation_mnk(self):
        """M permutation that gives each CTA of a two-CTA B-reuse tile a contiguous row block.

        Coordinate ``(i, j, k)`` maps to tile row ``i + instruction_m * j + (instruction_m // 2) * k``,
        so the atom consumes ``(i, j)`` -- 128 lanes on each of the two CTAs -- and the trailing
        ``k`` becomes the repeat mode. Without it the two halves of a CTA sit 256 rows apart
        because the instruction's own M spans both CTAs, which splits every epilogue store.

        One-CTA B-reuse needs no permutation: the plain tiling already puts the second half
        directly below the first. Applying it there would interleave the halves 64 rows at a
        time and make each epilogue subtile two disjoint row ranges.
        """
        if not (self.enable_breuse and self.use_2cta_instrs):
            return (1, 1, 1)
        m_layout = cute.make_layout(
            shape=(self.instruction_m // 2, 2, 2),
            stride=(1, self.instruction_m, self.instruction_m // 2),
        )
        return (m_layout, self.mma_tiler_mnk[1], self.instruction_k)

    def make_tiled_mma(self, b_collector_op=None) -> cute.TiledMma:
        """Create a context-local block-scaled MMA object.

        With B reuse the B collector op defaults to DISCARD (the primary MMA); the
        ``(bkeep, breuse)`` pair passes FILL / LASTUSE explicitly.
        """
        cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        optional_arguments = {}
        if self.enable_breuse:
            optional_arguments["permutation_mnk"] = self.mma_permutation_mnk()
            optional_arguments["a_collector_op"] = tcgen05.CollectorOp.DISCARD
            optional_arguments["b_collector_op"] = (
                tcgen05.CollectorOp.DISCARD if b_collector_op is None else b_collector_op
            )
        return sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.ab_sf_vec_size,
            cta_group,
            (self.instruction_m, self.mma_tiler_mnk[1], self.instruction_k),
            **optional_arguments,
        )

    def make_tiled_mma_breuse_pair(self) -> Tuple[cute.TiledMma, cute.TiledMma]:
        """Create the ``(bkeep, breuse)`` MMA pair that shares one B operand.

        ``bkeep`` fills the B collector buffer and ``breuse`` consumes it with LASTUSE, so the
        second instruction never re-reads B from SMEM.
        """
        collector_ops = (tcgen05.CollectorOp.FILL, tcgen05.CollectorOp.LASTUSE)
        bkeep, breuse = (self.make_tiled_mma(b_collector_op) for b_collector_op in collector_ops)
        return bkeep, breuse

    def materialize_codegen_members(self) -> None:
        """Rebuild and validate context-local CuTe MMA and layout objects."""
        self.tiled_mma = self.make_tiled_mma()
        self.tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.ab_sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.sfb_instruction_shape_mnk,
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (self.tiled_mma.thr_id.shape,)
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (self.tiled_mma_sfb.thr_id.shape,)
        )
        if self.cluster_layout_vmnk.shape != self.cluster_layout_shape_vmnk:
            raise ValueError(
                f"Main cluster layout mismatch: Python "
                f"{self.cluster_layout_shape_vmnk}, CuTe "
                f"{self.cluster_layout_vmnk.shape}."
            )
        if self.cluster_layout_sfb_vmnk.shape != self.cluster_layout_sfb_shape_vmnk:
            raise ValueError(
                f"SFB cluster layout mismatch: Python "
                f"{self.cluster_layout_sfb_shape_vmnk}, CuTe "
                f"{self.cluster_layout_sfb_vmnk.shape}."
            )
        # Without mixed CGA the fallback cluster is the preferred one: alias the layouts.
        self.fallback_cluster_layout_vmnk = self.cluster_layout_vmnk
        self.fallback_cluster_layout_sfb_vmnk = self.cluster_layout_sfb_vmnk
        if self.is_mixed_cga:
            self.fallback_cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout((*self.resolved_fallback_cluster_shape_mn, 1)), (self.tiled_mma.thr_id.shape,)
            )
            self.fallback_cluster_layout_sfb_vmnk = cute.tiled_divide(
                cute.make_layout((*self.resolved_fallback_cluster_shape_mn, 1)), (self.tiled_mma_sfb.thr_id.shape,)
            )

        self.a_smem_composed_layout_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler_mnk, self.a_dtype, self.num_ab_pipeline_stages
        )
        self.b_smem_composed_layout_staged = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler_mnk, self.b_dtype, self.num_ab_pipeline_stages
        )
        self.a_smem_layout_staged = self.a_smem_composed_layout_staged.outer
        self.b_smem_layout_staged = self.b_smem_composed_layout_staged.outer
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            self.tiled_mma, self.mma_tiler_mnk, self.ab_sf_vec_size, self.num_ab_pipeline_stages
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            self.tiled_mma, self.mma_tiler_mnk, self.ab_sf_vec_size, self.num_ab_pipeline_stages
        )
        for tensor_name, materialized_layout, planned_region in (
            ("A", self.a_smem_layout_staged, self.a_smem_region),
            ("B", self.b_smem_layout_staged, self.b_smem_region),
            ("SFA", self.sfa_smem_layout_staged, self.sfa_smem_region),
            ("SFB", self.sfb_smem_layout_staged, self.sfb_smem_region),
        ):
            if materialized_layout.shape != planned_region.shape or materialized_layout.stride != planned_region.stride:
                raise ValueError(
                    f"{tensor_name} SMEM plan mismatch: Python "
                    f"{planned_region.shape}:{planned_region.stride}, CuTe "
                    f"{materialized_layout.shape}:{materialized_layout.stride}."
                )

        accumulator_partition_shape = self.tiled_mma.partition_shape_C(self.mma_tiler_mnk[:2])
        accumulator_fragment = self.tiled_mma.make_fragment_C(
            cute.append(accumulator_partition_shape, self.num_accumulator_stages)
        )
        self.accumulator_mma_layout_staged = accumulator_fragment.layout
        self.validate_accumulator_layout(accumulator_fragment.layout)
        self.validate_scale_factor_columns()

    def validate_scale_factor_columns(self) -> None:
        """Assert the planned scale-factor budget covers the real TMEM layouts.

        SFB is placed at ``acc + num_accumulator + num_sfa``, so an SFA under-count does not
        merely waste columns -- it lands SFB on top of SFA and the MMA reads scale factors
        the s2t copies never wrote. On a B-reuse tile SFA doubles with the M repeat while
        SFB, being shared, does not, which is exactly the asymmetry worth pinning down.

        Measured in TMEM columns: recast the layout to 32 bits so cosize counts columns
        rather than scale-factor bytes, then mask off the TMEM lane bits, since a TMEM
        address is ``(lane << 16) | column``.
        """
        sf_window_tiler_mnk = (self.mma_tiler_mnk[0], self.mma_tiler_mnk[1], self.sf_window_k)
        for operand_name, make_smem_layout, make_tmem_layout, planned_columns in (
            ("SFA", blockscaled_utils.make_smem_layout_sfa, blockscaled_utils.make_tmem_layout_sfa, self.num_sfa_tmem_cols),
            ("SFB", blockscaled_utils.make_smem_layout_sfb, blockscaled_utils.make_tmem_layout_sfb, self.num_sfb_tmem_cols),
        ):
            window_smem_layout = cute.slice_(
                make_smem_layout(self.tiled_mma, sf_window_tiler_mnk, self.ab_sf_vec_size, 1),
                (None, None, None, 0),
            )
            tmem_layout = make_tmem_layout(
                self.tiled_mma, sf_window_tiler_mnk, self.ab_sf_vec_size, window_smem_layout
            )
            measured_columns = cute.cosize(cute.recast_layout(32, self.sf_dtype.width, tmem_layout)) & 0x0000FFFF
            if measured_columns > planned_columns:
                raise ValueError(
                    f"{operand_name} occupies {measured_columns} TMEM columns but the plan "
                    f"reserved {planned_columns}."
                )

    def validate_accumulator_layout(self, accumulator_layout) -> None:
        """Assert the MMA accumulator matches the kernel's shared TMEM view.

        The MMA writes through ``make_fragment_C`` while the epilogue reads through
        the kernel-resolved accumulator shape/stride. Check the generated fragment
        against that view: a divergent M-repeat or stage stride would corrupt output.
        """
        # Mainloop and epilogue receive the same layout from the kernel.
        epilogue_m_shape, epilogue_n_shape, epilogue_stage_shape = self.accumulator_shape
        epilogue_m_stride, epilogue_n_stride, epilogue_stage_stride = self.accumulator_stride
        if self.m_repeat == 1:
            epilogue_m_shape, epilogue_m_stride = (epilogue_m_shape, 1), (epilogue_m_stride, 0)
        expected_modes = (
            ("atom M", [0, 0], epilogue_m_shape[0], epilogue_m_stride[0]),
            ("atom N", [0, 1], epilogue_n_shape, epilogue_n_stride),
            ("M repeat", [1], epilogue_m_shape[1], epilogue_m_stride[1]),
            ("N repeat", [2], 1, 0),
            ("stage", [3], epilogue_stage_shape, epilogue_stage_stride),
        )
        for mode_name, mode, expected_size, expected_stride in expected_modes:
            mode_layout = cute.get(accumulator_layout, mode=mode)
            actual_size = cute.size(mode_layout)
            actual_stride = mode_layout.stride
            if actual_size != expected_size or (expected_size > 1 and actual_stride != expected_stride):
                raise ValueError(
                    f"Accumulator TMEM {mode_name} mode mismatch: expected "
                    f"{expected_size}:{expected_stride}, CuTe {actual_size}:{actual_stride}."
                )

    def prepare_tma_load_params(
        self, a: cute.Tensor, b: cute.Tensor, sfa: cute.Tensor, sfb: cute.Tensor, override_roles: Tuple[str, ...] = ()
    ) -> Tuple:
        """Build A, B, SFA, and SFB preferred/fallback TMA atom pairs.

        ``override_roles`` names the operands whose atoms are TMA override variants: their
        launch-time tensor is only a template (dtype, box, swizzle) and every copy supplies the
        expert's global address, extent and strides in the instruction.
        """
        a_stage_layout = cute.slice_(self.a_smem_composed_layout_staged, (None, None, None, 0))
        b_stage_layout = cute.slice_(self.b_smem_composed_layout_staged, (None, None, None, 0))
        sfa_stage_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_stage_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        a_operation = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, self.tiled_mma.thr_id)
        b_operation = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, self.tiled_mma.thr_id)
        sfa_operation = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, self.tiled_mma.thr_id)
        sfb_operation = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, self.tiled_mma.thr_id)
        fallback_a_operation = fallback_b_operation = fallback_sfa_operation = fallback_sfb_operation = None
        if self.is_mixed_cga:
            fallback_a_operation = sm100_utils.cluster_shape_to_tma_atom_A(
                self.resolved_fallback_cluster_shape_mn, self.tiled_mma.thr_id
            )
            fallback_b_operation = sm100_utils.cluster_shape_to_tma_atom_B(
                self.resolved_fallback_cluster_shape_mn, self.tiled_mma.thr_id
            )
            fallback_sfa_operation = sm100_utils.cluster_shape_to_tma_atom_A(
                self.resolved_fallback_cluster_shape_mn, self.tiled_mma.thr_id
            )
            fallback_sfb_operation = sm100_utils.cluster_shape_to_tma_atom_SFB(
                self.resolved_fallback_cluster_shape_mn, self.tiled_mma.thr_id
            )
        override_roles = tuple(override_roles)
        # The launch-time override atoms are unguarded (no per-copy fully-out-of-bounds branch).
        # Whether a box is fully out of bounds depends only on the work tile's M / N position (along K
        # a work tile issues ceil(token_count / tile K) K tiles, so its last K tile still holds a
        # token): ``load_one_tma_tile`` checks it once per work tile and only such tiles take the
        # guarded twins, see ``override_loads_never_fully_oob``.
        if "a" in override_roles:
            a_operation = override_load_op(a_operation)
            fallback_a_operation = None if fallback_a_operation is None else override_load_op(fallback_a_operation)
        if "b" in override_roles:
            b_operation = override_load_op(b_operation)
            fallback_b_operation = None if fallback_b_operation is None else override_load_op(fallback_b_operation)
        if "sfa" in override_roles:
            sfa_operation = override_load_op(sfa_operation)
            fallback_sfa_operation = (
                None if fallback_sfa_operation is None else override_load_op(fallback_sfa_operation)
            )
        if "sfb" in override_roles:
            sfb_operation = override_load_op(sfb_operation)
            fallback_sfb_operation = (
                None if fallback_sfb_operation is None else override_load_op(fallback_sfb_operation)
            )

        def make_tma_maybe_pair(
            make_plain_atom, operation, fallback_operation, gmem_tensor, smem_layout, mma_tiler, tiled_mma,
            preferred_cluster_layout, fallback_cluster_layout, internal_type=None,
        ):
            def make_atom(op, *args, **kwargs):
                if isinstance(op, (cpasync.CopyBulkTensorTileG2SOp, cpasync.CopyBulkTensorTileG2SMulticastOp)):
                    return make_plain_atom(op, *args, **kwargs)
                return make_override_tma_atom(make_plain_atom, op, *args, no_fully_oob_tile=True, **kwargs)

            preferred_atom, preferred_tensor = make_atom(
                operation, gmem_tensor, smem_layout, mma_tiler, tiled_mma,
                preferred_cluster_layout.shape, internal_type=internal_type,
            )
            if self.is_mixed_cga:
                fallback_atom, _ = make_atom(
                    fallback_operation, gmem_tensor, smem_layout, mma_tiler, tiled_mma,
                    fallback_cluster_layout.shape, internal_type=internal_type,
                )
                return preferred_tensor, (preferred_atom, fallback_atom)
            return preferred_tensor, preferred_atom

        a_tensor, a_atom = make_tma_maybe_pair(
            cute.nvgpu.make_tiled_tma_atom_A, a_operation, fallback_a_operation, a, a_stage_layout,
            self.mma_tiler_mnk, self.tiled_mma, self.cluster_layout_vmnk,
            self.fallback_cluster_layout_vmnk,
        )
        b_tensor, b_atom = make_tma_maybe_pair(
            cute.nvgpu.make_tiled_tma_atom_B, b_operation, fallback_b_operation, b, b_stage_layout,
            self.mma_tiler_mnk, self.tiled_mma, self.cluster_layout_vmnk,
            self.fallback_cluster_layout_vmnk,
        )
        # SF atoms use a 16-bit internal element type: with it the per-expert residue tensor of the
        # override copies keeps the ((16,4),rest) K profile of the SF data after the MMA partition
        # (a 64-bit internal type coalesces it and ``tma_partition`` rejects the pair).
        sfa_tensor, sfa_atom = make_tma_maybe_pair(
            cute.nvgpu.make_tiled_tma_atom_A, sfa_operation, fallback_sfa_operation, sfa, sfa_stage_layout,
            self.mma_tiler_mnk, self.tiled_mma, self.cluster_layout_vmnk,
            self.fallback_cluster_layout_vmnk, cutlass.Int16,
        )
        sfb_tensor, sfb_atom = make_tma_maybe_pair(
            cute.nvgpu.make_tiled_tma_atom_B, sfb_operation, fallback_sfb_operation, sfb, sfb_stage_layout,
            self.mma_tiler_sfb, self.tiled_mma_sfb, self.cluster_layout_sfb_vmnk,
            self.fallback_cluster_layout_sfb_vmnk, cutlass.Int16,
        )
        return (a_tensor, a_atom, b_tensor, b_atom, sfa_tensor, sfa_atom, sfb_tensor, sfb_atom)

    @property
    def override_loads_never_fully_oob(self) -> bool:
        """Whether no override-load box can lie fully out of bounds.

        Needs static M (``hidden``) and N (``intermediate``) that tile exactly into whole clusters:
        M a multiple of the tile M with the M-tile count a multiple of the cluster's M-tile count,
        N a multiple of the 256-wide tile N (also the SFB tile) with the N-tile count a multiple of
        the cluster N, for the preferred and (mixed CGA) the fallback cluster. Otherwise a cluster
        overhanging the grid has CTAs past M or N that still issue their (multicast) loads, with
        boxes fully out of bounds, and the guarded copies stay.
        """
        tile_m, tile_n = self.mma_tiler_mnk[0], self.mma_tiler_mnk[1]
        if not (isinstance(self.hidden, int) and isinstance(self.intermediate, int)):
            return False
        if tile_n != 256 or self.hidden % tile_m != 0 or self.intermediate % tile_n != 0:
            return False
        tiles_m, tiles_n = self.hidden // tile_m, self.intermediate // tile_n
        clusters = [tuple(self.cluster_shape_mn)]
        if self.is_mixed_cga:
            clusters.append(tuple(self.resolved_fallback_cluster_shape_mn))
        return all(
            tiles_m % (cluster_m // self.mma_cta_count) == 0 and tiles_n % cluster_n == 0
            for cluster_m, cluster_n in clusters
        )

    @cute.jit
    def assign_device_members(
        self,
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        cta_coord_in_cluster: cute.Coord,
        is_fallback_cluster: Boolean,
        hidden,
        intermediate,
    ) -> None:
        """Root context-local SMEM tensors and CTA coordinates."""
        if cutlass.const_expr(not isinstance(self.hidden, int)):
            self.hidden = hidden
        if cutlass.const_expr(not isinstance(self.intermediate, int)):
            self.intermediate = intermediate
        self.a_smem_tensor = smem_workspace.tensor(self.a_smem_tensor_region, smem_base)
        self.b_smem_tensor = smem_workspace.tensor(self.b_smem_tensor_region, smem_base)
        self.sfa_smem_tensor = smem_workspace.tensor(self.sfa_smem_tensor_region, smem_base)
        self.sfb_smem_tensor = smem_workspace.tensor(self.sfb_smem_tensor_region, smem_base)
        self.cta_coord_in_cluster = cta_coord_in_cluster
        self.is_fallback_cluster = is_fallback_cluster
        self.mma_cta_index = cta_coord_in_cluster[0] % self.mma_cta_count
        self.is_leader_cta = self.mma_cta_index == 0
        self.main_vmnk_coord = (
            self.mma_cta_index,
            cta_coord_in_cluster[0] // self.mma_cta_count,
            cta_coord_in_cluster[1],
            cta_coord_in_cluster[2],
        )
        self.sfb_vmnk_coord = (0, cta_coord_in_cluster[0], cta_coord_in_cluster[1], cta_coord_in_cluster[2])

    @cute.jit
    def create_ab_pipeline(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer):
        """Create the combined A/B/SFA/SFB TMA readiness pipeline."""
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        if cutlass.const_expr(self.is_mixed_cga):
            return PipelineTmaUmmaMixedCga.create(
                barrier_storage=smem_workspace.ptr(self.ab_pipeline_mbarriers_region, smem_base),
                num_stages=self.num_ab_pipeline_stages,
                producer_group=producer_group,
                num_mma_consumer_warps=1,
                tx_count=self.num_tma_load_bytes,
                preferred_cta_layout_vmnk=self.cluster_layout_vmnk,
                fallback_cta_layout_vmnk=self.fallback_cluster_layout_vmnk,
                cta_coord_vmnk=self.main_vmnk_coord,
                is_fallback_cluster=self.is_fallback_cluster,
                mcast_mode_mn=(1, 1),
                defer_sync=True,
            )
        num_tma_consumers = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_consumers)
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=smem_workspace.ptr(self.ab_pipeline_mbarriers_region, smem_base),
            num_stages=self.num_ab_pipeline_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=self.cluster_layout_vmnk,
            defer_sync=True,
        )

    @cute.jit
    def create_acc_pipeline(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer):
        """Create the UMMA-to-epilogue accumulator pipeline."""
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_thread_count = self.num_accumulator_consumer_warps_per_cta * 32 * self.mma_cta_count
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, consumer_thread_count)
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=smem_workspace.ptr(self.acc_pipeline_mbarriers_region, smem_base),
            num_stages=self.num_accumulator_pipeline_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            cta_layout_vmnk=self.cluster_layout_vmnk,
            defer_sync=True,
        )

    @cute.jit
    def create_tmem_allocator(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer, *, allocator_warp_id: int):
        """Bind the TMEM allocator to the finalized mainloop regions."""
        allocation_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_allocation_barrier_id,
            num_threads=32 * (1 + self.num_accumulator_consumer_warps_per_cta),
        )
        return utils.TmemAllocator(
            smem_workspace.ptr(self.tmem_holding_buffer_region, smem_base),
            barrier_for_retrieve=allocation_barrier,
            allocator_warp_id=allocator_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=smem_workspace.ptr(self.tmem_deallocation_mbarrier_region, smem_base),
            arch=self.architecture,
        )

    def _s2t_copy_and_partition(self, smem_tensor: cute.Tensor, tmem_tensor: cute.Tensor):
        compact_smem_tensor = cute.filter_zeros(smem_tensor)
        compact_tmem_tensor = cute.filter_zeros(tmem_tensor)
        cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        copy_atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(cta_group), self.sf_dtype)
        tiled_copy = tcgen05.make_s2t_copy(copy_atom, compact_tmem_tensor)
        thread_copy = tiled_copy.get_slice(0)

        # Rubin x4 UTCCP needs the MN broadcast represented explicitly in its source view.
        mn_mode = cute.get(compact_smem_tensor.layout, mode=[0, 0])
        mn_mode = cute.append(mn_mode, cute.make_layout((4,), stride=(0,)))
        broadcast_layout = cute.append(cute.group_modes(mn_mode, 0), cute.get(compact_smem_tensor.layout, mode=[0, 1]))
        broadcast_layout = cute.append(
            cute.group_modes(broadcast_layout, 0), cute.get(compact_smem_tensor.layout, mode=[1])
        )
        broadcast_layout = cute.append(broadcast_layout, cute.get(compact_smem_tensor.layout, mode=[2]))
        broadcast_layout = cute.append(broadcast_layout, cute.get(compact_smem_tensor.layout, mode=[3]))
        broadcast_smem_tensor = cute.make_tensor(compact_smem_tensor.iterator, broadcast_layout)

        partitioned_smem = thread_copy.partition_S(broadcast_smem_tensor)
        partitioned_smem = tcgen05.get_s2t_smem_desc_tensor(tiled_copy, partitioned_smem)
        partitioned_tmem = thread_copy.partition_D(compact_tmem_tensor)
        return tiled_copy, partitioned_smem, partitioned_tmem

    @cute.jit
    def load_one_tma_tile(
        self, *, tma_atoms: tuple, active_cluster_layouts: tuple, tma_tensors: tuple, real_tensors: tuple,
        kernel_extension, token_offset: Int32, token_count: Int32, multicast_masks: tuple, work_tile, ab_pipeline,
        ab_pipeline_state,
    ):
        """Partition and issue one work tile for one static physical cluster shape.

        Every operand tuple is ordered ``(a, b, sfa, sfb)``; ``real_tensors`` holds the launch-time
        GMEM tensors of the override operands (None elsewhere); ``active_cluster_layouts`` is
        ``(cluster_layout_vmnk, cluster_layout_sfb_vmnk)`` of the cluster shape being loaded. Each
        operand is resolved to its expert view by the extension: a coordinate view of the
        launch-time tensor for plain TMA operands, or a real GMEM tensor plus a TMA residue tensor
        for override operands (the copy then carries the expert's address, remaining
        extent and strides in the instruction; the K tail past ``token_count`` zero-fills).
        """
        tma_a_atom, tma_b_atom, tma_sfa_atom, tma_sfb_atom = tma_atoms
        active_cluster_layout_vmnk, active_cluster_layout_sfb_vmnk = active_cluster_layouts
        tensor_roles = ("a", "b", "sfa", "sfb")
        views = tuple(
            kernel_extension.operand_view(tensor_role, tma_atom, tma_tensor, real_tensor, token_offset, token_count)
            for tensor_role, tma_atom, tma_tensor, real_tensor in zip(tensor_roles, tma_atoms, tma_tensors, real_tensors)
        )
        thread_mma = self.tiled_mma.get_slice(self.mma_cta_index)
        thread_mma_sfb = self.tiled_mma_sfb.get_slice(self.mma_cta_index)
        a_tiler = cute.slice_(self.mma_tiler_mnk, (None, 0, None))
        b_tiler = cute.slice_(self.mma_tiler_mnk, (0, None, None))
        sfb_tiler = cute.slice_(self.mma_tiler_sfb, (0, None, None))
        tilers = (a_tiler, b_tiler, a_tiler, sfb_tiler)
        partitions = (thread_mma.partition_A, thread_mma.partition_B, thread_mma.partition_A, thread_mma_sfb.partition_B)
        a_cta_layout = cute.make_layout(cute.slice_(active_cluster_layout_vmnk, (0, 0, None, 0)).shape)
        b_cta_layout = cute.make_layout(cute.slice_(active_cluster_layout_vmnk, (0, None, 0, 0)).shape)
        sfb_cta_layout = cute.make_layout(cute.slice_(active_cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
        cta_coords = (self.main_vmnk_coord[2], self.main_vmnk_coord[1], self.main_vmnk_coord[2], self.sfb_vmnk_coord[1])
        cta_layouts = (a_cta_layout, b_cta_layout, a_cta_layout, sfb_cta_layout)
        smem_tensors = (self.a_smem_tensor, self.b_smem_tensor, self.sfa_smem_tensor, self.sfb_smem_tensor)
        mma_tile_m = work_tile.tile_m_idx // self.mma_cta_count
        tile_coords = (mma_tile_m, work_tile.tile_n_idx, mma_tile_m, work_tile.tile_n_idx)

        smem_stages = []
        global_slices = []
        residue_slices = []
        for operand_index in cutlass.range_constexpr(4):
            real_tensor, residue_tensor = views[operand_index]
            partitioned_global = cute.group_modes(
                partitions[operand_index](cute.local_tile(real_tensor, tilers[operand_index], (None, None, None))), 0, 3
            )
            grouped_smem = cute.group_modes(smem_tensors[operand_index], 0, 3)
            if cutlass.const_expr(residue_tensor is None):
                partitioned_smem, partitioned_global = cpasync.tma_partition(
                    tma_atoms[operand_index], cta_coords[operand_index], cta_layouts[operand_index],
                    grouped_smem, partitioned_global,
                )
                partitioned_residue = None
            else:
                partitioned_residue = cute.group_modes(
                    partitions[operand_index](
                        cute.local_tile(residue_tensor, tilers[operand_index], (None, None, None))
                    ), 0, 3,
                )
                partitioned_smem, partitioned_global, partitioned_residue = cpasync.tma_partition(
                    tma_atoms[operand_index], cta_coords[operand_index], cta_layouts[operand_index],
                    grouped_smem, (partitioned_global, partitioned_residue),
                )
            if cutlass.const_expr(operand_index >= 2):
                # Scale-factor partitions carry zero-stride modes; the residues keep their profile.
                partitioned_smem = cute.filter_zeros(partitioned_smem)
                partitioned_global = cute.filter_zeros(partitioned_global)
            tile_coord = (None, tile_coords[operand_index], None, 0)
            smem_stages.append(partitioned_smem)
            global_slices.append(partitioned_global[tile_coord])
            residue_slices.append(None if partitioned_residue is None else partitioned_residue[tile_coord])

        k_tile_count = work_tile.k_tile_count
        issue_args = dict(
            global_slices=global_slices, residue_slices=residue_slices, smem_stages=smem_stages,
            multicast_masks=multicast_masks, k_tile_count=k_tile_count, ab_pipeline=ab_pipeline,
        )
        override_operands = [i for i in range(4) if residue_slices[i] is not None]
        if cutlass.const_expr(self.override_loads_never_fully_oob or not override_operands):
            ab_pipeline_state = self.issue_work_tile_k_tiles(
                tma_atoms=tma_atoms, ab_pipeline_state=ab_pipeline_state, **issue_args
            )
        else:
            # Once per work tile: a CTA whose M / N box lies past the tensor (cluster overhanging the
            # grid) takes the guarded twins for all its K tiles; every other tile issues unguarded.
            has_fully_oob_box = is_fully_oob_tile(residue_slices[override_operands[0]][(None, 0)])
            for i in override_operands[1:]:
                has_fully_oob_box = has_fully_oob_box | is_fully_oob_tile(residue_slices[i][(None, 0)])
            if has_fully_oob_box:
                guarded_atoms = tuple(
                    tma_atoms[i] if residue_slices[i] is None else guarded_override_atom(tma_atoms[i]) for i in range(4)
                )
                ab_pipeline_state = self.issue_work_tile_k_tiles(
                    tma_atoms=guarded_atoms, ab_pipeline_state=ab_pipeline_state, **issue_args
                )
            else:
                ab_pipeline_state = self.issue_work_tile_k_tiles(
                    tma_atoms=tma_atoms, ab_pipeline_state=ab_pipeline_state, **issue_args
                )
        return ab_pipeline_state

    @cute.jit
    def issue_work_tile_k_tiles(
        self, *, tma_atoms: tuple, global_slices: list, residue_slices: list, smem_stages: list,
        multicast_masks: tuple, k_tile_count, ab_pipeline, ab_pipeline_state,
    ):
        """The K loop of one work tile, all K tiles through the given (guarded or unguarded) atoms."""
        empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
        for k_tile_index in cutlass.range(k_tile_count, unroll=1):
            ab_pipeline.producer_acquire(ab_pipeline_state, empty_status)
            tma_barrier = ab_pipeline.producer_get_barrier(ab_pipeline_state)
            stage_index = ab_pipeline_state.index
            ab_pipeline_state.advance()
            if k_tile_index + 1 < k_tile_count:
                empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
            issue_wgrad_k_tile(
                k_tile_index, tma_atoms, global_slices, residue_slices, smem_stages, multicast_masks, self.is_mixed_cga,
                tma_barrier, stage_index,
            )
        return ab_pipeline_state

    @cute.jit
    def run_tma(
        self,
        *,
        tma_a_tensor: cute.Tensor,
        tma_a_atom: TmaAtomOrPair,
        tma_b_tensor: cute.Tensor,
        tma_b_atom: TmaAtomOrPair,
        tma_sfa_tensor: cute.Tensor,
        tma_sfa_atom: TmaAtomOrPair,
        tma_sfb_tensor: cute.Tensor,
        tma_sfb_atom: TmaAtomOrPair,
        real_tensors: tuple,
        expert_token_offsets: cute.Tensor,
        ab_pipeline,
        ab_pipeline_state: pipeline.PipelineState,
        sched_consumer: SchedulerConsumer,
        kernel_extension,
    ) -> None:
        """Load A, B and their scale factors into the same pipeline stage.

        ``real_tensors`` = ``(a, b, sfa, sfb)`` launch-time GMEM tensors for the override
        operands (None for plain TMA operands).
        """
        multicast_masks = (None, None, None, None)
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or self.use_2cta_instrs):
            fallback_shape = self.resolved_fallback_cluster_shape_mn if self.is_mixed_cga else None
            # The multicast helper takes the preferred polarity; dynamic Booleans negate with ~.
            is_preferred = ~self.is_fallback_cluster if self.is_mixed_cga else None
            multicast_masks = tuple(
                tma_multicast_mask(
                    self.cluster_shape_mn, fallback_shape, self.cta_coord_in_cluster,
                    is_preferred, self.use_2cta_instrs, tensor_role,
                )
                for tensor_role in ("a", "b", "sfa", "sfb")
            )

        tma_atoms = (tma_a_atom, tma_b_atom, tma_sfa_atom, tma_sfb_atom)
        tma_tensors = (tma_a_tensor, tma_b_tensor, tma_sfa_tensor, tma_sfb_tensor)
        preferred_layouts = (self.cluster_layout_vmnk, self.cluster_layout_sfb_vmnk)
        fallback_layouts = (self.fallback_cluster_layout_vmnk, self.fallback_cluster_layout_sfb_vmnk)
        work_tile = sched_consumer.consume_work()
        while work_tile.is_valid_tile:
            # Exact token range of the expert: the override copies zero-fill the K tail past it.
            token_offset, token_count = kernel_extension.expert_token_range(expert_token_offsets, work_tile.expert_idx)
            if cutlass.const_expr(self.is_mixed_cga):
                # Each atom is a (preferred, fallback) pair; the physical cluster picks one.
                if self.is_fallback_cluster:
                    ab_pipeline_state = self.load_one_tma_tile(
                        tma_atoms=tuple(atom[1] for atom in tma_atoms), active_cluster_layouts=fallback_layouts,
                        tma_tensors=tma_tensors, real_tensors=real_tensors, kernel_extension=kernel_extension,
                        token_offset=token_offset, token_count=token_count, multicast_masks=multicast_masks,
                        work_tile=work_tile, ab_pipeline=ab_pipeline, ab_pipeline_state=ab_pipeline_state,
                    )
                else:
                    ab_pipeline_state = self.load_one_tma_tile(
                        tma_atoms=tuple(atom[0] for atom in tma_atoms), active_cluster_layouts=preferred_layouts,
                        tma_tensors=tma_tensors, real_tensors=real_tensors, kernel_extension=kernel_extension,
                        token_offset=token_offset, token_count=token_count, multicast_masks=multicast_masks,
                        work_tile=work_tile, ab_pipeline=ab_pipeline, ab_pipeline_state=ab_pipeline_state,
                    )
            else:
                ab_pipeline_state = self.load_one_tma_tile(
                    tma_atoms=tma_atoms, active_cluster_layouts=preferred_layouts,
                    tma_tensors=tma_tensors, real_tensors=real_tensors, kernel_extension=kernel_extension,
                    token_offset=token_offset, token_count=token_count, multicast_masks=multicast_masks,
                    work_tile=work_tile, ab_pipeline=ab_pipeline, ab_pipeline_state=ab_pipeline_state,
                )

            work_tile = sched_consumer.consume_work()

        ab_pipeline.producer_tail(ab_pipeline_state)

    @cute.jit
    def run_mma(
        self,
        *,
        tmem_allocator,
        ab_pipeline,
        ab_pipeline_state: pipeline.PipelineState,
        acc_pipeline,
        sched_consumer: SchedulerConsumer,
    ) -> None:
        """Copy scale factors to TMEM and execute block-scaled weight-gradient MMA."""
        tiled_mma = self.make_tiled_mma()
        tiled_mma_bkeep = None
        tiled_mma_breuse = None
        if cutlass.const_expr(self.enable_breuse):
            tiled_mma_bkeep, tiled_mma_breuse = self.make_tiled_mma_breuse_pair()
        fragment_a = tiled_mma.make_fragment_A(self.a_smem_tensor)
        fragment_b = tiled_mma.make_fragment_B(self.b_smem_tensor)

        tmem_allocator.wait_for_alloc()
        accumulator_pointer = tmem_allocator.retrieve_ptr(self.acc_dtype)
        accumulator_base = cute.make_tensor(accumulator_pointer, self.accumulator_mma_layout_staged)

        sf_window_tiler_mnk = (self.mma_tiler_mnk[0], self.mma_tiler_mnk[1], self.sf_window_k)
        sfa_window_smem_layout = cute.slice_(
            blockscaled_utils.make_smem_layout_sfa(tiled_mma, sf_window_tiler_mnk, self.ab_sf_vec_size, 1),
            (None, None, None, 0),
        )
        sfb_window_smem_layout = cute.slice_(
            blockscaled_utils.make_smem_layout_sfb(tiled_mma, sf_window_tiler_mnk, self.ab_sf_vec_size, 1),
            (None, None, None, 0),
        )

        sfa_pointer = cute.recast_ptr(accumulator_pointer + self.num_accumulator_tmem_cols, dtype=self.sf_dtype)
        sfa_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, sf_window_tiler_mnk, self.ab_sf_vec_size, sfa_window_smem_layout
        )
        tmem_sfa = cute.make_tensor(sfa_pointer, sfa_layout)

        sfb_pointer = cute.recast_ptr(
            accumulator_pointer + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols, dtype=self.sf_dtype
        )
        sfb_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, sf_window_tiler_mnk, self.ab_sf_vec_size, sfb_window_smem_layout
        )
        tmem_sfb = cute.make_tensor(sfb_pointer, sfb_layout)
        (tiled_copy_sfa, partitioned_smem_sfa, partitioned_tmem_sfa) = self._s2t_copy_and_partition(
            self.sfa_smem_tensor, tmem_sfa
        )
        (tiled_copy_sfb, partitioned_smem_sfb, partitioned_tmem_sfb) = self._s2t_copy_and_partition(
            self.sfb_smem_tensor, tmem_sfb
        )
        # UTCCP partitions K at copy granularity, not instruction granularity.
        # For MXFP8 a K128 SF window needs one copy but feeds two K64 MMAs.
        sf_copies_per_window = cute.size(partitioned_tmem_sfa, mode=[3])
        assert cute.size(partitioned_tmem_sfb, mode=[3]) == sf_copies_per_window
        assert cute.size(partitioned_smem_sfa, mode=[3]) == (
            self.num_sf_windows_per_ab_stage * sf_copies_per_window
        )
        assert cute.size(partitioned_smem_sfb, mode=[3]) == (
            self.num_sf_windows_per_ab_stage * sf_copies_per_window
        )

        acc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_accumulator_pipeline_stages
        )
        work_tile = sched_consumer.consume_work()
        while work_tile.is_valid_tile:
            k_tile_count = work_tile.k_tile_count
            accumulator_stage_index = acc_producer_state.index

            if self.is_leader_cta:
                accumulator = accumulator_base[(None, None, None, accumulator_stage_index)]
                full_status = cutlass.Boolean(1)
                next_ab_pipeline_state = ab_pipeline_state.clone()
                next_ab_pipeline_state.advance()
                if k_tile_count > 0:
                    full_status = ab_pipeline.consumer_try_wait(ab_pipeline_state)
                    acc_pipeline.producer_acquire(acc_producer_state)

                for k_tile_index in cutlass.range(k_tile_count, unroll=1):
                    ab_pipeline.consumer_wait(ab_pipeline_state, full_status)
                    next_full_status = cutlass.Boolean(1)
                    if k_tile_index + 1 < k_tile_count:
                        next_full_status = ab_pipeline.consumer_try_wait(next_ab_pipeline_state)

                    sf_stage_index = ab_pipeline_state.index

                    if cutlass.const_expr(not self.enable_breuse and self.num_sf_windows_per_ab_stage == 1):
                        # One SF window per stage: stage every scale factor, then one MMA call
                        # covers the whole K of the stage.
                        s2t_stage_coord = (None, None, None, None, sf_stage_index)
                        cute.copy(tiled_copy_sfa, partitioned_smem_sfa[s2t_stage_coord], partitioned_tmem_sfa)
                        cute.copy(tiled_copy_sfb, partitioned_smem_sfb[s2t_stage_coord], partitioned_tmem_sfb)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile_index != 0)
                        ab_stage_coord = (None, None, None, ab_pipeline_state.index)
                        cute.gemm(
                            tiled_mma,
                            accumulator,
                            [fragment_a[ab_stage_coord], tmem_sfa],
                            [fragment_b[ab_stage_coord], tmem_sfb],
                            accumulator,
                        )
                    else:
                        # Stage one SF window, then issue one MMA per instruction K block and
                        # per M repeat. With B reuse the two repeats share one B operand: bkeep
                        # fills the collector buffer, breuse consumes it. SFA is per repeat (the
                        # repeat mode of the S2T copy stays free, so one copy stages both), SFB
                        # is shared.
                        for sf_window_index in cutlass.range_constexpr(self.num_sf_windows_per_ab_stage):
                            for copy_k in cutlass.range_constexpr(sf_copies_per_window):
                                source_copy_k = sf_window_index * sf_copies_per_window + copy_k
                                s2t_source_coord = (None, None, None, source_copy_k, sf_stage_index)
                                s2t_destination_coord = (None, None, None, copy_k)
                                cute.copy(
                                    tiled_copy_sfa,
                                    partitioned_smem_sfa[s2t_source_coord],
                                    partitioned_tmem_sfa[s2t_destination_coord],
                                )
                                cute.copy(
                                    tiled_copy_sfb,
                                    partitioned_smem_sfb[s2t_source_coord],
                                    partitioned_tmem_sfb[s2t_destination_coord],
                                )

                            for window_k_block in cutlass.range_constexpr(self.num_mma_instructions_per_sf_window):
                                stage_k_block = (
                                    sf_window_index * self.num_mma_instructions_per_sf_window + window_k_block
                                )
                                accumulate = k_tile_index != 0 if stage_k_block == 0 else True
                                b_coord = (None, 0, stage_k_block, ab_pipeline_state.index)
                                for m_repeat_index in cutlass.range_constexpr(self.m_repeat):
                                    repeat_mma = tiled_mma
                                    if cutlass.const_expr(self.enable_breuse):
                                        repeat_mma = tiled_mma_bkeep if m_repeat_index == 0 else tiled_mma_breuse
                                    repeat_mma.set(tcgen05.Field.ACCUMULATE, accumulate)
                                    repeat_accumulator = accumulator[(None, m_repeat_index, 0)]
                                    cute.gemm(
                                        repeat_mma,
                                        repeat_accumulator,
                                        [
                                            fragment_a[(None, m_repeat_index, stage_k_block, ab_pipeline_state.index)],
                                            tmem_sfa[(None, m_repeat_index, window_k_block)],
                                        ],
                                        [fragment_b[b_coord], tmem_sfb[(None, 0, window_k_block)]],
                                        repeat_accumulator,
                                    )

                    ab_pipeline.consumer_release(ab_pipeline_state)
                    ab_pipeline_state.advance()
                    next_ab_pipeline_state.advance()
                    full_status = next_full_status

                if k_tile_count > 0:
                    acc_pipeline.producer_commit(acc_producer_state)

            if k_tile_count > 0:
                acc_producer_state.advance()
            work_tile = sched_consumer.consume_work()

        acc_pipeline.producer_tail(acc_producer_state)


__all__ = ["BlockScaledMoeWgradMainloop"]
