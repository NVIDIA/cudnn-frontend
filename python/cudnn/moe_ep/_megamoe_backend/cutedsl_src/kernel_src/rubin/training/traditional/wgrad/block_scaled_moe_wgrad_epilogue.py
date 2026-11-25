"""Autonomous epilogue for block-scaled MoE weight-gradient GEMM."""
# Needs the TMA descriptor overrides (cute.copy on the *Override* TMA atoms with a residue
# tensor); helpers/dsl_helpers.py builds them on the public cuTeDSL release wheel.

from typing import Any, ClassVar, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import Int16, Int32, Int64
from cutlass.utils.gemm.sm100 import (
    epilogue_smem_copy_and_partition,
    epilogue_tmem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

from ......api import ImplDesc, KernelComponent, ProblemDesc, StaticOrRuntimeIntegerType
from ......helpers.dsl_helpers import is_fully_oob_tile, make_override_tma_atom
from ......helpers.ptx_helpers import tma_reduce_add_override
from ......helpers.smem_workspace import SmemRegion, SmemWorkspace
from .....schedulers.base import SchedulerConsumer


def _make_epilogue_smem_region(
    output_dtype: type, output_layout: utils.LayoutEnum, epilogue_tile_shape: Tuple[int, int], stage_count: int
) -> SmemRegion:
    """Derive the staged C layout without constructing CuTe IR objects."""
    tile_m, tile_n = epilogue_tile_shape
    leading_extent = tile_n if output_layout.is_n_major_c() else tile_m
    leading_bits = leading_extent * output_dtype.width
    if leading_bits % 8 != 0:
        raise ValueError("The C epilogue leading dimension must occupy whole bytes.")

    leading_bytes = leading_bits // 8
    swizzle_bytes = next((candidate for candidate in (128, 64, 32, 16) if leading_bytes % candidate == 0), None)
    if swizzle_bytes is None:
        raise ValueError(f"Unsupported C epilogue leading dimension size {leading_bytes} bytes.")

    swizzle_by_bytes = {16: (0, 4, 3), 32: (1, 4, 3), 64: (2, 4, 3), 128: (3, 4, 3)}
    swizzle = swizzle_by_bytes[swizzle_bytes]
    if output_layout.is_m_major_c() and output_dtype.width == 32 and swizzle_bytes == 128:
        swizzle = (2, 5, 2)

    contiguous_elements = swizzle_bytes * 8 // output_dtype.width
    if leading_extent % contiguous_elements != 0:
        raise ValueError(f"C leading extent {leading_extent} must be divisible by {contiguous_elements} elements.")
    leading_repetitions = leading_extent // contiguous_elements
    stage_stride = tile_m * tile_n

    if output_layout.is_n_major_c():
        m_shape = tile_m
        m_stride = contiguous_elements if leading_repetitions > 1 else tile_n
        if leading_repetitions > 1:
            n_shape = (contiguous_elements, leading_repetitions)
            n_stride = (1, tile_m * contiguous_elements)
        else:
            n_shape = tile_n
            n_stride = 1
    else:
        n_shape = tile_n
        n_stride = contiguous_elements if leading_repetitions > 1 else tile_m
        if leading_repetitions > 1:
            m_shape = (contiguous_elements, leading_repetitions)
            m_stride = (1, tile_n * contiguous_elements)
        else:
            m_shape = tile_m
            m_stride = 1

    return SmemRegion(
        name="",
        kind="tensor",
        dtype=output_dtype,
        shape=(m_shape, n_shape, stage_count),
        stride=(m_stride, n_stride, stage_stride if stage_count > 1 else 0),
        swizzle=swizzle,
        byte_alignment=128,
    )


class BlockScaledMoeWgradEpilogue(KernelComponent):
    """Store wgrad accumulators for dense or discrete expert outputs."""

    c_smem_tensor_region: ClassVar[str] = "rubin.training.traditional.block_scaled_moe_wgrad.epilogue.c"
    epilogue_sync_barrier_id: ClassVar[int] = 1
    epilogue_warp_count: ClassVar[int] = 4
    # Register-staged (M512) epilogue geometry, measured on GR100 for store and reduce-add:
    # the 256 x tile-N BF16 per-CTA output is split over two warp groups (128 registers
    # each at tile N 256);
    # N32 subtiles halve the LDTM/STS/barrier/TMA steps of N16 for the same 16 KiB of C
    # smem; one C slot per warp group with a per-warp-group store barrier is the shortest
    # store chain behind the single accumulator stage.
    register_staging_wg_count: ClassVar[int] = 2
    register_staging_tile_n: ClassVar[int] = 32
    register_staging_c_stages_per_wg: ClassVar[int] = 1

    @staticmethod
    def use_register_staging(problem_desc, impl_desc):
        """Register staging for the M512 B-reuse tile: store or additive TMA writeback.

        The 512-row tile has a single accumulator stage, so the plain epilogue would hold
        TMEM for the whole store chain; staging converts the tile to BF16 in registers
        (two warp groups, 128 registers each), releases the accumulator, then stores. The
        256-row tile keeps two accumulator stages and the plain epilogue, which overlaps
        its stores with the next tile's MMA already (register staging measured 1-4%
        slower there at K<=2048 on GR100). Every block-scaled format, output layout and
        storage mode takes this path; the FP4 per-expert global scale is applied in
        registers. Output accumulation is performed by the TMA reduce-add atom, never by
        loading old C into registers.
        """
        # Two warp groups hold the converted per-CTA tile (256 rows x tile N): 128 registers
        # per thread at tile N 256 with a 16-bit output; a 32-bit output would need 256 and
        # does not fit. Tile N 128 halves the register image.
        return problem_desc["output_dtype"].width == 16 and impl_desc["mma_tiler_mnk"][0] == 512

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "expert_count": int,
            "intermediate": StaticOrRuntimeIntegerType,
            "hidden": StaticOrRuntimeIntegerType,
            "output_dtype": type,
            "acc_dtype": type,
            "output_layout": utils.LayoutEnum,
            "global_scale_mode": str,
            "output_storage_mode": str,
            "accumulate_on_output": bool,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            "mma_tiler_mnk": tuple,
            "cluster_shape_mn": tuple,
            "use_2cta_instrs": bool,
            "accumulator_shape": tuple,
            "accumulator_stride": tuple,
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.expert_count = problem_desc["expert_count"]
        self.intermediate = problem_desc["intermediate"]
        self.hidden = problem_desc["hidden"]
        self.output_dtype = problem_desc["output_dtype"]
        self.acc_dtype = problem_desc["acc_dtype"]
        self.output_layout = problem_desc["output_layout"]
        self.global_scale_mode = problem_desc["global_scale_mode"]
        self.output_storage_mode = problem_desc["output_storage_mode"]
        self.accumulate_on_output = problem_desc["accumulate_on_output"]
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.cluster_shape_mn = impl_desc["cluster_shape_mn"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        self.epilogue_register_staging = self.use_register_staging(problem_desc, impl_desc)
        if self.epilogue_register_staging:
            self.epilogue_warp_count = 4 * self.register_staging_wg_count

        self._validate_configuration()

        self.mma_cta_count = 2 if self.use_2cta_instrs else 1
        self.cta_tile_m = self.mma_tiler_mnk[0] // self.mma_cta_count
        self.cta_tile_n = self.mma_tiler_mnk[1]
        self.mma_tile_k = self.mma_tiler_mnk[2]
        tmem_plan = impl_desc["tmem_plan"]
        self.num_sfa_tmem_cols = tmem_plan.sfa_columns
        self.num_sfb_tmem_cols = tmem_plan.sfb_columns
        self.num_sf_tmem_cols = tmem_plan.sfa_columns + tmem_plan.sfb_columns
        self.num_tmem_alloc_cols = tmem_plan.allocation_columns
        self.num_accumulator_stages = tmem_plan.accumulator_stage_count
        self.num_accumulator_pipeline_stages = tmem_plan.accumulator_pipeline_stages
        if tmem_plan.accumulator_stage_stride_columns != tmem_plan.accumulator_stage_columns:
            raise ValueError("Rubin Wgrad does not support overlapping accumulator stages.")
        if tmem_plan.accumulator_pipeline_stages != tmem_plan.accumulator_stage_count:
            raise ValueError("Rubin Wgrad requires one pipeline stage per accumulator stage.")
        self.num_accumulator_tmem_cols = tmem_plan.accumulator_columns
        self.m_repeat = tmem_plan.m_repeat
        self.accumulator_shape = impl_desc["accumulator_shape"]
        self.accumulator_stride = impl_desc["accumulator_stride"]
        self.cta_tile_shape_mnk = (self.cta_tile_m, self.cta_tile_n, self.mma_tile_k)

        self.c_dtype = self.output_dtype
        self.c_layout = self.output_layout
        self.num_c_stages = (
            self.register_staging_c_stages_per_wg * self.register_staging_wg_count
            if self.epilogue_register_staging else 2
        )
        self.epilogue_tile_shape = (
            (128, self.register_staging_tile_n) if self.epilogue_register_staging else (128, 64)
        )
        if self.cta_tile_n % self.epilogue_tile_shape[1] != 0:
            raise ValueError("The epilogue tile N must divide the CTA tile N.")
        # A B-reuse tile adds a second epilogue subtile row, so M contributes to the count too.
        self.epilogue_subtile_count = self.m_repeat * (self.cta_tile_n // self.epilogue_tile_shape[1])
        if self.epilogue_subtile_count % self.num_c_stages != 0:
            raise ValueError("The epilogue subtile count must be divisible by the C pipeline stage count.")
        self.c_smem_region = _make_epilogue_smem_region(
            self.output_dtype, self.output_layout, self.epilogue_tile_shape, self.num_c_stages
        )
        self.c_stage_bytes = self.c_smem_region.nbytes // self.num_c_stages

        self.num_acc_stage = self.num_accumulator_stages
        self.num_acc_pipeline_stages = self.num_accumulator_pipeline_stages

    def _validate_configuration(self) -> None:
        if self.epilogue_register_staging:
            if not self.use_2cta_instrs or self.mma_tiler_mnk[0] != 512 or self.mma_tiler_mnk[1] not in (128, 256):
                raise ValueError("Register-staged epilogue requires the 2-CTA M512 tile with N128 or N256.")
            if self.output_dtype.width != 16:
                raise ValueError("Full-tile register epilogue requires a 16-bit output dtype.")
        if isinstance(self.expert_count, bool) or self.expert_count <= 0:
            raise ValueError("expert_count must be a positive int.")
        for field_name in ("intermediate", "hidden"):
            value = getattr(self, field_name)
            if isinstance(value, int) and (isinstance(value, bool) or value <= 0):
                raise ValueError(f"{field_name} must be a positive int.")

        if self.global_scale_mode not in ("none", "per_expert"):
            raise ValueError(f"global_scale_mode must be 'none' or 'per_expert', got {self.global_scale_mode!r}.")
        if self.output_storage_mode not in ("dense", "discrete"):
            raise ValueError(f"output_storage_mode must be 'dense' or 'discrete', got {self.output_storage_mode!r}.")
        if self.acc_dtype is not cutlass.Float32:
            raise ValueError("BlockScaledMoeWgradEpilogue requires Float32 accumulators.")
        if self.output_dtype not in (cutlass.BFloat16, cutlass.Float16, cutlass.Float32):
            raise ValueError("BlockScaledMoeWgradEpilogue supports BF16, FP16, or FP32 output.")

        if len(self.mma_tiler_mnk) != 3:
            raise ValueError("mma_tiler_mnk must contain three dimensions.")
        if len(self.cluster_shape_mn) != 2:
            raise ValueError("cluster_shape_mn must contain two dimensions.")
        if any(
            isinstance(dimension, bool) or not isinstance(dimension, int)
            for dimension in (*self.mma_tiler_mnk, *self.cluster_shape_mn)
        ):
            raise TypeError("MMA tile and cluster dimensions must be Python ints.")
        if any(dimension <= 0 for dimension in self.mma_tiler_mnk):
            raise ValueError("mma_tiler_mnk dimensions must be positive.")
        if any(dimension <= 0 for dimension in self.cluster_shape_mn):
            raise ValueError("cluster_shape_mn dimensions must be positive.")

        mma_cta_count = 2 if self.use_2cta_instrs else 1
        if self.mma_tiler_mnk[0] % mma_cta_count != 0:
            raise ValueError("mma_tiler M must be divisible by the MMA CTA count.")
        if self.cluster_shape_mn[0] % mma_cta_count != 0:
            raise ValueError("cluster M must be divisible by the MMA CTA count.")
        cta_tile_m = self.mma_tiler_mnk[0] // mma_cta_count
        if cta_tile_m not in (128, 256):
            raise ValueError(
                f"The Rubin epilogue requires per-CTA M=128, or 256 for the B-reuse tile, got {cta_tile_m}."
            )
        if self.mma_tiler_mnk[1] not in (128, 256):
            raise ValueError(f"The Rubin epilogue supports mma_tiler N in (128, 256), got {self.mma_tiler_mnk[1]}.")

    def register_smem_regions(self, smem_workspace: SmemWorkspace) -> None:
        """Register the two-stage typed C scratch tensor."""
        smem_workspace.register_tensor(
            self.c_smem_tensor_region,
            self.c_smem_region.dtype,
            self.c_smem_region.shape,
            stride=self.c_smem_region.stride,
            swizzle=self.c_smem_region.swizzle,
            byte_alignment=self.c_smem_region.byte_alignment,
        )

    @cute.jit
    def assign_device_members(self, hidden, intermediate) -> None:
        """Bind dynamic logical dimensions inside the device kernel."""
        if cutlass.const_expr(not isinstance(self.hidden, int)):
            self.hidden = hidden
        if cutlass.const_expr(not isinstance(self.intermediate, int)):
            self.intermediate = intermediate

    def materialize_codegen_members(self) -> None:
        """Create and verify context-local CuTe epilogue layouts."""
        self.epi_tile = self.epilogue_tile_shape
        materialized_tile_shape = (cute.size(self.epi_tile[0]), cute.size(self.epi_tile[1]))
        if materialized_tile_shape != self.epilogue_tile_shape:
            raise ValueError(
                f"C epilogue tile plan mismatch: Python {self.epilogue_tile_shape}, CuTe {materialized_tile_shape}."
            )

        composed_layout = sm100_utils.make_smem_layout_epi(
            self.output_dtype, self.output_layout, self.epi_tile, self.num_c_stages
        )
        if not isinstance(composed_layout, cute.ComposedLayout):
            raise TypeError("The Rubin C staging layout must be composed.")
        self.c_smem_composed_layout_staged = cute.coalesce(composed_layout, target_profile=(1, 1, 1))
        self.c_smem_layout_staged = self.c_smem_composed_layout_staged.outer

        if (
            self.c_smem_layout_staged.shape != self.c_smem_region.shape
            or self.c_smem_layout_staged.stride != self.c_smem_region.stride
        ):
            raise ValueError(
                "C SMEM plan mismatch: Python "
                f"{self.c_smem_region.shape}:{self.c_smem_region.stride}, CuTe "
                f"{self.c_smem_layout_staged.shape}:"
                f"{self.c_smem_layout_staged.stride}."
            )
        expected_swizzle = cute.make_swizzle(*self.c_smem_region.swizzle)
        if str(self.c_smem_composed_layout_staged.inner) != str(expected_swizzle):
            raise ValueError(
                "C SMEM swizzle mismatch: Python "
                f"{self.c_smem_region.swizzle}, CuTe "
                f"{self.c_smem_composed_layout_staged.inner}."
            )

    def make_tma_store_operation(self, override: bool = False) -> Any:
        """Return a fresh C store or additive-reduction operation.

        ``override`` selects the override store whose global address, extent and strides come
        from the instruction (discrete output blocks). The reduce-add has no override DSL op:
        a discrete beta = 1 output keeps the override *store* atom (same tensor map, and the
        DSL residue helper only accepts override atoms) and ``tma_reduce_add_override`` issues
        the reduce-add instruction with that atom's descriptor.
        """
        if override:
            # Default ``no_fully_oob_tile=False``: with True the override store wrote wrong results
            # (tail-tile discrete stores); fully out-of-bounds subtiles are skipped in
            # ``_issue_output_copy`` anyway.
            return cpasync.CopyBulkTensorTileS2GOverrideOp()
        if self.accumulate_on_output:
            return cpasync.CopyReduceBulkTensorTileS2GOp()
        return cpasync.CopyBulkTensorTileS2GOp()

    def prepare_tma_store_params(
        self, output_template: cute.Tensor, override: bool = False
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        """Build launch-time C TMA parameters with a fresh operation."""
        self.materialize_codegen_members()
        if output_template.element_type is not self.output_dtype:
            raise TypeError(f"C output dtype must be {self.output_dtype}, got {output_template.element_type}.")
        actual_layout = utils.LayoutEnum.from_tensor(output_template)
        if actual_layout is not self.output_layout:
            raise ValueError(f"C output layout must be {self.output_layout}, got {actual_layout}.")

        tma_operation = self.make_tma_store_operation(override)
        epilogue_smem_layout = cute.select(self.c_smem_composed_layout_staged, mode=[0, 1])
        if override:
            tma_atom, tma_tensor = make_override_tma_atom(
                cpasync.make_tiled_tma_atom, tma_operation, output_template, epilogue_smem_layout, self.epi_tile
            )
        else:
            tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
                tma_operation, output_template, epilogue_smem_layout, self.epi_tile
            )
        return tma_atom, tma_tensor

    @cute.jit
    def _issue_output_copy(
        self,
        kernel_extension: Any,
        tma_atom_c: cute.CopyAtom,
        reduce_pitch_bytes: Optional[Int64],
        partitioned_tma_smem: cute.Tensor,
        partitioned_tma_output: cute.Tensor,
        partitioned_tma_residue: Optional[cute.Tensor],
        c_stage_index,
        subtile_idx,
    ) -> None:
        """Issue one C subtile: dense store/reduce, discrete override store, or discrete override reduce-add.

        ``reduce_pitch_bytes`` is the byte stride of the strided C dimension (discrete beta = 1 only).
        """
        smem_subtile = partitioned_tma_smem[(None, c_stage_index)]
        output_subtile = partitioned_tma_output[(None, subtile_idx)]
        if cutlass.const_expr(kernel_extension.discrete_reduce_output):
            # Inline PTX twin of the DSL override store, one instruction per TMA box of the subtile
            # (an M-major subtile of 128 x 32 bf16 is two 64 x 32 boxes under the 128B swizzle): box
            # corner address, per-dimension sizes ``min(remaining, 256) - 1`` from the residue,
            # template pitch, zero coordinates. A fully out-of-bounds box has nothing to add and is
            # skipped.
            residue_subtile = partitioned_tma_residue[(None, subtile_idx)]
            # The TMA-partitioned subtile has one mode ``(box elements, boxes)``.
            box_elements, box_count = smem_subtile.shape[0]
            box_bytes = cute.size(box_elements) * (self.output_dtype.width // 8)
            # Shared-window address of the stage's first box: the swizzled staged layout keeps the
            # stage offset out of the sliced tensor's iterator, so compute it from the stage
            # footprint (every stage holds one epilogue tile); boxes follow each other in SMEM.
            stage_bytes = cute.size_in_bytes(
                self.output_dtype, cute.select(self.c_smem_composed_layout_staged, mode=[0, 1])
            )
            stage_address = Int32(partitioned_tma_smem.iterator.toint()) + Int32(c_stage_index) * Int32(stage_bytes)
            for box_idx in cutlass.range_constexpr(box_count):
                remaining = residue_subtile[((None, box_idx),)][0]
                remaining_0 = Int32(remaining[0])
                remaining_1 = Int32(remaining[1])
                if (remaining_0 > 0) & (remaining_1 > 0):
                    sizes = (
                        Int16(cutlass.min(remaining_0, Int32(256)) - Int32(1)),
                        Int16(cutlass.min(remaining_1, Int32(256)) - Int32(1)),
                        Int16(0),
                    )
                    smem_address = stage_address + Int32(box_idx * box_bytes)
                    # One issuing thread per warp, as the DSL does for its own TMA copies.
                    with cute.arch.elect_one():
                        tma_reduce_add_override(
                            tma_atom_c, output_subtile[((None, box_idx),)].iterator, sizes,
                            (reduce_pitch_bytes, Int64(0)), smem_address,
                        )
        elif cutlass.const_expr(partitioned_tma_residue is not None):
            # Discrete override store: a fully out-of-bounds box writes nothing and its corner
            # address lies past the expert block, so skip it.
            residue_subtile = partitioned_tma_residue[(None, subtile_idx)]
            if not is_fully_oob_tile(residue_subtile):
                cute.copy(tma_atom_c, smem_subtile, (output_subtile, residue_subtile))
        else:
            cute.copy(tma_atom_c, smem_subtile, output_subtile)

    @cute.jit
    def run(
        self,
        *,
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        tmem_ptr: cute.Pointer,
        acc_pipeline,
        sched_consumer: SchedulerConsumer,
        kernel_extension: Any,
        tiled_mma: cute.TiledMma,
        tma_atom_c: cute.CopyAtom,
        output_template: cute.Tensor,
        tidx: Int32,
        real_output: Optional[cute.Tensor] = None,
        global_scale_a: Optional[cute.Tensor] = None,
        global_scale_b: Optional[cute.Tensor] = None,
    ) -> None:
        """Consume scheduler work and store all C epilogue subtiles.

        ``real_output`` is the launch-time GMEM tensor behind the TMA ``output_template`` (discrete
        C: its iterator is the Int64 per-expert pointer array); None for dense C.
        """
        if cutlass.const_expr(not smem_workspace.finalized):
            raise RuntimeError("BlockScaledMoeWgradEpilogue.run requires finalized SMEM.")
        if cutlass.const_expr(self.global_scale_mode == "none"):
            if cutlass.const_expr(global_scale_a is not None or global_scale_b is not None):
                raise ValueError("global_scale_mode='none' requires both scale tensors to be None.")
        elif cutlass.const_expr(global_scale_a is None or global_scale_b is None):
            raise ValueError("global_scale_mode='per_expert' requires both scale tensors.")

        # A B-reuse tile spreads the epilogue subtiles over M as well as N. Pin the Python
        # count to the partitioned store so a mismatch fails here rather than storing short.
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        template_by_epilogue_tile = cute.flat_divide(
            transform_partitioned_tensor_layout(
                tiled_mma.get_slice(0).partition_C(
                    cute.local_tile(
                        output_template, cute.slice_(self.mma_tiler_mnk, (None, None, 0)), (None, None, None)
                    )
                )
            ),
            self.epi_tile,
        )
        materialized_subtile_count = cute.size(template_by_epilogue_tile, mode=[2]) * cute.size(
            template_by_epilogue_tile, mode=[3]
        )
        if cutlass.const_expr(materialized_subtile_count != self.epilogue_subtile_count):
            raise ValueError(
                f"The C store covers {materialized_subtile_count} epilogue subtiles, "
                f"expected {self.epilogue_subtile_count}."
            )

        c_smem = smem_workspace.tensor(self.c_smem_tensor_region, smem_base)
        accumulator = cute.make_tensor(
            cute.recast_ptr(tmem_ptr, dtype=self.acc_dtype),
            cute.make_layout(self.accumulator_shape, stride=self.accumulator_stride),
        )
        epilogue_thread_idx = tidx % (self.epilogue_warp_count * 32)
        epilogue_warp_idx = epilogue_thread_idx // 32
        epilogue_group_idx = Int32(0)
        if cutlass.const_expr(self.epilogue_register_staging):
            epilogue_group_idx = epilogue_thread_idx // 128
            epilogue_thread_idx = epilogue_thread_idx % 128
            epilogue_warp_idx = epilogue_thread_idx // 32
        mma_tile_coord_v = cute.arch.block_idx()[0] % self.mma_cta_count

        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_accumulator_pipeline_stages
        )
        epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilogue_sync_barrier_id, num_threads=self.epilogue_warp_count * 32
        )
        # TMEM release still rendezvous all epilogue warps. The register-staged store
        # loop only synchronizes the WG owning the C slot.
        if cutlass.const_expr(self.epilogue_register_staging):
            store_barrier = pipeline.NamedBarrier(barrier_id=Int32(3) + epilogue_group_idx, num_threads=128)
        else:
            store_barrier = epilogue_barrier

        work_tile_info = sched_consumer.consume_work()
        while work_tile_info.is_valid_tile:
            expert_idx = work_tile_info.expert_idx
            k_tile_count = work_tile_info.k_tile_count

            # Dense C: coordinate view of the launch-time tensor. Discrete C: the expert block as a
            # real tensor (+ TMA residue for the override store), or the template's coordinates
            # plus the block's base address for the override reduce-add.
            expert_output, output_residue = kernel_extension.output_view(
                tma_atom_c, output_template, real_output, expert_idx
            )
            reduce_pitch_bytes = None
            if cutlass.const_expr(kernel_extension.discrete_reduce_output):
                # Byte stride of the strided (non-contiguous) C dimension of every expert block.
                pitch_elements = (
                    real_output.stride[0] if cutlass.const_expr(self.output_layout.is_n_major_c()) else real_output.stride[1]
                )
                reduce_pitch_bytes = Int64(pitch_elements) * Int64(self.output_dtype.width // 8)

            mma_slice = tiled_mma.get_slice(mma_tile_coord_v)
            c_tiler = cute.slice_(self.mma_tiler_mnk, (None, None, 0))
            tiled_output = cute.local_tile(expert_output, c_tiler, (None, None, None))
            partitioned_output = mma_slice.partition_C(tiled_output)
            transformed_output = transform_partitioned_tensor_layout(partitioned_output)
            transformed_residue = None
            if cutlass.const_expr(output_residue is not None):
                transformed_residue = transform_partitioned_tensor_layout(
                    mma_slice.partition_C(cute.local_tile(output_residue, c_tiler, (None, None, None)))
                )
            mma_tile_coord_mnl = (work_tile_info.tile_m_idx // self.mma_cta_count, work_tile_info.tile_n_idx, Int32(0))

            tiled_copy_t2r, partitioned_accumulator, accumulator_registers = epilogue_tmem_copy_and_partition(
                self, epilogue_thread_idx, accumulator, transformed_output, self.epi_tile, self.use_2cta_instrs
            )
            output_registers = cute.make_rmem_tensor(accumulator_registers.shape, self.output_dtype)
            tiled_copy_r2s, retiled_output_registers, partitioned_smem = epilogue_smem_copy_and_partition(
                self, tiled_copy_t2r, output_registers, epilogue_thread_idx, c_smem
            )
            full_output_registers = None
            packed_output_registers = None
            if cutlass.const_expr(self.epilogue_register_staging):
                # Divide the complete packed BF16 output among epilogue WGs.
                # Reuse one FP32 subtile temporary for conversion, never hold
                # a complete FP32 accumulator in registers.
                packed_output_registers = cute.recast_tensor(retiled_output_registers, cutlass.Uint32)
                full_output_registers = cute.make_rmem_tensor(
                    (*packed_output_registers.shape, self.epilogue_subtile_count // self.register_staging_wg_count), cutlass.Uint32
                )

            output_by_epilogue_tile = cute.flat_divide(transformed_output, self.epi_tile)
            partitioned_tma_residue = None
            if cutlass.const_expr(transformed_residue is None):
                partitioned_tma_smem, partitioned_tma_output = cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(c_smem, 0, 2),
                    cute.group_modes(output_by_epilogue_tile, 0, 2),
                )
            else:
                residue_by_epilogue_tile = cute.flat_divide(transformed_residue, self.epi_tile)
                partitioned_tma_smem, partitioned_tma_output, partitioned_tma_residue = cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(c_smem, 0, 2),
                    (cute.group_modes(output_by_epilogue_tile, 0, 2), cute.group_modes(residue_by_epilogue_tile, 0, 2)),
                )
                partitioned_tma_residue = partitioned_tma_residue[(None, None, None, *mma_tile_coord_mnl)]
                partitioned_tma_residue = cute.group_modes(partitioned_tma_residue, 1, cute.rank(partitioned_tma_residue))
            partitioned_tma_output = partitioned_tma_output[(None, None, None, *mma_tile_coord_mnl)]
            partitioned_tma_output = cute.group_modes(partitioned_tma_output, 1, cute.rank(partitioned_tma_output))
            subtile_count = self.epilogue_subtile_count

            if cutlass.const_expr(self.epilogue_register_staging):
                # Interleave subtiles across WGs so each group owns one packed
                # register image while preserving a common release rendezvous.
                if k_tile_count > Int32(0):
                    accumulator_stage_index = acc_consumer_state.index
                    current_partitioned_accumulator = partitioned_accumulator[
                        (None, None, None, None, None, accumulator_stage_index)
                    ]
                    if cutlass.const_expr(self.global_scale_mode == "per_expert"):
                        # The scale depends only on the work tile: load it before
                        # the ACC-full wait so it is off the drain critical path.
                        alpha = cute.arch.load((global_scale_a.iterator + expert_idx).llvm_ptr, cutlass.Float32) * cute.arch.load(
                            (global_scale_b.iterator + expert_idx).llvm_ptr, cutlass.Float32
                        )
                    acc_pipeline.consumer_wait(acc_consumer_state)
                    current_partitioned_accumulator = cute.group_modes(
                        current_partitioned_accumulator, 3, cute.rank(current_partitioned_accumulator)
                    )
                    for local_subtile in cutlass.range_constexpr(
                        self.epilogue_subtile_count // self.register_staging_wg_count
                    ):
                        subtile_idx = epilogue_group_idx + self.register_staging_wg_count * local_subtile
                        cute.copy(
                            tiled_copy_t2r,
                            current_partitioned_accumulator[(None, None, None, subtile_idx)],
                            accumulator_registers,
                        )
                        accumulator_vector = tiled_copy_r2s.retile(accumulator_registers).load()
                        if cutlass.const_expr(self.global_scale_mode == "per_expert"):
                            accumulator_vector = accumulator_vector * alpha
                        retiled_output_registers.store(accumulator_vector.to(self.output_dtype))
                        full_output_registers[(None, None, None, local_subtile)].store(
                            packed_output_registers.load()
                        )
                    cute.arch.fence_view_async_tmem_load()
                    # No named barrier here: every epilogue thread is in the ACC consumer group and
                    # arrives on the empty barrier itself once its own TMEM loads are fenced.
                    acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()
                    for local_subtile in cutlass.range_constexpr(self.epilogue_subtile_count // self.register_staging_wg_count):
                        subtile_idx = epilogue_group_idx + self.register_staging_wg_count * local_subtile
                        c_stage_index = epilogue_group_idx  # one C slot per WG
                        packed_output_registers.store(
                            full_output_registers[(None, None, None, local_subtile)].load()
                        )
                        # The slot is reused every step: STS must not overwrite it while
                        # the previous TMA store is still reading it.
                        if epilogue_warp_idx == Int32(0):
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                        store_barrier.arrive_and_wait()
                        cute.copy(tiled_copy_r2s, retiled_output_registers,
                                  partitioned_smem[(None, None, None, c_stage_index)])
                        cute.arch.fence_proxy("async.shared", space="cta")
                        store_barrier.arrive_and_wait()
                        if epilogue_warp_idx == Int32(0):
                            self._issue_output_copy(
                                kernel_extension, tma_atom_c, reduce_pitch_bytes, partitioned_tma_smem,
                                partitioned_tma_output, partitioned_tma_residue, c_stage_index, subtile_idx,
                            )
                            cute.arch.cp_async_bulk_commit_group()
                elif cutlass.const_expr(not self.accumulate_on_output):
                    # Zero-token expert, beta = 0: dW is an empty sum, store zeros. (beta = 1 keeps C
                    # as is; the branch is compiled out.) Keep zero experts separate: merging a
                    # zero-filled register-staged epilogue buffer with the nonempty path creates a
                    # large live phi set.
                    c_stage_index = epilogue_group_idx  # one C slot per WG
                    retiled_output_registers.store(cute.zeros_like(retiled_output_registers))
                    # Empty experts can follow a nonempty persistent tile.
                    if epilogue_warp_idx == Int32(0):
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                    store_barrier.arrive_and_wait()
                    cute.copy(tiled_copy_r2s, retiled_output_registers,
                              partitioned_smem[(None, None, None, c_stage_index)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    store_barrier.arrive_and_wait()
                    if epilogue_warp_idx == Int32(0):
                        for local_subtile in cutlass.range_constexpr(self.epilogue_subtile_count // self.register_staging_wg_count):
                            subtile_idx = epilogue_group_idx + self.register_staging_wg_count * local_subtile
                            self._issue_output_copy(
                                kernel_extension, tma_atom_c, reduce_pitch_bytes, partitioned_tma_smem,
                                partitioned_tma_output, partitioned_tma_residue, c_stage_index, subtile_idx,
                            )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                    store_barrier.arrive_and_wait()
            elif k_tile_count > Int32(0):
                accumulator_stage_index = acc_consumer_state.index
                current_partitioned_accumulator = partitioned_accumulator[
                    (None, None, None, None, None, accumulator_stage_index)
                ]
                acc_pipeline.consumer_wait(acc_consumer_state)
                current_partitioned_accumulator = cute.group_modes(
                    current_partitioned_accumulator, 3, cute.rank(current_partitioned_accumulator)
                )

                if cutlass.const_expr(self.global_scale_mode == "per_expert"):
                    scale_a_ptr = global_scale_a.iterator + expert_idx
                    scale_b_ptr = global_scale_b.iterator + expert_idx
                    alpha = cute.arch.load(scale_a_ptr.llvm_ptr, cutlass.Float32) * cute.arch.load(
                        scale_b_ptr.llvm_ptr, cutlass.Float32
                    )
                else:
                    alpha = None

                for subtile_idx in cutlass.range(subtile_count, unroll=self.num_c_stages):
                    accumulator_subtile = current_partitioned_accumulator[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, accumulator_subtile, accumulator_registers)

                    accumulator_vector = tiled_copy_r2s.retile(accumulator_registers).load()
                    if cutlass.const_expr(self.global_scale_mode == "per_expert"):
                        accumulator_vector = accumulator_vector * alpha
                    retiled_output_registers.store(accumulator_vector.to(self.output_dtype))

                    c_stage_index = subtile_idx % self.num_c_stages
                    cute.copy(
                        tiled_copy_r2s, retiled_output_registers, partitioned_smem[(None, None, None, c_stage_index)]
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    epilogue_barrier.arrive_and_wait()

                    if epilogue_warp_idx == Int32(0):
                        self._issue_output_copy(
                            kernel_extension, tma_atom_c, reduce_pitch_bytes, partitioned_tma_smem,
                            partitioned_tma_output, partitioned_tma_residue, c_stage_index, subtile_idx,
                        )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(self.num_c_stages - 1, read=True)
                    epilogue_barrier.arrive_and_wait()

                acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
            elif cutlass.const_expr(not self.accumulate_on_output):
                # Zero-token expert, beta = 0: store zeros (beta = 1 leaves C untouched).
                zero_vector = cute.zeros_like(tiled_copy_r2s.retile(accumulator_registers), dtype=self.output_dtype)
                retiled_output_registers.store(zero_vector)
                cute.copy(tiled_copy_r2s, retiled_output_registers, partitioned_smem[(None, None, None, 0)])
                cute.arch.fence_proxy("async.shared", space="cta")
                epilogue_barrier.arrive_and_wait()

                if epilogue_warp_idx == Int32(0):
                    for subtile_idx in cutlass.range(subtile_count, unroll=1):
                        self._issue_output_copy(
                            kernel_extension, tma_atom_c, reduce_pitch_bytes, partitioned_tma_smem,
                            partitioned_tma_output, partitioned_tma_residue, Int32(0), subtile_idx,
                        )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                epilogue_barrier.arrive_and_wait()

            work_tile_info = sched_consumer.consume_work()

        cute.arch.cp_async_bulk_wait_group(0, read=True)


__all__ = ["BlockScaledMoeWgradEpilogue"]
