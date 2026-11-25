"""Top-level composition for Rubin block-scaled MoE weight gradients."""
# Needs the TMA descriptor overrides (cute.copy on the *Override* TMA atoms with a residue
# tensor); helpers/dsl_helpers.py builds them on the public cuTeDSL release wheel.

from typing import ClassVar, Optional, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cutlass_dsl import Boolean, Int32
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from ......api import ImplDesc, KernelClass, OptionalRequirement, ProblemDesc, StaticOrRuntimeIntegerType
from ......helpers.cute_py_helpers import Tcgen05MmaInstruction, make_tcgen05_tmem_plan
from ......helpers.device_workspace import DeviceWorkspace
from ......helpers.smem_workspace import SmemWorkspace
from ......quant_def import QuantKind
from .....schedulers.non_clc_mixed_cga import NonClcMixedCgaConfig
from .....schedulers.wgrad_scheduler import BlackwellMoeWgradScheduler as MoeWgradScheduler
from ....custom_mix_cga_helpers import TmaAtomOrPair, pipeline_init_arrive_mixed_cga, pipeline_init_wait_mixed_cga
from .block_scaled_moe_wgrad_epilogue import BlockScaledMoeWgradEpilogue
from .block_scaled_moe_wgrad_extension import BlockScaledMoeWgradExtension
from .block_scaled_moe_wgrad_mainloop import BlockScaledMoeWgradMainloop


_aot_symbol_prefix = "rubin_moe_wgrad_aot"


class BlockScaledMoeWgradKernel(KernelClass):
    """Compose launch-time TMA templates, full-grid scheduling, MMA, and output stores.

    Per-expert TMA fields (base address, token extent, strides, discrete output block) travel in
    the TMA instructions (descriptor overrides), so there is no descriptor init kernel and no
    tensormap fence; see ``block_scaled_moe_wgrad_extension.py``.

    Memory contract of the override loads: a TMA box that is only partly inside the per-expert
    view (a ragged M / N / K tail that is not a multiple of the box, e.g. hidden 144 or an SF
    m-block pair at hidden 640 with an M = 512 tile) is clipped with the instruction's size
    operands, and the TMA then issues the clipped elements' requests to a hashed address near
    the box instead of dropping them (measured up to ~48 KiB past the box).
    Fully out-of-bounds boxes are safe (the compiler falls back to a plain TMA through the template
    descriptor). Callers must therefore keep >= 64 KiB of readable memory after each A / B / SF
    plane unless hidden and intermediate are multiples of 128 and the token padding is a multiple
    of the K tile, in which case no box is ever clipped.
    """

    mma_warp_id: ClassVar[int] = 4
    tma_warp_id: ClassVar[int] = 5
    scheduler_warp_id: ClassVar[int] = 6
    threads_per_cta: ClassVar[int] = 7 * 32
    scheduler_consumer_thread_count: ClassVar[int] = 6 * 32
    accumulator_consumer_warp_count: ClassVar[int] = 4

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "expert_count": int,
            "intermediate": StaticOrRuntimeIntegerType,
            "hidden": StaticOrRuntimeIntegerType,
            "a_dtype": type,
            "b_dtype": type,
            "sf_dtype": type,
            "output_dtype": type,
            "acc_dtype": type,
            "ab_quant_kind": str,
            "a_major_mode": OperandMajorMode,
            "b_major_mode": OperandMajorMode,
            "output_layout": utils.LayoutEnum,
            "input_order": str,
            "output_storage_mode": str,
            "global_scale_mode": str,
            "accumulate_on_output": bool,
            "fix_pad_size": OptionalRequirement(int),
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            "mma_tiler_mnk": tuple,
            "cluster_shape_mn": tuple,
            "use_2cta_instrs": bool,
            "launch_cluster_count": OptionalRequirement(Optional[int]),
            "fallback_cluster_shape_mn": OptionalRequirement(Optional[tuple]),
            "preferred_cluster_count": OptionalRequirement(Optional[int]),
            "fallback_cluster_count": OptionalRequirement(Optional[int]),
        }

    @classmethod
    def implementation_error(cls, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> Optional[str]:
        """Host-only feasibility check using the exact constructor's validation.

        Includes operand types, epilogue compatibility, MMA geometry and the
        TMEM/SMEM resource plan. Does not compile, allocate device memory or run
        a kernel. Runtime pointers/token metadata and performance are not checked.
        A successful plan is not a claim of exhaustive device validation.
        """
        try:
            cls(problem_desc, impl_desc)
        except (TypeError, ValueError, KeyError) as error:
            return f"{type(error).__name__}: {error}"
        return None

    @classmethod
    def can_implement(cls, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> bool:
        """Return whether the current implementation can construct this plan."""
        return cls.implementation_error(problem_desc, impl_desc) is None

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.expert_count = problem_desc["expert_count"]
        self.intermediate = problem_desc["intermediate"]
        self.hidden = problem_desc["hidden"]
        self.a_dtype = problem_desc["a_dtype"]
        self.b_dtype = problem_desc["b_dtype"]
        self.sf_dtype = problem_desc["sf_dtype"]
        self.output_dtype = problem_desc["output_dtype"]
        self.acc_dtype = problem_desc["acc_dtype"]
        self.ab_quant_kind = QuantKind(problem_desc["ab_quant_kind"])
        self.ab_sf_vec_size = self.ab_quant_kind.sf_vec_size
        self.fix_pad_size = problem_desc.get("fix_pad_size", 256)
        self.a_major_mode = problem_desc["a_major_mode"]
        self.b_major_mode = problem_desc["b_major_mode"]
        self.output_layout = problem_desc["output_layout"]
        self.input_order = problem_desc["input_order"]
        self.output_storage_mode = problem_desc["output_storage_mode"]
        self.global_scale_mode = problem_desc["global_scale_mode"]
        self.accumulate_on_output = problem_desc["accumulate_on_output"]

        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.cluster_shape_mn = impl_desc["cluster_shape_mn"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        requested_fallback_cluster_shape_mn = impl_desc.get("fallback_cluster_shape_mn")
        self.mixed_cga_config = None
        if requested_fallback_cluster_shape_mn is not None and requested_fallback_cluster_shape_mn != self.cluster_shape_mn:
            self.mixed_cga_config = NonClcMixedCgaConfig(
                preferred_cluster_shape=self.cluster_shape_mn,
                fallback_cluster_shape=requested_fallback_cluster_shape_mn,
                launch_cluster_count=impl_desc.get("launch_cluster_count"),
                preferred_cluster_count=impl_desc.get("preferred_cluster_count"),
                fallback_cluster_count=impl_desc.get("fallback_cluster_count"),
            )
        self.fallback_cluster_shape_mn = (
            None if self.mixed_cga_config is None else self.mixed_cga_config.fallback_cluster_shape
        )
        self.resolved_fallback_cluster_shape_mn = (
            self.cluster_shape_mn if self.fallback_cluster_shape_mn is None else self.fallback_cluster_shape_mn
        )
        self.is_mixed_cga = self.fallback_cluster_shape_mn is not None
        self.preferred_cluster_count = (
            None if not self.is_mixed_cga else self.mixed_cga_config.preferred_cluster_count
        )
        self.fallback_cluster_count = (
            None if not self.is_mixed_cga else self.mixed_cga_config.fallback_cluster_count
        )
        self.launch_cluster_count = (
            None if not self.is_mixed_cga else self.mixed_cga_config.launch_cluster_cnt_merge_as_preferred
        )

        self.architecture = "sm_107"
        self.occupancy = 1
        self._validate_configuration()

        mma_cta_count = 2 if self.use_2cta_instrs else 1
        instruction_m = 128 * mma_cta_count
        instruction_k = 128 if self.ab_quant_kind in ("nvfp4", "mxfp4") else 64
        # SF window K is the reduction span whose scales reside in TMEM at once;
        # A/B still use the full tile K in SMEM. Cap the window for B-reuse tiles
        # (duplicated SFA) and the existing NVFP4/N256 policy to leave more TMEM
        # for accumulators. This does not guarantee an extra accumulator stage.
        m_repeat = self.mma_tiler_mnk[0] // instruction_m
        compact_sf = (
            m_repeat == 2 or (self.ab_quant_kind == "nvfp4" and self.mma_tiler_mnk[1] == 256)
        )
        sf_window_k = min(self.mma_tiler_mnk[2], 2 * instruction_k) if compact_sf else self.mma_tiler_mnk[2]
        mma_instruction = Tcgen05MmaInstruction(
            a_type=self.a_dtype,
            b_type=self.b_dtype,
            acc_type=self.acc_dtype,
            instruction_mnk=(instruction_m, self.mma_tiler_mnk[1], instruction_k),
            participates=mma_cta_count,
            sfa_type=self.sf_dtype,
            sfb_type=self.sf_dtype,
            sf_vec_size=self.ab_sf_vec_size,
        )
        sf_window_tiler_mnk = (self.mma_tiler_mnk[0], self.mma_tiler_mnk[1], sf_window_k)
        tmem_plan = make_tcgen05_tmem_plan(mma_instruction, self.architecture, sf_window_tiler_mnk)
        # Both consumers use this per-CTA TMEM view. M repeats share the
        # 128 physical lanes and occupy separate accumulator column groups.
        tile_n = self.mma_tiler_mnk[1]
        accumulator_m_shape = 128 if tmem_plan.m_repeat == 1 else (128, tmem_plan.m_repeat)
        accumulator_m_stride = 1 << 16 if tmem_plan.m_repeat == 1 else (1 << 16, tile_n)
        accumulator_shape = (accumulator_m_shape, tile_n, tmem_plan.accumulator_stage_count)
        accumulator_stride = (accumulator_m_stride, 1, tmem_plan.accumulator_stage_stride_columns)
        resolved_problem_desc = ProblemDesc({**problem_desc, "ab_sf_vec_size": self.ab_sf_vec_size})
        resolved_impl_desc = ImplDesc(
            {
                **impl_desc,
                "fallback_cluster_shape_mn": self.fallback_cluster_shape_mn,
                "preferred_cluster_count": self.preferred_cluster_count,
                "fallback_cluster_count": self.fallback_cluster_count,
                "launch_cluster_count": self.launch_cluster_count,
                "sf_window_k": sf_window_k,
                "tmem_plan": tmem_plan,
                "accumulator_shape": accumulator_shape,
                "accumulator_stride": accumulator_stride,
            }
        )
        self.epilogue = BlockScaledMoeWgradEpilogue(resolved_problem_desc, resolved_impl_desc)
        self.epilogue_register_staging = self.epilogue.epilogue_register_staging
        epi_warps = self.epilogue.epilogue_warp_count
        self.epilogue_warp_ids = tuple(range(epi_warps))
        self.mma_warp_id, self.tma_warp_id, self.scheduler_warp_id = epi_warps, epi_warps + 1, epi_warps + 2
        # Register staging needs a complete auxiliary WG for register donation.
        auxiliary_warps = 4 if self.epilogue_register_staging else 3
        self.threads_per_cta = (epi_warps + auxiliary_warps) * 32
        self.scheduler_consumer_thread_count = (epi_warps + 2) * 32
        self.accumulator_consumer_warp_count = epi_warps
        resolved_impl_desc = ImplDesc(
            {
                **resolved_impl_desc,
                "num_scheduler_consumer_threads": (self.scheduler_consumer_thread_count),
                "num_accumulator_consumer_warps_per_cta": (self.accumulator_consumer_warp_count),
            }
        )
        self.extension = BlockScaledMoeWgradExtension(resolved_problem_desc, resolved_impl_desc)
        self.scheduler = MoeWgradScheduler(resolved_problem_desc, resolved_impl_desc)

        self._device_workspace = self._build_device_workspace()
        self.mainloop, self._smem_workspace = self._build_mainloop_and_smem(resolved_problem_desc, resolved_impl_desc)

    def _validate_configuration(self) -> None:
        if isinstance(self.expert_count, bool) or self.expert_count <= 0:
            raise ValueError("expert_count must be a positive int.")
        for dimension_name in ("intermediate", "hidden"):
            dimension = getattr(self, dimension_name)
            if isinstance(dimension, int) and (isinstance(dimension, bool) or dimension <= 0):
                raise ValueError(f"{dimension_name} must be a positive int.")
        for dtype_name in ("a_dtype", "b_dtype", "sf_dtype", "output_dtype", "acc_dtype"):
            dtype = getattr(self, dtype_name)
            if not hasattr(dtype, "width"):
                raise TypeError(f"{dtype_name} must be a CUTLASS numeric type.")
            if not isinstance(dtype.width, int) or dtype.width <= 0:
                raise ValueError(f"{dtype_name} must have a positive static width.")

        if self.input_order not in ("tensor2d", "tensor_ragged"):
            raise ValueError("input_order must be 'tensor2d' or 'tensor_ragged'.")
        if self.output_storage_mode not in ("dense", "discrete"):
            raise ValueError("output_storage_mode must be 'dense' or 'discrete'.")
        if self.global_scale_mode not in ("none", "per_expert"):
            raise ValueError("global_scale_mode must be 'none' or 'per_expert'.")
        minimum_sf_padding = 4 * self.ab_sf_vec_size
        if isinstance(self.fix_pad_size, bool) or self.fix_pad_size <= 0:
            raise ValueError("fix_pad_size must be a positive int.")
        if self.fix_pad_size % minimum_sf_padding != 0:
            raise ValueError(f"fix_pad_size must be divisible by {minimum_sf_padding} for {self.ab_quant_kind}.")

        if len(self.mma_tiler_mnk) != 3:
            raise ValueError("mma_tiler_mnk must contain three dimensions.")
        if len(self.cluster_shape_mn) != 2:
            raise ValueError("cluster_shape_mn must contain two dimensions.")
        for field_name, dimensions in (
            ("mma_tiler_mnk", self.mma_tiler_mnk),
            ("cluster_shape_mn", self.cluster_shape_mn),
        ):
            if not all(isinstance(dimension, int) and not isinstance(dimension, bool) for dimension in dimensions):
                raise TypeError(f"{field_name} dimensions must be Python ints.")
            if any(dimension <= 0 for dimension in dimensions):
                raise ValueError(f"{field_name} dimensions must be positive.")

        cluster_m, cluster_n = self.cluster_shape_mn
        if cluster_m & (cluster_m - 1) or cluster_n & (cluster_n - 1):
            raise ValueError("cluster_shape_mn dimensions must be powers of two.")
        if cluster_m * cluster_n > 16:
            raise ValueError("A Rubin CTA cluster may contain at most 16 CTAs.")

        mma_cta_count = 2 if self.use_2cta_instrs else 1
        if self.fallback_cluster_shape_mn is not None:
            fallback_m, fallback_n = self.fallback_cluster_shape_mn
            if fallback_m & (fallback_m - 1) or fallback_n & (fallback_n - 1):
                raise ValueError("fallback_cluster_shape_mn dimensions must be powers of two.")
            if fallback_m * fallback_n > 16:
                raise ValueError("A Rubin fallback CTA cluster may contain at most 16 CTAs.")
            if fallback_m % mma_cta_count != 0:
                raise ValueError("fallback cluster M must be divisible by the MMA CTA count.")
        # Tile/instruction compatibility is checked when constructing the TMEM
        # plan and components; do not duplicate their supported M shapes here.
        if cluster_m % mma_cta_count != 0:
            raise ValueError("cluster M must be divisible by the MMA CTA count.")
        # A cluster that is wider than the problem's tile grid can never be fully occupied: every
        # scheduler ticket covers cluster_m x cluster_n CTA tiles, so the surplus CTAs of each cluster
        # would only ever draw out-of-range tiles. Reject such shapes here so can_implement() filters
        # them for every caller (static shapes only; runtime M/N skip the check).
        tile_m, tile_n = self.mma_tiler_mnk[0], self.mma_tiler_mnk[1]
        if isinstance(self.hidden, int) and not isinstance(self.hidden, bool):
            m_tiles = -(-self.hidden // tile_m)  # tile M already spans the whole CTA pair
            if cluster_m // mma_cta_count > m_tiles:
                raise ValueError(
                    f"cluster M {cluster_m} spans {cluster_m // mma_cta_count} M tiles of {tile_m} rows but hidden={self.hidden} "
                    f"has only {m_tiles}; the cluster can never be filled."
                )
        if isinstance(self.intermediate, int) and not isinstance(self.intermediate, bool):
            n_tiles = -(-self.intermediate // tile_n)
            if cluster_n > n_tiles:
                raise ValueError(
                    f"cluster N {cluster_n} spans {cluster_n} N tiles of {tile_n} columns but intermediate={self.intermediate} "
                    f"has only {n_tiles}; the cluster can never be filled."
                )

    def _build_device_workspace(self) -> DeviceWorkspace:
        device_workspace = DeviceWorkspace()
        self.scheduler.register_device_workspace(device_workspace)
        self.extension.observe_scheduler_workspace(device_workspace)
        self.extension.register_device_workspace(device_workspace)
        device_workspace.finalize()
        return device_workspace

    def _build_mainloop_and_smem(
        self, problem_desc: ProblemDesc, resolved_impl_desc: ImplDesc
    ) -> Tuple[BlockScaledMoeWgradMainloop, SmemWorkspace]:
        smem_limit = cutlass.memory.get_smem_capacity_in_bytes(self.architecture) // self.occupancy
        smem_workspace = SmemWorkspace()
        self.scheduler.register_smem_regions(smem_workspace)
        self.epilogue.register_smem_regions(smem_workspace)

        alignment_shift_bytes = max(region.byte_alignment for region in smem_workspace.regions())
        mainloop_smem_budget_bytes = smem_limit - smem_workspace.estimate_total_bytes() - alignment_shift_bytes
        if mainloop_smem_budget_bytes <= 0:
            raise ValueError("Scheduler and Epilogue leave no SMEM budget for Mainloop.")

        mainloop_impl_desc = ImplDesc(
            {
                **resolved_impl_desc,
                "mainloop_smem_budget_bytes": mainloop_smem_budget_bytes,
                "num_scheduler_consumer_threads": (self.scheduler_consumer_thread_count),
                "num_accumulator_consumer_warps_per_cta": (self.accumulator_consumer_warp_count),
            }
        )
        mainloop = BlockScaledMoeWgradMainloop(problem_desc, mainloop_impl_desc)

        mainloop.register_smem_regions(smem_workspace)
        smem_workspace.finalize(max_bytes=smem_limit)
        return mainloop, smem_workspace

    def get_workspace_size(self) -> int:
        """Return required local workspace bytes."""
        return self._device_workspace.total_bytes("local")

    @property
    def require_zero_workspace_leading_bytes(self) -> int:
        return self._device_workspace.zero_on_allocate_bytes("local")

    def name(self) -> str:
        def dtype_name(dtype: type) -> str:
            return getattr(dtype, "__name__", str(dtype)).lower()

        tile = "x".join(str(dimension) for dimension in self.mma_tiler_mnk)
        cluster = "x".join(str(dimension) for dimension in self.cluster_shape_mn)
        mixed_suffix = ""
        if self.is_mixed_cga:
            fallback = "x".join(str(dimension) for dimension in self.fallback_cluster_shape_mn)
            mixed_suffix = (
                f"_mixed_clc_fallback{fallback}_prefclusters{self.preferred_cluster_count}_"
                f"fallbackclusters{self.fallback_cluster_count}_launchclusters{self.launch_cluster_count}"
            )
        a_major = self.a_major_mode.name.lower()
        b_major = self.b_major_mode.name.lower()
        output_layout = self.output_layout.name.lower()
        store_kind = "reduce" if self.accumulate_on_output else "store"
        cta_kind = "2cta" if self.use_2cta_instrs else "1cta"
        hidden = str(self.hidden) if isinstance(self.hidden, int) else "dynamic"
        intermediate = str(self.intermediate) if isinstance(self.intermediate, int) else "dynamic"
        return (
            f"sm107_block_scaled_moe_wgrad_{self.ab_quant_kind}_"
            f"{dtype_name(self.a_dtype)}_{dtype_name(self.b_dtype)}_"
            f"{dtype_name(self.output_dtype)}_e{self.expert_count}_"
            f"h{hidden}_i{intermediate}_tile{tile}_"
            f"cluster{cluster}_{self.input_order}_"
            f"a{a_major}_b{b_major}_c{output_layout}_"
            f"{self.output_storage_mode}_{self.global_scale_mode}_"
            f"pad{self.fix_pad_size}_{store_kind}_{cta_kind}"
            f"{f'_epi_register_staged_{len(self.epilogue_warp_ids) // 4}wg_n{self.epilogue.register_staging_tile_n}_c{self.epilogue.register_staging_c_stages_per_wg}' if self.epilogue_register_staging else ''}"
            f"{mixed_suffix}"
        )

    def aot_compile(self, out_path: Optional[str] = None, **_compile_kwargs):
        """Compile a dynamic-token TVM-FFI callable and optionally export it."""
        from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor, make_ptr
        from cutlass.cute.typing import AddressSpace, sym_int64

        dynamic_tokens = sym_int64(divisibility=self.fix_pad_size)
        dynamic_scale_columns = dynamic_tokens // self.ab_sf_vec_size
        hidden = int(self.hidden)
        intermediate = int(self.intermediate)
        hidden_padded = (hidden + 127) // 128 * 128
        intermediate_padded = (intermediate + 127) // 128 * 128

        def fake_tensor(dtype, shape, stride_order, alignment):
            return make_fake_compact_tensor(dtype, shape, stride_order=stride_order, assumed_align=alignment)

        mat_a_order = (1, 0) if self.a_major_mode == OperandMajorMode.K else (0, 1)
        mat_b_order = (0, 1) if self.b_major_mode == OperandMajorMode.K else (1, 0)
        mat_a = fake_tensor(self.a_dtype, (hidden, dynamic_tokens), mat_a_order, 16)
        mat_b = fake_tensor(self.b_dtype, (dynamic_tokens, intermediate), mat_b_order, 16)
        scale_a = fake_tensor(self.sf_dtype, (hidden_padded, dynamic_scale_columns), (1, 0), 16)
        scale_b = fake_tensor(self.sf_dtype, (intermediate_padded, dynamic_scale_columns), (1, 0), 16)

        output_order = (2, 1, 0) if self.output_layout.is_n_major_c() else (2, 0, 1)
        if self.output_storage_mode == "dense":
            output = fake_tensor(self.output_dtype, (self.expert_count, hidden, intermediate), output_order, 16)
        else:
            output = make_fake_tensor(
                self.output_dtype, (self.expert_count, hidden, intermediate), stride=(0, 0, 0), assumed_align=8
            )

        expert_token_prefix_sum = fake_tensor(cutlass.Int32, (self.expert_count,), (0,), 4)
        local_workspace = make_ptr(cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=128)
        stream = make_fake_stream()
        if self.global_scale_mode == "per_expert":
            global_scale_a = fake_tensor(cutlass.Float32, (self.expert_count,), (0,), 4)
            global_scale_b = fake_tensor(cutlass.Float32, (self.expert_count,), (0,), 4)
        else:
            global_scale_a = None
            global_scale_b = None

        compiled = cute.compile[cute.EnableTVMFFI(True)](
            self,
            mat_a=mat_a,
            mat_b=mat_b,
            scale_a=scale_a,
            scale_b=scale_b,
            output=output,
            expert_token_prefix_sum=expert_token_prefix_sum,
            local_workspace=local_workspace,
            stream=stream,
            global_scale_a=global_scale_a,
            global_scale_b=global_scale_b,
        )
        if out_path is None:
            return compiled
        compiled.export_to_c(out_path, function_name=_aot_symbol_prefix, export_only_tvm_ffi_symbols=True)
        return out_path

    @staticmethod
    def load_compiled(path: str):
        from cutlass.cute.runtime import load_module

        return load_module(path, enable_tvm_ffi=True)[_aot_symbol_prefix]

    @cute.jit
    def __call__(
        self,
        mat_a: cute.Tensor,  # logical: (hidden, sum_tokens)
        mat_b: cute.Tensor,  # logical: (sum_tokens, intermediate)
        scale_a: cute.Tensor,  # logical per expert: (hidden, expert_tokens / ab_sf_vec_size)
        scale_b: cute.Tensor,  # logical per expert: (intermediate, expert_tokens / ab_sf_vec_size)
        output: cute.Tensor,  # logical: (experts, hidden, intermediate)
        expert_token_prefix_sum: cute.Tensor,  # logical: (experts,), cumulative token ends
        local_workspace: cute.Pointer,
        stream: cuda.CUstream,
        global_scale_a: Optional[cute.Tensor] = None,  # logical: (experts,)
        global_scale_b: Optional[cute.Tensor] = None,  # logical: (experts,)
    ) -> None:
        """Build GEMM views and the TMA templates, then launch Wgrad."""
        if cutlass.const_expr(cute.rank(mat_a.layout) != 2):
            raise ValueError("mat_a must have shape (hidden, tokens).")
        if cutlass.const_expr(cute.rank(mat_b.layout) != 2):
            raise ValueError("mat_b must have shape (tokens, intermediate).")
        if cutlass.const_expr(cute.rank(scale_a.layout) != 2):
            raise ValueError("scale_a must be a two-dimensional packed tensor.")
        if cutlass.const_expr(cute.rank(scale_b.layout) != 2):
            raise ValueError("scale_b must be a two-dimensional packed tensor.")
        if cutlass.const_expr(cute.rank(expert_token_prefix_sum.layout) != 1):
            raise ValueError("expert_token_prefix_sum must contain cumulative expert ends.")
        if cutlass.const_expr(cute.rank(output.layout) != 3):
            raise ValueError("output must have logical shape (experts, hidden, intermediate).")
        if cutlass.const_expr(output.element_type is not self.output_dtype):
            raise TypeError("output dtype does not match ProblemDesc.")
        if cutlass.const_expr(isinstance(output.shape[0], int) and output.shape[0] != self.expert_count):
            raise ValueError("output expert extent is incorrect.")

        if cutlass.const_expr(isinstance(self.hidden, int)):
            hidden = self.hidden
        else:
            hidden = output.shape[1]
        if cutlass.const_expr(isinstance(self.intermediate, int)):
            intermediate = self.intermediate
        else:
            intermediate = output.shape[2]

        if cutlass.const_expr(mat_a.element_type is not self.a_dtype):
            raise TypeError("mat_a has a dtype incompatible with ProblemDesc.")
        if cutlass.const_expr(mat_b.element_type is not self.b_dtype):
            raise TypeError("mat_b has a dtype incompatible with ProblemDesc.")
        if cutlass.const_expr(scale_a.element_type is not self.sf_dtype):
            raise TypeError("scale_a has a dtype incompatible with ProblemDesc.")
        if cutlass.const_expr(scale_b.element_type is not self.sf_dtype):
            raise TypeError("scale_b has a dtype incompatible with ProblemDesc.")
        if cutlass.const_expr(expert_token_prefix_sum.element_type is not cutlass.Int32):
            raise TypeError("expert_token_prefix_sum must use Int32 elements.")

        if cutlass.const_expr(
            isinstance(self.hidden, int) and isinstance(mat_a.shape[0], int) and mat_a.shape[0] != self.hidden
        ):
            raise ValueError("mat_a leading extent must equal hidden.")
        if cutlass.const_expr(
            isinstance(self.intermediate, int)
            and isinstance(mat_b.shape[1], int)
            and mat_b.shape[1] != self.intermediate
        ):
            raise ValueError("mat_b trailing extent must equal intermediate.")
        if cutlass.const_expr(
            isinstance(output.shape[1], int) and isinstance(hidden, int) and output.shape[1] != hidden
        ):
            raise ValueError("output hidden extent is incorrect.")
        if cutlass.const_expr(
            isinstance(output.shape[2], int) and isinstance(intermediate, int) and output.shape[2] != intermediate
        ):
            raise ValueError("output intermediate extent is incorrect.")
        if cutlass.const_expr(
            isinstance(mat_a.shape[1], int) and isinstance(mat_b.shape[0], int) and mat_a.shape[1] != mat_b.shape[0]
        ):
            raise ValueError("mat_a and mat_b token extents must match.")
        if cutlass.const_expr(
            isinstance(expert_token_prefix_sum.shape[0], int) and expert_token_prefix_sum.shape[0] != self.expert_count
        ):
            raise ValueError("expert_token_prefix_sum must have one cumulative end per expert.")

        scales_are_paired = (global_scale_a is None) == (global_scale_b is None)
        if cutlass.const_expr(not scales_are_paired):
            raise ValueError("global_scale_a and global_scale_b must be provided together.")
        if cutlass.const_expr(self.global_scale_mode == "none"):
            if cutlass.const_expr(global_scale_a is not None):
                raise ValueError("global_scale_mode='none' does not accept global scales.")
        elif cutlass.const_expr(global_scale_a is None):
            raise ValueError("global_scale_mode='per_expert' requires both global scales.")
        if cutlass.const_expr(global_scale_a is not None):
            if cutlass.const_expr(cute.rank(global_scale_a.layout) != 1 or cute.rank(global_scale_b.layout) != 1):
                raise ValueError("Global scales must be one-dimensional.")
            if cutlass.const_expr(
                global_scale_a.element_type is not cutlass.Float32 or global_scale_b.element_type is not cutlass.Float32
            ):
                raise TypeError("Global scales must use Float32 elements.")
            if cutlass.const_expr(
                isinstance(global_scale_a.shape[0], int) and global_scale_a.shape[0] != self.expert_count
            ):
                raise ValueError("global_scale_a must have one value per expert.")
            if cutlass.const_expr(
                isinstance(global_scale_b.shape[0], int) and global_scale_b.shape[0] != self.expert_count
            ):
                raise ValueError("global_scale_b must have one value per expert.")

        singleton = cutlass.Int32(1)
        zero = cutlass.Int32(0)
        runtime_hidden_extent = cutlass.Int32(hidden)
        runtime_intermediate_extent = cutlass.Int32(intermediate)
        runtime_expert_extent = cutlass.Int32(self.expert_count)
        token_extent_a = mat_a.shape[1]
        token_extent_b = mat_b.shape[0]
        a_gemm = cute.make_tensor(
            mat_a.iterator,
            cute.make_layout(
                (runtime_hidden_extent, token_extent_a, singleton), stride=(mat_a.stride[0], mat_a.stride[1], zero)
            ),
        )
        b_gemm = cute.make_tensor(
            mat_b.iterator,
            cute.make_layout(
                (runtime_intermediate_extent, token_extent_b, singleton),
                stride=(mat_b.stride[1], mat_b.stride[0], zero),
            ),
        )
        if cutlass.const_expr(utils.LayoutEnum.from_tensor(a_gemm).mma_major_mode() != self.a_major_mode):
            raise ValueError("mat_a layout does not match a_major_mode.")
        if cutlass.const_expr(utils.LayoutEnum.from_tensor(b_gemm).mma_major_mode() != self.b_major_mode):
            raise ValueError("mat_b layout does not match b_major_mode.")

        scale_a_non_k_extent = cutlass.Int32(scale_a.shape[0])
        scale_b_non_k_extent = cutlass.Int32(scale_b.shape[0])
        scale_a_k_extent = scale_a.shape[1] * self.ab_sf_vec_size
        scale_b_k_extent = scale_b.shape[1] * self.ab_sf_vec_size
        if cutlass.const_expr(
            isinstance(scale_a_non_k_extent, int) and isinstance(hidden, int) and scale_a_non_k_extent < hidden
        ):
            raise ValueError("scale_a does not cover the hidden dimension.")
        if cutlass.const_expr(
            isinstance(scale_b_non_k_extent, int)
            and isinstance(intermediate, int)
            and scale_b_non_k_extent < intermediate
        ):
            raise ValueError("scale_b does not cover the intermediate dimension.")
        if cutlass.const_expr(
            isinstance(scale_a_k_extent, int)
            and isinstance(scale_b_k_extent, int)
            and scale_a_k_extent != scale_b_k_extent
        ):
            raise ValueError("scale_a and scale_b packed K extents must match.")
        sfa_gemm = cute.make_tensor(
            scale_a.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (scale_a_non_k_extent, scale_a_k_extent, singleton), self.ab_sf_vec_size
            ),
        )
        sfb_gemm = cute.make_tensor(
            scale_b.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (scale_b_non_k_extent, scale_b_k_extent, singleton), self.ab_sf_vec_size
            ),
        )

        if cutlass.const_expr(self.output_storage_mode == "dense"):
            c_stride = (output.stride[1], output.stride[2], output.stride[0])
        else:
            # Discrete: ``output`` is the Int64 pointer array; the template only lends dtype,
            # shape and stride to the C descriptor, every store overrides the global address.
            c_stride = (
                (cutlass.Int64(runtime_intermediate_extent), 1, cutlass.Int64(0))
                if cutlass.const_expr(self.output_layout.is_n_major_c())
                else (1, cutlass.Int64(runtime_hidden_extent), cutlass.Int64(0))
            )

        c_gemm = cute.make_tensor(
            output.iterator,
            cute.make_layout(
                (runtime_hidden_extent, runtime_intermediate_extent, runtime_expert_extent), stride=c_stride
            ),
        )

        if cutlass.const_expr(utils.LayoutEnum.from_tensor(c_gemm) is not self.output_layout):
            raise ValueError("Output tensor layout does not match output_layout.")

        self.mainloop.materialize_codegen_members()
        (
            tma_a_tensor,
            tma_a_atom,
            tma_b_tensor,
            tma_b_atom,
            tma_sfa_tensor,
            tma_sfa_atom,
            tma_sfb_tensor,
            tma_sfb_atom,
        ) = self.mainloop.prepare_tma_load_params(
            a_gemm, b_gemm, sfa_gemm, sfb_gemm, override_roles=self.extension.override_roles
        )
        tma_c_atom, tma_c_tensor = self.epilogue.prepare_tma_store_params(
            c_gemm, override=self.extension.uses_override("c")
        )
        # The override copies take the expert's address / extent from these GMEM tensors in the
        # kernel (the TMA tensors above only carry coordinates); plain TMA operands pass None.
        real_a = a_gemm if cutlass.const_expr(self.extension.uses_override("a")) else None
        real_b = b_gemm if cutlass.const_expr(self.extension.uses_override("b")) else None
        real_c = c_gemm if cutlass.const_expr(self.extension.uses_override("c")) else None

        actual_expert_shape = (self.expert_count, intermediate, hidden)
        grid = self.scheduler.get_grid_shape(problem_desc=actual_expert_shape)
        runtime_hidden = None if cutlass.const_expr(isinstance(self.hidden, int)) else hidden
        runtime_intermediate = None if cutlass.const_expr(isinstance(self.intermediate, int)) else intermediate
        kernel = self._kernel(
            tma_a_tensor,
            tma_a_atom,
            tma_b_tensor,
            tma_b_atom,
            tma_sfa_tensor,
            tma_sfa_atom,
            tma_sfb_tensor,
            tma_sfb_atom,
            tma_c_atom,
            tma_c_tensor,
            real_a,
            real_b,
            sfa_gemm,
            sfb_gemm,
            real_c,
            runtime_hidden,
            runtime_intermediate,
            expert_token_prefix_sum,
            local_workspace,
            global_scale_a,
            global_scale_b,
        )
        if cutlass.const_expr(self.is_mixed_cga):
            kernel.launch(
                grid=grid,
                block=(self.threads_per_cta, 1, 1),
                cluster=(*self.cluster_shape_mn, 1),
                fallback_cluster=(*self.resolved_fallback_cluster_shape_mn, 1),
                stream=stream,
                min_blocks_per_mp=self.occupancy,
            )
        else:
            kernel.launch(
                grid=grid,
                block=(self.threads_per_cta, 1, 1),
                cluster=(*self.cluster_shape_mn, 1),
                stream=stream,
                min_blocks_per_mp=self.occupancy,
            )

    @cute.kernel
    def _kernel(
        self,
        tma_a_tensor: cute.Tensor,
        tma_a_atom: TmaAtomOrPair,
        tma_b_tensor: cute.Tensor,
        tma_b_atom: TmaAtomOrPair,
        tma_sfa_tensor: cute.Tensor,
        tma_sfa_atom: TmaAtomOrPair,
        tma_sfb_tensor: cute.Tensor,
        tma_sfb_atom: TmaAtomOrPair,
        tma_c_atom: cute.CopyAtom,
        output_template: cute.Tensor,
        real_a: Optional[cute.Tensor],
        real_b: Optional[cute.Tensor],
        real_sfa: cute.Tensor,
        real_sfb: cute.Tensor,
        real_c: Optional[cute.Tensor],
        runtime_hidden,
        runtime_intermediate,
        expert_token_prefix_sum: cute.Tensor,
        local_workspace: cute.Pointer,
        global_scale_a: Optional[cute.Tensor],
        global_scale_b: Optional[cute.Tensor],
    ):
        """Run the warp topology requested by the epilogue."""
        self.mainloop.materialize_codegen_members()
        self.epilogue.materialize_codegen_members()

        storage_type = self._smem_workspace.storage_class()
        smem_allocator = utils.SmemAllocator()
        storage = smem_allocator.allocate(storage_type)
        smem_base = storage.buffer.data_ptr()
        self._device_workspace.assign_device_members(local_workspace)

        thread_idx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(thread_idx // 32)
        block_idx = cute.arch.block_idx()
        grid_dim = cute.arch.grid_dim()
        active_cluster_m = Int32(self.cluster_shape_mn[0])
        active_cluster_n = Int32(self.cluster_shape_mn[1])
        is_fallback_cluster = Boolean(False)
        if cutlass.const_expr(self.is_mixed_cga):
            active_cluster_m, active_cluster_n, _ = cute.arch.block_in_cluster_dim()
            is_fallback_cluster = (active_cluster_m == Int32(self.resolved_fallback_cluster_shape_mn[0])) & (
                active_cluster_n == Int32(self.resolved_fallback_cluster_shape_mn[1])
            )
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cta_coord_in_cluster = (
            cta_rank_in_cluster % active_cluster_m,
            cta_rank_in_cluster // active_cluster_m,
            Int32(0),
        )
        if cutlass.const_expr(isinstance(self.hidden, int)):
            hidden = self.hidden
        else:
            hidden = runtime_hidden
        if cutlass.const_expr(isinstance(self.intermediate, int)):
            intermediate = self.intermediate
        else:
            intermediate = runtime_intermediate
        actual_expert_shape = (self.expert_count, intermediate, hidden)

        self.scheduler.assign_device_members(
            offs=expert_token_prefix_sum,
            actual_expert_shape=actual_expert_shape,
            block_idx=block_idx,
            grid_dim=grid_dim,
            smem_workspace=self._smem_workspace,
            smem_base=smem_base,
            device_workspace=self._device_workspace,
            is_fallback_cluster=is_fallback_cluster,
        )
        self.mainloop.assign_device_members(
            self._smem_workspace, smem_base, cta_coord_in_cluster, is_fallback_cluster, hidden, intermediate
        )
        self.epilogue.assign_device_members(hidden, intermediate)

        ab_pipeline = self.mainloop.create_ab_pipeline(self._smem_workspace, smem_base)
        tma_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mainloop.num_ab_pipeline_stages
        )
        mma_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mainloop.num_ab_pipeline_stages
        )
        acc_pipeline = self.mainloop.create_acc_pipeline(self._smem_workspace, smem_base)
        tmem_allocator = self.mainloop.create_tmem_allocator(
            self._smem_workspace, smem_base, allocator_warp_id=self.epilogue_warp_ids[0]
        )

        if cutlass.const_expr(self.is_mixed_cga):
            pipeline_init_arrive_mixed_cga(
                self.cluster_shape_mn, self.resolved_fallback_cluster_shape_mn, is_fallback_cluster, is_relaxed=True
            )
            pipeline_init_wait_mixed_cga(
                self.cluster_shape_mn, self.resolved_fallback_cluster_shape_mn, is_fallback_cluster
            )
        else:
            pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        if cutlass.const_expr(self.epilogue_register_staging):
            # Auxiliary warps donate registers to the epilogue warp group(s).
            if warp_idx >= len(self.epilogue_warp_ids):
                cute.arch.warpgroup_reg_dealloc(80)
        scheduler = self.scheduler

        if warp_idx == self.scheduler_warp_id:
            scheduler.gen_next_work()
            while scheduler.current_work.is_valid_tile:
                scheduler.publish_work(scheduler.current_work)
                scheduler.gen_next_work()
            scheduler.publish_work(scheduler.current_work)
            scheduler.produce_tail()

        if warp_idx == self.tma_warp_id:
            sched_consumer = scheduler.make_consumer()
            self.mainloop.run_tma(
                tma_a_tensor=tma_a_tensor,
                tma_a_atom=tma_a_atom,
                tma_b_tensor=tma_b_tensor,
                tma_b_atom=tma_b_atom,
                tma_sfa_tensor=tma_sfa_tensor,
                tma_sfa_atom=tma_sfa_atom,
                tma_sfb_tensor=tma_sfb_tensor,
                tma_sfb_atom=tma_sfb_atom,
                real_tensors=(real_a, real_b, real_sfa, real_sfb),
                expert_token_offsets=expert_token_prefix_sum,
                ab_pipeline=ab_pipeline,
                ab_pipeline_state=tma_pipeline_state,
                sched_consumer=sched_consumer,
                kernel_extension=self.extension,
            )

        if warp_idx == self.mma_warp_id:
            sched_consumer = scheduler.make_consumer()
            self.mainloop.run_mma(
                tmem_allocator=tmem_allocator,
                ab_pipeline=ab_pipeline,
                ab_pipeline_state=mma_pipeline_state,
                acc_pipeline=acc_pipeline,
                sched_consumer=sched_consumer,
            )

        if warp_idx < len(self.epilogue_warp_ids):
            if cutlass.const_expr(self.epilogue_register_staging):
                # Keep the increase inside the epilogue branch: merging it
                # with the 80-register auxiliary branch before dispatch makes
                # ptxas conservatively spill the later epilogue at that limit.
                cute.arch.warpgroup_reg_alloc(208)
            sched_consumer = scheduler.make_consumer()
            tmem_allocator.allocate(self.mainloop.num_tmem_alloc_cols)
            tmem_allocator.wait_for_alloc()
            tmem_pointer = tmem_allocator.retrieve_ptr(self.acc_dtype)
            self.epilogue.run(
                smem_workspace=self._smem_workspace,
                smem_base=smem_base,
                tmem_ptr=tmem_pointer,
                acc_pipeline=acc_pipeline,
                sched_consumer=sched_consumer,
                kernel_extension=self.extension,
                tiled_mma=self.mainloop.tiled_mma,
                tma_atom_c=tma_c_atom,
                output_template=output_template,
                tidx=thread_idx,
                real_output=real_c,
                global_scale_a=global_scale_a,
                global_scale_b=global_scale_b,
            )
            tmem_allocator.relinquish_alloc_permit()
            tmem_allocator.free(tmem_pointer, self.mainloop.num_tmem_alloc_cols)

        if cutlass.const_expr(self.extension.needs_tail_reset):
            # Mixed-CGA scheduling claims work through workspace counters. Once every warp of this
            # CTA is done (the scheduler warp made its last claim), count the CTA; the last CTA of
            # the grid zeroes the counters for the next launch (no init kernel).
            cute.arch.sync_threads()
            if thread_idx == Int32(0):
                self.extension.arrive_and_reset_tail(self._device_workspace)


__all__ = ["BlockScaledMoeWgradKernel"]
