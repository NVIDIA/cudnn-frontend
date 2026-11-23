"""Full-grid CLC scheduling for fixed and mixed-CGA MoE Wgrad."""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Boolean, Int32, Integer, extract_mlir_values, new_from_mlir_values

from ...api import ImplDesc, OptionalRequirement, ProblemDesc, StaticOrRuntimeIntegerType
from ...helpers.device_workspace import DeviceWorkspace
from ...helpers.smem_workspace import SmemWorkspace
from .base import SchedulerBase, SchedulerWorkTileBase
from .clc_mixed_cga import ClcMixedCgaSchedulerWorker
from .non_clc_mixed_cga import NonClcMixedCgaConfig
from .work_id_claim import ClusterLaunchControlWorkIdState, claim_work_id
from .wgrad_mapping import WgradWorkTileInfo, compute_expert_token_range, make_wgrad_done_tile, make_wgrad_work_tile

GROUPED_RASTER_L2_FRACTION = 4


def _current_device_l2_bytes() -> int:
    """L2 size of the device of the current CUDA context (device 0 when no context is current)."""
    from cuda.bindings import driver

    err, device = driver.cuCtxGetDevice()
    if err != driver.CUresult.CUDA_SUCCESS:
        (err,) = driver.cuInit(0)
        if err == driver.CUresult.CUDA_SUCCESS:
            err, device = driver.cuDeviceGet(0)
    if err == driver.CUresult.CUDA_SUCCESS:
        err, l2_bytes = driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device)
    if err != driver.CUresult.CUDA_SUCCESS or l2_bytes <= 0:
        raise RuntimeError(f"raster_order 'grouped' needs the device L2 size: {err}")
    return int(l2_bytes)


class BlackwellMoeWgradScheduler(SchedulerBase):
    """Use hardware CLC for both fixed and preferred/fallback cluster launches."""

    pipeline_mbarriers_region = "blackwell.wgrad.scheduler.pipeline_mbarriers"
    work_tiles_region = "blackwell.wgrad.scheduler.work_tiles"
    clc_mbarriers_region = "blackwell.wgrad.scheduler.clc_mbarriers"
    clc_response_region = "blackwell.wgrad.scheduler.clc_response"
    grid_z_limit = 65535
    raster_orders = ("along_m", "along_n", "grouped")

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {"expert_count": int, "intermediate": StaticOrRuntimeIntegerType, "hidden": StaticOrRuntimeIntegerType}

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            **super().impl_desc_require(),
            "mma_tiler_mnk": tuple,
            "cluster_shape_mn": tuple,
            "use_2cta_instrs": bool,
            "launch_cluster_count": OptionalRequirement(Optional[int]),
            "fallback_cluster_shape_mn": OptionalRequirement(Optional[tuple]),
            "preferred_cluster_count": OptionalRequirement(Optional[int]),
            "fallback_cluster_count": OptionalRequirement(Optional[int]),
            "raster_order": OptionalRequirement(str),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        super().__init__(problem_desc, impl_desc)
        self.expert_count = problem_desc["expert_count"]
        self.intermediate = problem_desc["intermediate"]
        self.hidden = problem_desc["hidden"]
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.cluster_shape_mn = impl_desc["cluster_shape_mn"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        fallback_cluster_shape_mn = impl_desc.get("fallback_cluster_shape_mn")
        self.is_mixed_cga = fallback_cluster_shape_mn is not None and fallback_cluster_shape_mn != self.cluster_shape_mn
        self.mixed_cga_config = None
        self._work_id_worker = None
        if self.is_mixed_cga:
            # Shared launch geometry, also used by the upstream mixed CLC worker.
            # Its occupancy counts do not size the full CLC ticket grid.
            self.mixed_cga_config = NonClcMixedCgaConfig(
                preferred_cluster_shape=self.cluster_shape_mn,
                fallback_cluster_shape=fallback_cluster_shape_mn,
                launch_cluster_count=impl_desc.get("launch_cluster_count"),
                preferred_cluster_count=impl_desc.get("preferred_cluster_count"),
                fallback_cluster_count=impl_desc.get("fallback_cluster_count"),
            )
            self._work_id_worker = ClcMixedCgaSchedulerWorker(config=self.mixed_cga_config)
        self.raster_order = impl_desc.get("raster_order", "along_m")
        # Operand widths (bits) for the grouped raster's L2 budget; 16 when the problem does not name them.
        self.a_bits = getattr(problem_desc.get("a_dtype"), "width", 16)
        self.b_bits = getattr(problem_desc.get("b_dtype"), "width", 16)
        self.work_tile_type = WgradWorkTileInfo

        self._validate_configuration()
        self.mma_cta_count = 2 if self.use_2cta_instrs else 1
        self.cta_tile_shape_mnk = (
            self.mma_tiler_mnk[0] // self.mma_cta_count,
            self.mma_tiler_mnk[1],
            self.mma_tiler_mnk[2],
        )
        if self.is_mixed_cga and isinstance(self.intermediate, int) and isinstance(self.hidden, int):
            # Reject an oversized static ticket grid during construction/can_implement.
            self.get_grid_shape()
        self._plan_grouped_raster()

    def _plan_grouped_raster(self) -> None:
        """Static part of the "grouped" raster (MoK-style supergroups).

        Clusters are visited in bands of G clusters along the smaller output dimension (the "fast" one); a band is
        filled fast-dimension-first and sweeps the whole slow dimension (alternating direction band by band) before
        the next band starts, so the concurrently running tiles share G slices of the operand indexed by the fast
        dimension and those stay in L2 across the sweep.  G is chosen per expert at run time from its token count
        K: G_max = the most bands whose held slices, G x cluster extent x K elements, fit L2 /
        GROUPED_RASTER_L2_FRACTION (at most the fast extent, at least 1); G = the largest divisor of the fast extent
        not above G_max (bands of equal width), unless that divisor is below G_max / 2 (a fast extent with few
        divisors, e.g. a prime), then G = G_max with a narrower last band.  G equal to the fast extent is the plain
        raster along that dimension.
        """
        self.grouped_along_n = False
        self.grouped_clusters_m = self.grouped_clusters_n = self.grouped_fast = 0
        self.grouped_divisors = (1,)
        self.grouped_budget_rows = 0
        if self.raster_order != "grouped":
            return
        cluster_m, cluster_n = self.cluster_shape_mn
        extent_m = self.cta_tile_shape_mnk[0] * cluster_m
        extent_n = self.cta_tile_shape_mnk[1] * cluster_n
        self.grouped_clusters_m = (self.hidden + extent_m - 1) // extent_m
        self.grouped_clusters_n = (self.intermediate + extent_n - 1) // extent_n
        self.grouped_along_n = self.intermediate < self.hidden
        fast = self.grouped_clusters_n if self.grouped_along_n else self.grouped_clusters_m
        self.grouped_fast = fast
        self.grouped_divisors = tuple(d for d in range(1, fast + 1) if fast % d == 0)
        bits_per_k_row = (extent_n * self.b_bits) if self.grouped_along_n else (extent_m * self.a_bits)
        budget_bytes = _current_device_l2_bytes() // GROUPED_RASTER_L2_FRACTION
        self.grouped_budget_rows = budget_bytes * 8 // bits_per_k_row

    def _validate_configuration(self) -> None:
        if isinstance(self.expert_count, bool) or self.expert_count <= 0:
            raise ValueError("expert_count must be positive.")
        for field_name in ("intermediate", "hidden"):
            value = getattr(self, field_name)
            if isinstance(value, int) and (isinstance(value, bool) or value <= 0):
                raise ValueError(f"{field_name} must be positive.")
        if self.expert_count > self.grid_z_limit:
            raise ValueError(f"expert_count must not exceed grid Z limit {self.grid_z_limit}, got {self.expert_count}.")

        if len(self.mma_tiler_mnk) != 3:
            raise ValueError("mma_tiler_mnk must contain three dimensions.")
        if not all(isinstance(dimension, int) for dimension in self.mma_tiler_mnk):
            raise TypeError("mma_tiler_mnk dimensions must be Python ints.")
        if any(dimension <= 0 for dimension in self.mma_tiler_mnk):
            raise ValueError("mma_tiler_mnk dimensions must be positive.")

        if len(self.cluster_shape_mn) != 2:
            raise ValueError("cluster_shape_mn must contain two dimensions.")
        if not all(isinstance(dimension, int) for dimension in self.cluster_shape_mn):
            raise TypeError("cluster_shape_mn dimensions must be Python ints.")
        if any(dimension <= 0 for dimension in self.cluster_shape_mn):
            raise ValueError("cluster_shape_mn dimensions must be positive.")
        if any(dimension & (dimension - 1) for dimension in self.cluster_shape_mn):
            raise ValueError("cluster_shape_mn dimensions must be powers of two.")
        if self.cluster_shape_mn[0] * self.cluster_shape_mn[1] > 16:
            raise ValueError("A Blackwell CTA cluster may contain at most 16 CTAs.")

        mma_cta_count = 2 if self.use_2cta_instrs else 1
        if self.mma_tiler_mnk[0] % mma_cta_count != 0:
            raise ValueError("mma_tiler M must be divisible by the MMA CTA count.")
        if self.cluster_shape_mn[0] % mma_cta_count != 0:
            raise ValueError("cluster M must be divisible by the MMA CTA count.")

        if self.raster_order not in self.raster_orders:
            raise ValueError(f"raster_order must be one of {self.raster_orders}, got {self.raster_order!r}.")
        if self.is_mixed_cga and self.raster_order != "along_m":
            # The mixed-CGA CLC path decodes its own M-fastest ticket order; the remap applies to fixed launches only.
            raise ValueError(f"raster_order {self.raster_order!r} is not supported with a mixed-CGA launch.")
        if self.raster_order == "grouped" and not (isinstance(self.hidden, int) and isinstance(self.intermediate, int)):
            raise ValueError("raster_order 'grouped' requires static hidden / intermediate extents.")

    def get_grid_shape(self, *, max_active_clusters: Optional[int] = None, problem_desc=None) -> Tuple[int, int, int]:
        """Return all output work, not just the number of resident clusters.

        Mixed CLC uses one preferred-cluster output tile per Z ticket, as in
        FC12's shared CLC worker. Runtime dimensions are checked by CUDA launch
        validation; host-known ticket counts are checked here as well.
        """
        if problem_desc is None:
            if not isinstance(self.intermediate, int) or not isinstance(self.hidden, int):
                raise ValueError("Dynamic Wgrad dimensions require an actual expert shape.")
            expert_count = self.expert_count
            intermediate = self.intermediate
            hidden = self.hidden
        else:
            expert_count, intermediate, hidden = problem_desc
        cta_tile_m, cta_tile_n, _ = self.cta_tile_shape_mnk
        cluster_m, cluster_n = self.cluster_shape_mn
        cluster_tiles_m = (hidden + cta_tile_m * cluster_m - 1) // (cta_tile_m * cluster_m)
        cluster_tiles_n = (intermediate + cta_tile_n * cluster_n - 1) // (cta_tile_n * cluster_n)
        if self.is_mixed_cga:
            ticket_count = expert_count * cluster_tiles_m * cluster_tiles_n
            if isinstance(ticket_count, int) and ticket_count > self.grid_z_limit:
                raise ValueError(
                    f"Mixed-CGA CLC ticket count {ticket_count} exceeds grid Z limit {self.grid_z_limit}."
                )
            return cluster_m, cluster_n, ticket_count
        return cluster_tiles_m * cluster_m, cluster_tiles_n * cluster_n, expert_count

    def register_smem_regions(self, smem_workspace: SmemWorkspace) -> None:
        """Register common transport before the CLC regions."""
        super().register_smem_regions(smem_workspace)
        if self.is_mixed_cga:
            self._work_id_worker.register_smem_regions(smem_workspace)
        else:
            smem_workspace.register_mbarrier(self.clc_mbarriers_region, 2)
            smem_workspace.register_tensor(self.clc_response_region, cutlass.Int32, (4,), byte_alignment=16)

    def register_device_workspace(self, device_workspace: DeviceWorkspace) -> None:
        """Register scheduler workspace when required by the selected backend."""
        if self.is_mixed_cga:
            self._work_id_worker.register_device_workspace(device_workspace)

    @cute.jit
    def assign_device_members(
        self,
        *,
        offs: cute.Tensor,
        actual_expert_shape: Tuple,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        device_workspace: Optional[DeviceWorkspace] = None,
        is_fallback_cluster: Optional[Boolean] = None,
    ) -> None:
        """Create CTA-lifetime transport, CLC, and bootstrap state."""
        if cutlass.const_expr(not isinstance(self.intermediate, int)):
            self.intermediate = actual_expert_shape[1]
        if cutlass.const_expr(not isinstance(self.hidden, int)):
            self.hidden = actual_expert_shape[2]
        self.create_scheduler_pipelines(smem_workspace, smem_base)
        self.offs = offs
        self.current_work = make_wgrad_done_tile()
        if cutlass.const_expr(self.is_mixed_cga):
            self.cluster_tile_count_m = (
                self.hidden + Int32(self.cta_tile_shape_mnk[0] * self.cluster_shape_mn[0]) - Int32(1)
            ) // Int32(self.cta_tile_shape_mnk[0] * self.cluster_shape_mn[0])
            self.cluster_tile_count_n = (
                self.intermediate + Int32(self.cta_tile_shape_mnk[1] * self.cluster_shape_mn[1]) - Int32(1)
            ) // Int32(self.cta_tile_shape_mnk[1] * self.cluster_shape_mn[1])
            self._work_id_worker.assign_device_members(
                is_fallback_cluster=is_fallback_cluster,
                block_idx=block_idx,
                smem_workspace=smem_workspace,
                smem_base=smem_base,
                device_workspace=device_workspace,
            )
            return

        block_m, block_n, block_expert = block_idx
        cluster_m, cluster_n = self.cluster_shape_mn
        cta_coord_in_cluster = (Int32(block_m % cluster_m), Int32(block_n % cluster_n), Int32(0))
        is_leader_cta = (cta_coord_in_cluster[0] + cta_coord_in_cluster[1] + cta_coord_in_cluster[2]) == Int32(0)
        cluster_size = cluster_m * cluster_n
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=smem_workspace.ptr(self.clc_mbarriers_region, smem_base),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * cluster_size),
            tx_count=16,
            cta_layout_vmnk=cute.make_layout((1, cluster_m, cluster_n, 1)),
            defer_sync=True,
        )

        producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.ProducerConsumer, 1)
        consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        self._work_id_state = ClusterLaunchControlWorkIdState(
            response_pending=Boolean(True),
            grid_m=Int32(block_m),
            grid_n=Int32(block_n),
            grid_l=Int32(block_expert),
            response_is_valid=Boolean(True),
            cta_coord_in_cluster=cta_coord_in_cluster,
            cluster_pipeline=clc_pipeline,
            producer_state=producer_state,
            consumer_state=consumer_state,
            is_leader_cta=is_leader_cta,
            response_pointer=smem_workspace.ptr(self.clc_response_region, smem_base),
        )

    @cute.jit
    def gen_next_work(self) -> SchedulerWorkTileBase:
        """Consume bootstrap once, then fetch and map the selected backend."""
        if cutlass.const_expr(self.is_mixed_cga):
            work_id = self._work_id_worker.claim_next_work()
            cluster_tile_count_mn = self.cluster_tile_count_m * self.cluster_tile_count_n
            next_work = make_wgrad_done_tile()
            # A failed CLC claim returns the harmless ID 0. Validity must be
            # tested separately or it would replay the first output tile.
            if self._work_id_worker.claimed_is_valid & (work_id < cluster_tile_count_mn * Int32(self.expert_count)):
                cluster_tile_m = work_id % self.cluster_tile_count_m
                work_id_m_rest = work_id // self.cluster_tile_count_m
                cluster_tile_n = work_id_m_rest % self.cluster_tile_count_n
                expert_idx = work_id_m_rest // self.cluster_tile_count_n
                # Use the freshly claimed slot: fallback clusters are not
                # bound to a fixed sub-tile and tickets need not be monotonic.
                logical_cta = self._work_id_worker.cta_coord_in_preferred_cluster
                next_work = make_wgrad_work_tile(
                    offs=self.offs,
                    expert_idx=expert_idx,
                    tile_m_idx=cluster_tile_m * Int32(self.cluster_shape_mn[0]) + logical_cta[0],
                    tile_n_idx=cluster_tile_n * Int32(self.cluster_shape_mn[1]) + logical_cta[1],
                    cta_tile_k=self.cta_tile_shape_mnk[2],
                )
            self.current_work = next_work
            return self.current_work

        work_id, self._work_id_state = claim_work_id(self._work_id_state)
        next_work = make_wgrad_done_tile()
        if work_id.is_valid:
            tile_m_idx = work_id.grid_m
            tile_n_idx = work_id.grid_n
            if cutlass.const_expr(self.raster_order == "grouped"):
                # Band width G for this expert from its token count (see _plan_grouped_raster); then the M-fastest
                # launch index of this cluster, in cluster units, is remapped to the banded order.  The CTA's
                # offset inside its cluster is kept, so 2CTA pairs and multicast peers still share one tile.
                cluster_m, cluster_n = self.cluster_shape_mn
                clusters_m = self.grouped_clusters_m
                fast_extent = self.grouped_clusters_m
                slow_extent = self.grouped_clusters_n
                if cutlass.const_expr(self.grouped_along_n):
                    fast_extent = self.grouped_clusters_n
                    slow_extent = self.grouped_clusters_m
                _, expert_rows = compute_expert_token_range(self.offs, work_id.grid_l)
                group_max = Int32(self.grouped_budget_rows) // cutlass.max(expert_rows, Int32(1))
                group_max = cutlass.max(cutlass.min(group_max, Int32(fast_extent)), Int32(1))
                group = Int32(1)
                for divisor_idx in cutlass.range_constexpr(len(self.grouped_divisors)):
                    if Int32(self.grouped_divisors[divisor_idx]) <= group_max:
                        group = Int32(self.grouped_divisors[divisor_idx])
                if group * Int32(2) < group_max:  # few divisors: unequal bands beat much narrower ones
                    group = group_max
                launch_index = (work_id.grid_n // cluster_n) * Int32(clusters_m) + work_id.grid_m // cluster_m
                full_fast = (Int32(fast_extent) // group) * group
                band_width = group
                band = launch_index // (Int32(slow_extent) * group)
                band_start = band * Int32(slow_extent) * group
                if launch_index >= full_fast * Int32(slow_extent):  # the narrower last band
                    band_width = Int32(fast_extent) - full_fast
                    band = full_fast // group
                    band_start = full_fast * Int32(slow_extent)
                offset = launch_index - band_start
                slow = offset // band_width
                fast = band * group + (offset - slow * band_width)
                if band % 2 == 1:
                    slow = Int32(slow_extent - 1) - slow
                cluster_m_idx = fast
                cluster_n_idx = slow
                if cutlass.const_expr(self.grouped_along_n):
                    cluster_m_idx = slow
                    cluster_n_idx = fast
                tile_m_idx = cluster_m_idx * cluster_m + work_id.grid_m % cluster_m
                tile_n_idx = cluster_n_idx * cluster_n + work_id.grid_n % cluster_n
            elif cutlass.const_expr(self.raster_order == "along_n"):
                # Remap the M-fastest launch index of this cluster, in cluster units, to N-fastest; the CTA's offset
                # inside its cluster is kept, so 2CTA pairs and multicast peers still share one tile.
                cluster_m, cluster_n = self.cluster_shape_mn
                grid_dim = cute.arch.grid_dim()
                clusters_m = Int32(grid_dim[0]) // cluster_m
                clusters_n = Int32(grid_dim[1]) // cluster_n
                launch_index = (work_id.grid_n // cluster_n) * clusters_m + work_id.grid_m // cluster_m
                tile_m_idx = (launch_index // clusters_n) * cluster_m + work_id.grid_m % cluster_m
                tile_n_idx = (launch_index % clusters_n) * cluster_n + work_id.grid_n % cluster_n
            next_work = make_wgrad_work_tile(
                offs=self.offs,
                expert_idx=work_id.grid_l,
                tile_m_idx=tile_m_idx,
                tile_n_idx=tile_n_idx,
                cta_tile_k=self.cta_tile_shape_mnk[2],
            )
        else:
            next_work = make_wgrad_done_tile()
        self.current_work = next_work
        return self.current_work

    @cute.jit
    def produce_tail(self) -> None:
        """Drain work transport and the leader-owned CLC producer."""
        super().produce_tail()
        if cutlass.const_expr(self.is_mixed_cga):
            self._work_id_worker.produce_tail()
            return
        work_id_state = self._work_id_state
        if self._work_id_state.is_leader_cta:
            self._work_id_state.cluster_pipeline.producer_tail(self._work_id_state.producer_state)
        else:
            self._work_id_state = work_id_state

    def __extract_mlir_values__(self) -> list:
        values = super().__extract_mlir_values__()
        states = (self.offs, self.current_work, self._work_id_worker) if self.is_mixed_cga else (
            self.offs, self.current_work, self._work_id_state
        )
        for state in states:
            values.extend(extract_mlir_values(state))
        if self.is_mixed_cga:
            values.extend(extract_mlir_values(self.cluster_tile_count_m))
            values.extend(extract_mlir_values(self.cluster_tile_count_n))
        return values

    def __new_from_mlir_values__(self, values: list) -> "BlackwellMoeWgradScheduler":
        base_value_count = len(super().__extract_mlir_values__())
        if len(values) < base_value_count:
            raise ValueError(
                "BlackwellMoeWgradScheduler MLIR value count is smaller "
                f"than its base state: expected at least "
                f"{base_value_count}, got {len(values)}."
            )
        result = super().__new_from_mlir_values__(values[:base_value_count])
        value_index = base_value_count

        def rebuild(state):
            nonlocal value_index
            state_value_count = len(extract_mlir_values(state))
            rebuilt_state = new_from_mlir_values(state, values[value_index : value_index + state_value_count])
            value_index += state_value_count
            return rebuilt_state

        result.offs = rebuild(self.offs)
        result.current_work = rebuild(self.current_work)
        if self.is_mixed_cga:
            result._work_id_worker = rebuild(self._work_id_worker)
            result.cluster_tile_count_m = rebuild(self.cluster_tile_count_m)
            result.cluster_tile_count_n = rebuild(self.cluster_tile_count_n)
        else:
            result._work_id_state = rebuild(self._work_id_state)
        if value_index != len(values):
            raise ValueError(f"BlackwellMoeWgradScheduler consumed {value_index} MLIR values, got {len(values)}.")

        for field_name in (
            "expert_count",
            "intermediate",
            "hidden",
            "mma_tiler_mnk",
            "cluster_shape_mn",
            "use_2cta_instrs",
            "raster_order",
            "a_bits",
            "b_bits",
            "grouped_along_n",
            "grouped_clusters_m",
            "grouped_clusters_n",
            "grouped_fast",
            "grouped_divisors",
            "grouped_budget_rows",
            "mma_cta_count",
            "cta_tile_shape_mnk",
            "is_mixed_cga",
            "mixed_cga_config",
        ):
            setattr(result, field_name, getattr(self, field_name))
        return result

__all__ = ["BlackwellMoeWgradScheduler"]
