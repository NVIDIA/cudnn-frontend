# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Private per-lane state for graph-capable stateless MXFP8 training."""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist

from ..._contracts import ForwardConfig
from ..._math import round_up
from ..._types import MoeEpNativeWeightLayout
from .._comm import SymmetricMemoryProvider
from .._plan import PreparedResources
from .._runtime import (
    RuntimeHandle,
    RuntimeManager,
    _RuntimeWatchdog,
    _runtime_debug,
    get_runtime_manager,
)
from .._workspace import (
    BufferRegion,
    LocalMemoryProvider,
    WorkspaceOwner,
    WorkspaceRequirements,
    WorkspaceViews,
)
from ._adapter import _typed_view
from ._backward_compile import PreparedMxfp8BackwardKernel
from ._compile import PreparedMxfp8Kernel
from ._fingerprint import canonical_json_sha256, source_tree_sha256
from ._training_stage import Mxfp8TrainingStager

_DATA_DTYPE = torch.float8_e4m3fn
_SCALE_DTYPE = torch.float8_e8m0fnu

_ROUTING_SYMMETRIC = frozenset({"topk_weights"})
_ROUTING_LOCAL = frozenset({"topk_idx"})
_CALLER_OWNED_FORWARD_LOCAL = frozenset({"col_quant_data", "col_quant_sf"})
_FORWARD_PRIVATE_SYMMETRIC = frozenset({"output_data", *_ROUTING_SYMMETRIC})
_FORWARD_PRIVATE_LOCAL = frozenset({"overflow_flag", *_CALLER_OWNED_FORWARD_LOCAL, *_ROUTING_LOCAL})
_BACKWARD_PRIVATE_SYMMETRIC = frozenset({"output_data", "backward_dprob", *_ROUTING_SYMMETRIC})
_BACKWARD_PRIVATE_LOCAL = frozenset({"overflow_flag", "backward_aux_data", "backward_aux_scale", *_ROUTING_LOCAL})


def _lane_name(
    lane: int,
    phase: str,
    space: str,
    name: str,
) -> str:
    return f"lane.{lane}.{phase}.{space}.{name}"


def _lane_fallback_name(
    lane: int,
    space: str,
    name: str,
) -> str:
    return f"lane.{lane}.fallback.{space}.{name}"


def _clone_region(name: str, region: BufferRegion) -> BufferRegion:
    return BufferRegion(
        name=name,
        nbytes=region.nbytes,
        alignment=region.alignment,
    )


def _region_map(
    requirements: WorkspaceRequirements,
    space: str,
) -> dict[str, BufferRegion]:
    regions = requirements.symmetric_regions if space == "symmetric" else requirements.local_regions
    return {region.name: region for region in regions}


def _add_lane_regions(
    output: list[BufferRegion],
    requirements: WorkspaceRequirements,
    *,
    lane: int,
    phase: str,
    space: str,
    excluded_names: frozenset[str],
) -> None:
    regions = requirements.symmetric_regions if space == "symmetric" else requirements.local_regions
    for region in regions:
        if region.name in excluded_names:
            continue
        output.append(
            _clone_region(
                _lane_name(lane, phase, space, region.name),
                region,
            )
        )


def build_training_workspace_requirements(
    config: ForwardConfig,
    forward: PreparedMxfp8Kernel,
    backward: PreparedMxfp8BackwardKernel,
    *,
    lane_count: int,
) -> WorkspaceRequirements:
    """Build one deterministic root layout for private execution lanes."""

    if isinstance(lane_count, bool) or not isinstance(lane_count, int) or lane_count <= 0:
        raise ValueError(f"lane_count must be a positive integer, got {lane_count!r}")
    if not config.generate_c:
        raise ValueError("training preparation requires generate_c=True")
    if forward.pool_token_capacity != backward.pool_token_capacity:
        raise ValueError("forward/backward pool capacities must match, got " f"{forward.pool_token_capacity} and " f"{backward.pool_token_capacity}")

    forward_requirements = forward.workspace_requirements
    backward_requirements = backward.workspace_requirements
    symmetric_regions: list[BufferRegion] = []
    local_regions: list[BufferRegion] = []

    for lane in range(lane_count):
        local_regions.extend(
            (
                BufferRegion(
                    _lane_name(
                        lane,
                        "finalizer",
                        "local",
                        "global_overflow",
                    ),
                    torch.int32.itemsize,
                    16,
                ),
                BufferRegion(
                    _lane_name(
                        lane,
                        "finalizer",
                        "local",
                        "overflow_ok",
                    ),
                    torch.bool.itemsize,
                    16,
                ),
            )
        )
        _add_lane_regions(
            symmetric_regions,
            forward_requirements,
            lane=lane,
            phase="forward",
            space="symmetric",
            excluded_names=_FORWARD_PRIVATE_SYMMETRIC,
        )
        _add_lane_regions(
            local_regions,
            forward_requirements,
            lane=lane,
            phase="forward",
            space="local",
            excluded_names=_FORWARD_PRIVATE_LOCAL,
        )
        _add_lane_regions(
            symmetric_regions,
            backward_requirements,
            lane=lane,
            phase="backward",
            space="symmetric",
            excluded_names=_BACKWARD_PRIVATE_SYMMETRIC,
        )
        _add_lane_regions(
            local_regions,
            backward_requirements,
            lane=lane,
            phase="backward",
            space="local",
            excluded_names=_BACKWARD_PRIVATE_LOCAL,
        )

    forward_symmetric = _region_map(forward_requirements, "symmetric")
    forward_local = _region_map(forward_requirements, "local")
    backward_symmetric = _region_map(backward_requirements, "symmetric")
    backward_local = _region_map(backward_requirements, "local")
    fc1_c_shape = tuple(int(extent) for extent in forward.kernel.get_aux_output_shapes()["fc1_c"])
    backward_fc1_preact_shape = tuple(int(extent) for extent in backward.kernel.get_fc1_preact_shape())
    if fc1_c_shape != backward_fc1_preact_shape:
        raise ValueError("forward fc1_c and backward fc1_preact shapes differ: " f"{fc1_c_shape} != {backward_fc1_preact_shape}")
    for lane in range(lane_count):
        for name in sorted(_FORWARD_PRIVATE_SYMMETRIC):
            if name in _ROUTING_SYMMETRIC:
                continue
            symmetric_regions.append(
                _clone_region(
                    _lane_name(lane, "forward", "symmetric", name),
                    forward_symmetric[name],
                )
            )
        for name in sorted(_BACKWARD_PRIVATE_SYMMETRIC):
            if name in _ROUTING_SYMMETRIC:
                continue
            symmetric_regions.append(
                _clone_region(
                    _lane_name(lane, "backward", "symmetric", name),
                    backward_symmetric[name],
                )
            )
        for name in sorted(_FORWARD_PRIVATE_LOCAL):
            if name in _ROUTING_LOCAL or name in _CALLER_OWNED_FORWARD_LOCAL:
                continue
            region = forward_local.get(name)
            if region is not None:
                local_regions.append(
                    _clone_region(
                        _lane_name(lane, "forward", "local", name),
                        region,
                    )
                )
        for name in sorted(_BACKWARD_PRIVATE_LOCAL):
            if name in _ROUTING_LOCAL:
                continue
            local_regions.append(
                _clone_region(
                    _lane_name(lane, "backward", "local", name),
                    backward_local[name],
                )
            )
        local_regions.append(
            BufferRegion(
                _lane_fallback_name(lane, "local", "routing_topk_idx"),
                int(config.max_tokens_per_rank) * config.top_k * torch.int32.itemsize,
                alignment=16,
            )
        )
        symmetric_regions.append(
            BufferRegion(
                _lane_fallback_name(lane, "symmetric", "routing_topk_weights"),
                int(config.max_tokens_per_rank) * config.top_k * torch.float32.itemsize,
                alignment=16,
            )
        )
    return WorkspaceRequirements(
        max_tokens_per_rank=int(config.max_tokens_per_rank),
        symmetric_regions=tuple(symmetric_regions),
        local_regions=tuple(local_regions),
    )


def _harmonize_symmetric_regions(
    requirements: WorkspaceRequirements,
    runtime: RuntimeHandle,
    device: torch.device,
) -> WorkspaceRequirements:
    """Make every peer-visible region size and offset identical on all ranks."""

    if runtime.world_size <= 1:
        return requirements

    regions = requirements.symmetric_regions
    count = torch.tensor([len(regions)], dtype=torch.int64, device=device)
    minimum_count = count.clone()
    maximum_count = count.clone()
    dist.all_reduce(minimum_count, op=dist.ReduceOp.MIN, group=runtime.group)
    dist.all_reduce(maximum_count, op=dist.ReduceOp.MAX, group=runtime.group)
    if int(minimum_count.item()) != int(maximum_count.item()):
        raise RuntimeError("symmetric workspace region counts differ across EP ranks: " f"min={int(minimum_count.item())}, max={int(maximum_count.item())}")

    metadata = "\0".join(f"{region.name}:{region.alignment}" for region in regions).encode()
    signature_value = int.from_bytes(
        hashlib.blake2b(metadata, digest_size=8).digest(),
        "little",
    ) & ((1 << 63) - 1)
    signature = torch.tensor(
        [signature_value],
        dtype=torch.int64,
        device=device,
    )
    minimum_signature = signature.clone()
    maximum_signature = signature.clone()
    dist.all_reduce(
        minimum_signature,
        op=dist.ReduceOp.MIN,
        group=runtime.group,
    )
    dist.all_reduce(
        maximum_signature,
        op=dist.ReduceOp.MAX,
        group=runtime.group,
    )
    if int(minimum_signature.item()) != int(maximum_signature.item()):
        raise RuntimeError(
            "symmetric workspace region names, order, or alignments differ "
            "across EP ranks: "
            f"local_signature={signature_value}, "
            "local_regions="
            f"{tuple((region.name, region.alignment) for region in regions)}"
        )

    local_sizes = torch.tensor(
        [region.nbytes for region in regions],
        dtype=torch.int64,
        device=device,
    )
    maximum_sizes = local_sizes.clone()
    dist.all_reduce(maximum_sizes, op=dist.ReduceOp.MAX, group=runtime.group)
    harmonized_sizes = tuple(int(value) for value in maximum_sizes.cpu().tolist())
    changes = tuple(
        f"{region.name}:{region.nbytes}->{harmonized_size}" for region, harmonized_size in zip(regions, harmonized_sizes) if region.nbytes != harmonized_size
    )
    _runtime_debug(
        "training-state.symmetric-layout-harmonized",
        region_count=len(regions),
        changed_regions=changes,
    )
    if not changes:
        return requirements

    return WorkspaceRequirements(
        max_tokens_per_rank=requirements.max_tokens_per_rank,
        symmetric_regions=tuple(
            BufferRegion(
                region.name,
                harmonized_size,
                alignment=region.alignment,
            )
            for region, harmonized_size in zip(regions, harmonized_sizes)
        ),
        local_regions=requirements.local_regions,
    )


def _workspace_abi(requirements: WorkspaceRequirements) -> dict[str, object]:
    def regions(values) -> list[dict[str, object]]:
        return [
            {
                "name": region.name,
                "nbytes": int(region.nbytes),
                "alignment": int(region.alignment),
            }
            for region in values
        ]

    return {
        "max_tokens_per_rank": requirements.max_tokens_per_rank,
        "symmetric_regions": regions(requirements.symmetric_regions),
        "local_regions": regions(requirements.local_regions),
    }


def _prepared_kernel_abi(prepared) -> dict[str, object]:
    kernel = prepared.kernel
    return {
        "name": str(kernel.name()),
        "architecture": list(prepared.architecture),
        "effective_config": prepared.config.effective_config(prepared.launch_cluster_count),
        "launch": {
            "cluster_count": int(prepared.launch_cluster_count),
            "threads_per_cta": int(kernel.threads_per_cta),
            "occupancy": int(kernel.occupancy),
            "smem_capacity": int(kernel.smem_capacity),
        },
        "workspace": _workspace_abi(prepared.workspace_requirements),
        "pool_token_capacity": int(prepared.pool_token_capacity),
    }


def _build_training_abi_facts(
    config: ForwardConfig,
    forward: PreparedMxfp8Kernel,
    backward: PreparedMxfp8BackwardKernel,
    requirements: WorkspaceRequirements,
    *,
    lane_count: int,
    source_tree_digest: str | None = None,
) -> dict[str, object]:
    """Return rank-independent JSON-safe facts for the stateless training ABI."""

    if source_tree_digest is None:
        source_root = Path(__file__).resolve().parents[1] / "cutedsl_src"
        source_tree_digest = source_tree_sha256(source_root)
    return {
        "schema_version": 2,
        "source_tree_sha256": source_tree_digest,
        "ep": {
            "size": int(config.ep_size),
            "global_ranks": list(config.ep_global_ranks),
        },
        "geometry": {
            "num_experts": int(config.num_experts),
            "experts_per_rank": int(config.experts_per_rank),
            "hidden": int(config.hidden_size),
            "intermediate": int(config.intermediate_size),
            "top_k": int(config.top_k),
            "max_tokens_per_rank": int(config.max_tokens_per_rank),
            "max_recv_size_per_rank": int(forward.config.max_recv_size_per_rank),
        },
        "policy": {
            "drop_on_overflow": bool(config.drop_on_overflow),
            "combine_format": config.combine_format,
            "output_format": config.output_format,
            "apply_topk_in_fc1": bool(config.apply_topk_in_fc1),
            "fc1_weight_layout": config.fc1_weight_layout.value,
            "gate_up_clamp": config.gate_up_clamp,
        },
        "resources": {
            "lane_count": int(lane_count),
            "workspace": _workspace_abi(requirements),
        },
        "native_weight_layouts": [layout.value for layout in MoeEpNativeWeightLayout],
        "forward_kernel": _prepared_kernel_abi(forward),
        "backward_kernel": _prepared_kernel_abi(backward),
    }


def _verify_training_abi_across_ranks(
    facts: dict[str, object],
    runtime: RuntimeHandle,
    device: torch.device,
) -> str:
    """Collectively reject rank-divergent training ABI before allocation."""

    digest = canonical_json_sha256(facts)
    if runtime.world_size <= 1:
        return digest
    digest_value = int(digest[:16], 16) & ((1 << 63) - 1)
    minimum = torch.tensor([digest_value], dtype=torch.int64, device=device)
    maximum = minimum.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=runtime.group)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=runtime.group)
    if int(minimum.item()) == int(maximum.item()):
        return digest

    rank_digests: list[Any] = [None] * runtime.world_size
    dist.all_gather_object(rank_digests, digest, group=runtime.group)
    raise RuntimeError(
        "MoeEp training ABI differs across expert-parallel ranks before " "workspace allocation: " f"digests={rank_digests}, local_facts={facts}"
    )


@dataclass(frozen=True)
class Mxfp8TrainingLaneScratch:
    """Private fixed-capacity transport and routing tensors for one lane."""

    index: int
    routing_topk_idx: torch.Tensor
    routing_topk_weights: torch.Tensor
    forward_output: torch.Tensor
    backward_output: torch.Tensor
    dprob: torch.Tensor
    forward_overflow: torch.Tensor
    backward_overflow: torch.Tensor


@dataclass(frozen=True)
class Mxfp8TrainingExecutionViews:
    """Prepared workspaces and private scratch for one execution lane."""

    scratch: Mxfp8TrainingLaneScratch
    forward: PreparedResources
    backward: PreparedResources
    forward_expert_size_snapshot: torch.Tensor


class Mxfp8TrainingState:
    """Own only private runtime and per-lane training scratch."""

    def __init__(
        self,
        config: ForwardConfig,
        device: torch.device,
        forward: PreparedMxfp8Kernel,
        backward: PreparedMxfp8BackwardKernel,
        *,
        lane_count: int,
        runtime_manager: Optional[RuntimeManager] = None,
        symmetric_provider: Optional[SymmetricMemoryProvider] = None,
        local_provider: Optional[LocalMemoryProvider] = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.forward_prepared = forward
        self.backward_prepared = backward
        self.stager = Mxfp8TrainingStager(config.hidden_size, config.top_k)
        self.beta = torch.ones(
            (config.experts_per_rank,),
            dtype=torch.float32,
            device=self.device,
        )
        self.lane_count = lane_count
        self.requirements = build_training_workspace_requirements(
            config,
            forward,
            backward,
            lane_count=lane_count,
        )
        self._runtime_manager = runtime_manager or get_runtime_manager()
        self._symmetric_provider = symmetric_provider
        self._local_provider = local_provider
        self._runtime: RuntimeHandle | None = None
        self._workspace: WorkspaceOwner | None = None
        self._closed = False
        self._lock = threading.RLock()

    def prepare(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("private training state is closed")
            if self._runtime is not None and self._workspace is not None and self._workspace.allocated:
                return
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("private training state must be prepared before CUDA graph capture")
            _runtime_debug(
                "training-state.prepare.begin",
                lane_count=self.lane_count,
                local_bytes=sum(region.nbytes for region in self.requirements.local_regions),
                symmetric_bytes=sum(region.nbytes for region in self.requirements.symmetric_regions),
            )
            _runtime_debug("training-state.runtime-acquire.begin")
            runtime = self._runtime_manager.acquire(self.config, self.device)
            _runtime_debug(
                "training-state.runtime-acquire.end",
                runtime_ref_count=self._runtime_manager.ref_count,
            )
            self._runtime = runtime
            try:
                layout_watchdog = _RuntimeWatchdog("training-state.symmetric-layout-harmonize")
                layout_watchdog.start()
                _runtime_debug("training-state.symmetric-layout-harmonize.begin")
                try:
                    self.requirements = _harmonize_symmetric_regions(
                        self.requirements,
                        runtime,
                        self.device,
                    )
                finally:
                    layout_watchdog.close()
                    _runtime_debug("training-state.symmetric-layout-harmonize.end")
                if runtime.world_size > 1:
                    abi_watchdog = _RuntimeWatchdog("training-state.abi-handshake")
                    abi_watchdog.start()
                    _runtime_debug("training-state.abi-handshake.begin")
                    try:
                        abi_facts = _build_training_abi_facts(
                            self.config,
                            self.forward_prepared,
                            self.backward_prepared,
                            self.requirements,
                            lane_count=self.lane_count,
                        )
                        abi_fingerprint = _verify_training_abi_across_ranks(
                            abi_facts,
                            runtime,
                            self.device,
                        )
                    finally:
                        abi_watchdog.close()
                    _runtime_debug(
                        "training-state.abi-handshake.end",
                        fingerprint=abi_fingerprint,
                    )
                _runtime_debug("training-state.workspace-create.begin")
                workspace = WorkspaceOwner(
                    self.requirements,
                    runtime,
                    symmetric_provider=self._symmetric_provider,
                    local_provider=self._local_provider,
                )
                _runtime_debug(
                    "training-state.workspace-create.end",
                    local_bytes=workspace.local_layout.total_bytes,
                    symmetric_bytes=workspace.symmetric_layout.total_bytes,
                )
                self._workspace = workspace
                allocation_watchdog = _RuntimeWatchdog("training-state.workspace-allocate")
                allocation_watchdog.start()
                try:
                    workspace.ensure_allocated()
                finally:
                    allocation_watchdog.close()
                _runtime_debug("training-state.workspace-allocate.end")
                if runtime.world_size > 1:
                    # Symmetric-root zeroing is asynchronous. No rank may
                    # enter the first device barrier until every peer has
                    # completed allocation and root initialization.
                    stream_watchdog = _RuntimeWatchdog("training-state.stream-synchronize")
                    stream_watchdog.start()
                    _runtime_debug("training-state.stream-synchronize.begin")
                    try:
                        torch.cuda.current_stream(self.device).synchronize()
                    finally:
                        stream_watchdog.close()
                    _runtime_debug("training-state.stream-synchronize.end")

                    barrier_watchdog = _RuntimeWatchdog("training-state.rank-barrier")
                    barrier_watchdog.start()
                    _runtime_debug("training-state.rank-barrier.begin")
                    try:
                        dist.barrier(group=runtime.group)
                    finally:
                        barrier_watchdog.close()
                    _runtime_debug("training-state.rank-barrier.end")
            except Exception:
                if self._workspace is not None:
                    self._workspace.close()
                    self._workspace = None
                runtime.close()
                self._runtime = None
                raise
            _runtime_debug("training-state.prepare.end")

    def _flat_views(self, token_count: int) -> WorkspaceViews:
        self.prepare()
        assert self._workspace is not None
        return self._workspace.views(token_count)

    @staticmethod
    def _phase_workspace(
        flat: WorkspaceViews,
        requirements: WorkspaceRequirements,
        *,
        lane: int,
        phase: str,
    ) -> WorkspaceViews:
        symmetric = {}
        local = {}
        for region in requirements.symmetric_regions:
            if region.name in _ROUTING_SYMMETRIC:
                symmetric[region.name] = flat.symmetric[_lane_fallback_name(lane, "symmetric", "routing_topk_weights")]
                continue
            symmetric[region.name] = flat.symmetric[_lane_name(lane, phase, "symmetric", region.name)]
        for region in requirements.local_regions:
            if phase == "forward" and region.name in _CALLER_OWNED_FORWARD_LOCAL:
                continue
            if region.name in _ROUTING_LOCAL:
                local[region.name] = flat.local[_lane_fallback_name(lane, "local", "routing_topk_idx")]
                continue
            local[region.name] = flat.local[_lane_name(lane, phase, "local", region.name)]
        return WorkspaceViews(
            token_count=flat.token_count,
            symmetric=MappingProxyType(symmetric),
            local=MappingProxyType(local),
            peer_mapping=flat.peer_mapping,
        )

    def _lane_scratch_views(
        self,
        flat: WorkspaceViews,
        lane: int,
    ) -> Mxfp8TrainingLaneScratch:
        config = self.config
        capacity = int(config.max_tokens_per_rank)
        bwd_shapes = {name: tuple(int(extent) for extent in shape) for name, shape in self.backward_prepared.kernel.get_aux_output_shapes().items()}

        def local_bytes(name: str) -> torch.Tensor:
            return flat.local[_lane_fallback_name(lane, "local", name)]

        return Mxfp8TrainingLaneScratch(
            index=lane,
            routing_topk_idx=_typed_view(
                local_bytes("routing_topk_idx"),
                torch.int32,
                (capacity, config.top_k),
            ),
            routing_topk_weights=_typed_view(
                flat.symmetric[
                    _lane_fallback_name(
                        lane,
                        "symmetric",
                        "routing_topk_weights",
                    )
                ],
                torch.float32,
                (capacity, config.top_k),
            ),
            forward_output=_typed_view(
                flat.symmetric[
                    _lane_name(
                        lane,
                        "forward",
                        "symmetric",
                        "output_data",
                    )
                ],
                torch.bfloat16,
                (capacity, config.hidden_size),
            ),
            backward_output=_typed_view(
                flat.symmetric[
                    _lane_name(
                        lane,
                        "backward",
                        "symmetric",
                        "output_data",
                    )
                ],
                torch.bfloat16,
                (capacity, config.hidden_size),
            ),
            dprob=_typed_view(
                flat.symmetric[
                    _lane_name(
                        lane,
                        "backward",
                        "symmetric",
                        "backward_dprob",
                    )
                ],
                torch.float32,
                bwd_shapes["dprob"],
            ),
            forward_overflow=_typed_view(
                flat.local[
                    _lane_name(
                        lane,
                        "forward",
                        "local",
                        "overflow_flag",
                    )
                ],
                torch.int32,
                (1,),
            ),
            backward_overflow=_typed_view(
                flat.local[
                    _lane_name(
                        lane,
                        "backward",
                        "local",
                        "overflow_flag",
                    )
                ],
                torch.int32,
                (1,),
            ),
        )

    def public_requirements(
        self,
    ) -> Mapping[
        str,
        tuple[tuple[int, ...], tuple[int, ...], torch.dtype, int],
    ]:
        """Return exact caller-owned output contracts without buffer objects."""

        config = self.config
        capacity = int(config.max_tokens_per_rank)
        pool_rows = int(self.forward_prepared.pool_token_capacity)
        forward_shapes = {name: tuple(int(extent) for extent in shape) for name, shape in self.forward_prepared.kernel.get_aux_output_shapes().items()}
        backward_shapes = {name: tuple(int(extent) for extent in shape) for name, shape in self.backward_prepared.kernel.get_aux_output_shapes().items()}
        fc1_sfa_rows = round_up(config.hidden_size, 128)
        fc1_sfa_elements = math.prod(forward_shapes["col_quant_sf"])
        if fc1_sfa_elements % fc1_sfa_rows:
            raise ValueError("forward fc1_sfa producer size is not atom aligned")
        fc2_sfb_rows = round_up(config.hidden_size, 128)
        fc2_sfb_elements = math.prod(backward_shapes["grad_y2_sf"])
        if fc2_sfb_elements % fc2_sfb_rows:
            raise ValueError("backward fc2_sfb producer size is not atom aligned")
        requirements = {
            "output": (
                (capacity, config.hidden_size),
                (config.hidden_size, 1),
                torch.bfloat16,
                16,
            ),
            "fc1_preact": (
                forward_shapes["fc1_c"],
                (forward_shapes["fc1_c"][1], 1),
                torch.bfloat16,
                128,
            ),
            "fc1_a": (
                (config.hidden_size, pool_rows),
                (pool_rows, 1),
                _DATA_DTYPE,
                128,
            ),
            "fc1_sfa": (
                (fc1_sfa_rows, fc1_sfa_elements // fc1_sfa_rows),
                (fc1_sfa_elements // fc1_sfa_rows, 1),
                _SCALE_DTYPE,
                128,
            ),
            "valid_route_counts": (
                (config.experts_per_rank,),
                (1,),
                torch.int32,
                16,
            ),
            "expert_offsets": (
                (config.experts_per_rank,),
                (1,),
                torch.int32,
                16,
            ),
            "grad_activation": (
                (capacity, config.hidden_size),
                (config.hidden_size, 1),
                torch.float32,
                16,
            ),
            "dprob": (
                backward_shapes["dprob"],
                (backward_shapes["dprob"][1], 1),
                torch.float32,
                16,
            ),
            "fc1_b": (
                backward_shapes["fc1_col_output"],
                (2 * config.intermediate_size, 1),
                _DATA_DTYPE,
                128,
            ),
            "fc1_sfb": (
                backward_shapes["fc1_col_output_sf"],
                (backward_shapes["fc1_col_output_sf"][1], 1),
                _SCALE_DTYPE,
                128,
            ),
            "fc2_a": (
                (config.intermediate_size, pool_rows),
                (1, config.intermediate_size),
                _DATA_DTYPE,
                128,
            ),
            "fc2_sfa": (
                backward_shapes["fc1_recompute_sf"],
                (backward_shapes["fc1_recompute_sf"][1], 1),
                _SCALE_DTYPE,
                128,
            ),
            "fc2_b": (
                backward_shapes["grad_y2"],
                (1, pool_rows),
                _DATA_DTYPE,
                128,
            ),
            "fc2_sfb": (
                (fc2_sfb_rows, fc2_sfb_elements // fc2_sfb_rows),
                (fc2_sfb_elements // fc2_sfb_rows, 1),
                _SCALE_DTYPE,
                128,
            ),
        }
        return MappingProxyType(requirements)

    def views(
        self,
        *,
        lane: int,
        token_count: int,
    ) -> Mxfp8TrainingExecutionViews:
        with self._lock:
            if lane < 0 or lane >= self.lane_count:
                raise ValueError(f"lane {lane} is outside [0, {self.lane_count})")
            col_quant_sizes_offset = self.forward_prepared.col_quant_sizes_offset
            if col_quant_sizes_offset is None:
                raise RuntimeError("training preparation requires a persistent col-quant expert-size snapshot")
            flat = self._flat_views(token_count)
            forward_workspace = self._phase_workspace(
                flat,
                self.forward_prepared.workspace_requirements,
                lane=lane,
                phase="forward",
            )
            backward_workspace = self._phase_workspace(
                flat,
                self.backward_prepared.workspace_requirements,
                lane=lane,
                phase="backward",
            )
            snapshot_bytes = forward_workspace.local["kernel_local_workspace"].narrow(
                0,
                col_quant_sizes_offset,
                self.forward_prepared.col_quant_sizes_bytes,
            )
            snapshot = _typed_view(
                snapshot_bytes,
                torch.int32,
                (self.config.experts_per_rank,),
            )
            assert self._runtime is not None
            return Mxfp8TrainingExecutionViews(
                scratch=self._lane_scratch_views(flat, lane),
                forward=PreparedResources(
                    runtime=self._runtime,
                    workspace=forward_workspace,
                ),
                backward=PreparedResources(
                    runtime=self._runtime,
                    workspace=backward_workspace,
                ),
                forward_expert_size_snapshot=snapshot,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._workspace is not None:
                self._workspace.close()
                self._workspace = None
            if self._runtime is not None:
                self._runtime.close()
                self._runtime = None
            self._closed = True

    def apply_overflow(
        self,
        *,
        lane: int,
        phase: str,
    ) -> torch.Tensor:
        """Apply the configured policy to one phase's private overflow flag."""

        if lane < 0 or lane >= self.lane_count:
            raise ValueError(f"lane {lane} is outside [0, {self.lane_count})")
        if phase not in ("forward", "backward"):
            raise ValueError(f"phase must be 'forward' or 'backward', got {phase!r}")
        flat = self._flat_views(0)
        global_overflow = _typed_view(
            flat.local[
                _lane_name(
                    lane,
                    "finalizer",
                    "local",
                    "global_overflow",
                )
            ],
            torch.int32,
            (1,),
        )
        flag = _typed_view(
            flat.local[
                _lane_name(
                    lane,
                    phase,
                    "local",
                    "overflow_flag",
                )
            ],
            torch.int32,
            (1,),
        )
        global_overflow.copy_(flag)
        assert self._runtime is not None
        if self._runtime.world_size > 1:
            dist.all_reduce(
                global_overflow,
                op=dist.ReduceOp.MAX,
                group=self._runtime.group,
            )
        if not self.config.drop_on_overflow:
            assert_async = getattr(torch, "_assert_async", None)
            if assert_async is None:
                raise RuntimeError("drop_on_overflow=False training requires torch._assert_async")
            overflow_ok = _typed_view(
                flat.local[
                    _lane_name(
                        lane,
                        "finalizer",
                        "local",
                        "overflow_ok",
                    )
                ],
                torch.bool,
                (1,),
            )
            torch.eq(global_overflow, 0, out=overflow_ok)
            assert_async(
                overflow_ok,
                f"Rubin MegaMoE receive route-pool overflow; the {phase} " "outputs are invalid",
            )
        return global_overflow


__all__ = [
    "Mxfp8TrainingExecutionViews",
    "Mxfp8TrainingState",
]
