# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed-capacity slot/lane resources for graph-capable MXFP8 training."""

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
from ..._types import MoeEpTrainingWeights
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
from ._adapter import _typed_k_major_view, _typed_view
from ._backward_compile import PreparedMxfp8BackwardKernel
from ._compile import PreparedMxfp8Kernel
from ._fingerprint import canonical_json_sha256, source_tree_sha256
from ._training_stage import Mxfp8TrainingStager
from ._training_weights import Mxfp8TrainingWeightBindings
from ._training_wgrad import Mxfp8TrainingWgradExporter

_DATA_DTYPE = torch.float8_e4m3fn
_SCALE_DTYPE = torch.float8_e8m0fnu

_ROUTING_SYMMETRIC = frozenset({"topk_weights"})
_ROUTING_LOCAL = frozenset({"topk_idx"})
_FORWARD_SLOT_SYMMETRIC = frozenset({"output_data", *_ROUTING_SYMMETRIC})
_FORWARD_SLOT_LOCAL = frozenset({"overflow_flag", "col_quant_data", "col_quant_sf", *_ROUTING_LOCAL})
_BACKWARD_SLOT_SYMMETRIC = frozenset({"output_data", "backward_dprob", *_ROUTING_SYMMETRIC})
_BACKWARD_SLOT_LOCAL = frozenset({"overflow_flag", "backward_aux_data", "backward_aux_scale", *_ROUTING_LOCAL})


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _align_scale_columns(token_capacity: int) -> int:
    return _round_up((token_capacity + 31) // 32, 4)


def _lane_name(
    lane: int,
    phase: str,
    space: str,
    name: str,
) -> str:
    return f"lane.{lane}.{phase}.{space}.{name}"


def _slot_name(
    slot: int,
    phase: str,
    space: str,
    name: str,
) -> str:
    return f"slot.{slot}.{phase}.{space}.{name}"


def _custom_slot_name(slot: int, name: str) -> str:
    return f"slot.{slot}.persistent.local.{name}"


def _custom_slot_symmetric_name(slot: int, name: str) -> str:
    return f"slot.{slot}.persistent.symmetric.{name}"


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


def _required_region(
    requirements: WorkspaceRequirements,
    space: str,
    name: str,
) -> BufferRegion:
    try:
        return _region_map(requirements, space)[name]
    except KeyError as exc:
        raise ValueError(f"{space} workspace requirements do not contain {name!r}") from exc


def _add_lane_regions(
    output: list[BufferRegion],
    requirements: WorkspaceRequirements,
    *,
    lane: int,
    phase: str,
    space: str,
    slot_names: frozenset[str],
) -> None:
    regions = requirements.symmetric_regions if space == "symmetric" else requirements.local_regions
    for region in regions:
        if region.name in slot_names:
            continue
        if phase == "backward" and space == "local" and region.name == ("backward_fc1_preact"):
            # The graph path aliases forward's raw receiver pool directly.
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
    slot_count: int,
    lane_count: int,
) -> WorkspaceRequirements:
    """Build one deterministic root layout for N slots and M lanes."""

    for name, value in (("slot_count", slot_count), ("lane_count", lane_count)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if not config.generate_c:
        raise ValueError("training resources require generate_c=True")
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
            slot_names=_FORWARD_SLOT_SYMMETRIC,
        )
        _add_lane_regions(
            local_regions,
            forward_requirements,
            lane=lane,
            phase="forward",
            space="local",
            slot_names=_FORWARD_SLOT_LOCAL,
        )
        _add_lane_regions(
            symmetric_regions,
            backward_requirements,
            lane=lane,
            phase="backward",
            space="symmetric",
            slot_names=_BACKWARD_SLOT_SYMMETRIC,
        )
        _add_lane_regions(
            local_regions,
            backward_requirements,
            lane=lane,
            phase="backward",
            space="local",
            slot_names=_BACKWARD_SLOT_LOCAL,
        )

    forward_symmetric = _region_map(forward_requirements, "symmetric")
    forward_local = _region_map(forward_requirements, "local")
    backward_symmetric = _region_map(backward_requirements, "symmetric")
    backward_local = _region_map(backward_requirements, "local")
    fc1_c_shape = tuple(int(extent) for extent in forward.kernel.get_aux_output_shapes()["fc1_c"])
    fc1_c_bytes = math.prod(fc1_c_shape) * torch.bfloat16.itemsize
    backward_preact = _required_region(
        backward_requirements,
        "local",
        "backward_fc1_preact",
    )
    if fc1_c_bytes != backward_preact.nbytes:
        raise ValueError("forward fc1_c and backward preactivation byte sizes differ: " f"{fc1_c_bytes} != {backward_preact.nbytes}")
    aux_shapes = {name: tuple(int(extent) for extent in shape) for name, shape in backward.kernel.get_aux_output_shapes().items()}
    aux_dtypes = {
        "fc1_recompute": _DATA_DTYPE,
        "fc1_recompute_sf": _SCALE_DTYPE,
        "fc1_col_output": _DATA_DTYPE,
        "fc1_col_output_sf": _SCALE_DTYPE,
        "grad_y2": _DATA_DTYPE,
        "grad_y2_sf": torch.uint8,
    }
    scale_columns = _align_scale_columns(forward.pool_token_capacity)
    wgrad_shapes = {
        "wgrad_fc1_sfa": (
            _round_up(config.hidden_size, 128),
            scale_columns,
        ),
        "wgrad_fc1_sfb": (
            _round_up(2 * config.intermediate_size, 128),
            scale_columns,
        ),
        "wgrad_fc2_sfa": (
            _round_up(config.intermediate_size, 128),
            scale_columns,
        ),
        "wgrad_fc2_sfb": (
            _round_up(config.hidden_size, 128),
            scale_columns,
        ),
    }

    for slot in range(slot_count):
        for name in sorted(_FORWARD_SLOT_SYMMETRIC):
            if name in _ROUTING_SYMMETRIC:
                continue
            symmetric_regions.append(
                _clone_region(
                    _slot_name(slot, "forward", "symmetric", name),
                    forward_symmetric[name],
                )
            )
        for name in sorted(_BACKWARD_SLOT_SYMMETRIC):
            if name in _ROUTING_SYMMETRIC:
                continue
            symmetric_regions.append(
                _clone_region(
                    _slot_name(slot, "backward", "symmetric", name),
                    backward_symmetric[name],
                )
            )
        for name in sorted(_FORWARD_SLOT_LOCAL):
            if name in _ROUTING_LOCAL:
                continue
            region = forward_local.get(name)
            if region is not None:
                local_regions.append(
                    _clone_region(
                        _slot_name(slot, "forward", "local", name),
                        region,
                    )
                )
        for name in sorted(_BACKWARD_SLOT_LOCAL):
            if name in _ROUTING_LOCAL:
                continue
            local_regions.append(
                _clone_region(
                    _slot_name(slot, "backward", "local", name),
                    backward_local[name],
                )
            )
        local_regions.extend(
            (
                BufferRegion(
                    _custom_slot_name(slot, "fc1_preact"),
                    fc1_c_bytes,
                    alignment=128,
                ),
                BufferRegion(
                    _custom_slot_name(slot, "routing_topk_idx"),
                    int(config.max_tokens_per_rank) * config.top_k * torch.int32.itemsize,
                    alignment=16,
                ),
                BufferRegion(
                    _custom_slot_name(slot, "valid_route_counts"),
                    config.experts_per_rank * torch.int32.itemsize,
                    alignment=16,
                ),
                BufferRegion(
                    _custom_slot_name(slot, "expert_offsets"),
                    config.experts_per_rank * torch.int32.itemsize,
                    alignment=16,
                ),
                BufferRegion(
                    _custom_slot_name(slot, "grad_activation"),
                    int(config.max_tokens_per_rank) * config.hidden_size * torch.float32.itemsize,
                    alignment=16,
                ),
            )
        )
        symmetric_regions.append(
            BufferRegion(
                _custom_slot_symmetric_name(slot, "routing_topk_weights"),
                int(config.max_tokens_per_rank) * config.top_k * torch.float32.itemsize,
                alignment=16,
            )
        )
        for name, dtype in aux_dtypes.items():
            local_regions.append(
                BufferRegion(
                    _custom_slot_name(slot, name),
                    math.prod(aux_shapes[name]) * dtype.itemsize,
                    alignment=128 if name != "grad_y2_sf" else 16,
                )
            )
        for name, shape in wgrad_shapes.items():
            local_regions.append(
                BufferRegion(
                    _custom_slot_name(slot, name),
                    math.prod(shape),
                    alignment=128,
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
        "training-resources.symmetric-layout-harmonized",
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


def _block_scaled_tensor_abi(tensor) -> dict[str, object]:
    return {
        "format": tensor.format.value,
        "axis": int(tensor.axis),
        "logical_shape": list(tensor.logical_shape),
        "data": {
            "shape": list(tensor.data.shape),
            "stride": list(tensor.data.stride()),
            "dtype": str(tensor.data.dtype),
        },
        "scale": {
            "shape": list(tensor.scale.shape),
            "stride": list(tensor.scale.stride()),
            "dtype": str(tensor.scale.dtype),
        },
    }


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
            "occupancy": int(getattr(kernel, "occupancy", 1)),
            "smem_capacity": int(getattr(kernel, "smem_capacity", 0)),
        },
        "workspace": _workspace_abi(prepared.workspace_requirements),
        "pool_token_capacity": int(prepared.pool_token_capacity),
    }


def _build_training_abi_facts(
    config: ForwardConfig,
    forward: PreparedMxfp8Kernel,
    backward: PreparedMxfp8BackwardKernel,
    weights: MoeEpTrainingWeights,
    requirements: WorkspaceRequirements,
    *,
    slot_count: int,
    lane_count: int,
    source_tree_digest: str | None = None,
) -> dict[str, object]:
    """Return rank-independent JSON-safe facts for one training resource ABI."""

    if source_tree_digest is None:
        source_root = Path(__file__).resolve().parents[1] / "cutedsl_src"
        source_tree_digest = source_tree_sha256(source_root)
    weight_facts = {
        name: _block_scaled_tensor_abi(getattr(weights, name))
        for name in (
            "forward_fc1",
            "forward_fc2",
            "backward_w2_transpose",
            "backward_w1_transpose",
        )
    }
    return {
        "schema_version": 1,
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
            "gate_up_clamp": config.gate_up_clamp,
        },
        "resources": {
            "slot_count": int(slot_count),
            "lane_count": int(lane_count),
            "workspace": _workspace_abi(requirements),
        },
        "weights": weight_facts,
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
class Mxfp8TrainingSlotViews:
    """Persistent tensors that survive from forward through wgrad consumption."""

    index: int
    routing_topk_idx: torch.Tensor
    routing_topk_weights: torch.Tensor
    fc1_preact: torch.Tensor
    col_quant_data: torch.Tensor | None
    col_quant_sf: torch.Tensor | None
    valid_route_counts: torch.Tensor
    expert_offsets: torch.Tensor
    forward_output: torch.Tensor
    backward_output: torch.Tensor
    grad_activation: torch.Tensor
    dprob: torch.Tensor
    forward_overflow: torch.Tensor
    backward_overflow: torch.Tensor
    fc1_recompute: torch.Tensor
    fc1_recompute_sf: torch.Tensor
    fc1_col_output: torch.Tensor
    fc1_col_output_sf: torch.Tensor
    grad_y2: torch.Tensor
    grad_y2_sf: torch.Tensor
    wgrad_fc1_sfa: torch.Tensor
    wgrad_fc1_sfb: torch.Tensor
    wgrad_fc2_sfa: torch.Tensor
    wgrad_fc2_sfb: torch.Tensor


@dataclass(frozen=True)
class Mxfp8TrainingExecutionViews:
    """One slot bound to one mutable execution lane."""

    slot: Mxfp8TrainingSlotViews
    forward: PreparedResources
    backward: PreparedResources
    forward_expert_size_snapshot: torch.Tensor | None


class Mxfp8TrainingResourceOwner:
    """Own one combined symmetric/local root for N slots and M lanes."""

    def __init__(
        self,
        config: ForwardConfig,
        device: torch.device,
        forward: PreparedMxfp8Kernel,
        backward: PreparedMxfp8BackwardKernel,
        weights: MoeEpTrainingWeights,
        *,
        slot_count: int,
        lane_count: int,
        runtime_manager: Optional[RuntimeManager] = None,
        symmetric_provider: Optional[SymmetricMemoryProvider] = None,
        local_provider: Optional[LocalMemoryProvider] = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.forward_prepared = forward
        self.backward_prepared = backward
        self.weight_bindings = Mxfp8TrainingWeightBindings(
            weights,
            weight_interleave_size=config.weight_interleave_size,
        )
        self.stager = Mxfp8TrainingStager(config.hidden_size, config.top_k)
        self.wgrad_exporter = Mxfp8TrainingWgradExporter(
            experts=config.experts_per_rank,
            hidden=config.hidden_size,
            intermediate=config.intermediate_size,
            sf_padding=backward.config.sf_padding_block,
        )
        self.beta = torch.ones(
            (config.experts_per_rank,),
            dtype=torch.float32,
            device=self.device,
        )
        self.slot_count = slot_count
        self.lane_count = lane_count
        self.requirements = build_training_workspace_requirements(
            config,
            forward,
            backward,
            slot_count=slot_count,
            lane_count=lane_count,
        )
        self._runtime_manager = runtime_manager or get_runtime_manager()
        self._symmetric_provider = symmetric_provider
        self._local_provider = local_provider
        self._runtime: RuntimeHandle | None = None
        self._workspace: WorkspaceOwner | None = None
        self._abi_fingerprint: str | None = None
        self._closed = False
        self._lock = threading.RLock()

    @property
    def prepared(self) -> bool:
        return not self._closed and self._runtime is not None and self._workspace is not None and self._workspace.allocated

    def prepare(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("training resources are closed")
            if self.prepared:
                return
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("training resources must be prepared before CUDA graph capture")
            _runtime_debug(
                "training-resources.prepare.begin",
                slot_count=self.slot_count,
                lane_count=self.lane_count,
                local_bytes=(
                    self.requirements.local_layout.total_bytes
                    if hasattr(self.requirements, "local_layout")
                    else sum(region.nbytes for region in self.requirements.local_regions)
                ),
                symmetric_bytes=sum(region.nbytes for region in self.requirements.symmetric_regions),
            )
            _runtime_debug("training-resources.runtime-acquire.begin")
            runtime = self._runtime_manager.acquire(self.config, self.device)
            _runtime_debug(
                "training-resources.runtime-acquire.end",
                runtime_ref_count=getattr(
                    self._runtime_manager,
                    "ref_count",
                    "?",
                ),
            )
            self._runtime = runtime
            try:
                layout_watchdog = _RuntimeWatchdog("training-resources.symmetric-layout-harmonize")
                layout_watchdog.start()
                _runtime_debug("training-resources.symmetric-layout-harmonize.begin")
                try:
                    self.requirements = _harmonize_symmetric_regions(
                        self.requirements,
                        runtime,
                        self.device,
                    )
                finally:
                    layout_watchdog.close()
                _runtime_debug("training-resources.symmetric-layout-harmonize.end")
                if runtime.world_size > 1:
                    abi_watchdog = _RuntimeWatchdog("training-resources.abi-handshake")
                    abi_watchdog.start()
                    _runtime_debug("training-resources.abi-handshake.begin")
                    try:
                        abi_facts = _build_training_abi_facts(
                            self.config,
                            self.forward_prepared,
                            self.backward_prepared,
                            self.weight_bindings.weights,
                            self.requirements,
                            slot_count=self.slot_count,
                            lane_count=self.lane_count,
                        )
                        self._abi_fingerprint = _verify_training_abi_across_ranks(
                            abi_facts,
                            runtime,
                            self.device,
                        )
                    finally:
                        abi_watchdog.close()
                    _runtime_debug(
                        "training-resources.abi-handshake.end",
                        fingerprint=self._abi_fingerprint,
                    )
                _runtime_debug("training-resources.workspace-create.begin")
                workspace = WorkspaceOwner(
                    self.requirements,
                    runtime,
                    symmetric_provider=self._symmetric_provider,
                    local_provider=self._local_provider,
                )
                _runtime_debug(
                    "training-resources.workspace-create.end",
                    local_bytes=workspace.local_layout.total_bytes,
                    symmetric_bytes=workspace.symmetric_layout.total_bytes,
                )
                self._workspace = workspace
                allocation_watchdog = _RuntimeWatchdog("training-resources.workspace-allocate")
                allocation_watchdog.start()
                try:
                    workspace.ensure_allocated()
                finally:
                    allocation_watchdog.close()
                _runtime_debug("training-resources.workspace-allocate.end")
                if runtime.world_size > 1:
                    # Symmetric-root zeroing is asynchronous. No rank may
                    # enter the first device barrier until every peer has
                    # completed allocation and root initialization.
                    stream_watchdog = _RuntimeWatchdog("training-resources.stream-synchronize")
                    stream_watchdog.start()
                    _runtime_debug("training-resources.stream-synchronize.begin")
                    try:
                        torch.cuda.current_stream(self.device).synchronize()
                    finally:
                        stream_watchdog.close()
                    _runtime_debug("training-resources.stream-synchronize.end")

                    barrier_watchdog = _RuntimeWatchdog("training-resources.rank-barrier")
                    barrier_watchdog.start()
                    _runtime_debug("training-resources.rank-barrier.begin")
                    try:
                        dist.barrier(group=runtime.group)
                    finally:
                        barrier_watchdog.close()
                    _runtime_debug("training-resources.rank-barrier.end")
            except Exception:
                if self._workspace is not None:
                    self._workspace.close()
                    self._workspace = None
                runtime.close()
                self._runtime = None
                raise
            _runtime_debug("training-resources.prepare.end")

    def _flat_views(self, token_count: int) -> WorkspaceViews:
        self.prepare()
        assert self._workspace is not None
        return self._workspace.views(token_count)

    @staticmethod
    def _phase_workspace(
        flat: WorkspaceViews,
        requirements: WorkspaceRequirements,
        *,
        slot: int,
        lane: int,
        phase: str,
    ) -> WorkspaceViews:
        symmetric = {}
        local = {}
        slot_symmetric = _FORWARD_SLOT_SYMMETRIC if phase == "forward" else _BACKWARD_SLOT_SYMMETRIC
        slot_local = _FORWARD_SLOT_LOCAL if phase == "forward" else _BACKWARD_SLOT_LOCAL
        for region in requirements.symmetric_regions:
            if region.name in _ROUTING_SYMMETRIC:
                symmetric[region.name] = flat.symmetric[_custom_slot_symmetric_name(slot, "routing_topk_weights")]
                continue
            scope_name = (
                _slot_name(slot, phase, "symmetric", region.name) if region.name in slot_symmetric else _lane_name(lane, phase, "symmetric", region.name)
            )
            symmetric[region.name] = flat.symmetric[scope_name]
        for region in requirements.local_regions:
            if region.name in _ROUTING_LOCAL:
                local[region.name] = flat.local[_custom_slot_name(slot, "routing_topk_idx")]
                continue
            if phase == "backward" and region.name == "backward_fc1_preact":
                local[region.name] = flat.local[_custom_slot_name(slot, "fc1_preact")]
                continue
            scope_name = _slot_name(slot, phase, "local", region.name) if region.name in slot_local else _lane_name(lane, phase, "local", region.name)
            local[region.name] = flat.local[scope_name]
        return WorkspaceViews(
            token_count=flat.token_count,
            symmetric=MappingProxyType(symmetric),
            local=MappingProxyType(local),
            peer_mapping=flat.peer_mapping,
        )

    def _slot_views(
        self,
        flat: WorkspaceViews,
        slot: int,
    ) -> Mxfp8TrainingSlotViews:
        config = self.config
        capacity = int(config.max_tokens_per_rank)
        fwd_shapes = {name: tuple(int(extent) for extent in shape) for name, shape in self.forward_prepared.kernel.get_aux_output_shapes().items()}
        bwd_shapes = {name: tuple(int(extent) for extent in shape) for name, shape in self.backward_prepared.kernel.get_aux_output_shapes().items()}
        scale_columns = _align_scale_columns(self.forward_prepared.pool_token_capacity)

        def local_bytes(name: str) -> torch.Tensor:
            return flat.local[_custom_slot_name(slot, name)]

        col_quant_data = None
        col_quant_sf = None
        col_data_name = _slot_name(
            slot,
            "forward",
            "local",
            "col_quant_data",
        )
        if col_data_name in flat.local:
            col_quant_data = _typed_k_major_view(
                flat.local[col_data_name],
                _DATA_DTYPE,
                fwd_shapes["col_quant_data"],
            )
            col_quant_sf = _typed_view(
                flat.local[
                    _slot_name(
                        slot,
                        "forward",
                        "local",
                        "col_quant_sf",
                    )
                ],
                torch.uint8,
                fwd_shapes["col_quant_sf"],
            )

        return Mxfp8TrainingSlotViews(
            index=slot,
            routing_topk_idx=_typed_view(
                local_bytes("routing_topk_idx"),
                torch.int32,
                (capacity, config.top_k),
            ),
            routing_topk_weights=_typed_view(
                flat.symmetric[
                    _custom_slot_symmetric_name(
                        slot,
                        "routing_topk_weights",
                    )
                ],
                torch.float32,
                (capacity, config.top_k),
            ),
            fc1_preact=_typed_view(
                local_bytes("fc1_preact"),
                torch.bfloat16,
                fwd_shapes["fc1_c"],
            ),
            col_quant_data=col_quant_data,
            col_quant_sf=col_quant_sf,
            valid_route_counts=_typed_view(
                local_bytes("valid_route_counts"),
                torch.int32,
                (config.experts_per_rank,),
            ),
            expert_offsets=_typed_view(
                local_bytes("expert_offsets"),
                torch.int32,
                (config.experts_per_rank,),
            ),
            forward_output=_typed_view(
                flat.symmetric[
                    _slot_name(
                        slot,
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
                    _slot_name(
                        slot,
                        "backward",
                        "symmetric",
                        "output_data",
                    )
                ],
                torch.bfloat16,
                (capacity, config.hidden_size),
            ),
            grad_activation=_typed_view(
                local_bytes("grad_activation"),
                torch.float32,
                (capacity, config.hidden_size),
            ),
            dprob=_typed_view(
                flat.symmetric[
                    _slot_name(
                        slot,
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
                    _slot_name(
                        slot,
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
                    _slot_name(
                        slot,
                        "backward",
                        "local",
                        "overflow_flag",
                    )
                ],
                torch.int32,
                (1,),
            ),
            fc1_recompute=_typed_view(
                local_bytes("fc1_recompute"),
                _DATA_DTYPE,
                bwd_shapes["fc1_recompute"],
            ),
            fc1_recompute_sf=_typed_view(
                local_bytes("fc1_recompute_sf"),
                _SCALE_DTYPE,
                bwd_shapes["fc1_recompute_sf"],
            ),
            fc1_col_output=_typed_view(
                local_bytes("fc1_col_output"),
                _DATA_DTYPE,
                bwd_shapes["fc1_col_output"],
            ),
            fc1_col_output_sf=_typed_view(
                local_bytes("fc1_col_output_sf"),
                _SCALE_DTYPE,
                bwd_shapes["fc1_col_output_sf"],
            ),
            grad_y2=_typed_k_major_view(
                local_bytes("grad_y2"),
                _DATA_DTYPE,
                bwd_shapes["grad_y2"],
            ),
            grad_y2_sf=_typed_view(
                local_bytes("grad_y2_sf"),
                torch.uint8,
                bwd_shapes["grad_y2_sf"],
            ),
            wgrad_fc1_sfa=_typed_view(
                local_bytes("wgrad_fc1_sfa"),
                _SCALE_DTYPE,
                (_round_up(config.hidden_size, 128), scale_columns),
            ),
            wgrad_fc1_sfb=_typed_view(
                local_bytes("wgrad_fc1_sfb"),
                _SCALE_DTYPE,
                (
                    _round_up(2 * config.intermediate_size, 128),
                    scale_columns,
                ),
            ),
            wgrad_fc2_sfa=_typed_view(
                local_bytes("wgrad_fc2_sfa"),
                _SCALE_DTYPE,
                (
                    _round_up(config.intermediate_size, 128),
                    scale_columns,
                ),
            ),
            wgrad_fc2_sfb=_typed_view(
                local_bytes("wgrad_fc2_sfb"),
                _SCALE_DTYPE,
                (_round_up(config.hidden_size, 128), scale_columns),
            ),
        )

    def views(
        self,
        *,
        slot: int,
        lane: int,
        token_count: int,
    ) -> Mxfp8TrainingExecutionViews:
        with self._lock:
            if slot < 0 or slot >= self.slot_count:
                raise ValueError(f"slot {slot} is outside [0, {self.slot_count})")
            if lane < 0 or lane >= self.lane_count:
                raise ValueError(f"lane {lane} is outside [0, {self.lane_count})")
            flat = self._flat_views(token_count)
            forward_workspace = self._phase_workspace(
                flat,
                self.forward_prepared.workspace_requirements,
                slot=slot,
                lane=lane,
                phase="forward",
            )
            backward_workspace = self._phase_workspace(
                flat,
                self.backward_prepared.workspace_requirements,
                slot=slot,
                lane=lane,
                phase="backward",
            )
            snapshot = None
            if self.forward_prepared.col_quant_sizes_offset is not None:
                snapshot_bytes = forward_workspace.local["kernel_local_workspace"].narrow(
                    0,
                    self.forward_prepared.col_quant_sizes_offset,
                    self.forward_prepared.col_quant_sizes_bytes,
                )
                snapshot = _typed_view(
                    snapshot_bytes,
                    torch.int32,
                    (self.config.experts_per_rank,),
                )
            assert self._runtime is not None
            return Mxfp8TrainingExecutionViews(
                slot=self._slot_views(flat, slot),
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

    def refresh_weights(self) -> None:
        """Enqueue fixed-address layout refreshes from the bound MXFP8 pack."""

        with self._lock:
            if self._closed:
                raise RuntimeError("training resources are closed")
            self.weight_bindings.refresh()

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

    def finalize_overflow(
        self,
        slots: tuple[int, ...],
        *,
        lane: int,
    ) -> torch.Tensor:
        """Aggregate slot flags and apply the public error/drop policy."""

        if not slots:
            raise ValueError("finalize_overflow requires at least one slot")
        if len(set(slots)) != len(slots):
            raise ValueError("finalize_overflow slots must be unique")
        for slot in slots:
            if slot < 0 or slot >= self.slot_count:
                raise ValueError(f"slot {slot} is outside [0, {self.slot_count})")
        if lane < 0 or lane >= self.lane_count:
            raise ValueError(f"lane {lane} is outside [0, {self.lane_count})")
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
        global_overflow.zero_()
        for slot in slots:
            for phase in ("forward", "backward"):
                flag = _typed_view(
                    flat.local[
                        _slot_name(
                            slot,
                            phase,
                            "local",
                            "overflow_flag",
                        )
                    ],
                    torch.int32,
                    (1,),
                )
                torch.maximum(global_overflow, flag, out=global_overflow)
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
                raise RuntimeError("drop_on_overflow=False training resources require " "torch._assert_async")
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
                "Rubin MegaMoE receive route-pool overflow; " "the fixed-slot outputs are invalid",
            )
        return global_overflow


__all__ = [
    "Mxfp8TrainingExecutionViews",
    "Mxfp8TrainingResourceOwner",
    "Mxfp8TrainingSlotViews",
    "build_training_workspace_requirements",
]
