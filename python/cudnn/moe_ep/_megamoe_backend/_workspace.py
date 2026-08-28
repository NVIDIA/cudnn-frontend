# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Stable local and symmetric workspace ownership for MegaMoE."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Protocol, Sequence

import torch

from .._contracts import ForwardConfig
from ._comm import (
    PeerMapping,
    SymmetricMemoryProvider,
    SymmetricSlab,
    _TorchMemoryProvider,
)
from ._runtime import RuntimeHandle, _runtime_debug


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def padded_mxfp8_scale_columns(hidden: int) -> int:
    """Return the E8M0 row width required by Rubin's 16-byte token-in copy."""

    logical_columns = (hidden + 31) // 32
    return _align_up(logical_columns, 16)


@dataclass(frozen=True)
class BufferRegion:
    """One named byte region within a stable root allocation."""

    name: str
    nbytes: int
    alignment: int = 256

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("workspace region name must not be empty")
        if self.nbytes < 0:
            raise ValueError(
                f"workspace region {self.name!r} has negative size {self.nbytes}"
            )
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise ValueError(
                f"workspace region {self.name!r} alignment must be a power of two"
            )


@dataclass(frozen=True)
class BufferPlacement:
    """Resolved byte offset for one region."""

    name: str
    offset: int
    nbytes: int


@dataclass(frozen=True)
class BufferLayout:
    """Deterministic aligned layout for one root byte allocation."""

    placements: tuple[BufferPlacement, ...]
    total_bytes: int

    @classmethod
    def build(cls, regions: Sequence[BufferRegion]) -> "BufferLayout":
        names: set[str] = set()
        placements = []
        offset = 0
        max_alignment = 1
        for region in regions:
            if region.name in names:
                raise ValueError(f"duplicate workspace region {region.name!r}")
            names.add(region.name)
            offset = _align_up(offset, region.alignment)
            placements.append(
                BufferPlacement(
                    name=region.name,
                    offset=offset,
                    nbytes=region.nbytes,
                )
            )
            offset += region.nbytes
            max_alignment = max(max_alignment, region.alignment)
        return cls(
            placements=tuple(placements),
            total_bytes=_align_up(offset, max_alignment),
        )

    def placement(self, name: str) -> BufferPlacement:
        for placement in self.placements:
            if placement.name == name:
                return placement
        raise KeyError(name)


@dataclass(frozen=True)
class WorkspaceRequirements:
    """Capacity-driven regions supplied before runtime allocation.

    The executable backend obtains exact Rubin kernel workspace sizes, then
    passes them here without making the runtime owner import or instantiate
    CuTeDSL kernels.
    """

    max_tokens_per_rank: int
    symmetric_regions: tuple[BufferRegion, ...]
    local_regions: tuple[BufferRegion, ...]

    def __post_init__(self) -> None:
        if self.max_tokens_per_rank < 0:
            raise ValueError("max_tokens_per_rank must be non-negative")
        symmetric_names = {region.name for region in self.symmetric_regions}
        local_names = {region.name for region in self.local_regions}
        duplicates = symmetric_names & local_names
        if duplicates:
            raise ValueError(
                "workspace region names must be unique across roots: "
                f"{sorted(duplicates)}"
            )

    @classmethod
    def for_mxfp8(
        cls,
        config: ForwardConfig,
        *,
        kernel_local_workspace_bytes: int,
        kernel_shared_workspace_bytes: int,
        col_quant_data_bytes: int = 0,
        col_quant_sf_bytes: int = 0,
        backward_fc1_preact_bytes: int = 0,
        backward_dprob_bytes: int = 0,
        backward_aux_data_bytes: int = 0,
        backward_aux_scale_bytes: int = 0,
    ) -> "WorkspaceRequirements":
        if config.max_tokens_per_rank is None:
            raise ValueError("MXFP8 workspace requires max_tokens_per_rank")
        for name, value in (
            ("kernel_local_workspace_bytes", kernel_local_workspace_bytes),
            ("kernel_shared_workspace_bytes", kernel_shared_workspace_bytes),
            ("col_quant_data_bytes", col_quant_data_bytes),
            ("col_quant_sf_bytes", col_quant_sf_bytes),
            ("backward_fc1_preact_bytes", backward_fc1_preact_bytes),
            ("backward_dprob_bytes", backward_dprob_bytes),
            ("backward_aux_data_bytes", backward_aux_data_bytes),
            ("backward_aux_scale_bytes", backward_aux_scale_bytes),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if bool(col_quant_data_bytes) != bool(col_quant_sf_bytes):
            raise ValueError(
                "column requant data and scale workspace must be enabled together"
            )
        backward_sizes = (
            backward_fc1_preact_bytes,
            backward_dprob_bytes,
            backward_aux_data_bytes,
            backward_aux_scale_bytes,
        )
        if any(backward_sizes) and not all(backward_sizes):
            raise ValueError(
                "backward preactivation, dprob, data, and scale workspace "
                "must be enabled together"
            )

        tokens = config.max_tokens_per_rank
        hidden = config.hidden_size
        top_k = config.top_k
        kernel_sf_columns = padded_mxfp8_scale_columns(hidden)

        backward_symmetric_regions = (
            (BufferRegion("backward_dprob", backward_dprob_bytes),)
            if backward_dprob_bytes
            else ()
        )
        symmetric_regions = (
            BufferRegion("activation_data", tokens * hidden),
            BufferRegion("activation_scale", tokens * kernel_sf_columns),
            BufferRegion("topk_weights", tokens * top_k * 4),
            BufferRegion("output_data", tokens * hidden * 2),
            *backward_symmetric_regions,
            BufferRegion(
                "kernel_shared_workspace",
                kernel_shared_workspace_bytes,
            ),
            # Test-visible tail canary placed immediately after the opaque
            # peer-visible kernel workspace. It does not enter the kernel ABI.
            BufferRegion("symmetric_guard", 256, alignment=1),
        )
        col_quant_regions = (
            (
                BufferRegion("col_quant_data", col_quant_data_bytes),
                BufferRegion("col_quant_sf", col_quant_sf_bytes),
            )
            if col_quant_data_bytes
            else ()
        )
        backward_local_regions = (
            (
                BufferRegion(
                    "backward_fc1_preact",
                    backward_fc1_preact_bytes,
                    alignment=128,
                ),
                BufferRegion(
                    "backward_aux_data",
                    backward_aux_data_bytes,
                    alignment=128,
                ),
                BufferRegion(
                    "backward_aux_scale",
                    backward_aux_scale_bytes,
                    alignment=128,
                ),
            )
            if backward_fc1_preact_bytes
            else ()
        )
        local_regions = (
            BufferRegion("topk_idx", tokens * top_k * 4),
            BufferRegion("overflow_flag", 4),
            *col_quant_regions,
            *backward_local_regions,
            BufferRegion("kernel_local_workspace", kernel_local_workspace_bytes),
            BufferRegion("local_guard", 256, alignment=1),
        )
        return cls(
            max_tokens_per_rank=tokens,
            symmetric_regions=symmetric_regions,
            local_regions=local_regions,
        )

class LocalMemoryProvider(Protocol):
    """Injectable local allocation boundary."""

    def allocate(self, nbytes: int, device: torch.device) -> torch.Tensor: ...

    def free(self, tensor: torch.Tensor) -> None: ...


class _LocalSlab:
    def __init__(
        self,
        nbytes: int,
        device: torch.device,
        provider: LocalMemoryProvider,
    ) -> None:
        if nbytes <= 0:
            raise ValueError(f"local slab size must be positive, got {nbytes}")
        self._provider = provider
        self._nbytes = nbytes
        _runtime_debug("local-slab.allocate.begin", nbytes=nbytes, device=device)
        root = provider.allocate(nbytes, device)
        _runtime_debug(
            "local-slab.allocate.end",
            nbytes=nbytes,
            data_ptr=hex(root.data_ptr()) if isinstance(root, torch.Tensor) else "?",
        )
        self._root: Optional[torch.Tensor] = None
        try:
            if not isinstance(root, torch.Tensor):
                raise TypeError("local memory provider must return a torch.Tensor")
            if root.dtype is not torch.uint8 or root.numel() < nbytes:
                raise ValueError(
                    "local root must be a uint8 tensor with at least "
                    f"{nbytes} elements"
                )
            if root.device != device:
                raise ValueError(
                    "local root device does not match runtime device: "
                    f"root={root.device}, runtime={device}"
                )
            if not root.is_contiguous():
                raise ValueError("local root tensor must be contiguous")
            _runtime_debug("local-slab.zero.begin", nbytes=nbytes)
            root.zero_()
            _runtime_debug("local-slab.zero.enqueued", nbytes=nbytes)
        except Exception:
            if isinstance(root, torch.Tensor):
                provider.free(root)
            raise
        self._root = root

    @property
    def root(self) -> torch.Tensor:
        if self._root is None:
            raise RuntimeError("local workspace slab is closed")
        return self._root

    def byte_view(self, offset: int, nbytes: int) -> torch.Tensor:
        if offset < 0 or nbytes < 0 or offset + nbytes > self._nbytes:
            raise ValueError(
                f"byte view [{offset}, {offset + nbytes}) exceeds "
                f"local slab size {self._nbytes}"
            )
        return self.root.narrow(0, offset, nbytes)

    def close(self) -> None:
        root = self._root
        if root is None:
            return
        self._provider.free(root)
        self._root = None


@dataclass(frozen=True)
class WorkspaceViews:
    """Stable full-capacity byte views for one prepared request."""

    token_count: int
    symmetric: Mapping[str, torch.Tensor]
    local: Mapping[str, torch.Tensor]
    peer_mapping: PeerMapping


class WorkspaceOwner:
    """Own local and symmetric slabs for one static execution plan."""

    def __init__(
        self,
        requirements: WorkspaceRequirements,
        runtime: RuntimeHandle,
        *,
        symmetric_provider: Optional[SymmetricMemoryProvider] = None,
        local_provider: Optional[LocalMemoryProvider] = None,
    ) -> None:
        self.requirements = requirements
        self.runtime = runtime
        self.symmetric_layout = BufferLayout.build(
            requirements.symmetric_regions
        )
        self.local_layout = BufferLayout.build(requirements.local_regions)
        if self.symmetric_layout.total_bytes <= 0:
            raise ValueError("workspace requires at least one symmetric byte")
        if self.local_layout.total_bytes <= 0:
            raise ValueError("workspace requires at least one local byte")

        self._symmetric_provider = symmetric_provider
        self._local_provider = local_provider or _TorchMemoryProvider()
        self._symmetric: Optional[SymmetricSlab] = None
        self._local: Optional[_LocalSlab] = None
        self._closed = False
        self._cleanup_required = False
        self._lock = threading.RLock()

    @property
    def allocated(self) -> bool:
        return (
            not self._cleanup_required
            and self._symmetric is not None
            and self._symmetric.allocated
            and self._local is not None
        )

    @property
    def cleanup_required(self) -> bool:
        return self._cleanup_required

    @property
    def closed(self) -> bool:
        return self._closed

    def ensure_allocated(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("workspace owner is closed")
            if self._cleanup_required:
                raise RuntimeError(
                    "workspace owner requires cleanup before allocation"
                )
            if self.allocated:
                return
            self.runtime.ensure_open()

            _runtime_debug(
                "workspace.allocate.begin",
                local_bytes=self.local_layout.total_bytes,
                symmetric_bytes=self.symmetric_layout.total_bytes,
            )
            local = _LocalSlab(
                self.local_layout.total_bytes,
                self.runtime.device,
                self._local_provider,
            )
            self._local = local
            try:
                _runtime_debug("workspace.symmetric-slab.create.begin")
                symmetric = SymmetricSlab(
                    self.runtime,
                    self.symmetric_layout.total_bytes,
                    provider=self._symmetric_provider,
                )
                self._symmetric = symmetric
                _runtime_debug("workspace.symmetric-slab.ensure.begin")
                symmetric.ensure_allocated()
                _runtime_debug("workspace.symmetric-slab.ensure.end")
            except Exception:
                try:
                    if self._symmetric is not None:
                        self._symmetric.close()
                        self._symmetric = None
                    if self._local is not None:
                        self._local.close()
                        self._local = None
                except Exception:
                    self._cleanup_required = True
                    raise
                raise
            _runtime_debug("workspace.allocate.end")

    def views(self, token_count: int) -> WorkspaceViews:
        with self._lock:
            if token_count < 0:
                raise ValueError(
                    f"token_count must be non-negative, got {token_count}"
                )
            if token_count > self.requirements.max_tokens_per_rank:
                raise ValueError(
                    f"token count {token_count} exceeds "
                    f"max_tokens_per_rank={self.requirements.max_tokens_per_rank}"
                )
            self.ensure_allocated()
            assert self._symmetric is not None
            assert self._local is not None

            symmetric_views = {
                placement.name: self._symmetric.byte_view(
                    placement.offset,
                    placement.nbytes,
                )
                for placement in self.symmetric_layout.placements
            }
            local_views = {
                placement.name: self._local.byte_view(
                    placement.offset,
                    placement.nbytes,
                )
                for placement in self.local_layout.placements
            }
            return WorkspaceViews(
                token_count=token_count,
                symmetric=MappingProxyType(symmetric_views),
                local=MappingProxyType(local_views),
                peer_mapping=self._symmetric.mapping,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                if self._symmetric is not None:
                    self._symmetric.close()
                    self._symmetric = None
                if self._local is not None:
                    self._local.close()
                    self._local = None
            except Exception:
                self._cleanup_required = True
                raise
            self._cleanup_required = False
            self._closed = True


__all__ = [
    "BufferLayout",
    "BufferPlacement",
    "BufferRegion",
    "LocalMemoryProvider",
    "WorkspaceOwner",
    "WorkspaceRequirements",
    "WorkspaceViews",
    "padded_mxfp8_scale_columns",
]
