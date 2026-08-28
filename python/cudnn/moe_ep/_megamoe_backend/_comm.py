# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Symmetric-memory ownership and peer-pointer descriptors for MegaMoE."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Protocol

import torch

from ._runtime import (
    RuntimeHandle,
    RuntimeUnavailableError,
    _runtime_debug,
)


class SymmetricMemoryProvider(Protocol):
    """Injectable allocation boundary for a symmetric root slab."""

    def allocate(self, nbytes: int, device: torch.device) -> torch.Tensor: ...

    def free(self, tensor: torch.Tensor) -> None: ...

    def peer_address(self, tensor: torch.Tensor, peer: int) -> int: ...


class _TorchMemoryProvider:
    """CUDA tensor provider for local and single-rank symmetric memory."""

    def allocate(self, nbytes: int, device: torch.device) -> torch.Tensor:
        return torch.empty(nbytes, dtype=torch.uint8, device=device)

    def free(self, tensor: torch.Tensor) -> None:
        del tensor

    def peer_address(self, tensor: torch.Tensor, peer: int) -> int:
        if peer != 0:
            raise ValueError(f"single-rank symmetric memory has no peer {peer}")
        return int(tensor.data_ptr())


class _NvshmemMemoryProvider:
    """Lazy adapter over NVSHMEM symmetric tensor allocation."""

    @staticmethod
    def _core():
        try:
            import nvshmem.core as core
        except (ImportError, OSError) as exc:
            raise RuntimeUnavailableError(
                "symmetric workspace requires nvshmem4py and NVSHMEM libraries"
            ) from exc
        return core

    def allocate(self, nbytes: int, device: torch.device) -> torch.Tensor:
        del device  # NVSHMEM allocates on the device bound during runtime init.
        started_at = time.monotonic()
        _runtime_debug("symmetric.allocate.begin", nbytes=nbytes)
        try:
            tensor = self._core().tensor(
                (nbytes,),
                dtype=torch.uint8,
                release=False,
                except_on_del=True,
            )
        except Exception as exc:
            _runtime_debug(
                "symmetric.allocate.error",
                nbytes=nbytes,
                error_type=type(exc).__name__,
                error=repr(exc),
                elapsed_seconds=f"{time.monotonic() - started_at:.3f}",
            )
            raise RuntimeUnavailableError(
                f"failed to allocate {nbytes} bytes from the NVSHMEM symmetric heap"
            ) from exc
        _runtime_debug(
            "symmetric.allocate.end",
            nbytes=nbytes,
            data_ptr=hex(tensor.data_ptr()),
            elapsed_seconds=f"{time.monotonic() - started_at:.3f}",
        )
        return tensor

    def free(self, tensor: torch.Tensor) -> None:
        started_at = time.monotonic()
        _runtime_debug(
            "symmetric.free.begin",
            nbytes=tensor.numel() * tensor.element_size(),
            data_ptr=hex(tensor.data_ptr()),
        )
        try:
            self._core().free_tensor(tensor)
        except Exception as exc:
            _runtime_debug(
                "symmetric.free.error",
                error_type=type(exc).__name__,
                error=repr(exc),
                elapsed_seconds=f"{time.monotonic() - started_at:.3f}",
            )
            raise RuntimeUnavailableError(
                "failed to free the NVSHMEM symmetric root slab"
            ) from exc
        _runtime_debug(
            "symmetric.free.end",
            elapsed_seconds=f"{time.monotonic() - started_at:.3f}",
        )

    def peer_address(self, tensor: torch.Tensor, peer: int) -> int:
        started_at = time.monotonic()
        _runtime_debug(
            "symmetric.peer-map.begin",
            peer=peer,
            data_ptr=hex(tensor.data_ptr()),
        )
        try:
            peer_tensor = self._core().get_peer_tensor(tensor, peer)
        except Exception as exc:
            _runtime_debug(
                "symmetric.peer-map.error",
                peer=peer,
                error_type=type(exc).__name__,
                error=repr(exc),
                elapsed_seconds=f"{time.monotonic() - started_at:.3f}",
            )
            raise RuntimeUnavailableError(
                f"failed to map symmetric root slab for peer {peer}"
            ) from exc
        peer_pointer = int(peer_tensor.data_ptr())
        _runtime_debug(
            "symmetric.peer-map.end",
            peer=peer,
            peer_data_ptr=hex(peer_pointer),
            elapsed_seconds=f"{time.monotonic() - started_at:.3f}",
        )
        return peer_pointer


@dataclass(frozen=True)
class PeerMapping:
    """Dense EP-rank peer deltas packed into the vendored kernel mapper ABI."""

    base_address: int
    offsets: tuple[int, ...]
    rank: int

    def __post_init__(self) -> None:
        if len(self.offsets) == 0:
            raise ValueError("peer mapping requires at least one rank")
        if self.rank < 0 or self.rank >= len(self.offsets):
            raise ValueError(
                f"peer mapping rank {self.rank} is outside {len(self.offsets)} ranks"
            )
        if self.offsets[self.rank] != 0:
            raise ValueError(
                f"local peer offset must be zero, got {self.offsets[self.rank]}"
            )

    @property
    def world_size(self) -> int:
        return len(self.offsets)

    def to_sym_buffer_host(self):
        """Build the CuTeDSL host payload lazily at the launch boundary."""

        from .cutedsl_src.communication.nvlink_domain.symmetric_buffer import (
            SymmetricBufferHost,
        )

        return SymmetricBufferHost(
            base_address=self.base_address,
            offsets=self.offsets,
            rank=self.rank,
            max_ranks=self.world_size,
        )


class SymmetricSlab:
    """One stable root allocation shared by all peer-visible workspace views."""

    def __init__(
        self,
        runtime: RuntimeHandle,
        nbytes: int,
        *,
        provider: Optional[SymmetricMemoryProvider] = None,
    ) -> None:
        if nbytes <= 0:
            raise ValueError(f"symmetric slab size must be positive, got {nbytes}")
        runtime.ensure_open()

        self._runtime = runtime
        self._nbytes = nbytes
        self._provider = provider or (
            _NvshmemMemoryProvider()
            if runtime.nvshmem_enabled
            else _TorchMemoryProvider()
        )
        self._root: Optional[torch.Tensor] = None
        self._mapping: Optional[PeerMapping] = None
        self._cleanup_required = False

    def ensure_allocated(self) -> None:
        if self._cleanup_required:
            raise RuntimeError(
                "symmetric slab requires cleanup before allocation"
            )
        if self._root is not None and self._mapping is not None:
            return
        if self._root is not None:
            raise RuntimeError(
                "symmetric slab has an allocation pending cleanup"
            )

        _runtime_debug(
            "symmetric-slab.ensure.begin",
            nbytes=self._nbytes,
            world_size=self._runtime.world_size,
            ep_rank=self._runtime.rank,
        )
        root = self._provider.allocate(self._nbytes, self._runtime.device)
        if not isinstance(root, torch.Tensor):
            raise TypeError(
                "symmetric memory provider must return a torch.Tensor"
            )
        self._root = root
        try:
            if root.dtype is not torch.uint8 or root.numel() < self._nbytes:
                raise ValueError(
                    "symmetric root must be a uint8 tensor with at least "
                    f"{self._nbytes} elements"
                )
            if root.device != self._runtime.device:
                raise ValueError(
                    "symmetric root device does not match runtime device: "
                    f"root={root.device}, runtime={self._runtime.device}"
                )
            if not root.is_contiguous():
                raise ValueError("symmetric root tensor must be contiguous")
        except Exception:
            self._cleanup_required = True
            raise

        try:
            _runtime_debug(
                "symmetric-slab.zero.begin",
                nbytes=self._nbytes,
                data_ptr=hex(root.data_ptr()),
            )
            root.zero_()
            _runtime_debug("symmetric-slab.zero.enqueued")

            base_address = int(root.data_ptr())
            offsets = []
            for peer in range(self._runtime.world_size):
                if peer == self._runtime.rank:
                    offsets.append(0)
                    continue
                offsets.append(
                    self._provider.peer_address(root, peer) - base_address
                )
            mapping = PeerMapping(
                base_address=base_address,
                offsets=tuple(offsets),
                rank=self._runtime.rank,
            )
            if (
                mapping.world_size != self._runtime.world_size
                or mapping.rank != self._runtime.rank
            ):
                raise RuntimeError(
                    "symmetric peer mapping does not match the EP subgroup"
                )
        except Exception:
            self._cleanup_required = True
            raise

        self._mapping = mapping
        _runtime_debug(
            "symmetric-slab.ensure.end",
            nbytes=self._nbytes,
            base_address=hex(mapping.base_address),
            offsets=mapping.offsets,
        )

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def closed(self) -> bool:
        return self._root is None

    @property
    def allocated(self) -> bool:
        return (
            not self._cleanup_required
            and self._root is not None
            and self._mapping is not None
        )

    @property
    def mapping(self) -> PeerMapping:
        if self._cleanup_required:
            raise RuntimeError("symmetric slab requires cleanup")
        if self._mapping is None:
            raise RuntimeError("symmetric slab is closed")
        return self._mapping

    @property
    def root(self) -> torch.Tensor:
        if self._cleanup_required:
            raise RuntimeError("symmetric slab requires cleanup")
        if self._root is None:
            raise RuntimeError("symmetric slab is closed")
        return self._root

    def byte_view(self, offset: int, nbytes: int) -> torch.Tensor:
        if offset < 0 or nbytes < 0 or offset + nbytes > self._nbytes:
            raise ValueError(
                f"byte view [{offset}, {offset + nbytes}) exceeds "
                f"symmetric slab size {self._nbytes}"
            )
        return self.root.narrow(0, offset, nbytes)

    def close(self) -> None:
        root = self._root
        if root is None:
            self._cleanup_required = False
            return
        try:
            self._provider.free(root)
        except Exception:
            self._cleanup_required = True
            raise
        self._root = None
        self._mapping = None
        self._cleanup_required = False


__all__ = [
    "PeerMapping",
    "SymmetricMemoryProvider",
    "SymmetricSlab",
]
