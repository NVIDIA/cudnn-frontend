# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pluggable token-exchange interface for the EP dispatch/combine comm.

The layer only ever talks to :class:`TokenComm`; ``torch_dist`` (c10d
``all_to_all_single``) is the baseline oracle backend, and an NVSHMEM
symmetric-heap backend will implement the same interface later so the
transparent layer and the future megakernel share one comm contract.

Contract:

- ``all_to_all(x, output_splits, input_splits)`` is DIFFERENTIABLE. Its
  backward must be the adjoint exchange: the same all-to-all with
  ``input_splits`` and ``output_splits`` swapped (grads flow back to the
  ranks that sent the corresponding rows).
- ``all_to_all_no_grad`` is the same data movement for non-differentiable
  payloads (routing metadata such as expert ids).
- ``exchange_counts(send_counts)`` swaps a per-destination-rank int64 count
  vector so each rank learns its per-source receive counts. Not
  differentiable.

Split sizes are Python ints (row counts per rank), so building them forces a
device->host sync once per dispatch — accepted cost for the transparent
version.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Sequence

import torch


class TokenComm(ABC):
    """Row-wise all-to-all over the EP group."""

    @abstractmethod
    def all_to_all(
        self,
        x: torch.Tensor,
        output_splits: Sequence[int],
        input_splits: Sequence[int],
    ) -> torch.Tensor:
        """Differentiable uneven all-to-all along dim 0.

        ``x`` has ``sum(input_splits)`` rows grouped by destination rank in
        ascending rank order; the result has ``sum(output_splits)`` rows
        grouped by source rank in ascending rank order.
        """

    @abstractmethod
    def all_to_all_no_grad(
        self,
        x: torch.Tensor,
        output_splits: Sequence[int],
        input_splits: Sequence[int],
    ) -> torch.Tensor:
        """Same exchange for non-differentiable payloads (metadata)."""

    @abstractmethod
    def exchange_counts(self, send_counts: torch.Tensor) -> torch.Tensor:
        """Exchange a ``[ep_size]`` int64 per-rank count vector.

        Entry ``r`` of the input is the number of rows this rank will send to
        rank ``r``; entry ``r`` of the result is the number of rows this rank
        will receive from rank ``r``.
        """


_COMM_REGISTRY: Dict[str, Callable[..., TokenComm]] = {}


def register_comm(name: str) -> Callable[[Callable[..., TokenComm]], Callable[..., TokenComm]]:
    def deco(factory: Callable[..., TokenComm]) -> Callable[..., TokenComm]:
        if name in _COMM_REGISTRY:
            raise ValueError(f"comm backend {name!r} already registered")
        _COMM_REGISTRY[name] = factory
        return factory

    return deco


def create_comm(name: str, **kwargs) -> TokenComm:
    try:
        factory = _COMM_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown comm backend {name!r}; available: {sorted(_COMM_REGISTRY)}"
        ) from None
    return factory(**kwargs)
