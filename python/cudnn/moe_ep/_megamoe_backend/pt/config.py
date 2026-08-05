# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""EP sizing + sharding config for the transparent PyTorch MoE-EP layer.

Deliberately much smaller than flashinfer's ``BootstrapConfig`` /
``FleetParams``: no transport knobs, no capacity (``max_tokens_per_rank``) —
the training path is dropless, so buffers are sized by the actual routed
token counts each iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch.distributed


@dataclass(frozen=True)
class EpConfig:
    """Expert-parallel MoE sizing.

    ``ep_size`` / ``ep_rank`` describe the EP comm domain (analogue of
    ``BootstrapConfig.world_size`` / ``rank``). ``process_group`` is the
    torch process group to run collectives on; ``None`` means the default
    (WORLD) group. Expert ``e`` is owned by rank ``e // num_local_experts``
    (contiguous sharding, matching flashinfer moe_ep).
    """

    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    ep_size: int
    ep_rank: int
    process_group: Optional["torch.distributed.ProcessGroup"] = field(
        default=None, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        for name in ("num_experts", "top_k", "hidden_size", "intermediate_size", "ep_size"):
            v = getattr(self, name)
            if v <= 0:
                raise ValueError(f"EpConfig.{name} must be positive, got {v}")
        if not (0 <= self.ep_rank < self.ep_size):
            raise ValueError(f"ep_rank {self.ep_rank} not in [0, {self.ep_size})")
        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"ep_size ({self.ep_size})"
            )
        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k ({self.top_k}) must be <= num_experts ({self.num_experts})"
            )

    @property
    def num_local_experts(self) -> int:
        return self.num_experts // self.ep_size

    @property
    def first_local_expert(self) -> int:
        """Global id of this rank's first local expert."""
        return self.ep_rank * self.num_local_experts
