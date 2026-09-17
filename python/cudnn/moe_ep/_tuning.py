# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public, semantic-preserving performance tuning for :class:`MoeEp`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AutotuneMode = Literal[
    "inference",
    "training_forward",
    "training_backward",
]

TokenBackMode = Literal[
    "epi_warps",
    "standalone_warps",
    "reuse_dispatch_warps",
]

_TOKEN_BACK_MODES = frozenset(
    {
        "epi_warps",
        "standalone_warps",
        "reuse_dispatch_warps",
    }
)
_EPI_FLAG_BATCHES = frozenset(
    {
        (4, 2),
        (1, 1),
        (1, 2),
        (1, 4),
        (2, 1),
        (2, 2),
        (2, 4),
        (4, 4),
    }
)
_TOKEN_IN_FLAG_BATCHES = frozenset({1, 2, 4, 8, 16})
_GROUP_HINTS = frozenset({64, 128, 256, 512, 768, 1024})
_DGRAD_OPTIMIZATIONS = frozenset(
    {
        "baseline",
        "rolling",
        "ds3_ep4_v1",
    }
)


@dataclass(frozen=True, kw_only=True)
class MoeEpTuningConfig:
    """Validated Rubin MegaMoE performance knobs.

    These fields select scheduling and transport implementations without
    changing the public MoE mathematical contract. Every rank in an expert
    parallel group must use the same configuration.

    ``group_hint=None`` preserves the default behavior: the backend uses the
    number of hardware-resident CTA clusters.

    ``dgrad_optimization`` applies only to training backward. ``baseline``
    preserves the default grouped schedule, ``rolling`` selects the upstream
    rolling schedule, and ``ds3_ep4_v1`` selects the strictly qualified
    upstream preset. The DS3 profile owns its preset fields and canonicalizes
    ``epi_flag_batch`` to ``(4, 2)``.
    """

    token_back_mode: TokenBackMode = "epi_warps"
    epi_flag_batch: tuple[int, int] = (1, 1)
    token_in_flag_batch: int = 1
    group_hint: int | None = None
    reduce_topk_in_kernel: bool = False
    dgrad_optimization: Literal[
        "baseline",
        "rolling",
        "ds3_ep4_v1",
    ] = "baseline"

    def __post_init__(self) -> None:
        if not isinstance(self.token_back_mode, str) or self.token_back_mode not in _TOKEN_BACK_MODES:
            raise ValueError("token_back_mode must be one of " f"{tuple(sorted(_TOKEN_BACK_MODES))}, got " f"{self.token_back_mode!r}")
        if not isinstance(self.epi_flag_batch, tuple) or self.epi_flag_batch not in _EPI_FLAG_BATCHES:
            raise ValueError("epi_flag_batch must be one of " f"{tuple(sorted(_EPI_FLAG_BATCHES))}, got " f"{self.epi_flag_batch!r}")
        if isinstance(self.token_in_flag_batch, bool) or self.token_in_flag_batch not in _TOKEN_IN_FLAG_BATCHES:
            raise ValueError("token_in_flag_batch must be one of " f"{tuple(sorted(_TOKEN_IN_FLAG_BATCHES))}, got " f"{self.token_in_flag_batch!r}")
        if self.group_hint is not None and (isinstance(self.group_hint, bool) or self.group_hint not in _GROUP_HINTS):
            raise ValueError("group_hint must be None or one of " f"{tuple(sorted(_GROUP_HINTS))}, got {self.group_hint!r}")
        if not isinstance(self.reduce_topk_in_kernel, bool):
            raise ValueError("reduce_topk_in_kernel must be a bool, got " f"{self.reduce_topk_in_kernel!r}")
        if (
            not isinstance(self.dgrad_optimization, str)
            or self.dgrad_optimization not in _DGRAD_OPTIMIZATIONS
        ):
            raise ValueError(
                "dgrad_optimization must be one of "
                f"{tuple(sorted(_DGRAD_OPTIMIZATIONS))}, got "
                f"{self.dgrad_optimization!r}"
            )
        if self.reduce_topk_in_kernel and self.token_back_mode != "epi_warps":
            raise ValueError("reduce_topk_in_kernel requires " "token_back_mode='epi_warps'")
        if self.dgrad_optimization != "ds3_ep4_v1":
            return
        if self.token_back_mode != "epi_warps":
            raise ValueError(
                "dgrad_optimization='ds3_ep4_v1' requires "
                "token_back_mode='epi_warps'"
            )
        if self.epi_flag_batch not in ((1, 1), (4, 2)):
            raise ValueError(
                "dgrad_optimization='ds3_ep4_v1' requires "
                "epi_flag_batch=(1, 1) or (4, 2)"
            )
        if self.token_in_flag_batch != 1:
            raise ValueError(
                "dgrad_optimization='ds3_ep4_v1' requires "
                "token_in_flag_batch=1"
            )
        if self.group_hint is not None:
            raise ValueError(
                "dgrad_optimization='ds3_ep4_v1' requires group_hint=None"
            )
        if self.reduce_topk_in_kernel:
            raise ValueError(
                "dgrad_optimization='ds3_ep4_v1' requires "
                "reduce_topk_in_kernel=False"
            )
        object.__setattr__(self, "epi_flag_batch", (4, 2))


@dataclass(frozen=True)
class MoeEpAutotuneCandidateResult:
    """Measured slow-rank latency for one successfully evaluated candidate."""

    tuning: MoeEpTuningConfig
    latency_ms: float
    samples_ms: tuple[float, ...]


@dataclass(frozen=True)
class MoeEpAutotuneResult:
    """Winner and measurements produced by one explicit tuning sweep."""

    mode: AutotuneMode
    winner: MoeEpTuningConfig
    candidates: tuple[MoeEpAutotuneCandidateResult, ...]

    @property
    def evaluated_candidates(self) -> int:
        return len(self.candidates)


__all__ = [
    "MoeEpAutotuneCandidateResult",
    "MoeEpAutotuneResult",
    "MoeEpTuningConfig",
]
