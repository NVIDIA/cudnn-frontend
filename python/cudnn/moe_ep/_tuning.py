# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Public, semantic-preserving performance tuning for :class:`MoeEp`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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


@dataclass(frozen=True, kw_only=True)
class MoeEpTuningConfig:
    """Validated Rubin MegaMoE performance knobs.

    These fields select scheduling and transport implementations without
    changing the public MoE mathematical contract. Every rank in an expert
    parallel group must use the same configuration.

    ``group_hint=None`` preserves the default behavior: the backend uses the
    number of hardware-resident CTA clusters.
    """

    token_back_mode: TokenBackMode = "epi_warps"
    epi_flag_batch: tuple[int, int] = (1, 1)
    token_in_flag_batch: int = 1
    group_hint: int | None = None
    reduce_topk_in_kernel: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.token_back_mode, str)
            or self.token_back_mode not in _TOKEN_BACK_MODES
        ):
            raise ValueError(
                "token_back_mode must be one of "
                f"{tuple(sorted(_TOKEN_BACK_MODES))}, got "
                f"{self.token_back_mode!r}"
            )
        if (
            not isinstance(self.epi_flag_batch, tuple)
            or self.epi_flag_batch not in _EPI_FLAG_BATCHES
        ):
            raise ValueError(
                "epi_flag_batch must be one of "
                f"{tuple(sorted(_EPI_FLAG_BATCHES))}, got "
                f"{self.epi_flag_batch!r}"
            )
        if (
            isinstance(self.token_in_flag_batch, bool)
            or self.token_in_flag_batch not in _TOKEN_IN_FLAG_BATCHES
        ):
            raise ValueError(
                "token_in_flag_batch must be one of "
                f"{tuple(sorted(_TOKEN_IN_FLAG_BATCHES))}, got "
                f"{self.token_in_flag_batch!r}"
            )
        if self.group_hint is not None and (
            isinstance(self.group_hint, bool)
            or self.group_hint not in _GROUP_HINTS
        ):
            raise ValueError(
                "group_hint must be None or one of "
                f"{tuple(sorted(_GROUP_HINTS))}, got {self.group_hint!r}"
            )
        if not isinstance(self.reduce_topk_in_kernel, bool):
            raise ValueError(
                "reduce_topk_in_kernel must be a bool, got "
                f"{self.reduce_topk_in_kernel!r}"
            )
        if (
            self.reduce_topk_in_kernel
            and self.token_back_mode != "epi_warps"
        ):
            raise ValueError(
                "reduce_topk_in_kernel requires "
                "token_back_mode='epi_warps'"
            )

__all__ = ["MoeEpTuningConfig"]
