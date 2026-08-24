# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Stateless lowering from the public backward stash to Rubin pool rows."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..._contracts import ForwardConfig, ValidatedBackwardRequest

@dataclass(frozen=True)
class Mxfp8BackwardLayout:
    """Stateless lowering from compact route metadata to the dGLU pool LUT."""

    preact_row_lut: torch.Tensor

    @classmethod
    def from_request(
        cls,
        request: ValidatedBackwardRequest,
    ) -> "Mxfp8BackwardLayout":
        config = request.config
        metadata = request.route_metadata
        if config.max_tokens_per_rank is None:
            raise ValueError("MXFP8 backward requires max_tokens_per_rank")

        bounds = (
            (metadata[:, 0], 0, config.experts_per_rank, "local expert"),
            (metadata[:, 1], 0, config.ep_size, "source rank"),
            (
                metadata[:, 2],
                0,
                config.max_tokens_per_rank,
                "source token",
            ),
            (metadata[:, 3], 0, config.top_k, "source top-k slot"),
        )
        for values, lower, upper, name in bounds:
            if values.numel() and bool(
                ((values < lower) | (values >= upper)).any().item()
            ):
                raise ValueError(
                    f"route_metadata contains an out-of-range {name}"
                )

        preact_row_lut = cls._build_preact_row_lut(config, metadata)
        return cls(
            preact_row_lut=preact_row_lut,
        )

    @staticmethod
    def _build_preact_row_lut(
        config: ForwardConfig,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        lut = torch.full(
            (
                config.ep_size,
                int(config.max_tokens_per_rank),
                config.top_k,
            ),
            -1,
            dtype=torch.int32,
            device=metadata.device,
        )
        if metadata.shape[0]:
            compact_rows = torch.arange(
                metadata.shape[0],
                dtype=torch.int32,
                device=metadata.device,
            )
            lut[
                metadata[:, 1].to(torch.int64),
                metadata[:, 2].to(torch.int64),
                metadata[:, 3].to(torch.int64),
            ] = compact_rows
        return lut

__all__ = ["Mxfp8BackwardLayout"]
