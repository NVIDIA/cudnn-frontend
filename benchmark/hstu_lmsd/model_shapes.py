# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named HSTU LMSD workloads used by micro and end-to-end benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LMSDShape:
    """One flattened HSTU LayerNorm-Multiply-SiLU-Dropout workload."""

    name: str
    num_rows: int
    hidden_size: int = 512
    u_storage_width: int = 2048
    eps: float = 1e-6
    dropout_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.num_rows <= 0:
            raise ValueError("num_rows must be positive")
        if self.hidden_size != 512:
            raise ValueError("the current HSTU LMSD kernels require hidden_size=512")
        if self.u_storage_width < self.hidden_size:
            raise ValueError("u_storage_width must be at least hidden_size")
        if not 0.0 <= self.dropout_ratio < 1.0:
            raise ValueError("dropout_ratio must be in [0, 1)")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


MODEL_SHAPES = {
    "smoke": LMSDShape(
        name="smoke",
        num_rows=8_192,
    ),
    "hstu_256k": LMSDShape(
        name="hstu_256k",
        num_rows=256_000,
    ),
    "hstu_production": LMSDShape(
        name="hstu_production",
        num_rows=2_739_421,
    ),
}

DEFAULT_SHAPE = "smoke"


def get_model_shape(name: str) -> LMSDShape:
    """Resolve a named workload and report valid choices on failure."""

    try:
        return MODEL_SHAPES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_SHAPES))
        raise ValueError(f"unknown shape {name!r}; choose one of: {choices}") from exc
