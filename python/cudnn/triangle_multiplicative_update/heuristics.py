# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The triangle multiplicative update's performance knobs and their MEASURED defaults -- the one place the API reads a tile or a tile order from.

Every axis here is performance-only: a plan computes the same function (bitwise, the tests pin it) under any value of its domain, so a caller
or an autotuner may pick freely.  Numerics modes (``affine_fold``, ``fuse_residual`` of the output stage) are NOT knobs: they change the
function and stay op attributes the API never sets.

  ``k2_tile``        the FROST GEMM tile config of the contraction (:data:`K2_TILE_DOMAIN`, catalog names): the 2-CTA 256 x 256 pair tile,
                     the same with the A operand multicast across 2 / 4 CTAs, the 1-CTA 128 x 128 tile, the same with B multicast across 2.
  ``k1_pass_order``  the projection stage's tile order (``"interleaved"``: both passes of a row block adjacent, the second ``x`` read an L2 hit;
                     ``"major"``: every row block of pass 0 before pass 1).
  ``k3_prologue``    the output stage's MN-major LayerNorm form (``"mn_major"`` in place; ``"transpose"`` into the K-major layout).
  ``clc_stages``     the two LayerNorm-prologue stages' scheduler ring depth.

:func:`recommend` is the table -- the best MEASURED configuration per geometry on the 212-SM Rubin perf-node class (per-N buckets, by hidden
width ``c`` and batch); a geometry nobody timed takes the nearest measured bucket's pick, never a guess outside the domain.  :func:`resolve`
merges a caller's ``k2_tile`` / ``knobs`` over it; :func:`validate_knobs` is the typed vocabulary check ``check_support`` runs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

from cudnn.gemm.frost.ln_prologue import A0_PROLOGUES, PASS_ORDERS

from .kernels import contraction as k2

#: The contraction tile configs the heuristics may pick and ``check_support`` admits (every one a FROST GEMM catalog name; every one measured).
K2_TILE_DOMAIN: Tuple[str, ...] = (
    k2.DESIGN_TILE_CONFIG,
    k2.MULTICAST_A2_TILE_CONFIG,
    k2.MULTICAST_A4_TILE_CONFIG,
    k2.SMALL_N_TILE_CONFIG,
    k2.SMALL_N_MULTICAST_B2_TILE_CONFIG,
)
#: The scheduler ring depths of the LayerNorm-prologue stages that were measured (the substrate admits 1..8).
CLC_STAGES_DOMAIN: Tuple[int, ...] = (1, 2, 4)
#: Defaults of the knob record (the c = 128 geometry's measured picks; :func:`recommend` is the table that decides per geometry).
DEFAULT_K1_PASS_ORDER = "interleaved"
DEFAULT_K3_PROLOGUE = "mn_major"
DEFAULT_CLC_STAGES = 2


@dataclass(frozen=True)
class TriMulKnobs:
    """One complete knob assignment (performance only; the output is the same function under every member of every domain)."""

    k2_tile: str = k2.DESIGN_TILE_CONFIG
    k1_pass_order: str = DEFAULT_K1_PASS_ORDER
    k3_prologue: str = DEFAULT_K3_PROLOGUE
    clc_stages: int = DEFAULT_CLC_STAGES


def validate_knobs(knobs: TriMulKnobs) -> None:
    """Typed vocabulary check (``ValueError`` names the axis and its domain); a performance knob outside its domain is a malformed declaration."""
    if not isinstance(knobs, TriMulKnobs):
        raise TypeError(f"knobs must be a TriMulKnobs, got {type(knobs).__name__}")
    if knobs.k2_tile not in K2_TILE_DOMAIN:
        raise ValueError(f"k2_tile must be one of the measured contraction tile configs {K2_TILE_DOMAIN}, got {knobs.k2_tile!r}")
    if knobs.k1_pass_order not in PASS_ORDERS:
        raise ValueError(f"k1_pass_order must be one of {PASS_ORDERS}, got {knobs.k1_pass_order!r}")
    if knobs.k3_prologue not in A0_PROLOGUES:
        raise ValueError(f"k3_prologue must be one of {A0_PROLOGUES}, got {knobs.k3_prologue!r}")
    if knobs.clc_stages not in CLC_STAGES_DOMAIN:
        raise ValueError(f"clc_stages must be one of {CLC_STAGES_DOMAIN}, got {knobs.clc_stages!r}")


# THE TABLE.  Per hidden width ``c``: descending N floors, each with the COMPLETE knob set measured best there (the first floor <= N wins; below the
# last floor the last entry serves).  Measured on the 212-SM Rubin perf-node class (lock 2376 MHz, sampled 2364), B in {1, 4}, both directions, with a
# control pair per cell: an entry replaces the previous pick only where it beat it by more than twice the control-pair spread AND by > 0.5 %, in both
# directions and both batches of its bucket; otherwise the bucket keeps the simpler default.  (Values: the S37 sweep; the prototype measurement set the
# contraction tiles at c = 128, B = 1 first.)
_PICKS: Dict[int, Tuple[Tuple[int, TriMulKnobs], ...]] = {
    128: (
        (1024, TriMulKnobs(k2_tile=k2.MULTICAST_A4_TILE_CONFIG, k1_pass_order="interleaved", k3_prologue="transpose", clc_stages=1)),
        (768, TriMulKnobs(k2_tile=k2.DESIGN_TILE_CONFIG, k1_pass_order="interleaved", k3_prologue="transpose", clc_stages=1)),
        (512, TriMulKnobs(k2_tile=k2.MULTICAST_A2_TILE_CONFIG, k1_pass_order="interleaved", k3_prologue="transpose", clc_stages=1)),
        (384, TriMulKnobs(k2_tile=k2.SMALL_N_TILE_CONFIG, k1_pass_order="interleaved", k3_prologue="transpose", clc_stages=1)),
        (256, TriMulKnobs(k2_tile=k2.DESIGN_TILE_CONFIG, k1_pass_order="interleaved", k3_prologue="transpose", clc_stages=2)),
        (129, TriMulKnobs(k2_tile=k2.DESIGN_TILE_CONFIG, k1_pass_order="interleaved", k3_prologue="transpose", clc_stages=1)),
        (0, TriMulKnobs(k2_tile=k2.SMALL_N_TILE_CONFIG, k1_pass_order="interleaved", k3_prologue="mn_major", clc_stages=2)),
    ),
    256: (
        (1024, TriMulKnobs(k2_tile=k2.MULTICAST_A4_TILE_CONFIG, k1_pass_order="major", k3_prologue="mn_major", clc_stages=1)),
        (768, TriMulKnobs(k2_tile=k2.DESIGN_TILE_CONFIG, k1_pass_order="major", k3_prologue="mn_major", clc_stages=1)),
        (512, TriMulKnobs(k2_tile=k2.MULTICAST_A2_TILE_CONFIG, k1_pass_order="major", k3_prologue="mn_major", clc_stages=1)),
        (384, TriMulKnobs(k2_tile=k2.SMALL_N_TILE_CONFIG, k1_pass_order="major", k3_prologue="mn_major", clc_stages=1)),
        (256, TriMulKnobs(k2_tile=k2.DESIGN_TILE_CONFIG, k1_pass_order="major", k3_prologue="mn_major", clc_stages=1)),
        (129, TriMulKnobs(k2_tile=k2.DESIGN_TILE_CONFIG, k1_pass_order="major", k3_prologue="mn_major", clc_stages=1)),
        (0, TriMulKnobs(k2_tile=k2.SMALL_N_TILE_CONFIG, k1_pass_order="major", k3_prologue="mn_major", clc_stages=2)),
    ),
}


def _nearest_width(c: int) -> int:
    widths = sorted(_PICKS)
    return min(widths, key=lambda w: (abs(w - int(c)), w))


def k2_tile_for(n_res: int, channels: int = 128, batch: int = 1) -> str:
    """The contraction's tile config for ``(N, c, B)`` from the measured table."""
    return recommend(n_res, channels, batch).k2_tile


def recommend(n_res: int, channels: int = 128, batch: int = 1, direction: str = "outgoing") -> TriMulKnobs:
    """The measured default knob set for this geometry (both directions share it: the stages' tile orders do not depend on the direction; the batch
    folds into the contraction's GEMM batch and the row count, which did not move a pick between B = 1 and B = 4)."""
    if direction not in k2.DIRECTIONS:
        raise ValueError(f"direction must be one of {k2.DIRECTIONS}, got {direction!r}")
    n_res = int(n_res)
    table = _PICKS[_nearest_width(channels)]
    for floor, knobs in table:
        if n_res >= floor:
            return knobs
    return table[-1][1]


def resolve(n_res: int, channels: int, batch: int, direction: str, *, k2_tile: Optional[str] = None, knobs: Optional[TriMulKnobs] = None) -> TriMulKnobs:
    """The knob set a plan runs: :func:`recommend` for the geometry, then the caller's ``knobs`` (whole set) and ``k2_tile`` (one axis) over it.
    No validation here -- ``check_support`` runs :func:`validate_knobs` so an out-of-vocabulary name is a typed decline at the usual place."""
    out = knobs if knobs is not None else recommend(n_res, channels, batch, direction)
    if k2_tile is not None:
        out = replace(out, k2_tile=str(k2_tile))
    return out


__all__ = [
    "CLC_STAGES_DOMAIN",
    "DEFAULT_CLC_STAGES",
    "DEFAULT_K1_PASS_ORDER",
    "DEFAULT_K3_PROLOGUE",
    "K2_TILE_DOMAIN",
    "TriMulKnobs",
    "k2_tile_for",
    "recommend",
    "resolve",
    "validate_knobs",
]
