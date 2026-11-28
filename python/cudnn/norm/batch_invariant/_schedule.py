# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the class from the row shape and the placement from the device and the row count."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from cudnn._device import device_info

from ._heuristic import PART_TABLE, Part, choose, resident_ctas
from ._knobs import Knobs, well_formed
from ._spec import RmsNormClass, class_for

ARCH_FAMILIES = {(10, 0): "sm_100", (10, 3): "sm_103", (10, 7): "sm_107"}


@dataclass(frozen=True)
class Schedule:
    """The class, its placement and the persistent grid of one launch shape."""

    cls: RmsNormClass
    knobs: Knobs
    grid: int


def part_for(device: int) -> tuple[Part, int]:
    info = device_info(device)
    family = ARCH_FAMILIES.get(info.compute_capability)
    if family is None:
        raise NotImplementedError(f"batch_invariant_rmsnorm requires SM100, SM103 or SM107, got {info.compute_capability}")
    return PART_TABLE[family], info.sm_count


def grid_for(knobs: Knobs, cls: RmsNormClass, rows: int, part: Part, sm_count: int) -> int:
    """One CTA per row group; a persistent placement caps it at the CTAs the device holds at once."""
    groups = -(-rows // knobs.rows_per_cta)
    if not knobs.persistent:
        return groups
    return max(1, min(groups, resident_ctas(knobs, cls, part) * sm_count))


def select_schedule(
    rows: int,
    hidden: int,
    in_dtype: str,
    device: int,
    knobs: Optional[Knobs] = None,
) -> Schedule:
    """The schedule of one launch shape.

    ``knobs`` pins a placement; any well-formed one gives the same bits.
    """
    if rows <= 0 or rows >= 2**31:
        raise NotImplementedError("batch_invariant_rmsnorm requires a positive int32 row count")
    cls = class_for(hidden, in_dtype)
    part, sm_count = part_for(device)
    if knobs is None:
        knobs = _pick(cls, rows, part, sm_count)
    elif not well_formed(knobs, cls):
        raise NotImplementedError(f"batch_invariant_rmsnorm: {knobs} is not a placement of {cls.name}")
    return Schedule(cls, knobs, grid_for(knobs, cls, rows, part, sm_count))


@lru_cache(maxsize=4096)
def _pick(cls: RmsNormClass, rows: int, part: Part, sm_count: int) -> Knobs:
    knobs = choose(cls, rows, part, sm_count)
    if not well_formed(knobs, cls):
        raise NotImplementedError(f"batch_invariant_rmsnorm has no placement of {cls.name} for {rows} rows")
    return knobs
