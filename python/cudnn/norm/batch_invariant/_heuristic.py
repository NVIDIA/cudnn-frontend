# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Placement choice for the batch-invariant RMSNorm, from occupancy and per-regime rules.

The kernel streams each row once. Its speed depends on how many row groups each
resident CTA walks ("waves"): with about one wave the launch is bound by one
row's load latency, and with many waves by how many load bytes each SM keeps in
flight. Below about one wave a CTA takes one row; a short launch stages it
through shared memory and a wide one loads it directly. With a few waves
persistent CTAs prefetch the next row groups with ``cp.async``. With many waves
each CTA stages one row group and the hardware scheduler keeps the SMs full,
which keeps more rows in flight than a persistent CTA's prefetch does, except
for the widest rows, which keep persistent CTAs. The rules in :func:`choose` rank
placements only; every placement computes the same bits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

from ._knobs import ROWS_PER_CTA, Knobs, stageable, well_formed
from ._spec import ELEMENT_BYTES, RmsNormClass


@dataclass(frozen=True)
class Part:
    """Per-SM limits of a part and its nominal SM count."""

    family: str
    sm_count: int
    max_threads_per_sm: int = 2048
    max_ctas_per_sm: int = 32
    registers_per_sm: int = 65536
    shared_bytes_per_sm: int = 228 * 1024


PART_TABLE: Dict[str, Part] = {
    "sm_100": Part("sm_100", 148),
    "sm_103": Part("sm_103", 148),
    "sm_107": Part("sm_107", 204),
}

#: Registers per thread outside the row's elements, and per element held across the reduction with its weight.
BASE_REGISTERS = 16
REGISTERS_PER_ELEMENT = 2
#: Registers per element when the row is read back from shared memory: its packed weight only.
WEIGHT_REGISTERS_PER_ELEMENT = 0.75
#: Shared memory the runtime reserves per CTA.
CTA_SHARED_RESERVE = 1024
#: Threads per row of a staged placement: four warps, the fewest that keep the per-row reduction short.
ROW_THREADS = 128
#: Rows narrower than this many elements load directly unless the launch has many waves, and then a CTA's
#: row group packs enough of them to carry at least this many elements.
NARROW_HIDDEN = 4096
#: Waves up to which a CTA takes one row; below ``DEEP_WAVES`` a persistent CTA prefetches two row groups, from it three;
#: from ``MANY_WAVES`` each CTA takes one row group instead of walking them persistently.
ONE_ROW_WAVES = 1.5
DEEP_WAVES = 4
MANY_WAVES = 8
#: Row count from which a launch of about one wave loads its rows directly.
WIDE_LAUNCH_ROWS = 1024
#: Threads of a CTA that packs several rows of a class of at most 32 lanes.
PACKED_CTA_THREADS = 128
#: Waves of packed CTAs (see :func:`packed_waves`) between which short rows load directly; fewer stage their rows,
#: more prefetch three row groups.
SHORT_DIRECT_WAVES = (0.4, 2.0)
#: Bytes of a row above which it takes the fewest threads that hold it and a prefetch as deep as shared memory allows.
WIDE_ROW_BYTES = 16384
#: Bytes of a row up to which many waves take one CTA per row group; wider rows keep persistent CTAs.
PER_GROUP_ROW_BYTES = 28672
#: Elements per thread up to which a row stays in registers between the two passes: in a persistent CTA of a wide
#: row at many waves, and in the direct loads of a row narrower than ``NARROW_HIDDEN``.
HELD_ELEMENTS = 40
#: Bytes such a persistent CTA prefetches ahead of the row group it reduces, at most, but at least one row.
PREFETCH_BYTES = 65536
#: Bytes of a granule below which rows stage only a small launch's row group (one buffer for a CTA that takes one row
#: group at many waves) and load directly where wider granules prefetch row groups ahead.
PREFETCH_GRANULE_BYTES = 8


def shared_bytes(knobs: Knobs, cls: RmsNormClass) -> int:
    """Shared memory of one CTA: the staged row-group buffers and the double-buffered group sums."""
    staged = knobs.stages * knobs.rows_per_cta * cls.hidden * ELEMENT_BYTES[cls.in_dtype]
    groups = 2 * knobs.rows_per_cta * cls.groups * 4 if cls.groups > 1 else 0
    return staged + groups


def resident_ctas(knobs: Knobs, cls: RmsNormClass, part: Part) -> int:
    """CTAs one SM holds at once under the thread, CTA, register and shared-memory limits."""
    per_element = REGISTERS_PER_ELEMENT if knobs.x_regs else WEIGHT_REGISTERS_PER_ELEMENT
    regs = min(255, -(-int(BASE_REGISTERS + per_element * knobs.elements_per_thread(cls)) // 8) * 8)
    by_threads = part.max_threads_per_sm // knobs.cta_threads
    by_regs = part.registers_per_sm // (regs * knobs.cta_threads)
    by_shared = part.shared_bytes_per_sm // (shared_bytes(knobs, cls) + CTA_SHARED_RESERVE)
    return max(1, min(by_threads, by_regs, by_shared, part.max_ctas_per_sm))


def waves(cls: RmsNormClass, rows: int, part: Part, sm_count: int) -> float:
    """Row groups each resident CTA walks with one row per CTA, one lane per thread and no staging."""
    return rows / (resident_ctas(Knobs(cls.lanes, 1, 0), cls, part) * sm_count)


def packed(cls: RmsNormClass) -> Knobs:
    """Direct loads of a class of at most 32 lanes, rows packed until a CTA has ``PACKED_CTA_THREADS`` threads."""
    least = max(1, 32 // cls.lanes)
    return Knobs(cls.lanes, max(least, min(ROWS_PER_CTA[-1], PACKED_CTA_THREADS // cls.lanes)), 0)


def packed_waves(cls: RmsNormClass, rows: int, part: Part, sm_count: int) -> float:
    """Row groups each resident CTA of :func:`packed` walks."""
    knobs = packed(cls)
    return rows / (resident_ctas(knobs, cls, part) * knobs.rows_per_cta * sm_count)


def _placed(cls: RmsNormClass, knobs: Knobs) -> Knobs:
    """``knobs`` as ``cls`` takes it: a class whose granules ``cp.async`` cannot copy loads directly instead of staging,
    with the row in registers, at the same threads, rows and grid."""
    return knobs if stageable(cls) else replace(knobs, stages=0, x_regs=True)


def _row_threads(cls: RmsNormClass, rows_per_cta: int, stages: int, persistent: bool) -> int:
    """``ROW_THREADS`` threads per row when the class has that many lanes and they hold the row, else more."""
    t = min(cls.lanes, ROW_THREADS)
    while t < cls.lanes and not well_formed(_placed(cls, Knobs(t, rows_per_cta, stages, False, persistent)), cls):
        t *= 2
    return t


def _held_threads(cls: RmsNormClass, stages: int) -> Optional[int]:
    """The fewest threads per row, from ``ROW_THREADS``, that keep at most ``HELD_ELEMENTS`` elements each, or None."""
    t = min(cls.lanes, ROW_THREADS)
    while t <= cls.lanes:
        knobs = Knobs(t, 1, stages)
        if well_formed(knobs, cls) and knobs.elements_per_thread(cls) <= HELD_ELEMENTS:
            return t
        t *= 2
    return None


def _narrow(cls: RmsNormClass) -> bool:
    """Whether the granules of ``cls`` are below ``PREFETCH_GRANULE_BYTES``."""
    return cls.vector * ELEMENT_BYTES[cls.in_dtype] < PREFETCH_GRANULE_BYTES


def _prefetching(cls: RmsNormClass, knobs: Knobs) -> Optional[Knobs]:
    """:func:`_staged` for a placement that prefetches row groups ahead; rows of narrow granules load directly."""
    if _narrow(cls):
        knobs = replace(knobs, stages=0, x_regs=True)
        return knobs if well_formed(knobs, cls) else None
    return _staged(cls, knobs)


def _staged(cls: RmsNormClass, knobs: Knobs) -> Optional[Knobs]:
    """``knobs`` with the deepest staging up to its own that the class admits (see :func:`_placed`), or None."""
    if not stageable(cls):
        knobs = _placed(cls, knobs)
        return knobs if well_formed(knobs, cls) else None
    for stages in range(knobs.stages, 0, -1):
        candidate = replace(knobs, stages=stages)
        if well_formed(candidate, cls):
            return candidate
    return None


def choose(cls: RmsNormClass, rows: int, part: Part, sm_count: int) -> Knobs:
    """The placement for ``rows`` rows of ``cls``: rows per CTA, staging depth and grid by the class width and by
    waves."""
    w = waves(cls, rows, part, sm_count)
    row_bytes = cls.hidden * ELEMENT_BYTES[cls.in_dtype]
    least = max(1, 32 // cls.lanes)  # rows that fill a warp
    direct = Knobs(cls.lanes, least, 0)
    knobs = None
    if cls.lanes <= 32:
        # Short rows, packed: about a wave of CTAs loads directly; a smaller launch stages its rows, and CTAs that
        # walk more row groups prefetch row groups ahead.
        direct = packed(cls)
        p = packed_waves(cls, rows, part, sm_count)
        if SHORT_DIRECT_WAVES[0] <= p <= SHORT_DIRECT_WAVES[1]:
            knobs = direct
        elif p < SHORT_DIRECT_WAVES[0]:
            knobs = _staged(cls, replace(direct, stages=2))
        else:
            knobs = _prefetching(cls, replace(direct, stages=3))
    elif row_bytes > WIDE_ROW_BYTES and (w < MANY_WAVES or row_bytes > PER_GROUP_ROW_BYTES):
        # Wide rows. Below many waves: the fewest threads that hold a row, the row read back from shared memory for
        # the store pass. At many waves: a prefetch of at most PREFETCH_BYTES, and the row kept in registers by as
        # few threads as keep at most HELD_ELEMENTS each (read back from shared memory if none do).
        if w < MANY_WAVES:
            stages = 2 if w <= ONE_ROW_WAVES else 3
            held = None
        else:
            stages = 1 + max(1, PREFETCH_BYTES // row_bytes)
            held = _held_threads(cls, stages)
        staged = _staged if w <= ONE_ROW_WAVES else _prefetching
        if held is not None:
            knobs = staged(cls, Knobs(held, 1, stages))
        else:
            knobs = staged(cls, Knobs(_row_threads(cls, 1, stages, True), 1, stages, x_regs=False))
    elif w <= ONE_ROW_WAVES:
        # About one row per CTA: staging shortens a small launch's load path, but costs a wide one.
        knobs = direct if rows >= WIDE_LAUNCH_ROWS else _staged(cls, Knobs(cls.lanes, least, 2, x_regs=False))
    elif w < MANY_WAVES and cls.hidden < NARROW_HIDDEN:
        # Narrow rows load directly while a thread holds at most HELD_ELEMENTS of a row; with more, rows packed until a
        # CTA has PACKED_CTA_THREADS threads prefetch two row groups and read the row back from shared memory.
        if direct.elements_per_thread(cls) <= HELD_ELEMENTS:
            knobs = direct
        else:
            r = max(least, min(ROWS_PER_CTA[-1], PACKED_CTA_THREADS // cls.lanes))
            knobs = _prefetching(cls, Knobs(cls.lanes, r, 3, x_regs=False))
    elif w < MANY_WAVES:
        stages = 3 if w < DEEP_WAVES else 4
        knobs = _prefetching(cls, Knobs(_row_threads(cls, least, stages, True), least, stages, x_regs=False))
    else:
        r = least
        while r * cls.hidden < NARROW_HIDDEN and r < ROWS_PER_CTA[-1]:
            r *= 2
        stages = 1 if _narrow(cls) else 2
        knobs = _staged(cls, Knobs(_row_threads(cls, r, stages, False), r, stages, x_regs=False, persistent=False))
    # A rule may not fit a class it was not measured on (shared memory of a long row); fall back to direct loads.
    return knobs if knobs is not None else direct


def reachable(cls: RmsNormClass, part: Part, sm_count: int) -> frozenset:
    """Every placement :func:`choose` returns for some row count of ``cls``.

    ``choose`` depends on the row count only through the waves thresholds (of :func:`waves`, and of
    :func:`packed_waves` for short rows) and ``WIDE_LAUNCH_ROWS``, so its answer is constant between consecutive
    breakpoints; evaluating it on both sides of each breakpoint covers every answer.
    """
    capacity = resident_ctas(Knobs(cls.lanes, 1, 0), cls, part) * sm_count
    bounds = [(w, capacity) for w in (ONE_ROW_WAVES, DEEP_WAVES, MANY_WAVES)]
    if cls.lanes <= 32:
        knobs = packed(cls)
        bounds += [(w, resident_ctas(knobs, cls, part) * knobs.rows_per_cta * sm_count) for w in SHORT_DIRECT_WAVES]
    edges = {1, WIDE_LAUNCH_ROWS, 2**31 - 1}
    for w, rows_per_wave in bounds:
        edge = int(w * rows_per_wave)
        edges.update((edge, edge + 1))
    return frozenset(choose(cls, rows, part, sm_count) for rows in edges if 1 <= rows < 2**31)
