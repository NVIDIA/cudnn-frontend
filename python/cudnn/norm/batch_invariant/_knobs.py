# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legal placements of a batch-invariant RMSNorm class.

A placement assigns the class's ``L`` lanes to ``threads_per_row`` threads
(thread ``t`` holds lanes ``t, t + T, t + 2T, ...``, one accumulator each) and
packs ``rows_per_cta`` rows into one CTA. With ``stages > 0`` row groups are
copied into ``stages`` shared-memory buffers with ``cp.async``, and the next
``stages - 1`` row groups stream in while the current one is reduced and stored
(one stage copies only the current row group); such a placement may keep the
widened row in registers between its two passes (``x_regs``) or read it back
from shared memory. A ``persistent`` placement launches the CTAs the device holds
at once and walks the row groups; otherwise each CTA takes one row group and the
hardware scheduler fills the SMs. The compiled kernel is the same either way.

Lane ``t + i * T`` lies in butterfly group ``(t + i * T) // 32``, and its partner
lane at offset ``< 32`` is held by the partner thread in the same slot ``i``, so
a warp shuffle on slot ``i`` performs the class butterfly of one group. The group
sums are then folded in the class order through shared memory. Every placement
therefore computes the same bits.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

from ._spec import ELEMENT_BYTES, MAX_ELEMENTS_PER_THREAD, RmsNormClass

MAX_CTA_THREADS = 1024
#: Shared memory one CTA may take for its staged row-group buffers.
MAX_STAGED_BYTES = 192 * 1024
#: Row-group buffers of a staged placement; 0 loads every row group straight from global memory.
STAGES = (0, 1, 2, 3, 4)
#: Bytes of the smallest granule ``cp.async`` copies, and so the smallest a staged placement takes.
MIN_STAGED_GRANULE_BYTES = 4

#: Lanes per thread a placement may fold into registers.
LANES_PER_THREAD = (1, 2, 4, 8)
ROWS_PER_CTA = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class Knobs:
    """``threads_per_row`` threads reduce one row; a CTA holds ``rows_per_cta`` rows.

    ``stages`` row-group buffers in shared memory carry a ``cp.async`` prefetch ``stages - 1`` row groups deep.
    ``x_regs`` keeps the widened row in registers between the reduction and the store pass; without it a staged
    placement reads the row back from shared memory for the store pass. ``persistent`` sizes the grid to the
    resident CTAs; otherwise the grid has one CTA per row group.
    """

    threads_per_row: int
    rows_per_cta: int = 1
    stages: int = 0
    x_regs: bool = True
    persistent: bool = True

    @property
    def cta_threads(self) -> int:
        return self.threads_per_row * self.rows_per_cta

    def lanes_per_thread(self, cls: RmsNormClass) -> int:
        return cls.lanes // self.threads_per_row

    def elements_per_thread(self, cls: RmsNormClass) -> int:
        return self.lanes_per_thread(cls) * cls.granules_per_lane * cls.vector


def stageable(cls: RmsNormClass) -> bool:
    """Whether ``cp.async`` copies the granules of ``cls``, so that its placements may stage rows."""
    return cls.vector * ELEMENT_BYTES[cls.in_dtype] >= MIN_STAGED_GRANULE_BYTES


def well_formed(knobs: Knobs, cls: RmsNormClass) -> bool:
    """Whether ``knobs`` places ``cls`` exactly.

    Threads per row divide the lanes; a row spans whole warps unless the class
    has fewer than 32 lanes, in which case each row is one butterfly group and
    a CTA consists of whole warps of such groups. A staged placement needs
    granules ``cp.async`` can copy.
    """
    t, r = knobs.threads_per_row, knobs.rows_per_cta
    if t <= 0 or r not in ROWS_PER_CTA or cls.lanes % t or cls.lanes // t not in LANES_PER_THREAD:
        return False
    if cls.lanes >= 32:
        if t % 32:
            return False
    elif t != cls.lanes:
        return False
    if knobs.cta_threads > MAX_CTA_THREADS or knobs.cta_threads % 32:
        return False
    if knobs.stages not in STAGES or knobs.stages * r * cls.hidden * ELEMENT_BYTES[cls.in_dtype] > MAX_STAGED_BYTES:
        return False
    if knobs.stages and not stageable(cls):
        return False
    if not knobs.x_regs and not knobs.stages:
        return False  # only a staged row has a copy to read back
    return knobs.elements_per_thread(cls) <= MAX_ELEMENTS_PER_THREAD


@lru_cache(maxsize=None)
def knob_space(cls: RmsNormClass) -> Tuple[Knobs, ...]:
    """Every well-formed placement of ``cls``: by prefetch depth, then widest rows first."""
    out = []
    for persistent in (True, False):
        for stages in STAGES:
            for x_regs in (True, False):
                for k in LANES_PER_THREAD:
                    if cls.lanes % k:
                        continue
                    for r in ROWS_PER_CTA:
                        knobs = Knobs(cls.lanes // k, r, stages, x_regs, persistent)
                        if well_formed(knobs, cls):
                            out.append(knobs)
    return tuple(out)
