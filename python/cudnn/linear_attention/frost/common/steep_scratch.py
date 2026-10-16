# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile-time storage contract for the steep-decay kernels.

Offsets are bytes, lengths are elements of the region's declared type. Forward
kernels keep separate arrays; backward kernels place these regions in bank fill.
Stage lifetimes and synchronization remain owned by each kernel's schedule.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScratchRegion:
    offset: int
    elements: int
    nbytes: int


@dataclass(frozen=True)
class SteepScratchLayout:
    corr: ScratchRegion
    dadm: ScratchRegion
    last: ScratchRegion
    scan: ScratchRegion
    tile: ScratchRegion
    norm: ScratchRegion
    nbytes: int

    def check_capacity(self, capacity: int) -> None:
        if self.nbytes > capacity:
            raise ValueError(f"steep scratch needs {self.nbytes} bytes, bank fill has {capacity}")


@dataclass(frozen=True)
class StateGradientScratchLayout:
    """Per-CTA global workspace: an FP32 state and a ring of bounded FP32 keys."""

    state: ScratchRegion
    keys: ScratchRegion
    nbytes: int


def state_gradient_scratch_layout(*, d_k: int, d_v: int, b_t: int, decay_stages: int) -> StateGradientScratchLayout:
    state = ScratchRegion(0, d_v * d_k, d_v * d_k * 4)
    keys = ScratchRegion(state.nbytes, decay_stages * b_t * d_k, decay_stages * b_t * d_k * 4)
    return StateGradientScratchLayout(state, keys, state.nbytes + keys.nbytes)


def steep_scratch_layout(
    *,
    b_t: int,
    d_k: int,
    scan_warps: int,
    scan_stages: int,
    last_stages: int,
    decay_stages: int,
    norm_stages: int,
    dadm_stages: int = 0,
    io_bytes: int = 2,
) -> SteepScratchLayout:
    """Lay out disjoint, 16-byte-aligned regions in the order used by backward."""
    offset = 0

    def region(elements, width=4):
        nonlocal offset
        offset = (offset + 15) // 16 * 16
        result = ScratchRegion(offset, elements, elements * width)
        offset += result.nbytes
        return result

    corr = region(decay_stages * 2 * b_t * b_t)
    dadm = region(dadm_stages * 2 * b_t * b_t, io_bytes)
    last = region(last_stages * d_k)
    scan = region(scan_stages * scan_warps)
    tile = region(decay_stages * scan_warps)
    norm = region(norm_stages * b_t)
    return SteepScratchLayout(corr, dadm, last, scan, tile, norm, (offset + 15) // 16 * 16)
