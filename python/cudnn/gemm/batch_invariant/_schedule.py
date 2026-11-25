# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from FlashInfer batch_invariant/{schedule,heuristic,api}.py (e2d163fb3d46).

"""Select the arithmetic from the weight geometry and the placement from the device and input layout."""

import logging
from dataclasses import dataclass
from functools import lru_cache

from cudnn._device import device_info, execution_context

from ._formats import FP8_BLOCKSCALE_FMTS, MX_MIXED_FMTS, chunk_unit
from ._heuristic import PART_TABLE, TOKEN_BOX_ROWS, decode_sane, estimate_cycles
from ._knobs import CLUSTERS, Knobs, can_implement, knob_product

logger = logging.getLogger(__name__)


SPLIT_MIN_CTAS = 256  # below this many N tiles at 32 wide, a K split is considered
SPLIT_MAX_CHUNKS = 128
SPLIT_MIN_KC = 256  # elements of K per chunk at least
# the mixed classes' chunk rule: the MX rule with a lower bound of MIXED_SPLIT_MIN_CHUNKS chunks for narrow, long weights -- at most
# MIXED_SPLIT_MAX_N columns (so few weight tiles that the unsplit kernel runs on a fraction of the part) and at least MIXED_SPLIT_MIN_K
# elements of K in whole runs of MIXED_SPLIT_MIN_CHUNKS granules; declared empirical, the same on every part, a function of (N, K) only
MIXED_SPLIT_MAX_N = 512
MIXED_SPLIT_MIN_K = 16384
MIXED_SPLIT_MIN_CHUNKS = 8


def chunks_for(N: int, K: int, in_fmt: str = "bf16") -> int:
    """Choose power-of-two K chunks from the weight shape, aligned to the format's MMA and scale-factor granules."""
    if in_fmt in FP8_BLOCKSCALE_FMTS:
        return 1  # the block-scale class has no split variant: its block fold already partitions K
    step = chunk_unit(in_fmt)  # the format's MMA step; for the block-scaled classes also a whole scale-factor atom column
    if N <= 0 or -(-N // 32) >= SPLIT_MIN_CTAS:
        return 1
    budget = K // (16 * N)
    C = 1
    while min(budget, SPLIT_MAX_CHUNKS) >= C * 2 and K % (step * C * 2) == 0 and K // (C * 2) >= SPLIT_MIN_KC:
        C *= 2
    if in_fmt in MX_MIXED_FMTS and N <= MIXED_SPLIT_MAX_N and K >= MIXED_SPLIT_MIN_K and K % (MIXED_SPLIT_MIN_CHUNKS * step) == 0:
        C = max(C, MIXED_SPLIT_MIN_CHUNKS)
    return C


#: The sm_107 instructions' classes of the MX formats (the sm_100 / sm_103 instructions' classes elsewhere).
MX_SM107_CLASS = {
    "mxfp8": "mxfp8_k64",
    "mxfp4": "mxfp4_k128",
    "mxfp4w8a": "mxfp4w8a_k64",
}


def mx_class_for(base: str, family: str) -> str:
    """SM107 uses its native MX instruction width for every batch size."""
    return MX_SM107_CLASS.get(base, base) if family == "sm_107" else base


def fp4_class_for(N: int, K: int, family: str) -> str:
    """The NVFP4 class a part runs for a weight shape: on the sm_107 parts the sm_107 instruction's class (128 elements per
    step, ``nvfp4_k128``) for every weight shape whose chunk rule at that step is unsplit, the sm_100 instruction's class
    (``nvfp4``) for the split shapes; ``nvfp4`` elsewhere. A function of (N, K) and the architecture family only, never of M
    (a token's bits may not depend on the batch), so a weight that takes the wider step at prefill takes it at decode too."""
    if family == "sm_107" and chunks_for(N, K, "nvfp4_k128") == 1:
        return "nvfp4_k128"
    return "nvfp4"


@dataclass(frozen=True)
class Geometry:
    fmt: str
    chunks: int
    knobs: Knobs
    total_clusters: int
    resident_clusters: int


@lru_cache(maxsize=None)
def _active_clusters(device, size):
    """Cache successful occupancy queries. A transient driver or context error is retried on the next plan."""
    from cutlass.utils import HardwareInfo

    with execution_context(None, device):
        count = int(HardwareInfo(device_id=device).get_max_active_clusters(size))
    if count <= 0:
        raise ValueError(f"invalid active cluster count {count}")
    return count


def _hardware(device):
    info = device_info(device)
    family = {(10, 0): "sm_100", (10, 3): "sm_103", (10, 7): "sm_107"}.get(info.compute_capability)
    if family is None:
        raise NotImplementedError(f"batch_invariant_gemm requires SM100, SM103 or SM107, got {info.compute_capability}")
    part = PART_TABLE[family]
    capacity = {}
    for size in sorted({cm * cn for cm, cn in CLUSTERS}):
        try:
            capacity[size] = _active_clusters(device, size)
        except (RuntimeError, ValueError) as exc:
            nominal = dict(part.capacity)[size]
            capacity[size] = max(1, round(nominal * info.sm_count / part.sm_count))
            logger.warning(
                "batch_invariant_gemm could not query occupancy for cluster size %d on device %d, using %d clusters: %s", size, device, capacity[size], exc
            )
    return part, info.sm_count, info.l2_cache_bytes, capacity


def _instruction_format(in_fmt, n, k, family):
    if in_fmt == "e4m3_block128":
        return "e4m3_bs"
    if in_fmt == "nvfp4":
        return fp4_class_for(n, k, family)
    return mx_class_for(in_fmt, family)


def _pick(m, n, k, b_major, fmt, chunks, part, sm_count, l2, capacity):
    candidates = [knobs for knobs in knob_product(cta_chunks=chunks > 1, in_fmt=fmt) if can_implement(knobs, m, n, k, b_major, chunks, fmt)]
    if not candidates:
        raise NotImplementedError(f"no batch_invariant_gemm placement for {(m, n, k)} in {fmt}, layout {b_major}, chunks {chunks}")
    swapped = any(knobs.swap_ab for knobs in candidates)
    blockscale = fmt in FP8_BLOCKSCALE_FMTS
    guarded = [knobs for knobs in candidates if decode_sane(knobs, m, swapped, blockscale)]
    if blockscale and m <= TOKEN_BOX_ROWS // 4:
        wide = [knobs for knobs in guarded if knobs.cluster_n == 4 and -(-n // (4 * knobs.tile_n)) <= capacity[4]]
        guarded = wide or guarded

    # min preserves enumeration order for ties, including the order of raster and swizzle choices.
    return min(guarded or candidates, key=lambda knobs: estimate_cycles(knobs, m, n, k, chunks, fmt, part, sm_count, capacity, l2))


def select_schedule(m, n, k, b_major, device, *, in_fmt="bf16", batch=1):
    """Select the instruction, chunk fold, persistent tile order and grid for a graph plan."""
    if min(m, n, k, batch) <= 0 or max(m, n, k, batch) >= 2**31:
        raise NotImplementedError("batch_invariant_gemm requires positive int32 dimensions")
    part, sm_count, l2, capacity = _hardware(device)
    return _schedule_for_profile(m, n, k, b_major, in_fmt, batch, part, sm_count, l2, tuple(sorted(capacity.items())))


@lru_cache(maxsize=4096)
def _schedule_for_profile(m, n, k, b_major, in_fmt, batch, part, sm_count, l2, capacities):
    # Key the placement by queried hardware facts so a plan built with nominal occupancy
    # does not mask a subsequent successful occupancy query.
    capacity = dict(capacities)
    fmt = _instruction_format(in_fmt, n, k, part.family)
    chunks = chunks_for(n, k, fmt)
    knobs = _pick(m, n, k, b_major, fmt, chunks, part, sm_count, l2, capacity)
    pm, pn = (n, m) if knobs.swap_ab else (m, n)
    cm, cn = knobs.cluster_shape_mn
    rows, cols = knobs.cta_tile_mn
    clusters_m = -(-pm // (rows * cm))
    clusters_n = -(-pn // (cols * cn))
    # Split BMM launches each batch separately. The tensor's third dimension indexes
    # workspace partials for a workspace fold and independent products for unsplit BMM.
    planes = batch if chunks == 1 else chunks if not knobs.cta_chunks and knobs.cluster_k == 1 else 1
    total = clusters_m * clusters_n * planes
    size = cm * cn * knobs.cluster_k
    resident = min(total, capacity[size])
    # The in-cluster fold launches all its output tiles on grid.z with adjacent K collaborators.
    advance = total if knobs.cluster_k > 1 else resident
    slow_axis = clusters_m if knobs.raster_along == "m" else clusters_n
    if max(total * size, total - 1 + advance, slow_axis * knobs.swizzle_size) >= 2**31:
        raise NotImplementedError("batch_invariant_gemm matrix geometry exceeds the persistent scheduler's int32 counters")
    return Geometry(fmt, chunks, knobs, total, resident)
