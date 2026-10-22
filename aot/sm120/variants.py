# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one place the SM120 AOT kernel matrix is declared.

A *shape class* is what a plan fixes: head dim, dtype, mask (+ window), sink. Within a class the compiled knob sets are
the union of

* every knob set FROST's own SM120 heuristic (``cudnn.sdpa.fwd.heuristics.recommend``) picks over ``GRID``, snapped
  into the class's compiled domain (``SPLITS``, ``PACK_GQA``), and
* ``EXTRA_KNOBS`` (unsplit tiles a caller may tune over).

The same heuristic pass emits the run-time selection table (``build.py``), so the table only ever names compiled
kernels. Edit this file to add a variant; nothing else holds the matrix.
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from typing import Optional, Tuple

DTYPES = ("f16", "bf16")

# The decode-shaped launches split KV; every split count is its own kernel. Larger heuristic picks (64 and 128, chosen
# only for b * h_kv <= 2 at the longest KV) snap down to 32.
SPLITS = (1, 2, 4, 8, 16, 32)

# PackGQA bakes G = h_q / h_kv into the kernel. G = 6 needs tile_q % G != 0 support (PR #1740).
PACK_GQA = (2, 4, 6, 8)


@dataclass(frozen=True)
class ShapeClass:
    d: int
    causal: bool
    window_left: Optional[int]  # None = unbounded; W keeps keys j >= i' - W (i' = bottom-right aligned position)
    sink: bool
    pack_gqa: Tuple[int, ...] = PACK_GQA

    @property
    def mask_name(self) -> str:
        if not self.causal:
            return "nomask"
        return "causal" if self.window_left is None else f"causal_swa{self.window_left + 1}"


CLASSES = tuple(ShapeClass(d, causal, None, False) for d in (64, 128, 256) for causal in (False, True)) + (
    # gpt-oss: d=64, 64 / 8 heads, attention sinks; full-causal and 128-token sliding-window layers alternate.
    ShapeClass(64, True, None, True, pack_gqa=(8,)),
    ShapeClass(64, True, 127, True, pack_gqa=(8,)),
)

# Unsplit knob sets compiled for every class regardless of the heuristic, so a caller can tune over them:
# (pack_gqa, sched_policy, split_kv, tile_m, tile_n); tile_n = 0 means "the class's preferred KV tile".
EXTRA_KNOBS = ((0, 0, 1, 64, 0), (0, 0, 1, 128, 0), (0, 2, 1, 64, 0), (0, 2, 1, 128, 0))

# The selection grid. Each axis is bucketed by ceil(log2(x)); a bucket's representative is its upper end 2^i. The C++
# lookup clamps past the last bucket. "units" is b * h_kv (the heuristic sees work through that product).
GRID_UNITS = tuple(2**i for i in range(8))  # 1 .. 128
GRID_SQ = tuple(2**i for i in range(15))  # 1 .. 16384
GRID_SKV = tuple(2**i for i in range(18))  # 1 .. 131072
# G = h_q / h_kv: one table per G in 1..8; G > 8 uses the table computed at GRID_G_LARGE (never packed).
GRID_G = tuple(range(1, 9))
GRID_G_LARGE = 16
# SM counts the tables are computed for (RTX 5090 = 170, RTX PRO 6000 Blackwell = 188); a device uses the nearest.
SM_COUNTS = (170, 188)

SCHED_NAMES = {0: "nat", 1: "lpt", 2: "lptl2"}
PREFIX = "cudnn_fe_aot_sm120_sdpa_fwd"
COMBINE_PREFIX = "cudnn_fe_aot_sm120_sdpa_combine"


@dataclass(frozen=True)
class Knobs:
    pack_gqa: int  # 0 = off, else the baked G
    sched: int
    split: int
    tile_m: int
    tile_n: int


@functools.lru_cache(maxsize=None)
def kv_tile(d: int) -> int:
    """The KV tile the SM120 templates use at head dim d (config_sm120.kv_tile_preference, first that fits SMEM)."""
    from cudnn.sdpa.fwd.config_sm120 import SMEM_CAPACITY_BYTES, kv_tile_preference, smem_bytes, tile_domain

    dom = tile_domain(d, d, False)
    for n in kv_tile_preference(d):
        if all((m, n) in dom and smem_bytes(d, d, m, n) <= SMEM_CAPACITY_BYTES for m in (64, 128)):
            return n
    raise ValueError(f"no KV tile fits both Q tiles at d={d}")


def variant_name(dtype: str, c: ShapeClass, k: Knobs) -> str:
    sink = "_sink" if c.sink else ""
    gqa = f"_pgqa{k.pack_gqa}" if k.pack_gqa else ""
    return f"{PREFIX}_d{c.d}_{dtype}_{c.mask_name}{sink}_m{k.tile_m}n{k.tile_n}_split{k.split}_{SCHED_NAMES[k.sched]}{gqa}"


def combine_name(dtype: str, sink: bool) -> str:
    return f"{COMBINE_PREFIX}_{dtype}{'_sink' if sink else ''}"


def in_domain(c: ShapeClass, g: int, k: Knobs) -> bool:
    """Whether knob set ``k`` (proposed at GQA ratio ``g``) is compiled for class ``c``: the class's KV tile; PackGQA only
    for the class's ratios and at tile_m 128; splits from ``SPLITS``, unpacked at tile_m 64; an unsplit causal set on
    LPT_L2 (LPT folds into it)."""
    if k.tile_n != kv_tile(c.d) or k.split not in SPLITS:
        return False
    if k.pack_gqa and (g not in c.pack_gqa or k.tile_m != 128):
        return False
    if k.split > 1:
        return k.sched == 0 and (k.pack_gqa or k.tile_m == 64)
    return k.sched == (2 if c.causal else 0)


def snap(c: ShapeClass, g: int, pick: Knobs) -> Knobs:
    """``pick`` moved into the compiled domain of class ``c`` (see :func:`in_domain`)."""
    pack = pick.pack_gqa if (pick.pack_gqa and g in c.pack_gqa) else 0
    split = max(s for s in SPLITS if s <= pick.split)
    tile_m = 128 if pack else (64 if split > 1 else pick.tile_m)
    sched = 0 if split > 1 else (2 if c.causal else 0)
    return Knobs(pack, sched, split, tile_m, kv_tile(c.d))


def extra_knobs(c: ShapeClass):
    n = kv_tile(c.d)
    for pack, sched, split, tm, tn in EXTRA_KNOBS:
        if sched != 0 and not c.causal:
            continue
        yield Knobs(pack, sched, split, tm, tn or n)


def grid_points():
    return itertools.product(range(len(GRID_UNITS)), range(len(GRID_SQ)), range(len(GRID_SKV)))
