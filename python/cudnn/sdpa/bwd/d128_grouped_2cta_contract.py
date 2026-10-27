# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped THD d128 workspace on the cta_group::2 datapath: the grouped THD layout with NINE per-sequence tensor maps.

Declaration-only arithmetic shared by the adapter and the traced host (no device reads).  The 2-CTA main kernel reads
Q / dO through 64-row boxes (its N-split operands) AND 128-row boxes (its d-split operands) and reduces dQ through a
64-row fp32 box, so the per-sequence map array holds ``MAPS_PER_SEQUENCE`` (9) descriptors instead of the 1-CTA path's 7;
everything else (compact FP32 dQ / Delta, metadata, origins, native dK / dV partials) is the grouped THD contract.
"""

from dataclasses import dataclass
from typing import Optional

from .config_sm100_d128_accum_2cta import CFG
from .d128_contract import ACCUMULATION_CCS, WORKSPACE_ALIGNMENT, WORKSPACE_SLACK
from .d128_grouped_contract import PARTIAL_ITEMSIZE
from .d128_grouped_contract import validate_thd_bounds as _validate_grouped_thd_bounds
from .d128_thd_contract import ORIGIN_PORTS

ABI_VERSION = 1
THD = "accumulation_grouped_thd_2cta"
MAPS_PER_SEQUENCE = CFG.MAPS_PER_SEQUENCE
UNIT_ROWS = CFG.THD_UNIT_ROWS  # 256 kv rows per CLC work item (one cga2 pair)

# --- the per-S HEURISTIC: which body serves a BF16 grouped THD request (the adapter's typed op-level field ``cta_pair``) --------------------
# One place for every number the selector reads (G10 config-as-truth; test_sdpa_bwd_d128_cta_pair_heuristic.py pins each).  The selector
# itself (:func:`select_cta_pair`) is a pure function of plan facts: the device reads (cc, the occupancy query) are handed in by the
# prepared module, so the rule is testable without a GPU and the probe runs only when the cheaper predicates passed.
CTA_PAIR_MODES = ("auto", "off", "on")
CTA_PAIR_MODE_DEFAULT = "auto"
# The smallest S where the pair body won the in-session step on GB200 (cc 10.0) against the one-CTA body -- 8K +6.5 %, 16K +7.0 %,
# 32K +8.8 %, 64K +10.4 %, varlen ramps +7..+8.5 % (2026-10-27); lowered when a shorter S wins (the 2K / 4K cells are measured separately).
# Both declared extents (S_q and S_kv: the batch's LONGEST sequence on each side) must reach it: the measured cells have them equal, so a
# lopsided batch serves the measured-safe one-CTA body.
CTA_PAIR_MIN_SEQ = 8192
# The measured part.  SM103 (cc 10.3) is admitted by the pair body's own support check but has no measurement: it serves the one-CTA body
# under ``auto`` and declines ``on`` by name until a row is measured there.
CTA_PAIR_CCS = ((10, 0),)
CTA_PAIR_CLUSTER = CFG.CGA_M  # the pair = one cluster of two CTAs; the device must admit at least one such cluster (the occupancy query)
CTA_PAIR_BODY = "s_tail"  # the heuristic's pair body (the measured winner) when the typed ``tmem_map`` is unset under ``auto``
PAIR_BODY = "2cta"
ONE_CTA_BODY = "1cta"
CTA_PAIR_PREDICATES = ("thd", "cc", "cluster", "seq")  # the names a decline / a fallback reason carries; ``seq`` under ``auto`` only


class CtaPairDecline(ValueError):
    """``cta_pair = "on"`` with a failing pair predicate: the text names the predicate and the value seen (the adapter's typed decline)."""


def resolve_cta_pair(name=None):
    """The adapter's ``cta_pair`` mode, typed: ``None`` -> ``CTA_PAIR_MODE_DEFAULT``; any value outside ``CTA_PAIR_MODES`` is a ``ValueError``
    BY NAME (a selector outside its domain declines, never a silent default)."""
    mode = CTA_PAIR_MODE_DEFAULT if name is None else name
    if type(mode) is not str or mode not in CTA_PAIR_MODES:
        raise ValueError(f"cta_pair must be one of {CTA_PAIR_MODES}; got {mode!r}")
    return mode


@dataclass(frozen=True)
class CtaPairResolution:
    """What the selector decided for one request: the mode it read, the body it chose, why, and the facts it read."""

    mode: str
    body: str  # PAIR_BODY | ONE_CTA_BODY
    reason: str
    s_q_max: int
    s_kv_max: int
    cc: Optional[tuple]
    cluster_admitted: Optional[bool]  # None = the occupancy query was not consulted (off, or a cheaper predicate failed first)
    pair_map: Optional[str] = None  # the pair body's TMEM map once the dispatch settled it (None on the one-CTA body)

    @property
    def token(self):
        """The ``resolved_impl`` token a driver pins by value: ``pair=<mode>:<body>``."""
        return f"pair={self.mode}:{self.body}"


def _active_pairs(max_active_pairs):
    value = max_active_pairs() if callable(max_active_pairs) else max_active_pairs
    if type(value) is not int or value < 0:
        raise ValueError(f"the occupancy query must yield a non-negative int of active clusters of {CTA_PAIR_CLUSTER}; got {value!r}")
    return value


def select_cta_pair(mode, *, thd, s_q_max, s_kv_max, cc, max_active_pairs):
    """The per-S heuristic as a pure function of plan facts (no device reads here: the caller hands in Q's cc -- ``None`` off CUDA -- and the
    occupancy query as an int or a zero-argument callable, consulted LAST and only when the cheaper predicates passed, because it compiles a
    probe kernel).

    ``off`` -> the one-CTA body, nothing probed.  ``on`` -> the pair body at ANY S (the threshold is the heuristic's, not a support predicate);
    a failing pair predicate (``thd`` / ``cc`` / ``cluster``) is a :class:`CtaPairDecline` naming it.  ``auto`` -> the pair body iff THD, cc in
    ``CTA_PAIR_CCS``, both extents >= ``CTA_PAIR_MIN_SEQ`` and the device admits a cluster of ``CTA_PAIR_CLUSTER``; else the one-CTA body with
    the failing predicate named in the reason.  The pair body's own support predicates (grouped heads, BF16, D 128, the pair bounds) run after
    this in the dispatch; under ``auto`` their decline falls back to the one-CTA body."""
    mode = resolve_cta_pair(mode)
    for name, value in (("s_q_max", s_q_max), ("s_kv_max", s_kv_max)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"sequence extents must be positive integers; got {name}={value!r}")
    cc = None if cc is None else tuple(cc)
    if mode == "off":
        return CtaPairResolution(mode, ONE_CTA_BODY, "cta_pair=off", s_q_max, s_kv_max, cc, None)
    failures = []
    if not thd:
        failures.append(("thd", "requires THD (the pair body has no dense twin)", "a dense declaration"))
    if cc not in CTA_PAIR_CCS:
        failures.append(("cc", f"requires cc in {CTA_PAIR_CCS} (the measured part)", "a device that is not CUDA" if cc is None else str(cc)))
    if mode == "auto" and min(s_q_max, s_kv_max) < CTA_PAIR_MIN_SEQ:
        failures.append(("seq", f"both declared S_q and S_kv >= {CTA_PAIR_MIN_SEQ}", f"({s_q_max}, {s_kv_max})"))
    admitted = None
    if not failures:
        pairs = _active_pairs(max_active_pairs)
        admitted = pairs >= 1
        if not admitted:
            failures.append(("cluster", f"requires a cluster of {CTA_PAIR_CLUSTER} CTAs admitted by the device's occupancy query", str(pairs)))
    if failures:
        name, text, got = failures[0]
        if mode == "on":
            raise CtaPairDecline(f"cta_pair 'on' {text}; got {got}")
        return CtaPairResolution(mode, ONE_CTA_BODY, f"cta_pair=auto: the one-CTA body ({name}: {text}; got {got})", s_q_max, s_kv_max, cc, admitted)
    if mode == "on":
        return CtaPairResolution(mode, PAIR_BODY, "cta_pair=on", s_q_max, s_kv_max, cc, admitted)
    reason = f"cta_pair=auto: the pair body (S_q {s_q_max}, S_kv {s_kv_max} >= {CTA_PAIR_MIN_SEQ}; cc {cc}; cluster of {CTA_PAIR_CLUSTER} admitted)"
    return CtaPairResolution(mode, PAIR_BODY, reason, s_q_max, s_kv_max, cc, admitted)


def _align(value):
    return (value + WORKSPACE_ALIGNMENT - 1) // WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT


def validate_thd_bounds(batch, heads, kv_heads, sq, skv, tq, tkv, tq_phys, tkv_phys):
    """The grouped THD bounds plus the pair envelope: 256-row rounding and a grid x of pairs x CGA_M."""
    _validate_grouped_thd_bounds(batch, heads, kv_heads, sq, skv, tq, tkv, tq_phys, tkv_phys)
    if max(sq, skv) > CFG.INT32_ROUNDING_BOUND:
        raise ValueError("sequence envelope exceeds safe Int32 pair-unit rounding")
    if batch * heads * ((skv + UNIT_ROWS - 1) // UNIT_ROWS) * CFG.CGA_M > 2**31 - 1:
        raise ValueError("pair launch envelope exceeds the grid or Int32 work-id range")


def thd_workspace_offsets(batch, heads, tq, tkv):
    """Shared host/trace arithmetic; no value reads from device metadata.  Byte offsets from the 256-B aligned payload."""
    rounded_q = (tq + 127) // 128 * 128
    delta = 4 * rounded_q * heads * 128
    meta = delta + 4 * rounded_q * heads
    origins = meta + _align(4 * (4 * batch + 4))
    maps = origins + _align(8 * len(ORIGIN_PORTS) * batch)
    prefix_end = maps + MAPS_PER_SEQUENCE * batch * 128
    rounded_k = (tkv + 127) // 128 * 128
    dk = _align(prefix_end)
    size = rounded_k * heads * 128 * PARTIAL_ITEMSIZE
    dv = dk + _align(size)
    return rounded_q, rounded_k, delta, meta, origins, maps, dk, dv, dv + size


def thd_workspace_layout(batch, heads, tq, tkv):
    """Compact FP32 dQ / Delta, metadata, INT64 origins, nine per-sequence maps, native compact-KV partials."""
    if any(type(n) is not int or n <= 0 for n in (batch, heads, tq, tkv)):
        raise ValueError("grouped 2-CTA d128 packed workspace dimensions must be positive integers")
    rounded_q, rounded_k, delta, meta, origins, maps, dk, dv, total = thd_workspace_offsets(batch, heads, tq, tkv)
    if total + WORKSPACE_SLACK > 2**63 - 1:
        raise ValueError("grouped 2-CTA d128 packed workspace exceeds the signed pointer range")
    kv_shape, kv_strides = (1, rounded_k, heads, 128), (rounded_k * heads * 128, heads * 128, 128, 1)
    regions = (
        (0, (1, rounded_q, heads, 128), (rounded_q * heads * 128, heads * 128, 128, 1)),
        (delta, (1, heads, rounded_q), (heads * rounded_q, rounded_q, 1)),
        (meta, (4 * batch + 4,), (1,)),
        (origins, (len(ORIGIN_PORTS), batch), (batch, 1)),
        (maps, (MAPS_PER_SEQUENCE * batch * 16,), (1,)),
        (dk, kv_shape, kv_strides),
        (dv, kv_shape, kv_strides),
    )
    return regions, total + WORKSPACE_SLACK
