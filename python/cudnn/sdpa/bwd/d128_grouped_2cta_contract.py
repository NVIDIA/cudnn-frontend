# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped d128 workspace on the cta_group::2 datapath: the grouped THD layout with NINE per-sequence tensor maps, and its dense twin.

Declaration-only arithmetic shared by the adapter and the traced host (no device reads).  The 2-CTA main kernel reads
Q / dO through 64-row boxes (its N-split operands) AND 128-row boxes (its d-split operands) and reduces dQ through a
64-row fp32 box, so the per-sequence map array holds ``MAPS_PER_SEQUENCE`` (9) descriptors instead of the 1-CTA path's 7;
everything else (compact FP32 dQ / Delta, metadata, origins, native dK / dV partials) is the grouped THD contract.

The DENSE twin (``LAYOUT_FORMS``: "dense"; implementation ``DENSE``) is the SAME five launches over the same workspace: a dense
compact-BSHD tensor (B, H, S, D) with strides (S*H*D, D, H*D, 1) IS the packed token-major view (1, B*S, H, D) with uniform lengths
S and cu[b] = b * S, so the setup kernel's dense arm writes the metadata [kv_lens | cu_q | cu_k] arithmetically from the two declared
extents instead of reading length tensors, the eight data ports keep the compact prefix as their origin, and the ONE port whose
dense layout is not the packed one -- Stats, contiguous (B, H, S_q, 1) -- is bound as the head-major packed form (1, H, S_q) with
pitch S_q plus a per-sequence origin ``org[stats][b] = b * H * S_q`` (``DENSE_STATS_ORIGIN_SPEC``): head * S_q + (b * H * S_q + row)
== b * H * S_q + head * S_q + row, the contiguous index, so every stats reader of the chain (the delta launch's LSE fetch, the ldg
softmax's `_stats`) addresses the right word through the origin it already reads.  The main kernel is untouched (its rendering pins
hold); the THD rendering of the setup kernel is byte-identical (the dense operands are ``Optional``, None under THD).
"""

from dataclasses import dataclass
from typing import Optional

from .config_sm100_d128_accum_2cta import CFG
from .d128_contract import WORKSPACE_ALIGNMENT, WORKSPACE_SLACK
from .d128_grouped_contract import PARTIAL_ITEMSIZE
from .d128_grouped_contract import validate_thd_bounds as _validate_grouped_thd_bounds
from .d128_thd_contract import ORIGIN_PORTS

ABI_VERSION = 2  # 2: the ``stats_form = "tma_ring"`` workspace form (the padded pre-scaled stats region + the padded row offsets; header section 14)
THD = "accumulation_grouped_thd_2cta"
DENSE = "accumulation_grouped_dense_2cta"  # the dense compact-BSHD twin: the THD chain under the setup kernel's dense arm (module docstring)
# --- the LAYOUT FORM of the chain: a typed internal selector derived from the declaration (``thd`` -> "thd", else "dense"), threaded by NAME
# into the setup kernel (``layout_form_code``: the Constexpr arm that synthesizes the metadata), the prepared host's compile key and the
# implementation token.  Not a user field: the adapter's ``thd`` flag decides it.
LAYOUT_FORMS = ("thd", "dense")
LAYOUT_FORM_CODES = {"thd": 0, "dense": 1}
IMPLEMENTATIONS = {"thd": THD, "dense": DENSE}
# The dense Stats origin spec: ``origin_specs[8]`` of the dense form -- (row 8 = the stats port, multiplier 1, token stride 1): the row EXISTS
# (``_origin_row`` resolves it to 8 for every stats reader, as for a THD declaration that binds a stats ragged offset) but no offset tensor is
# read: the setup kernel's dense arm writes ``org[8][b] = b * H * S_q`` itself, so the multiplier / stride are the identity.
DENSE_STATS_ORIGIN_SPEC = (8, 1, 1)
DENSE_ORIGIN_SPECS = (None,) * 8 + (DENSE_STATS_ORIGIN_SPEC,)
DENSE_ORIGIN_DTYPES = (None,) * 9
MAPS_PER_SEQUENCE = CFG.MAPS_PER_SEQUENCE
UNIT_ROWS = CFG.THD_UNIT_ROWS  # 256 kv rows per CLC work item (one cga2 pair)
# --- stats_form = "tma_ring" (the s_tail body's header section 11; config_sm100_d128_accum_2cta_stail.CFG.STATS_FORMS): the delta launch
# writes the softmax's stat words PRE-SCALED and q-tile-PADDED into ``stats2`` = fp32 [heads][PAD_TOTAL / N][STATS_KINDS][N] -- kind 0 =
# -log2(e) * LSE (0 for an LSE of -inf), kind 1 = -scale * delta; the pad rows [S_b, ceil(S_b / N) * N) of every sequence hold 0.0 = the ldg
# form's `stat = 0` for a row past the sequence, bit for bit -- and the main kernel's loader bulk-copies ONE (head, q tile) block = STATS_KINDS
# x N x 4 B = 1 KiB per tile into its SMEM stats ring (no tensor map, no clip: a tile never crosses its sequence's padded block).  PAD_TOTAL =
# rounded_q + N * batch is the host bound on sum_b ceil(S_b / N) * N; ``rowoff[b] = sum_{i < b} ceil(S_i / N) * N`` (+ the total at [batch])
# is written by the setup kernel (tile_dsl.thd.write_thd_row_offsets) into its own Int32 [batch + 1] region -- the 4B+4 metadata tensor
# keeps its shape (a tensor's extent is part of the kernel's mangled signature: the ldg rendering pin).  Under "ldg" the compact delta region
# and the shipped layout are unchanged.
STATS_KINDS = 2


def stats2_pad_total(batch, rounded_q):
    """The padded token total of the tma_ring stats region: every sequence rounded UP to a whole q tile (N), bounded on the host."""
    return rounded_q + CFG.N * batch


def stats2_shape_strides(heads, pad_total):
    """The ``stats2`` tensor as the delta launch writes it and the loader addresses it: [heads][pad_total / N tiles][STATS_KINDS][N] fp32,
    one (head, tile) block contiguous = STATS_KINDS * N elements = the SMEM ring stage image [LSE row | delta row]."""
    tiles = pad_total // CFG.N
    return (heads, tiles, STATS_KINDS, CFG.N), (tiles * STATS_KINDS * CFG.N, STATS_KINDS * CFG.N, CFG.N, 1)


def resolve_layout_form(name):
    """The chain's layout form, typed: a name of ``LAYOUT_FORMS`` ("thd" = the packed chain reading device lengths and ragged offsets; "dense" =
    the compact-BSHD twin synthesizing them); any other value is a ``ValueError`` BY NAME (no default: the dispatch derives it from ``thd``)."""
    if type(name) is not str or name not in LAYOUT_FORMS:
        raise ValueError(f"layout must be one of {LAYOUT_FORMS}; got {name!r}")
    return name


def layout_form_code(name):
    """The setup kernel's ``layout_code`` Constexpr[int] for a layout form (``LAYOUT_FORM_CODES``: 0 = thd, 1 = dense); declines as
    :func:`resolve_layout_form`."""
    return LAYOUT_FORM_CODES[resolve_layout_form(name)]


def layout_form_of(thd):
    """The layout form a declaration selects: ``thd`` True -> "thd", else "dense"."""
    return "thd" if thd else "dense"


# --- the per-S HEURISTIC: which body serves a BF16 grouped request (the adapter's typed op-level field ``cta_pair``) ------------------------
# One place for every number the selector reads (G10 config-as-truth; test_sdpa_bwd_d128_cta_pair_heuristic.py pins each).  The selector
# itself (:func:`select_cta_pair`) is a pure function of plan facts: the device reads (cc, the occupancy query) are handed in by the
# prepared module, so the rule is testable without a GPU and the probe runs only when the cheaper predicates passed.
CTA_PAIR_MODES = ("auto", "off", "on")
CTA_PAIR_MODE_DEFAULT = "auto"
# PER PART (cc -> S): the smallest S where the pair body won the in-session backward step against the one-CTA body.  Measured on GB200
# (cc 10.0): 8K +4.9 %, 16K +7.9 %, 32K +8.5 %, 64K +8.8 % uniform, the varlen ramps +6.5 % (16K) / +8.3 % (32K) (2026-10-27; the rows of
# record in the PR); below it the pair LOSES on the step (2K -3.9 %, 4K -3.5 %, measured) and the one-CTA body serves.  A part's value is
# lowered when a shorter S wins THERE; a carried value is replaced by that part's own measurement and the note beside it says where.
# Both declared extents (S_q and S_kv: the batch's LONGEST sequence on each side) must reach it: the measured cells have them equal, so a
# lopsided batch serves the measured-safe one-CTA body.  The dense twin (``LAYOUT_FORMS`` "dense": the same kernels over the same bytes as a
# uniform THD batch) runs under the SAME threshold; its own placement against the backend is not measured yet (the tracker says so).
# The keys are the admitted parts (``CTA_PAIR_CCS``): the same tcgen05 / cta_group::2 / TMEM / CLC feature set, the same per-S rule, the
# compile target following the device (``--gpu-arch sm_{cc}a`` in the prepared host, as for the one-CTA body).  Hopper, SM120 and any other
# line have no pair body: ``auto`` serves the one-CTA body there and ``on`` declines by name; an unadmitted cc has no threshold
# (:func:`cta_pair_min_seq` raises by name).
CTA_PAIR_MIN_SEQ = {
    # GB200: measured 2026-10-27 (the rows above)
    (10, 0): 8192,
    # GB300: the GB200 value carried with the cc 10.3 admission (2026-10-28); unmeasured on cc 10.3
    (10, 3): 8192,
    # Rubin: re-measured 2026-11-03 on a Rubin class node (212 SMs; A/B/A x3 x100, both build orders): the pair body wins the backward step
    # over the one-CTA body by +7.9 % (8K), +9.9 % (16K), +12.3 % (32K) and is not a win at 4K -> the GB200 threshold holds.  The axes the
    # rule ignores were measured there the same day and none moves the verdict at S >= 8K: the GQA ratio (hq / hkv 2, 4, 32: +8.0, +7.5,
    # +7.6 % at 8K; +9.6, +9.4 % at 16K -- the per-kv-tile iteration count does not drive the margin, S does), the dense twin (a thinner win
    # over its four-launch one-CTA base: +3.3 % at 8K, +6.5 % at 16K), non-causal (a larger one: +12.9 % at 8K, +11.8 % at 16K) and the
    # varlen ramps (+8.1 % at 16K, +10.1 % at 32K).  Under-fill cells (batch 1 or eight heads: steps under 1 ms) resolve no sign at this
    # protocol's resolution; the rule's choice there costs nothing measurable.
    (10, 7): 8192,
}
CTA_PAIR_CCS = tuple(CTA_PAIR_MIN_SEQ)
# The layouts the pair body serves: the packed chain and its dense compact-BSHD twin (``LAYOUT_FORMS``); the selector's ``layout`` predicate
# declines any other name.  Whether the DECLARATION fits the layout (compact strides, contiguous Stats, S % 128 on dense) is the pair body's
# own support check, after the selector.
CTA_PAIR_LAYOUTS = LAYOUT_FORMS
CTA_PAIR_CLUSTER = CFG.CGA_M  # the pair = one cluster of two CTAs; the device must admit at least one such cluster (the occupancy query)
# the pair body's identity token (the s_tail module, config_sm100_d128_accum_2cta_stail; its TMEM MAP is the typed, PART-KEYED op-level field
# ``tmem_map`` -- "s_tail" | "dq_576" -- resolved by the prepared module per device cc and carried by the compile key, never by this token)
CTA_PAIR_BODY = "s_tail"
PAIR_BODY = "2cta"
ONE_CTA_BODY = "1cta"
CTA_PAIR_PREDICATES = ("layout", "cc", "cluster", "seq")  # the names a decline / a fallback reason carries; ``seq`` under ``auto`` only


class CtaPairDecline(ValueError):
    """``cta_pair = "on"`` with a failing pair predicate: the text names the predicate and the value seen (the adapter's typed decline);
    ``predicate`` = its ``CTA_PAIR_PREDICATES`` name (the dispatch appends the occupancy query's own failure text to a ``cluster`` decline)."""

    def __init__(self, predicate, text):
        super().__init__(text)
        self.predicate = predicate


def cta_pair_min_seq(cc):
    """The per-S threshold of the admitted part ``cc`` (``CTA_PAIR_MIN_SEQ[cc]``); an unadmitted cc has none -- a ``ValueError`` by name,
    never a silent default (the selector reports the ``cc`` predicate before it reads a threshold)."""
    cc = None if cc is None else tuple(cc)
    if cc not in CTA_PAIR_MIN_SEQ:
        raise ValueError(f"no cta_pair threshold for cc {cc}: the pair body is admitted on {CTA_PAIR_CCS}")
    return CTA_PAIR_MIN_SEQ[cc]


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
    pair_map: Optional[str] = None  # the pair body's identity (``CTA_PAIR_BODY``) once the dispatch chose it (None on the one-CTA body)

    @property
    def token(self):
        """The ``resolved_impl`` token a driver pins by value: ``pair=<mode>:<body>``."""
        return f"pair={self.mode}:{self.body}"


def _active_pairs(max_active_pairs):
    value = max_active_pairs() if callable(max_active_pairs) else max_active_pairs
    if type(value) is not int or value < 0:
        raise ValueError(f"the occupancy query must yield a non-negative int of active clusters of {CTA_PAIR_CLUSTER}; got {value!r}")
    return value


def select_cta_pair(mode, *, layout, s_q_max, s_kv_max, cc, max_active_pairs):
    """The per-S heuristic as a pure function of plan facts (no device reads here: the caller hands in Q's cc -- ``None`` off CUDA -- and the
    occupancy query as an int or a zero-argument callable, consulted LAST and only when the cheaper predicates passed, because it compiles a
    probe kernel).  ``layout`` = the declaration's layout form (``layout_form_of(thd)``: "thd" | "dense").

    ``off`` -> the one-CTA body, nothing probed.  ``on`` -> the pair body at ANY S (the threshold is the heuristic's, not a support predicate);
    a failing pair predicate (``layout`` / ``cc`` / ``cluster``) is a :class:`CtaPairDecline` naming it.  ``auto`` -> the pair body iff the layout
    is one of ``CTA_PAIR_LAYOUTS`` (THD or dense compact BSHD), cc in ``CTA_PAIR_CCS``, both extents >= that part's ``CTA_PAIR_MIN_SEQ[cc]``
    (:func:`cta_pair_min_seq`; the ``cc`` predicate is reported first, so an unadmitted cc never reaches the threshold) and the device admits a
    cluster of ``CTA_PAIR_CLUSTER``; else the one-CTA body with the failing predicate named in the reason.  The pair body's own support
    predicates (grouped heads, BF16, D 128, compact storage, the pair bounds) run after this in the dispatch; under ``auto`` their decline
    falls back to the one-CTA body."""
    mode = resolve_cta_pair(mode)
    for name, value in (("s_q_max", s_q_max), ("s_kv_max", s_kv_max)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"sequence extents must be positive integers; got {name}={value!r}")
    cc = None if cc is None else tuple(cc)
    if mode == "off":
        return CtaPairResolution(mode, ONE_CTA_BODY, "cta_pair=off", s_q_max, s_kv_max, cc, None)
    failures = []
    if layout not in CTA_PAIR_LAYOUTS:
        failures.append(("layout", f"requires a layout in {CTA_PAIR_LAYOUTS} (the packed chain or its dense compact-BSHD twin)", repr(layout)))
    if cc not in CTA_PAIR_CCS:
        failures.append(
            (
                "cc",
                f"requires cc in {CTA_PAIR_CCS} (the admitted parts; the per-S threshold was measured on GB200, cc 10.0, and is carried per part)",
                "a device that is not CUDA" if cc is None else str(cc),
            )
        )
    elif mode == "auto" and min(s_q_max, s_kv_max) < cta_pair_min_seq(cc):  # the threshold exists for an admitted part only
        failures.append(("seq", f"both declared S_q and S_kv >= {cta_pair_min_seq(cc)}", f"({s_q_max}, {s_kv_max})"))
    admitted = None
    if not failures:
        pairs = _active_pairs(max_active_pairs)
        admitted = pairs >= 1
        if not admitted:
            failures.append(("cluster", f"requires a cluster of {CTA_PAIR_CLUSTER} CTAs admitted by the device's occupancy query", str(pairs)))
    if failures:
        name, text, got = failures[0]
        if mode == "on":
            raise CtaPairDecline(name, f"cta_pair 'on' {text}; got {got}")
        return CtaPairResolution(mode, ONE_CTA_BODY, f"cta_pair=auto: the one-CTA body ({name}: {text}; got {got})", s_q_max, s_kv_max, cc, admitted)
    if mode == "on":
        return CtaPairResolution(mode, PAIR_BODY, "cta_pair=on", s_q_max, s_kv_max, cc, admitted)
    reason = f"cta_pair=auto: the pair body (S_q {s_q_max}, S_kv {s_kv_max} >= {cta_pair_min_seq(cc)}; cc {cc}; cluster of {CTA_PAIR_CLUSTER} admitted)"
    return CtaPairResolution(mode, PAIR_BODY, reason, s_q_max, s_kv_max, cc, admitted)


def _align(value):
    return (value + WORKSPACE_ALIGNMENT - 1) // WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT


def validate_thd_bounds(batch, heads, kv_heads, sq, skv, tq, tkv, tq_phys, tkv_phys):
    """The grouped THD bounds plus the pair envelope: 256-row rounding and a grid x of pairs x CGA_M.  The dense twin passes its packed-view
    capacities ``tq = tq_phys = batch * sq`` and ``tkv = tkv_phys = batch * skv`` (``dense_capacities``)."""
    _validate_grouped_thd_bounds(batch, heads, kv_heads, sq, skv, tq, tkv, tq_phys, tkv_phys)
    if max(sq, skv) > CFG.INT32_ROUNDING_BOUND:
        raise ValueError("sequence envelope exceeds safe Int32 pair-unit rounding")
    if batch * heads * ((skv + UNIT_ROWS - 1) // UNIT_ROWS) * CFG.CGA_M > 2**31 - 1:
        raise ValueError("pair launch envelope exceeds the grid or Int32 work-id range")


def dense_capacities(batch, sq, skv):
    """The packed-view token capacities of a dense compact-BSHD declaration: ``(tq, tkv, tq_phys, tkv_phys)`` = (B*S_q, B*S_kv, B*S_q, B*S_kv)
    -- the dense bytes ARE the compact packed view, so the compact and physical capacities coincide (no gaps, no ragged offsets)."""
    if any(type(n) is not int or n <= 0 for n in (batch, sq, skv)):
        raise ValueError("dense grouped 2-CTA d128 capacities need positive integer batch and sequence extents")
    return batch * sq, batch * skv, batch * sq, batch * skv


def thd_workspace_offsets(batch, heads, tq, tkv, stats_ring=False):
    """Shared host/trace arithmetic; no value reads from device metadata.  Byte offsets from the 256-B aligned payload.  ``stats_ring``
    (``stats_form = "tma_ring"``): the compact delta region is replaced by the padded pre-scaled ``stats2`` region followed by the Int32
    ``rowoff`` [batch + 1] region (``None`` under the ldg form); every later offset follows.  Returns ``(rounded_q, rounded_k, delta_or_stats2,
    meta, origins, maps, dk, dv, regions_end, rowoff)``."""
    rounded_q = (tq + 127) // 128 * 128
    delta = 4 * rounded_q * heads * 128
    if stats_ring:
        rowoff = delta + _align(CFG.BPE_ACC * STATS_KINDS * stats2_pad_total(batch, rounded_q) * heads)
        meta = rowoff + _align(4 * (batch + 1))
    else:
        rowoff = None
        meta = delta + 4 * rounded_q * heads
    origins = meta + _align(4 * (4 * batch + 4))
    maps = origins + _align(8 * len(ORIGIN_PORTS) * batch)
    prefix_end = maps + MAPS_PER_SEQUENCE * batch * 128
    rounded_k = (tkv + 127) // 128 * 128
    dk = _align(prefix_end)
    size = rounded_k * heads * 128 * PARTIAL_ITEMSIZE
    dv = dk + _align(size)
    return rounded_q, rounded_k, delta, meta, origins, maps, dk, dv, dv + size, rowoff


def thd_workspace_layout(batch, heads, tq, tkv, stats_ring=False):
    """Compact FP32 dQ / Delta (or the padded pre-scaled stats region + the padded row offsets under ``stats_ring``), metadata, INT64 origins,
    nine per-sequence maps, native compact-KV partials."""
    if any(type(n) is not int or n <= 0 for n in (batch, heads, tq, tkv)):
        raise ValueError("grouped 2-CTA d128 packed workspace dimensions must be positive integers")
    rounded_q, rounded_k, delta, meta, origins, maps, dk, dv, total, rowoff = thd_workspace_offsets(batch, heads, tq, tkv, stats_ring)
    if total + WORKSPACE_SLACK > 2**63 - 1:
        raise ValueError("grouped 2-CTA d128 packed workspace exceeds the signed pointer range")
    kv_shape, kv_strides = (1, rounded_k, heads, 128), (rounded_k * heads * 128, heads * 128, 128, 1)
    if stats_ring:
        stats_shape, stats_strides = stats2_shape_strides(heads, stats2_pad_total(batch, rounded_q))
        stats_regions = ((delta, stats_shape, stats_strides), (rowoff, (batch + 1,), (1,)))
    else:
        stats_regions = ((delta, (1, heads, rounded_q), (heads * rounded_q, rounded_q, 1)),)
    regions = (
        (0, (1, rounded_q, heads, 128), (rounded_q * heads * 128, heads * 128, 128, 1)),
        *stats_regions,
        (meta, (4 * batch + 4,), (1,)),
        (origins, (len(ORIGIN_PORTS), batch), (batch, 1)),
        (maps, (MAPS_PER_SEQUENCE * batch * 16,), (1,)),
        (dk, kv_shape, kv_strides),
        (dv, kv_shape, kv_strides),
    )
    return regions, total + WORKSPACE_SLACK
