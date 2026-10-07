# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Canonical band-mask model shared by the SDPA graph analyzer and the engines.

Two objects, deliberately distinct:

``BandFacts``
    the band a GRAPH asks for, normalized out of
    :class:`cudnn.sdpa.graph_analyzer.SdpaGraphFacts`
    (:meth:`BandFacts.from_sdpa_facts`, also reachable as ``facts.band``).
``BandSupport``
    the band SET an ENGINE claims it can serve — the engine-side claim that
    lives on each ``Capabilities`` row as its appended ``band`` field.

The band is three independent axes plus an anchor, in this repository's own
vocabulary (no new window semantics are invented here):

============================ ==================================================
left bound                   absent, or a sliding window — the analyzer's
                             ``window_left`` OFFSET (cuDNN length - 1, so API
                             length 1 is offset 0)
right bound                  unbounded / causal (``right_bound == 0``, i.e.
                             ``use_causal_mask``) / a finite right band
                             (``right_bound > 0``, ``right_band_widening``)
anchor                       top-left, or bottom-right (the diagonal shifted by
                             ``S_kv - S_q``, ``diagonal_alignment.BOTTOM_RIGHT``)
============================ ==================================================

plus ONE real combination constraint: an anchor is a property *of a diagonal*,
so a bottom-right anchor with no right bound at all is incoherent (only a left
window present is not a diagonal).  That is the existing decline
``REASON_ANCHOR_WITHOUT_DIAGONAL`` — a graph-side constraint, not a capability,
and it stays expressible here rather than being folded away.

Why the model exists (#601).  The four legacy capability flags (``causal`` /
``bottom_right`` / ``swa`` / ``right_band_widening``) can only *widen* a row:
each defaults to False and a True is a yes.  They cannot express a RESTRICTED
row — "this kernel serves unmasked graphs only", which the coming SM89 d64 row
needs, or "causal but not bottom-right / SWA / right-band widening", which
``sdpa_bwd_sm100_mxfp8`` already is.  The flags stay the declaration spelling
of every pre-existing row; :meth:`BandSupport.from_legacy_flags` is the single
normalization layer that maps them, and the probes read the model, never the
flags.  A row may therefore declare ``band`` ALONE: only the legacy flags that
are actually SET constrain it, and each of those must be inside the declared
band (a set flag outside it raises at construction, see
``Capabilities.__post_init__`` in ``fwd/engines.py`` / ``bwd/engines.py`` —
never silently resolved by field order).  The all-default legacy block is the
spelling of "nothing declared", so it leaves an explicit claim alone.

This module is import-light on purpose (typing only): the engines' probes keep
working in a process that never imports the CuTe DSL.

Cost (measured, host-side, #601): normalizing facts into :class:`BandFacts` and
asking the three axis questions costs ~1.5 us per probe that reaches the band
block, against ~0.2 us for the four raw boolean reads it replaces (repeated CPU
probes on the L20 host, one fresh process per arm); the probe builds it only
after the arch / dtype / shape declines, and ``BandSupport`` itself is resolved
once at import.  That is a representation cost, not a speed win -- the value
here is that a RESTRICTED claim is expressible at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

# --- right-bound modes ------------------------------------------------------
RIGHT_UNBOUNDED = "unbounded"
RIGHT_CAUSAL = "causal"
RIGHT_FINITE = "finite_right"
RIGHT_MODES = frozenset({RIGHT_UNBOUNDED, RIGHT_CAUSAL, RIGHT_FINITE})

# --- left-bound modes -------------------------------------------------------
LEFT_NONE = "no_left_bound"
LEFT_WINDOW = "left_window"
LEFT_MODES = frozenset({LEFT_NONE, LEFT_WINDOW})

# --- diagonal anchor --------------------------------------------------------
ANCHOR_TOP_LEFT = "top_left"
ANCHOR_BOTTOM_RIGHT = "bottom_right"
ANCHORS = frozenset({ANCHOR_TOP_LEFT, ANCHOR_BOTTOM_RIGHT})

# --- decline vocabulary -----------------------------------------------------
# The probe reports the FIRST failing check as text; these labels are what the
# feature loop turns into "graph uses <label>, which this engine does not
# support".  Kept here so the model's own :meth:`BandSupport.decline` and the
# probes' spliced-in checks cannot drift into two vocabularies.
LABEL_RIGHT_FINITE = "causal right-band widening"
LABEL_RIGHT_CAUSAL = "causal mask"
LABEL_LEFT_WINDOW = "sliding window"
LABEL_ANCHOR = "bottom-right causal"
# The mirror labels: every axis is a SET on both sides, so a row that claims only
# part of an axis has to decline the part it does not claim -- including the mode
# a graph asks for by NOT setting a flag (unbounded right band, no left bound,
# top-left anchor).  Without these the absence of a graph flag would bypass a
# restricted claim.
LABEL_RIGHT_UNBOUNDED = "unmasked band"
LABEL_LEFT_NONE = "no left bound"
LABEL_ANCHOR_TOP_LEFT = "top-left causal"


def feature_reason(label: str) -> str:
    """The probe's decline sentence for a feature label."""
    return f"graph uses {label}, which this engine does not support"


LABEL_RIGHT = {
    RIGHT_UNBOUNDED: LABEL_RIGHT_UNBOUNDED,
    RIGHT_CAUSAL: LABEL_RIGHT_CAUSAL,
    RIGHT_FINITE: LABEL_RIGHT_FINITE,
}
LABEL_LEFT = {
    LEFT_NONE: LABEL_LEFT_NONE,
    LEFT_WINDOW: LABEL_LEFT_WINDOW,
}
REASON_RIGHT = {mode: feature_reason(label) for mode, label in LABEL_RIGHT.items()}
REASON_LEFT = {mode: feature_reason(label) for mode, label in LABEL_LEFT.items()}
REASON_LEFT_WINDOW = REASON_LEFT[LEFT_WINDOW]
REASON_ANCHOR_WITHOUT_DIAGONAL = "bottom-right alignment requires a causal upper bound (plain or right-widened)"


def anchor_reason(noun: str = "kernel", *, anchor: str = ANCHOR_BOTTOM_RIGHT) -> str:
    """The anchor decline sentence, for either anchor.

    The two probes have always worded the same decline differently -- the
    forward probe says "kernel", the backward probe "engine" -- and the string
    is user-visible, so the model keeps BOTH addressable through one definition
    instead of silently rewording one pass during the migration.  The top-left
    wording is the mirror case: a row that claims ONLY the bottom-right anchor
    declines a top-left graph instead of serving it by omission.
    """
    label = LABEL_ANCHOR_TOP_LEFT if anchor == ANCHOR_TOP_LEFT else LABEL_ANCHOR
    return f"graph uses {label}, which this {noun} does not support"


REASON_ANCHOR = anchor_reason()  # canonical wording (the forward probe's)


@dataclass(frozen=True, slots=True)
class BandFacts:
    """The band one graph asks for.

    ``right_bound`` keeps the analyzer's RAW resolved value (``0`` for causal,
    ``None`` when there is no right bound at all); the MODE is what decides
    semantics.  ``left_window`` is the offset, i.e. cuDNN length - 1.
    """

    right_mode: str = RIGHT_UNBOUNDED
    right_bound: Optional[int] = None
    left_mode: str = LEFT_NONE
    left_window: Optional[int] = None
    anchor: str = ANCHOR_TOP_LEFT

    # ---- construction ------------------------------------------------------
    @classmethod
    def from_sdpa_facts(cls, facts: "SdpaGraphFacts") -> "BandFacts":
        """Normalize the analyzer's resolved mask facts into the canonical band.

        Deliberately total (never raises): facts are descriptive and every
        engine must be able to decline them, so an odd combination is carried
        as-is rather than becoming an exception on the probe path.
        """
        if getattr(facts, "right_band_widening", False):
            right_mode, right_bound = RIGHT_FINITE, getattr(facts, "right_bound", None)
        elif getattr(facts, "causal", False):
            right_mode, right_bound = RIGHT_CAUSAL, getattr(facts, "right_bound", None)
        else:
            right_mode, right_bound = RIGHT_UNBOUNDED, None
        window_left = getattr(facts, "window_left", None)
        if window_left is None:
            left_mode = LEFT_NONE
        else:
            left_mode = LEFT_WINDOW
        anchor = ANCHOR_BOTTOM_RIGHT if getattr(facts, "bottom_right", False) else ANCHOR_TOP_LEFT
        # Positional: this is the probe's per-call path, and the keyword form
        # measurably costs more (the fields are the class's own order).
        return cls(right_mode, right_bound, left_mode, window_left, anchor)

    # ---- derived predicates ------------------------------------------------
    @property
    def has_diagonal(self) -> bool:
        """True when a causal upper bound exists (plain or right-widened)."""
        return self.right_mode in (RIGHT_CAUSAL, RIGHT_FINITE)

    @property
    def has_left_window(self) -> bool:
        return self.left_mode == LEFT_WINDOW

    @property
    def right_widening(self) -> int:
        """The right band's width R (0 for a plain causal diagonal)."""
        return int(self.right_bound or 0)

    @property
    def left_window_length(self) -> Optional[int]:
        """The sliding window in the API's spelling (offset + 1); None if absent."""
        return None if self.left_window is None else int(self.left_window) + 1

    def covers_kv_tail(self, s_q: int, s_kv: int) -> bool:
        """True when the band provably masks every KV column >= ``s_kv``.

        The last unmasked column is ``(S_q - 1) + R`` top-left or
        ``(S_kv - 1) + R`` bottom-right (R = the right band's width, 0 for
        plain causal), so a ragged KV tail cannot leak into the softmax.  This
        is a graph x geometry combination, which is exactly why it lives on the
        model instead of being re-derived at each call site.
        """
        if not self.has_diagonal:
            return False
        r = self.right_widening
        if self.anchor == ANCHOR_BOTTOM_RIGHT:
            return r == 0
        return s_q + r <= s_kv

    def as_dict(self) -> dict:
        """Stable, JSON-friendly view (used by tests and evidence tooling)."""
        return {
            "right_mode": self.right_mode,
            "right_bound": self.right_bound,
            "left_mode": self.left_mode,
            "left_window": self.left_window,
            "anchor": self.anchor,
        }


@dataclass(frozen=True, slots=True)
class BandSupport:
    """The band set one ENGINE can serve.

    ``right`` / ``left`` / ``anchors`` are SETS of the vocabulary above, which
    is what makes a restricted row expressible.  No axis accepts an empty set, so
    the DEFAULT ``BandSupport()`` is not "no band" but the NARROWEST claim —
    unbounded / no left bound / top-left, i.e. exactly ``unmasked_only()`` — and
    the convenience constructors below name the shapes rows actually take.
    """

    right: frozenset = frozenset({RIGHT_UNBOUNDED})
    left: frozenset = frozenset({LEFT_NONE})
    anchors: frozenset = frozenset({ANCHOR_TOP_LEFT})

    def __post_init__(self) -> None:
        for label, domain, values in (
            ("right", RIGHT_MODES, self.right),
            ("left", LEFT_MODES, self.left),
            ("anchors", ANCHORS, self.anchors),
        ):
            unknown = set(values) - domain
            if unknown:
                raise ValueError(f"BandSupport.{label} has unknown members {sorted(unknown)}; known: {sorted(domain)}")
            if not values:
                raise ValueError(f"BandSupport.{label} is empty; a row must claim at least one {label} member")

    # ---- the three declaration shapes --------------------------------------
    @classmethod
    def unmasked_only(cls) -> "BandSupport":
        """Unmasked (unbounded, top-left) graphs only — the restricted row shape."""
        return cls()

    @classmethod
    def causal_and_unmasked(cls) -> "BandSupport":
        """Unmasked plus a plain causal diagonal, top-left: ``sdpa_bwd_sm100_mxfp8``."""
        return cls(right=frozenset({RIGHT_UNBOUNDED, RIGHT_CAUSAL}))

    @classmethod
    def full_masks(cls) -> "BandSupport":
        """Every axis, both anchors — what the long-standing rows declare."""
        return cls(right=RIGHT_MODES, left=LEFT_MODES, anchors=ANCHORS)

    @classmethod
    def from_legacy_flags(
        cls,
        *,
        causal: bool = False,
        right_band_widening: bool = False,
        swa: bool = False,
        bottom_right: bool = False,
    ) -> "BandSupport":
        """THE normalization layer from the pre-model capability flags.

        Each axis widens independently, exactly as the flags did:
        ``causal`` adds the causal right mode, ``right_band_widening`` the
        finite one, ``swa`` the left window, ``bottom_right`` the bottom-right
        anchor.  Unbounded / no-left-bound / top-left are always served, which
        is what makes the mapping of a row with no flags at all the
        unmasked-only claim rather than an empty one.
        """
        right = {RIGHT_UNBOUNDED}
        if causal:
            right.add(RIGHT_CAUSAL)
        if right_band_widening:
            right.add(RIGHT_FINITE)
        left = {LEFT_NONE, LEFT_WINDOW} if swa else {LEFT_NONE}
        anchors = {ANCHOR_TOP_LEFT, ANCHOR_BOTTOM_RIGHT} if bottom_right else {ANCHOR_TOP_LEFT}
        return cls(right=frozenset(right), left=frozenset(left), anchors=frozenset(anchors))

    # ---- decisions ---------------------------------------------------------
    def serves_right_mode(self, mode: str) -> bool:
        return mode in self.right

    def serves_left_mode(self, mode: str) -> bool:
        return mode in self.left

    def serves_anchor(self, band: BandFacts) -> bool:
        """Whether this row serves the band's ANCHOR (coherence is checked separately)."""
        return band.anchor in self.anchors

    def decline(self, band: BandFacts) -> Optional[str]:
        """The canonical first reason this row cannot serve ``band``, or None.

        Order: right mode, left bound, coherence of the anchor, anchor support.
        The probes ask the same predicates in their own historical order (the
        non-band features interleave), so this is the band-only view of the
        decision, not a second implementation of it.
        """
        if not self.serves_right_mode(band.right_mode):
            return REASON_RIGHT[band.right_mode]
        if not self.serves_left_mode(band.left_mode):
            return REASON_LEFT[band.left_mode]
        if band.anchor == ANCHOR_BOTTOM_RIGHT and not band.has_diagonal:
            return REASON_ANCHOR_WITHOUT_DIAGONAL
        if not self.serves_anchor(band):
            return anchor_reason(anchor=band.anchor)
        return None

    def serves(self, band: BandFacts) -> bool:
        return self.decline(band) is None

    def as_dict(self) -> dict:
        return {"right": sorted(self.right), "left": sorted(self.left), "anchors": sorted(self.anchors)}
