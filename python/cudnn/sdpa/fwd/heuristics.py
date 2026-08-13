# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the SDPA-forward family ranks plans for a graph.

``engines/heuristics.rank`` hands over the parsed facts, this family's engine
ids, and the backend's entries tagged by mode; what :func:`recommend` returns is
``graph.plans``, position for position. The comparison is here because a cell
cannot see its siblings and neither side of the FROST/backend split can place
the other.

Per mode:

- **A** — candidates worth running, best guess first, runners-up behind it for
  a caller that autotunes.
- **FALLBACK** — the config expected to build where mode A's choice may not.
  Nothing here is chosen for speed.
- **OPENSOURCE** — mode A without the backend's recommendation, since these
  cells ARE the open-source implementation.
- **B** — answered as A: it asks for a wider search this family has none to
  give.

To add a rule for a cell: write the function (:func:`_sm120_tiles` is the worked
example), list the cell in ``_TILE_RULE_CELLS``, put the measurement in the
commit. A cell absent from that set runs its row's sole point per axis, which is
the honest answer while nobody has timed it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cudnn

from cudnn.engines.base import PlanConfig
from cudnn.sdpa.fwd.config_sm120 import SMEM_CAPACITY_BYTES, smem_bytes
from cudnn.sdpa.fwd.engines import ENGINE_SPECS, Capabilities, SdpaFwdKnobs, mismatch

# Cells timed against the backend's kernel and found SLOWER, so the backend's
# mode-A entries lead them. Empty: absent means faster OR never timed, and both
# keep the order this dispatch has always had. Moving a cell in needs a
# measurement.
_MEASURED_BEHIND: frozenset = frozenset()

# Cells whose (tile_m, tile_n) choice _sm120_tiles makes.
_TILE_RULE_CELLS = frozenset({"sdpa_fwd_prefill_sm120", "sdpa_fwd_prefill_sm120_fp8"})


def _sm120_tiles(caps: Capabilities, facts) -> Tuple[int, int]:
    """(tile_m, tile_n) for the SM120 SDPA-forward prefill cell.

    ``tile_m=64`` when the grid cannot fill the machine AND each CTA has enough
    KV tiles to amortize the extra Q-tile loop; a causal mask counts as a
    halved grid because it halves the work per CTA. ``tile_n`` is the largest
    that fits SMEM: 128 is fastest, but a wide head has no room for it (D>=208
    in half, further out in FP8 -- its KV tile is a byte per element).

    Fit is a KERNEL property, so ``config_sm120.smem_bytes`` is the one
    implementation and the adapter's check calls it too. So is the tuning: an
    earlier revision of this template staged P through SMEM and wanted
    ``tile_n=64``. Re-measure when a kernel changes -- the sweeps are in PR #528
    (f16: regret 1.009 geomean, 1.054 worst) and PR #509 (fp8: 1.0046 geomean,
    1.039 worst over 30 seeded cells).

    Read those worst cases with care. Most cells here are within the ~1%
    run-to-run floor of each other, so a single sweep's worst cell is often
    noise: an unseeded run of the SAME code reported 1.155 at B1xH16xS2048
    causal, which the seeded repeat shows as a tie. What survives repetition is
    that the misses cluster on CAUSAL shapes, where the triangular mask shifts
    the per-CTA balance in a way `grid` alone does not capture.
    """
    sm_count = facts.device_sm_count or 0
    grid = -(-facts.s_q // 128) * facts.h_q * facts.b
    if facts.causal:
        grid //= 2
    kv_tiles = -(-facts.s_kv // 128)
    fine = sm_count > 0 and (grid * 2 <= sm_count or (grid * 2 <= 3 * sm_count and kv_tiles >= 12))
    tile_m = 64 if fine else 128
    # FP8 stages a byte per KV element but still writes O in half, so the two
    # SMEM terms size differently -- see config_sm120.smem_bytes.
    qkv_itemsize, o_itemsize = (1, 2) if facts.is_fp8 else (2, 2)
    fits = [n for n in sorted(caps.tile_ns, reverse=True) if smem_bytes(facts.d_qk, facts.d_v, tile_m, n, qkv_itemsize, o_itemsize) <= SMEM_CAPACITY_BYTES]
    return tile_m, (fits[0] if fits else min(caps.tile_ns))


def _sole(values):
    """The only value on an axis, or None where the row declares no domain."""
    return next(iter(values)) if len(values) == 1 else None


def _knobs(caps: Capabilities, tile_m, tile_n) -> SdpaFwdKnobs:
    """A knob request for one point. A field is None ONLY where the capability
    row declares no domain for that axis — never "engine, pick for me", which
    is how the same choice ended up being made here and again in the adapter."""
    return SdpaFwdKnobs(sched_policy=_sole(caps.sched_policies), tile_m=tile_m, tile_n=tile_n, cga=_sole(caps.cgas))


def _eligible(facts, offered: Dict[str, int]):
    """(engine_id, spec) for each offered cell whose capability row admits ``facts``."""
    for spec in ENGINE_SPECS:
        engine_id = offered.get(spec.name)
        if engine_id is not None and mismatch(spec.capabilities, facts, None) is None:
            yield engine_id, spec


def _admissible(caps: Capabilities, facts, knobs: SdpaFwdKnobs) -> bool:
    return mismatch(caps, facts, knobs) is None


def _mode_a(facts, offered: Dict[str, int], mode) -> List[PlanConfig]:
    """Candidates worth timing, best guess first."""
    out = []
    for engine_id, spec in _eligible(facts, offered):
        caps = spec.capabilities
        if spec.name not in _TILE_RULE_CELLS:
            # No rule measured for this cell: its capability row has one point
            # per axis, so there is nothing to choose between anyway.
            knobs = _knobs(caps, _sole(caps.tile_ms), _sole(caps.tile_ns))
            if _admissible(caps, facts, knobs):
                out.append(PlanConfig(engine_id, knobs, mode=mode))
            continue
        best = _sm120_tiles(caps, facts)
        # The guess first, then the rest of the domain as autotune candidates:
        # the rule's regret is small but not zero, so the runners-up are worth
        # offering to a caller who measures. Configs the kernel cannot fit are
        # not runners-up -- they would sit in the list only to decline at build.
        qkv_itemsize, o_itemsize = (1, 2) if facts.is_fp8 else (2, 2)
        domain = [
            (m, n) for m in caps.tile_ms for n in caps.tile_ns if smem_bytes(facts.d_qk, facts.d_v, m, n, qkv_itemsize, o_itemsize) <= SMEM_CAPACITY_BYTES
        ]
        ordered = sorted(domain or [best], key=lambda mn: (mn != best, mn[1] != best[1], -mn[0]))
        for tile_m, tile_n in ordered:
            knobs = _knobs(caps, tile_m, tile_n)
            if _admissible(caps, facts, knobs):
                out.append(PlanConfig(engine_id, knobs, mode=mode))
    return out


def _mode_fallback(facts, offered: Dict[str, int]) -> List[PlanConfig]:
    """Configs expected to build where mode A's choice may not.

    TODO: today this is the smallest tile the row admits — the config that asks
    least of the device, which is the one thing a fallback must be. Once a cell
    has features its largest tiles cannot serve, this becomes the handful of
    configs that between them cover the whole plane, chosen from measurements.
    """
    out = []
    for engine_id, spec in _eligible(facts, offered):
        caps = spec.capabilities
        knobs = _knobs(caps, min(caps.tile_ms, default=None), min(caps.tile_ns, default=None))
        if _admissible(caps, facts, knobs):
            out.append(PlanConfig(engine_id, knobs, mode=cudnn.heur_mode.FALLBACK))
    return out


def _leads(offered: Dict[str, int], plans: List[PlanConfig]) -> bool:
    """Whether this family's mode-A plans outrank the backend's. See _MEASURED_BEHIND."""
    behind = {offered[name] for name in _MEASURED_BEHIND if name in offered}
    return bool(plans) and not all(cfg.engine_id in behind for cfg in plans)


def recommend(modes: List[Any], facts, offered: Dict[str, int], backend_plans: List[PlanConfig]) -> List[PlanConfig]:
    """The ranked plan list for this graph, mode by mode in the caller's order.

    Each mode contributes a block and the blocks concatenate, so asking for
    ``[A, FALLBACK]`` puts every tuned candidate — both sides' — ahead of every
    fallback. A plan repeated across modes keeps its first position: building
    the same config twice only costs the caller a JIT compile.
    """
    # An untagged backend entry is the delegating one: OSS candidates C++ holds
    # but never exposes as plans, so it cannot be enumerated. It belongs to no
    # mode, and it is NOT a pure OSS entry -- Graph::build_plans tries the OSS
    # engine and, if that one declines, falls through to the native
    # engine_configs already enqueued. So it leads the BACKEND's entries but not
    # ours: ahead of our OPENSOURCE block it would answer an OSS-coverage
    # question with a native kernel.
    delegating = [c for c in backend_plans if c.mode is None]
    out: List[PlanConfig] = []
    for mode in modes:
        if mode == cudnn.heur_mode.OPENSOURCE:
            out += _mode_a(facts, offered, cudnn.heur_mode.A) + delegating
        elif mode in (cudnn.heur_mode.A, cudnn.heur_mode.B):
            ours = _mode_a(facts, offered, mode)
            theirs = delegating + [c for c in backend_plans if c.mode == mode]
            out += (ours + theirs) if _leads(offered, ours) else (theirs + ours)
        elif mode == cudnn.heur_mode.FALLBACK:
            out += _mode_fallback(facts, offered) + delegating + [c for c in backend_plans if c.mode == mode]
    # A delegate with no mode asked for it (the backend has engines but exposed
    # no plans) would otherwise be dropped.
    out += delegating

    # Identity is (engine, knobs). cpp_index is only WHERE one backend query put
    # a plan, so keying on it would let [A, A], or one config both modes return,
    # through as two entries -- and an autotuner would build and time it twice.
    seen, ranked = set(), []
    for cfg in out:
        key = (cfg.engine_id, repr(cfg.knobs))
        if key not in seen:
            seen.add(key)
            ranked.append(cfg)
    return ranked
