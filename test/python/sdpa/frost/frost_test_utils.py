# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-condition markers for the FROST SDPA suites.

One gate for the suite, aligned with what the ENGINES declare
(``Capabilities.sm_lo``/``sm_hi`` in sdpa/fwd/engines.py) rather than
re-derived per file. Five files each carried their own copy pinned to exactly
(10, 0), so every one skipped on sm103 -- and would have on Rubin and Thor --
while the engines they test serve the whole line.
"""

import pytest


def _active_sm():
    import torch

    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


_SM = _active_sm()

requires_blackwell = pytest.mark.skipif(
    _SM is None or not (100 <= _SM <= 119),
    reason="needs an SM100-line GPU (100 <= SM <= 119), have " + ("none" if _SM is None else f"sm_{_SM}"),
)
# Pre-Rubin gate for the suites whose lowerings do not exist on the Rubin
# line (f16/bf16 and MXFP8 SM100 paths; Rubin serves per-tensor FP8 only) —
# these must SKIP on cc10.7 so the Rubin CI lane can run the whole frost
# directory (the lane's FROST_TEST_PATHS note asks exactly for this).
requires_pre_rubin_blackwell = pytest.mark.skipif(
    _SM is None or not (100 <= _SM <= 106),
    reason="needs a pre-Rubin SM100-line GPU (100 <= SM <= 106; no f16/MXFP8 Rubin lowering), have " + ("none" if _SM is None else f"sm_{_SM}"),
)
requires_blackwell_geforce = pytest.mark.skipif(
    _SM is None or not (120 <= _SM <= 129),
    reason="needs an SM120-line GPU, have " + ("none" if _SM is None else f"sm_{_SM}"),
)


def _dsl_usable():
    """``(usable, why_not)`` for the DSL these engines lower through.

    Version too, not just presence: the extra is deliberately NOT pinned to the
    floor (that would make cudnn-frontend incompatible with anything holding the
    DSL back -- quack-kernels pins ==4.6.0), so an environment can legitimately
    have an older one. The engines decline it; these tests must skip rather than
    fail, and for the same reason.
    """
    from cudnn.frost.buffers import CUTEDSL_MIN_VERSION, cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed:
        return False, "needs the cutedsl extra (nvidia-cutlass-dsl)"
    if cutedsl_too_old(version):
        want = ".".join(str(v) for v in CUTEDSL_MIN_VERSION)
        return False, f"needs nvidia-cutlass-dsl >= {want}, have {version[1]}"
    return True, ""


_DSL_OK, _DSL_WHY = _dsl_usable()
requires_dsl = pytest.mark.skipif(not _DSL_OK, reason=_DSL_WHY or "cutedsl available")


def _dsl_installed() -> bool:
    """For the few call sites that gate inside a test body rather than on it."""
    return _DSL_OK


def _is_plan_for(plan_name, engine) -> bool:
    """A plan reads ``<engine>[<knobs>]``: the heuristics name a concrete config
    for every entry, so match on the engine, not on the whole plan name."""
    return plan_name == engine or plan_name.startswith(engine + "[")


def select_engine(graph, name, tiles=None, pack_gqa=None):
    """Pin the ranked entry for engine ``name`` (graph.plans holds the backend's
    plans and the python engines' in one list). A pin is strict: check_support /
    build_plans raise if that engine declines the graph.

    The FIRST entry for that engine is the heuristics' own best guess for this
    shape. ``tiles`` / ``pack_gqa`` pin a different one, so a test can run a
    config the best guess would not choose. Filters match the STRUCTURED knobs,
    not the rendered plan name: substring matching a name would let a request
    for tile_n=128 select a tile_n=1280 plan, and the test would pass having
    run something else. Every filter — and each ``tiles`` component — is
    None-transparent (matches any value), so a test can pin just kv_tile
    (the auto-tile_m graph_api cases) or nothing at all (the best guess).
    """
    names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    want_m, want_n = tiles if tiles is not None else (None, None)

    def _wanted(i):
        if not _is_plan_for(names[i], name):
            return False
        return all(
            want is None or getattr(graph.plans[i].knobs, field, None) == want
            for field, want in (("tile_m", want_m), ("tile_n", want_n), ("pack_gqa", pack_gqa))
        )

    index = next((i for i in range(len(names)) if _wanted(i)), None)
    assert index is not None, f"no plan for engine {name!r} with tiles={tiles} pack_gqa={pack_gqa}; plans={names}"
    graph.select_plan(index)
    return graph


def offers_engine(graph, name) -> bool:
    """Whether any ranked entry is a plan for engine ``name``."""
    return any(_is_plan_for(graph.get_plan_name_at_index(i), name) for i in range(len(graph.plans)))


def make_dense_stats(batch: int, heads: int, sequence: int, layout: str):
    """Allocate dense Stats in compact or permuted-and-gapped storage."""
    import torch

    if layout == "contiguous":
        return torch.empty(batch, heads, sequence, 1, dtype=torch.float32, device="cuda")
    if layout == "strided":
        storage = torch.empty(sequence + 7, heads + 2, batch, dtype=torch.float32, device="cuda")
        stats = storage.permute(2, 1, 0)[:, :heads, :sequence].unsqueeze(-1)
        assert not stats.is_contiguous()
        return stats
    raise ValueError(f"unknown dense Stats layout {layout!r}")
