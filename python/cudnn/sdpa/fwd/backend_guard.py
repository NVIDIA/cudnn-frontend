# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graphs the SDPA-forward family keeps away from the cuDNN backend's planner
(``cudnn.engines.manifest.EngineFamily.backend_guard``).

One documented defect: on cc 10.7 the backend heuristics SIGSEGV while planning a block-scale MXFP8
(``sdpa_mxfp8``) forward graph with a single query row and no sink token -- measured on cuDNN 9.26.0.51 and
9.27.0.28 over the 21-contract trigger matrix of the detector named below: every d128 contract crashes (dense BSHD
and BHSD, THD (ragged) queries, Stats on or off, batch 1 / 2 / 4 and one head, KV length 128 / 2048 / 4096, no mask
or a causal top-left / bottom-right band, bf16 / f16 / E4M3 / E5M2 O, E4M3 and E5M2 inputs), while the d192x128 /
d256 / d512 contracts never reach the crash (the backend's engine-config step declines them with "No valid engine
configs" and the row plans alone).  Every head dim is treated alike, and a sink token makes the backend plan (three
plans).  The process
dies inside the C++ plan creation -- the ``create_execution_plans`` heuristics query, and equally the explicit
``create_execution_plan(engine_id, knobs)`` engine-config path when a backend record is replayed onto such a graph
(rc 139 before any check_support or build, 9.26.0.51) -- after the lowering, the C++ validate and
build_operation_graph all completed, so no exception can be caught: the FROST row serves these graphs, planning
records this reason as the backend's decline instead of asking, and an explicit backend pin raises it as a typed
decline (``_pygraph._refuse_guarded_backend``), below the first version shown not to crash.  Paged MXFP8 is
outside the domain on purpose: the backend's own C++ validate rejects paged MXFP8 before any heuristics run, and
its decline text must stay the one the user reads.

Import-light on purpose (no torch, no cutlass; ``cudnn`` is imported inside the function): planning consults it
for every SDPA-forward graph, twice.
"""

from __future__ import annotations

from typing import Optional

# First backend version shown NOT to crash, or None while every known build crashes.  Measured: 9.26.0.51 and
# 9.27.0.28 crash on 18 of the 21 contracts of the detector's trigger matrix (every d128 contract; the three larger
# head dims are declined by the backend's engine-config step before the crash -- 216-SM cc 10.7 board, 2026-10-08);
# cuDNN 9.28.0 plans cleanly: the detector's four sentinel contracts (dense / Stats / THD / BHSD at s_q == 1) all
# "PLANNED 3 plans" on the cc 10.7 CI lane (2026-10-08) and it failed with "record 92800", which is this value, and the
# detector -- which since walks the whole 21-contract matrix on every run of that lane -- is what keeps the record
# qualified on 9.28.0: a crash on any contract there fails the lane naming the contract, and the value goes back to
# None.  (9.28.0.16 could not be loaded on the measurement board itself: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED at
# the first descriptor finalize.)  Set it only from a measured, crash-free run of the trigger matrix -- never a guessed
# ceiling, which would re-arm the crash on the next build.
#
# Re-measure procedure (the constant is a measurement, so it is kept current by a detector rather than by memory):
# test/python/sdpa/graph/test_mhas_v2.py::test_sdpa_mxfp8_cc107_backend_planning_crash_guard_is_current_L0 walks, on a
# cc 10.7 device, the 21-contract trigger matrix (_P1_CRASH_MATRIX: dense BSHD / BHSD and THD at s_q == 1, Stats on
# and off, batch 1 / 2 / 4, KV 128 / 2048 / 4096, no mask / causal top-left / bottom-right, bf16 / f16 / E4M3 / E5M2 O,
# E4M3 and E5M2 inputs, d128 / d192x128 / d256 / d512; plus the sink contract as the control the backend plans) in a
# child interpreter with this constant set to 0 inside it, i.e. the guard disabled and the installed backend asked
# through the ordinary [A, FALLBACK] walk, restarting the child after every crash so every contract reports.  While
# this constant says the installed backend crashes, at least one contract must take its child down (rc 139); a backend
# that plans every contract cleanly FAILS the detector with the version to record here.  At or above a recorded
# version every contract must plan or decline cleanly, and the per-contract record is printed under "measured" in the
# run's terminal summary (the CI job log).  Record a version only after the whole matrix is clean on it, and leave the
# graphs the row serves on the row: this constant lifts the GUARD, not the placement verdict
# (``placement._place_sm107_mxfp8``).
SQ1_MXFP8_PLANNING_CRASH_FIXED_IN: Optional[int] = 92800


def backend_guard(graph, facts) -> Optional[str]:
    """The reason the backend must not be consulted for ``graph`` (its SDPA facts), or None."""
    import cudnn

    if facts is None or facts.invalid is not None or facts.is_backward or not facts.is_mxfp8:
        return None
    if facts.device_cc != (10, 7) or facts.s_q != 1 or facts.has_sink or facts.has_paged_kv:
        return None
    if SQ1_MXFP8_PLANNING_CRASH_FIXED_IN is not None and cudnn.backend_version() >= SQ1_MXFP8_PLANNING_CRASH_FIXED_IN:
        return None
    return (
        f"cuDNN {cudnn.backend_version_string()} backend heuristics crash while planning single-query MXFP8 SDPA graphs "
        f"on cc 10.7 without a sink token (measured on 9.26.0.51 and 9.27.0.28, clean from 9.28.0: dense and THD, Stats on or off); "
        f"the backend is not consulted for this graph"
    )
