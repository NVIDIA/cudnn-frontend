# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graphs the SDPA-forward family keeps away from the cuDNN backend's planner
(``cudnn.engines.manifest.EngineFamily.backend_guard``).

One documented defect: on cc 10.7 the backend heuristics SIGSEGV while planning a block-scale MXFP8
(``sdpa_mxfp8``) forward graph with a single query row and no sink token -- measured on cuDNN 9.26.0.51 and
9.27.0.28 for batch 1 and 4, KV length 128..2048, BSHD and BHSD, dense and THD (ragged) queries, causal or not,
Stats on or off, f16 / bf16 / E4M3 O, E4M3 and E5M2 inputs.  Every head dim is treated alike (d128 crashes;
d192x128 / d256 / d512 decline cleanly), and a sink token makes the backend plan (three plans).  The process
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

# First backend version shown NOT to crash, or None while every known build crashes.  Newest build checked:
# 9.27.0.28 crashes; 9.28.0.16 could not be loaded on the measurement board (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
# at the first descriptor finalize), so no crash-free build is known.  Set it only from a measured, crash-free
# run of the trigger matrix (one process per contract: s_q == 1, no sink, dense and THD, Stats on and off) --
# never a guessed ceiling, which would re-arm the crash on the next build.
#
# Re-measure procedure (the constant is a measurement, so it is kept current by a detector rather than by memory):
# test/python/sdpa/graph/test_mhas_v2.py::test_sdpa_mxfp8_cc107_backend_planning_crash_guard_is_current_L0 runs, on a
# cc 10.7 device, the trigger matrix (dense / Stats / THD / BHSD at s_q == 1, plus the sink contract as the control the
# backend plans) one interpreter per contract with this constant set to 0 inside it, i.e. the guard disabled and the
# installed backend asked through the ordinary [A, FALLBACK] walk.  While this constant says the installed backend
# crashes, at least one contract must take its process down (rc 139); a backend that plans every contract cleanly FAILS
# the detector with the version to record here.  At or above a recorded version every contract must plan or decline
# cleanly.  Record a version only after the whole matrix is clean on it (dense and THD, Stats on and off, BSHD and BHSD,
# every O dtype, E4M3 and E5M2 inputs), and leave the graphs the row serves on the row: this constant lifts the GUARD,
# not the placement verdict (``placement._place_sm107_mxfp8``).
SQ1_MXFP8_PLANNING_CRASH_FIXED_IN: Optional[int] = None


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
        f"on cc 10.7 without a sink token (measured on 9.26.0.51 and 9.27.0.28: dense and THD, Stats on or off); "
        f"the backend is not consulted for this graph"
    )
