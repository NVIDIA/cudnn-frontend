# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graphs the SDPA-forward family keeps away from the cuDNN backend's planner
(``cudnn.engines.manifest.EngineFamily.backend_guard``).

One documented defect: on cc 10.7 the backend heuristics SIGSEGV while planning a block-scale MXFP8
(``sdpa_mxfp8``) forward graph with a single query row and no sink token -- measured on cuDNN 9.26.0.51 and
9.27.0.28 for batch 1 and 4, KV length 128..2048, BSHD and BHSD, dense and THD (ragged) queries, causal or not,
Stats on or off, f16 / bf16 / E4M3 O, E4M3 and E5M2 inputs.  Every head dim is treated alike (d128 crashes;
d192x128 / d256 / d512 decline cleanly), and a sink token makes the backend plan (three plans).  The process
dies inside the C++ ``create_execution_plans`` heuristics query -- after the lowering, the C++ validate and
build_operation_graph all completed -- so no exception can be caught: the FROST row serves these graphs and the
backend is simply not consulted below the first version shown not to crash.  Paged MXFP8 is outside the domain
on purpose: the backend's own C++ validate rejects paged MXFP8 before any heuristics run, and its decline text
must stay the one the user reads.

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
