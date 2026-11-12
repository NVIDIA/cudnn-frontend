# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graphs the SDPA-forward family keeps away from the cuDNN backend's planner
(``cudnn.engines.manifest.EngineFamily.backend_guard``), and backend engines it keeps off a graph
(``EngineFamily.backend_engine_guard``).

One documented defect: on cc 10.7 the backend kills the process while planning or building a block-scale MXFP8
(``sdpa_mxfp8``) forward graph with a single query row and no sink token.  Measured on cuDNN 9.26.0.51 and 9.27.0.28
over the 21-contract trigger matrix of the detector named below: the heuristics SIGSEGV on every d128 contract (dense
BSHD and BHSD, THD (ragged) queries, Stats on or off, batch 1 / 2 / 4 and one head, KV length 128 / 2048 / 4096, no
mask or a causal top-left / bottom-right band, bf16 / f16 / E4M3 / E5M2 O, E4M3 and E5M2 inputs), while the d192x128 /
d256 / d512 contracts never reach the crash (the backend's engine-config step declines them with "No valid engine
configs" and the row plans alone).  On cuDNN 9.28.0 (the cc 10.7 CI lane) the heuristics plan every contract, and
building the backend's plan for the BHSD single-query graph killed the test process instead -- the crash moved from
the first stage of the backend's lifecycle to the last, so no build is clean yet.  Every head dim is treated alike,
and a sink token makes the backend plan and build (three plans).  The process
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

from typing import Dict, Optional, Tuple

# First backend version shown NOT to crash at any stage, measured per compute capability. Unmeasured architectures
# remain guarded; one architecture's result must not lift the guard on another. Historically, 9.26.0.51 and 9.27.0.28
# crash in the heuristics on 18 of the 21 contracts of the detector's trigger matrix (every d128 contract; the three
# larger head dims are declined by the backend's engine-config step before the crash -- 216-SM cc 10.7 board,
# 2026-10-08). cuDNN 9.28.0 is NOT clean either: on the cc 10.7 CI lane its heuristics planned all 21 contracts (a first
# detector revision that stopped at planning therefore reported "record 92800"), and the same run lost its xdist worker
# to a process crash while check_support / build_plans ran the backend's plan for the BHSD single-query graph (job
# 476641257, 2026-10-08). 9.28.0.16 could not be loaded on the measurement board itself:
# CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED at the first descriptor finalize.
# cc 10.0: 9.30.0 (93000), measured on GB100 with cuDNN 9.30.0.4 (job 543389412, 2026-11-11). All 18 applicable
# contracts built; the three larger-head contracts declined with no engine, and none crashed.
# cc 10.3 and cc 10.7: no clean complete measurement yet. The cc 10.7 run on 9.30.0.4 timed out before the detector
# completed (job 543389419, 2026-11-11).
# Record a version only after the whole trigger matrix is clean on that architecture; never infer a bound from another
# architecture or from planning alone, since the crash can happen during check_support or build_plans.
#
# Re-measure procedure (each per-CC bound is kept current by a detector rather than by memory):
# test/python/sdpa/graph/test_mhas_v2.py::test_sdpa_mxfp8_cc100_backend_planning_crash_guard_is_current_L0 walks, on a
# cc 10.0 / 10.3 device, the 21-contract trigger matrix (_P1_CRASH_MATRIX: dense BSHD / BHSD and THD at s_q == 1, Stats on
# and off, batch 1 / 2 / 4, KV 128 / 2048 / 4096, no mask / causal top-left / bottom-right, bf16 / f16 / E4M3 / E5M2 O,
# E4M3 and E5M2 inputs, d128 / d192x128 / d256 / d512; plus the sink contract as the control the backend plans and
# builds) in a child interpreter with all bounds set to 0 inside it -- the guard disabled, the installed backend
# asked through the ordinary [A, FALLBACK] walk, then, with the FROST row barred, its own plan taken through
# check_support and build_plans (nothing executes) -- restarting the child after every crash so every contract reports
# its stage (crash@plan / crash@support / crash@build, built, planned-but-declined, declined). While a device's bound is
# None, at least one contract must take its child down; a backend that gets every contract through planning and building
# cleanly FAILS that device's detector with the version to record. At or above a recorded per-device version no contract
# may crash, and the per-contract record is printed under "measured" in the run's terminal summary
# (the CI job log). Record a version only after the whole matrix is clean on that device, and leave the graphs the row
# serves on the row: these bounds lift the GUARD, not the placement verdict (``placement._place_sm107_mxfp8``).
SQ1_MXFP8_PLANNING_CRASH_FIXED_IN: Dict[Tuple[int, int], Optional[int]] = {
    (10, 0): 93000,
    (10, 3): None,
    (10, 7): None,
}

# The devices the crash is measured on.  cc 10.0 and 10.3 build the plan through the SM100 decode emitter and segfault in
# it (emit_code_native_sdpa_decode_sm100, which has no MXFP8 scale-factor path) for a dense d128 graph on every build
# tried: 9.23.1.3, 9.26.0.51, 9.27.0.42, 9.28.0.38 and the 9.30.0.2 nightly (148-SM cc 10.0, cc 10.3 B300, 2026-11-10).
# Record and lift the guard independently for each architecture after its detector completes cleanly.
SQ1_MXFP8_CRASH_DEVICES = frozenset({(10, 0), (10, 3), (10, 7)})


def backend_guard(graph, facts) -> Optional[str]:
    """The reason the backend must not be consulted for ``graph`` (its SDPA facts), or None."""
    import cudnn

    if facts is None or facts.invalid is not None or facts.is_backward or not facts.is_mxfp8:
        return None
    if facts.device_cc not in SQ1_MXFP8_CRASH_DEVICES or facts.s_q != 1 or facts.has_sink or facts.has_paged_kv:
        return None
    fixed_in = SQ1_MXFP8_PLANNING_CRASH_FIXED_IN.get(facts.device_cc)
    if fixed_in is not None and cudnn.backend_version() >= fixed_in:
        return None
    return (
        f"cuDNN {cudnn.backend_version_string()} backend crashes the process while planning or building single-query MXFP8 SDPA graphs "
        f"on cc 10.0 / 10.3 / 10.7 without a sink token (cc 10.7: heuristics on 9.26.0.51 and 9.27.0.28, plan build on 9.28.0; "
        f"cc 10.0 / 10.3: plan build on 9.23.1 through 9.30; dense and THD, Stats on or off); "
        f"the backend is not consulted for this graph"
    )


# The native SM100 / SM107 forward engines (global ids 10 and 18) build O from the K buffer when d_qk <= 128 < d_v: their
# P.V matmul's V descriptor names smem_K while V is staged in smem_V, and nothing raises (NVIDIA/cudnn-frontend#2401).
# Measured on cc 10.0, dense causal 128/256, GQA 16/4, S 1024, the backend's default plan against torch's fp32 SDPA: max
# |O - ref| 4.60 on 9.23.1.3, 9.24.0.43, 9.25.1.1 and 9.27.0.42, 0.0086 on 9.18.1.3, 9.21.1.3, 9.22.0.52 and 9.23.0.39.
# On 9.30.0.2 engines 10 and 18 are off by 0.49-0.67 against engine 8 on cc 10.0 and cc 10.7 alike, for dense 128/256 and
# 64/256 and paged 128/256 (cc 10.7 against torch: 4.88).  The backend's SM80 engine (8) is right, so these graphs keep
# the backend and lose engines 10 and 18 only.
WIDE_V_WRONG_O_SINCE = 92301
# First backend version shown to return the right O on these engines, or None while every build since 9.23.1 is wrong.
# A measurement, re-taken by test_sdpa_backend_guard.py::test_wide_v_engine_guard_is_current on a cc 10.x device (the
# engines run with the guard lifted); record a version only from that detector, never a guessed release.
WIDE_V_WRONG_O_FIXED_IN: Optional[int] = None
WIDE_V_WRONG_O_ENGINES = (10, 18)


def backend_engine_guard(graph, facts) -> Dict[int, str]:
    """Backend engines (global ids) that must not run ``graph`` (its SDPA facts), each with the reason."""
    import cudnn

    if facts is None or facts.invalid is not None or facts.is_backward:
        return {}
    if not facts.device_cc or facts.device_cc[0] != 10 or not facts.d_qk or not facts.d_v or not facts.d_qk <= 128 < facts.d_v:
        return {}
    version = cudnn.backend_version()
    if version < WIDE_V_WRONG_O_SINCE or (WIDE_V_WRONG_O_FIXED_IN is not None and version >= WIDE_V_WRONG_O_FIXED_IN):
        return {}
    return {
        engine: f"cuDNN {cudnn.backend_version_string()} engine {engine} returns a wrong O for SDPA forward with d_qk <= 128 < d_v on cc 10.x "
        f"(NVIDIA/cudnn-frontend#2401); the backend's other engines serve this graph"
        for engine in WIDE_V_WRONG_O_ENGINES
    }
