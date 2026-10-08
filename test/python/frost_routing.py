# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-side tally of which backend served each SDPA test graph.

We are transitioning ops from the native cuDNN backend to FROST engines; the
end state is every graph on FROST. The sdpa test harness records one entry per
graph after build_plans — i.e. after FROST auto-selection has resolved —
("frost:<engine name>" when a FROST engine serves the graph,
"native:<harness site>" otherwise), and conftest.py prints the
aggregated tally at the end of the run — so the remaining native population
stays visible per op family without touching the cudnn package itself.
"""

COUNTS: "dict[str, int]" = {}

# The most recently tallied graph: ``(engine name or None, its PlanConfig.knobs
# or None)``. Lets a test assert WHICH plan of an engine served (a kernel
# flavor selected by a knob value), where COUNTS only says which engine.
LAST_PLAN: "tuple" = (None, None)
# The softmax arms the most recently tallied FROST plan compiled (``softmax_arms`` on the compiled
# executor: "f32" / "f16" plus "+fold" / "+fused"), or None for the backend / an engine that does not
# record them.  Lets a sweep assert that the lever it requested is the arm that ran.
LAST_ARMS: "str | None" = None


def note(key: str) -> None:
    COUNTS[key] = COUNTS.get(key, 0) + 1


def snapshot() -> "dict[str, int]":
    return dict(COUNTS)


def reset() -> None:
    COUNTS.clear()
