# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KDA family heuristics: rank the offered engines for one graph.

Without a hook the family falls back to slot order, and slot order is identity,
not preference -- a slot is fixed forever because an autotune result is
``(engine_id, knobs)``. ``kda_cutile`` holds slot 1 because it shipped first,
so on Hopper it won every graph by default even though the two sm90 engines,
added later at slots 4 and 5, are measurably faster there.

This hook is ORDERING ONLY. It does not decide eligibility: it cannot, because
the hook signature sees facts and engine names but not the graph, and the
decisive conditions for the sm90 engines (NVRTC and a CUDA include tree, or the
CuTeDSL version) are environmental rather than facts. It does not need to.
``build_plans()`` walks the ranked list and a decline advances the walk, so an
engine ranked first that cannot serve this graph falls through to the next
entry; ranking can cost an extra probe, never a wrong answer.

The order outside Hopper is exactly the slot order this replaced, so no
non-Hopper graph changes engine.
"""

from __future__ import annotations

from typing import Dict, List

from cudnn.engines.base import PlanConfig

HOPPER_SM = 90

# Hopper preference, fastest first. Measured on H100 80GB HBM3 (SXM) at the
# production gate over ten shapes with FlashKDA as an in-run control:
# kda_hopper_cuda 2.09x FlashKDA, kda_hopper 1.19x, kda_cutile 0.61x. The
# backward ordering agrees and is a larger gap (kda_hopper_cuda is 2.82x
# kda_cutile, and is the only sm90 backward kernel), so one order serves both
# directions.
_HOPPER_FIRST = ("kda_hopper_cuda", "kda_hopper")


def _current_sm() -> int:
    """The running device's SM version, or 0 when it cannot be determined.

    0 falls through to slot order, which is the pre-hook behaviour -- the right
    answer when there is no device to prefer an engine for.
    """
    try:
        from cudnn.frost import buffers

        return int(buffers.current_sm())
    except Exception:  # noqa: BLE001 -- no device, no driver, no preference
        return 0


def _ordered_names(offered: Dict[str, int]) -> List[str]:
    """Offered engine names, best first."""
    # Slot order is the order `offered` already arrives in (manifest.offered_ids
    # builds it from the slots mapping), so preserving it for the tail keeps
    # every engine this hook has no opinion about exactly where it was.
    names = list(offered)
    if _current_sm() != HOPPER_SM:
        return names
    promoted = [n for n in _HOPPER_FIRST if n in offered]
    return promoted + [n for n in names if n not in promoted]


def recommend(kind: str, facts, offered: Dict[str, int]) -> List[PlanConfig]:
    """One default plan per offered engine, best first.

    ``kind`` does not change the answer: these engines expose no knobs, so there
    is no wider search to give for ``"A"`` and no separate ``"FALLBACK"``
    configuration -- the later entries in the list already are the fallback.
    """
    return [PlanConfig(offered[name], None) for name in _ordered_names(offered)]
