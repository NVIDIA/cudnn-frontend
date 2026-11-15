# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed SM90 body over the existing graph binding and launch services."""

from dataclasses import replace
from .contract import Geometry


def lower(spec, facts, knobs, *, lower_impl, knob_type):
    # An omitted scheduler keeps the existing adapter default. This resolves a
    # tuning default only; the implementation is already fixed by this host.
    knobs = knobs or knob_type()
    if knobs.sched_policy is None:
        from cudnn.sdpa.fwd.api_dsl import _causal_sched_policy

        scheduler = _causal_sched_policy(facts.s_kv, facts.d_qk, facts.d_v, 2) if not facts.thd and (facts.causal or facts.right_band_widening) else 0
        knobs = replace(knobs, sched_policy=scheduler)
    geometry = Geometry()
    return lower_impl(spec, facts, geometry.canonical_knobs(knobs), selected_kernel=geometry)
