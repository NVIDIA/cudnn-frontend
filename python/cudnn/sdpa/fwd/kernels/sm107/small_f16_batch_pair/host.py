# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lower a fixed work item through the injected shared adapter."""

from .contract import NAME
from ..small_half_contract import explicit_geometry

PACKED_HN_STATS_OVERRIDE = False


def lower(spec, facts, knobs, *, lower_impl, knob_type, shape_of=None):
    knobs = knobs or knob_type()
    shape = shape_of(spec.capabilities, facts, knobs) if shape_of is not None else (facts.d_qk, facts.d_v)
    geometry = explicit_geometry(NAME, facts, knobs, d_flavor=shape[0])
    return lower_impl(spec, facts, geometry.canonical_knobs(knobs), selected_kernel=geometry)
