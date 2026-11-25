# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed implementation lowered through common graph services."""

from .contract import Geometry

PACKED_HN_STATS_OVERRIDE = True


def lower(spec, facts, knobs, *, lower_impl, knob_type):
    knobs = knobs or knob_type()
    geometry = Geometry(cga=2 if knobs.cga is None else knobs.cga)
    return lower_impl(spec, facts, geometry.canonical_knobs(knobs), selected_kernel=geometry)
