# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed shared body lowered through injected graph services."""

from .contract import NAME, explicit_geometry

PACKED_HN_STATS_OVERRIDE = True


def lower(spec, facts, knobs, *, lower_impl, knob_type):
    knobs = knobs or knob_type()
    geometry = explicit_geometry(NAME, facts, knobs)
    return lower_impl(spec, facts, geometry.canonical_knobs(knobs), selected_kernel=geometry)
